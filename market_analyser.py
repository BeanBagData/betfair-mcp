"""
Market Analyser
Implements quantitative market analysis derived from the #theyknow Betfair
notebook, providing tick-aware price arithmetic, weight-of-money scoring,
market-spread measurement, and real-time steam/plunge detection.

All functions here are pure (no I/O) so they can be unit-tested and used
by both the MCP server and the Gemini agent.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  BETFAIR TICK LADDER
# ─────────────────────────────────────────────────────────────────────────────

def build_tick_ladder() -> list[float]:
    """
    Build the complete Betfair price ladder as a list of valid tick prices.
    Matches the exchange ladder exactly (1.01 → 1000 in variable increments).
    """
    increments = [
        (1.0,   2.0,   0.01),
        (2.0,   3.0,   0.02),
        (3.0,   4.0,   0.05),
        (4.0,   6.0,   0.1),
        (6.0,   10.0,  0.2),
        (10.0,  20.0,  0.5),
        (20.0,  30.0,  1.0),
        (30.0,  50.0,  2.0),
        (50.0,  100.0, 5.0),
        (100.0, 1000.0, 10.0),
    ]
    ladder: list[float] = []
    for lo, hi, step in increments:
        price = lo
        while price < hi:
            ladder.append(round(price, 2))
            price = round(price + step, 2)
    ladder.append(1000.0)
    return ladder


# Build once at module load and reuse.
_TICK_LADDER: list[float] = build_tick_ladder()
_TICK_INDEX: dict[float, int] = {p: i for i, p in enumerate(_TICK_LADDER)}


def tick_floor_index(price: float) -> tuple[int, float]:
    """
    Return (index, floored_tick) for *price* on the Betfair ladder.
    Prices that sit exactly on a tick are returned as-is.
    """
    price = round(price, 2)
    # Fast path: price is already a valid tick
    if price in _TICK_INDEX:
        return _TICK_INDEX[price], price
    # Binary search for the largest tick ≤ price
    lo, hi = 0, len(_TICK_LADDER) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _TICK_LADDER[mid] <= price:
            lo = mid
        else:
            hi = mid - 1
    return lo, _TICK_LADDER[lo]


def tick_delta(price_a: float, price_b: float) -> int:
    """
    Number of Betfair ticks between two prices.
    Positive when price_a > price_b (price drifted out).
    Negative when price_a < price_b (price steamed in).
    Returns 0 if either price is NaN/None.
    """
    if price_a is None or price_b is None:
        return 0
    try:
        idx_a, _ = tick_floor_index(float(price_a))
        idx_b, _ = tick_floor_index(float(price_b))
        return idx_a - idx_b
    except Exception:
        return 0


def nearest_tick(price: float) -> float:
    """Round price to the nearest valid Betfair tick."""
    _, floored = tick_floor_index(price)
    idx = _TICK_INDEX.get(floored, 0)
    # Check if the next tick is closer
    if idx + 1 < len(_TICK_LADDER):
        next_tick = _TICK_LADDER[idx + 1]
        if abs(price - next_tick) < abs(price - floored):
            return next_tick
    return floored


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LADDER ANALYSIS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ladder_from_api(rungs: list[dict]) -> tuple[list[float], list[float]]:
    """
    Convert Betfair API ladder (list of {price, size} dicts) into two
    parallel lists (prices, sizes).  Returns empty lists on bad input.
    """
    prices, sizes = [], []
    for rung in (rungs or []):
        try:
            prices.append(float(rung["price"]))
            sizes.append(float(rung["size"]))
        except (KeyError, TypeError, ValueError):
            continue
    return prices, sizes


def weight_of_money(
    atb_rungs: list[dict],
    atl_rungs: list[dict],
    depth: int = 5,
) -> dict:
    """
    Weight-of-money: fraction of top-N ladder liquidity on each side.

    Returns:
        {
          "back_wom":  0.62,   # 0-1 fraction; >0.5 → more money backing
          "lay_wom":   0.38,   # 0-1 fraction
          "back_volume": 4200, # absolute AUD in top-N back rungs
          "lay_volume":  2600,
          "total_volume": 6800,
          "signal": "BACK_HEAVY" | "LAY_HEAVY" | "BALANCED"
        }
    """
    back_prices, back_sizes = _ladder_from_api(atb_rungs[:depth])
    lay_prices,  lay_sizes  = _ladder_from_api(atl_rungs[:depth])

    back_vol  = sum(back_sizes)
    lay_vol   = sum(lay_sizes)
    total_vol = back_vol + lay_vol

    if total_vol == 0:
        return {
            "back_wom": 0.5, "lay_wom": 0.5,
            "back_volume": 0, "lay_volume": 0,
            "total_volume": 0, "signal": "NO_LIQUIDITY",
        }

    bwom = round(back_vol / total_vol, 4)
    lwom = round(lay_vol  / total_vol, 4)

    if bwom >= 0.60:
        signal = "BACK_HEAVY"       # market expects the horse to win
    elif lwom >= 0.60:
        signal = "LAY_HEAVY"        # market expects the horse to lose
    else:
        signal = "BALANCED"

    return {
        "back_wom":     bwom,
        "lay_wom":      lwom,
        "back_volume":  round(back_vol,  2),
        "lay_volume":   round(lay_vol,   2),
        "total_volume": round(total_vol, 2),
        "signal":       signal,
    }


def market_spread(
    best_back_price: Optional[float],
    best_lay_price:  Optional[float],
) -> dict:
    """
    Tick spread between best back and best lay for a runner.

    A tight spread (0-2 ticks) means the market is liquid and efficient.
    A wide spread (>10 ticks) means the market is illiquid — beware.

    Returns:
        {
          "spread_ticks": 3,
          "best_back": 4.0,
          "best_lay": 4.3,
          "assessment": "TIGHT" | "NORMAL" | "WIDE" | "NO_MARKET"
        }
    """
    if best_back_price is None or best_lay_price is None:
        return {
            "spread_ticks": None,
            "best_back": best_back_price,
            "best_lay": best_lay_price,
            "assessment": "NO_MARKET",
        }

    ticks = tick_delta(best_lay_price, best_back_price)

    if ticks <= 2:
        assessment = "TIGHT"
    elif ticks <= 8:
        assessment = "NORMAL"
    else:
        assessment = "WIDE"

    return {
        "spread_ticks": ticks,
        "best_back":    best_back_price,
        "best_lay":     best_lay_price,
        "assessment":   assessment,
    }


def vwap(rungs: list[dict], depth: int = 3) -> Optional[float]:
    """
    Volume-weighted average price across the top *depth* ladder rungs.
    Returns None if no liquidity.
    """
    prices, sizes = _ladder_from_api(rungs[:depth])
    total_vol = sum(sizes)
    if total_vol == 0:
        return None
    return round(sum(p * s for p, s in zip(prices, sizes)) / total_vol, 3)


def top_box_support(
    best_price: Optional[float],
    vwap_price: Optional[float],
    side: str = "back",
) -> Optional[float]:
    """
    Ratio of best price to VWAP, indicating how well the top-of-book is
    supported relative to the rest of the top-3 ladder.

    Back side:  ratio > 1 → top box undercut by deeper money (weak support)
                ratio ≈ 1 → market in agreement with best price
    Lay side:   ratio < 1 → best lay undercut by deeper offers (strong lay wall)
    """
    if best_price is None or vwap_price is None or vwap_price == 0:
        return None
    if side == "back":
        return round(vwap_price / best_price, 4)
    else:  # lay
        return round(best_price / vwap_price, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  VENUE INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────

# Average total volume traded per race by venue (approximate AUD, from notebook)
VENUE_PROFILE: dict[str, dict] = {
    "Flemington":   {"avg_volume": 480_000, "tier": "PREMIUM"},
    "Caulfield":    {"avg_volume": 350_000, "tier": "PREMIUM"},
    "Moonee Valley":{"avg_volume": 280_000, "tier": "MAJOR"},
    "Randwick":     {"avg_volume": 400_000, "tier": "PREMIUM"},
    "Rosehill":     {"avg_volume": 300_000, "tier": "MAJOR"},
    "Sandown":      {"avg_volume": 200_000, "tier": "MAJOR"},
    "Bendigo":      {"avg_volume": 120_000, "tier": "PROVINCIAL"},
    "Ballarat":     {"avg_volume": 100_000, "tier": "PROVINCIAL"},
    "Geelong":      {"avg_volume":  90_000, "tier": "PROVINCIAL"},
}

def venue_profile(venue_name: str) -> dict:
    """Return trading profile for a venue, with UNKNOWN fallback."""
    name = venue_name.strip().title()
    profile = VENUE_PROFILE.get(name, {"avg_volume": 80_000, "tier": "UNKNOWN"})
    return {"venue": venue_name, **profile}


def timing_advice(seconds_to_jump: float, venue: str = "") -> dict:
    """
    Based on the notebook's finding that ~70% of volume trades in the last
    5 minutes, advise on bet timing.

    Returns:
        {
          "seconds_to_jump": int,
          "window": "TOO_EARLY" | "MONITOR" | "OPTIMAL" | "LAST_CHANCE" | "INPLAY",
          "advice": str
        }
    """
    s = int(seconds_to_jump)
    if s > 1800:
        window, advice = "TOO_EARLY",    "Market hasn't formed yet. Wait until 30 min before jump."
    elif s > 600:
        window, advice = "MONITOR",      "Early window. Prices valid but sharp money hasn't arrived. Monitor for steam."
    elif s > 120:
        window, advice = "OPTIMAL",      "Prime window: 2–10 min before jump. Sharp money active. Best time to lay."
    elif s > 0:
        window, advice = "LAST_CHANCE",  "Under 2 min. Spreads tighten dramatically. Act now or skip."
    else:
        window, advice = "INPLAY",       "Race has started. Pre-race lay betting closed."

    return {
        "seconds_to_jump": s,
        "minutes_to_jump": round(s / 60, 1),
        "window":          window,
        "advice":          advice,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  STEAM / PLUNGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PriceSample:
    """A single price observation for a runner."""
    timestamp:  float          # Unix time
    best_back:  Optional[float]
    best_lay:   Optional[float]
    traded_vol: float = 0.0


@dataclass
class RunnerTracker:
    """
    Rolling price history for a single runner.
    Keeps the last *maxlen* samples (default: 120 = 2 min @ 1 Hz).
    """
    selection_id: int
    runner_name:  str = "Unknown"
    history:      deque = field(default_factory=lambda: deque(maxlen=120))

    def add(self, sample: PriceSample):
        self.history.append(sample)

    def ticks_moved(self, window_seconds: float = 30.0) -> Optional[int]:
        """
        Net tick movement (negative = steamed in / firmed) over the last
        *window_seconds* seconds.  Returns None if insufficient history.
        """
        now = time.time()
        cutoff = now - window_seconds
        # Most recent sample
        if not self.history:
            return None
        latest = self.history[-1]
        # Find the oldest sample within the window
        oldest = None
        for s in self.history:
            if s.timestamp >= cutoff:
                oldest = s
                break
        if oldest is None or oldest is latest:
            return None
        if latest.best_back is None or oldest.best_back is None:
            return None
        return tick_delta(latest.best_back, oldest.best_back)

    def volume_delta(self, window_seconds: float = 30.0) -> float:
        """Volume traded in the last *window_seconds*."""
        now = time.time()
        cutoff = now - window_seconds
        samples_in_window = [s for s in self.history if s.timestamp >= cutoff]
        if len(samples_in_window) < 2:
            return 0.0
        return samples_in_window[-1].traded_vol - samples_in_window[0].traded_vol


class SteamDetector:
    """
    Monitors one market, tracking price history for every runner.
    Call `update(market_book_data)` each time you fetch a fresh market book.
    Call `scan()` to get a list of active steam/drift signals.

    Steam  = horse firming quickly (negative ticks_moved) → potential lay target
    Drift  = horse drifting out  (positive ticks_moved)   → avoid laying
    """

    # Thresholds (configurable)
    STEAM_TICKS_THRESHOLD: int   = 5    # ≥5 ticks in in 30s = steam signal
    DRIFT_TICKS_THRESHOLD: int   = 5    # ≥5 ticks out = drift signal
    VOLUME_SURGE_FACTOR:   float = 2.5  # traded vol 2.5× recent average = surge

    def __init__(self, market_id: str):
        self.market_id = market_id
        self._runners: dict[int, RunnerTracker] = {}
        self._created_at = time.time()

    def update(self, market_book: dict):
        """
        Ingest a fresh market book response (from BetfairClient.get_market_book).
        """
        now = time.time()
        for runner in market_book.get("runners", []):
            sel_id = runner.get("selection_id")
            if sel_id is None:
                continue

            if sel_id not in self._runners:
                self._runners[sel_id] = RunnerTracker(
                    selection_id=sel_id,
                    runner_name=runner.get("runner_name", "Unknown"),
                )

            best_back = (runner.get("best_back") or {}).get("price")
            best_lay  = (runner.get("best_lay")  or {}).get("price")
            traded    = runner.get("total_matched", 0) or 0

            self._runners[sel_id].add(PriceSample(
                timestamp=now,
                best_back=best_back,
                best_lay=best_lay,
                traded_vol=float(traded),
            ))

    def scan(self, window_seconds: float = 30.0) -> list[dict]:
        """
        Return a list of signal dicts for runners showing notable movement.

        Each dict:
        {
          "selection_id": int,
          "runner_name": str,
          "signal": "STEAM" | "DRIFT" | "VOLUME_SURGE",
          "ticks_moved": int,          # negative = in, positive = out
          "current_back": float,
          "current_lay": float,
          "volume_delta_30s": float,
          "severity": "MILD" | "STRONG" | "EXTREME"
        }
        """
        signals = []
        for sel_id, tracker in self._runners.items():
            if not tracker.history:
                continue

            latest    = tracker.history[-1]
            ticks     = tracker.ticks_moved(window_seconds)
            vol_delta = tracker.volume_delta(window_seconds)

            if ticks is None:
                continue

            # Steam: price firming (ticks < -threshold)
            if ticks <= -self.STEAM_TICKS_THRESHOLD:
                abs_ticks = abs(ticks)
                severity  = "EXTREME" if abs_ticks >= 20 else "STRONG" if abs_ticks >= 10 else "MILD"
                signals.append({
                    "selection_id":    sel_id,
                    "runner_name":     tracker.runner_name,
                    "signal":          "STEAM",
                    "ticks_moved":     ticks,
                    "current_back":    latest.best_back,
                    "current_lay":     latest.best_lay,
                    "volume_delta_30s": round(vol_delta, 2),
                    "severity":        severity,
                    "interpretation":  (
                        f"{tracker.runner_name} has steamed {abs_ticks} ticks shorter "
                        f"in {int(window_seconds)}s. Potential lay opportunity if price "
                        "still above value — but sharp money is backing it. Be cautious."
                    ),
                })

            # Drift: price drifting (ticks > +threshold)
            elif ticks >= self.DRIFT_TICKS_THRESHOLD:
                severity = "EXTREME" if ticks >= 20 else "STRONG" if ticks >= 10 else "MILD"
                signals.append({
                    "selection_id":    sel_id,
                    "runner_name":     tracker.runner_name,
                    "signal":          "DRIFT",
                    "ticks_moved":     ticks,
                    "current_back":    latest.best_back,
                    "current_lay":     latest.best_lay,
                    "volume_delta_30s": round(vol_delta, 2),
                    "severity":        severity,
                    "interpretation":  (
                        f"{tracker.runner_name} has drifted {ticks} ticks longer "
                        f"in {int(window_seconds)}s. Classic lay candidate if fundamentals agree."
                    ),
                })

        # Sort by absolute magnitude
        signals.sort(key=lambda x: abs(x["ticks_moved"]), reverse=True)
        return signals

    def snapshot(self) -> dict:
        """Return a summary of all tracked runners with latest prices."""
        runners = []
        for sel_id, tracker in self._runners.items():
            if not tracker.history:
                continue
            latest = tracker.history[-1]
            ticks  = tracker.ticks_moved(30)
            runners.append({
                "selection_id": sel_id,
                "runner_name":  tracker.runner_name,
                "best_back":    latest.best_back,
                "best_lay":     latest.best_lay,
                "ticks_30s":    ticks,
                "samples":      len(tracker.history),
            })
        runners.sort(key=lambda r: r.get("best_back") or 9999)
        return {
            "market_id":    self.market_id,
            "tracking_secs": round(time.time() - self._created_at, 0),
            "runners":      runners,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMPREHENSIVE MARKET ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_market(
    market_book: dict,
    full_depth_book: Optional[dict] = None,
    seconds_to_jump: Optional[float] = None,
    venue: str = "",
) -> dict:
    """
    Produce a consolidated market analysis from one or two market book responses.

    Args:
        market_book:       Standard get_market_book response (3-rung depth).
        full_depth_book:   Optional get_market_depth response (10-rung depth).
                           If provided, WOM is calculated from this richer data.
        seconds_to_jump:   Seconds until scheduled race start (None = unknown).
        venue:             Venue name for profile lookup.

    Returns a dict with per-runner analysis plus market-level metrics.
    """
    runners_out = []
    depth_map   = {}

    # Build a map of selection_id → full depth rungs if available
    if full_depth_book:
        for r in full_depth_book.get("runners", []):
            depth_map[r["selection_id"]] = r

    for runner in market_book.get("runners", []):
        sel_id      = runner.get("selection_id")
        best_back_p = (runner.get("best_back") or {}).get("price")
        best_lay_p  = (runner.get("best_lay")  or {}).get("price")
        ltp         = runner.get("last_price_traded")

        # Use full depth data if available, otherwise fall back to 3-rung data
        depth_runner = depth_map.get(sel_id, runner)
        atb_rungs    = depth_runner.get("back_prices", runner.get("back_prices", []))
        atl_rungs    = depth_runner.get("lay_prices",  runner.get("lay_prices",  []))

        # Core metrics
        wom      = weight_of_money(atb_rungs, atl_rungs, depth=min(5, len(atb_rungs) or 3))
        spread   = market_spread(best_back_p, best_lay_p)
        back_vwap = vwap(atb_rungs, depth=3)
        lay_vwap  = vwap(atl_rungs, depth=3)
        back_support = top_box_support(best_back_p, back_vwap, side="back")
        lay_support  = top_box_support(best_lay_p,  lay_vwap,  side="lay")

        # Lay value assessment
        lay_analysis = runner.get("lay_analysis") or {}
        profit_ratio = lay_analysis.get("profit_ratio")

        # Composite lay recommendation
        lay_rec = _lay_recommendation(
            profit_ratio=profit_ratio,
            wom=wom,
            spread=spread,
            best_lay_price=best_lay_p,
        )

        runners_out.append({
            "selection_id":   sel_id,
            "runner_name":    runner.get("runner_name", "Unknown"),
            "status":         runner.get("status", "UNKNOWN"),
            "best_back":      best_back_p,
            "best_lay":       best_lay_p,
            "last_traded":    ltp,
            "total_matched":  runner.get("total_matched", 0),
            "profit_ratio":   profit_ratio,
            "weight_of_money": wom,
            "spread":          spread,
            "back_vwap":       back_vwap,
            "lay_vwap":        lay_vwap,
            "back_support":    back_support,
            "lay_support":     lay_support,
            "lay_recommendation": lay_rec,
        })

    # Market-level
    out = {
        "success":      True,
        "market_id":    market_book.get("market_id"),
        "status":       market_book.get("status"),
        "inplay":       market_book.get("inplay", False),
        "total_matched": market_book.get("total_matched", 0),
        "runners":      runners_out,
    }

    if venue:
        out["venue_profile"] = venue_profile(venue)

    if seconds_to_jump is not None:
        out["timing"] = timing_advice(seconds_to_jump, venue)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5b.  MARKET SUPPORT SIGNAL  (from #theyknow notebook — Angle 2)
# ─────────────────────────────────────────────────────────────────────────────

def market_support_signal(
    baseline_vwap: Optional[float],
    current_vwap:  Optional[float],
) -> dict:
    """
    Determine whether the market has shifted money onto (or off) a runner
    between two price snapshots.

    In the #theyknow notebook, "market support = Y" when wap_5m > wap_30s,
    meaning the horse shortened from the 5-minute mark to the 30-second mark
    (money came in to back it).

    Practical usage in the agent:
      1. Call get_weight_of_money early (e.g. 10+ min before jump) and store
         the back_vwap as the baseline.
      2. Call again close to jump and pass both values here.
      3. Use the "supported" flag to decide whether to lay or avoid.

    Args:
        baseline_vwap: Back VWAP from an earlier poll (higher = longer odds then)
        current_vwap:  Back VWAP from the most recent poll

    Returns:
        {
          "supported": bool,   # True = market backed horse (price shortened)
          "signal":    str,    # "SUPPORTED" | "DRIFTED" | "STABLE" | "UNKNOWN"
          "delta_ticks": int,  # Betfair ticks of movement (negative = shortened)
          "interpretation": str,
          "lay_implication": str,
        }
    """
    if baseline_vwap is None or current_vwap is None:
        return {
            "supported":      False,
            "signal":         "UNKNOWN",
            "delta_ticks":    0,
            "interpretation": "Insufficient data — only one VWAP reading available.",
            "lay_implication": "Cannot assess market support without a baseline reading.",
        }

    ticks = tick_delta(current_vwap, baseline_vwap)  # positive = drifted, negative = steamed

    if ticks <= -3:
        # Price shortened ≥3 ticks — market actively backed this horse
        return {
            "supported":      True,
            "signal":         "SUPPORTED",
            "delta_ticks":    ticks,
            "interpretation": (
                f"Horse shortened {abs(ticks)} ticks since baseline "
                f"({baseline_vwap} → {current_vwap}). "
                "Smart money has backed this horse."
            ),
            "lay_implication": (
                "CAUTION: Market support makes this a risky lay. "
                "Only lay if SP predictions show strong positive EV "
                "AND profit_ratio ≥ 1.5."
            ),
        }
    elif ticks >= 3:
        # Price drifted ≥3 ticks — market has lost confidence
        return {
            "supported":      False,
            "signal":         "DRIFTED",
            "delta_ticks":    ticks,
            "interpretation": (
                f"Horse drifted {ticks} ticks longer since baseline "
                f"({baseline_vwap} → {current_vwap}). "
                "Market is losing confidence in this runner."
            ),
            "lay_implication": (
                "POSITIVE: Drift is a classic lay signal. "
                "Drifting horses win less often than their current price implies. "
                "Reinforce with WOM and profit_ratio check."
            ),
        }
    else:
        return {
            "supported":      False,
            "signal":         "STABLE",
            "delta_ticks":    ticks,
            "interpretation": (
                f"Price movement minimal ({ticks:+d} ticks, "
                f"{baseline_vwap} → {current_vwap}). Market undecided."
            ),
            "lay_implication": "Neutral. Rely on WOM, profit_ratio, and SP predictions.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5c.  LAY CONTEXT HEURISTICS  (from #theyknow notebook — Angles 1 & 3)
# ─────────────────────────────────────────────────────────────────────────────

# Notebook Angle 1 finding: track/distance/barrier combinations with statistically
# significant lay profit in backtesting. Encoded as qualitative score adjustments.
# Higher is better for laying.

_BARRIER_LAY_SCORE: dict[str, int] = {
    # Inside draws are generally the shortest-priced runners;
    # they win more → RISKY to lay unless price is compelling
    "inside":    -5,
    "mid_field":  0,   # neutral
    "outside":   +8,   # wide runners win less often → better lay candidates
    "unknown":    0,
}

_DISTANCE_LAY_SCORE: dict[str, int] = {
    # Sprint races are more volatile (barrier matters more, jockey less)
    "sprint":    +5,
    "mid_short":  0,
    "mid_long":  -3,   # longer races — form holds better → riskier lay on fav
    "long":      -5,
    "unknown":    0,
}

_TRACK_CONDITION_LAY_SCORE: dict[str, int] = {
    # Heavy/Soft tracks create more upsets → better lay environment overall
    "Good":   0,
    "Good3":  0,
    "Good4":  0,
    "Soft":  +5,
    "Soft5": +5,
    "Soft6": +8,
    "Heavy": +10,
    "Heavy8":+10,
    "Heavy9":+12,
    "Synthetic": -3,   # Synthetic surfaces favour short-priced horses
}


def lay_context_hints(
    barrier_group:    Optional[str] = None,
    distance_group:   Optional[str] = None,
    track_condition:  Optional[str] = None,
    jockey:           Optional[str] = None,
    market_supported: Optional[bool] = None,
) -> dict:
    """
    Produce a qualitative lay context score from race metadata.

    Derived from the #theyknow notebook Angle 1 (track/distance/barrier) and
    Angle 2 (jockey × market support) analysis.

    Args:
        barrier_group:    "inside" | "mid_field" | "outside" | "unknown"
        distance_group:   "sprint" | "mid_short" | "mid_long" | "long" | "unknown"
        track_condition:  Raw track condition string from race metadata
        jockey:           Jockey name string (used for narrative; no lookup table
                          in live mode since model needs historical data)
        market_supported: From market_support_signal() — True = market backed horse

    Returns:
        {
          "context_score":  int,   # additive; positive = better lay candidate
          "barrier_score":  int,
          "distance_score": int,
          "condition_score": int,
          "support_score":  int,
          "insights":       [str], # bullet-point reasoning
          "context_verdict": "FAVOURABLE" | "NEUTRAL" | "UNFAVOURABLE"
        }
    """
    insights = []
    barrier_sc   = _BARRIER_LAY_SCORE.get(barrier_group or "unknown", 0)
    distance_sc  = _DISTANCE_LAY_SCORE.get(distance_group or "unknown", 0)

    # Fuzzy match track condition (e.g. "Good (4)" → "Good4")
    cond_key = (track_condition or "").replace(" ", "").replace("(", "").replace(")", "")
    cond_sc  = _TRACK_CONDITION_LAY_SCORE.get(cond_key,
               _TRACK_CONDITION_LAY_SCORE.get(cond_key[:4], 0))  # try 4-char prefix

    support_sc = 0
    if market_supported is True:
        support_sc = -10
        insights.append("Market backed this horse (SUPPORTED signal) → caution, risky lay")
    elif market_supported is False:
        support_sc = +8
        insights.append("Market did NOT support this horse (DRIFTED/STABLE) → lay-friendly")

    if barrier_group == "outside":
        insights.append(f"Wide draw ({barrier_group}) → horse must use extra energy; outsiders win less")
    elif barrier_group == "inside":
        insights.append("Inside draw → favoured in short races; riskier lay if sprint")

    if distance_group == "sprint":
        insights.append("Sprint distance: barriers and luck-in-running matter more; more variance")
    elif distance_group in ("mid_long", "long"):
        insights.append("Longer race: form and fitness more decisive; shorter prices are more reliable")

    if cond_sc >= 8:
        insights.append(f"Heavy/Soft track ({track_condition}) → upsets more likely, good lay environment")
    elif cond_sc <= -3:
        insights.append(f"Synthetic surface ({track_condition}) → favours consistent performers; riskier lay")

    if jockey:
        insights.append(
            f"Jockey: {jockey} — cross-reference against historical market-support profitability "
            "if backtested data is available (Angle 2 from notebook)"
        )

    total = barrier_sc + distance_sc + cond_sc + support_sc

    if total >= 10:
        verdict = "FAVOURABLE"
    elif total >= 0:
        verdict = "NEUTRAL"
    else:
        verdict = "UNFAVOURABLE"

    return {
        "context_score":   total,
        "barrier_score":   barrier_sc,
        "distance_score":  distance_sc,
        "condition_score": cond_sc,
        "support_score":   support_sc,
        "insights":        insights,
        "context_verdict": verdict,
    }


def _lay_recommendation(
    profit_ratio: Optional[float],
    wom: dict,
    spread: dict,
    best_lay_price: Optional[float],
) -> dict:
    """
    Produce a structured lay recommendation combining all signals.
    """
    issues = []
    score  = 0  # higher = better lay candidate

    # 1. Profit ratio gate (hard rule from original agent)
    if profit_ratio is None:
        issues.append("No lay price available")
        return {"verdict": "NO_BET", "score": 0, "issues": issues}

    if profit_ratio >= 1.5:
        score += 40
    elif profit_ratio >= 1.0:
        score += 15
        issues.append(f"Profit ratio {profit_ratio:.2f} < 1.5 threshold")
    else:
        score -= 20
        issues.append(f"Profit ratio {profit_ratio:.2f} is too low — liability exceeds potential win")

    # 2. Weight of money
    wom_signal = wom.get("signal", "BALANCED")
    if wom_signal == "LAY_HEAVY":
        score += 20  # market money agrees: horse expected to lose
    elif wom_signal == "BALANCED":
        score += 5
    else:  # BACK_HEAVY — market expects the horse to win → risky lay
        score -= 15
        issues.append("WOM is BACK_HEAVY — market participants are backing this horse")

    # 3. Market spread
    spread_assess = spread.get("assessment", "NO_MARKET")
    if spread_assess == "TIGHT":
        score += 20  # liquid market
    elif spread_assess == "NORMAL":
        score += 10
    elif spread_assess == "WIDE":
        score -= 10
        issues.append("Wide spread — illiquid market, bet may not fill")
    elif spread_assess == "NO_MARKET":
        score -= 30
        issues.append("No back/lay prices available")

    # 4. Price sanity
    if best_lay_price and best_lay_price > 10:
        issues.append(f"High lay price {best_lay_price} — high liability risk")

    # Verdict
    if score >= 55:
        verdict = "STRONG_BET"
    elif score >= 35:
        verdict = "BET"
    elif score >= 15:
        verdict = "MARGINAL"
    else:
        verdict = "NO_BET"

    return {
        "verdict":      verdict,
        "score":        score,
        "issues":       issues,
        "profit_ratio": profit_ratio,
        "wom_signal":   wom_signal,
        "spread":       spread_assess,
    }
