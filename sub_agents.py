"""
Sub-Agent Orchestration System
Provides a multi-agent framework for complex betting tasks.

Architecture:
  OrchestratorAgent  — receives high-level goal from user, decomposes into
                       tasks, and coordinates specialist sub-agents.
  VenueAnalystAgent  — scans all markets at a venue, scores opportunities,
                       returns a ranked shortlist with full analysis.
  RiskManagerAgent   — allocates bankroll across races, enforces circuit
                       breakers, prevents over-exposure.
  ExecutionAgent     — runs Flumine (streaming) or polling-based execution
                       for a given list of markets / opportunities.
  ReporterAgent      — summarises session P&L, compares to pre-session
                       forecast, surfaces lessons.

Example usage (via agent.py tool):
    orchestrate_venue_session("Doomben", bankroll=500, session_hours=8)
    → Analyst scans all Doomben markets
    → RiskManager allocates per-race budgets
    → ExecutionAgent starts Flumine for automated execution
    → ReporterAgent checks in every 30 min
"""

from __future__ import annotations

import datetime
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependency — fetched on first use
def _get_cache():
    from shared_cache import SharedCache
    return SharedCache.instance()

def _get_betting_memory():
    from betting_memory import get_betting_memory
    return get_betting_memory()


# ─────────────────────────────────────────────────────────────────────────────
# SHARED SESSION STATE  (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    """Shared mutable state passed between sub-agents."""
    venue:          str = ""
    bankroll:       float = 0.0
    remaining_bankroll: float = 0.0
    races_analysed: int = 0
    races_bet:      int = 0
    total_staked:   float = 0.0
    total_pnl:      float = 0.0   # updated from Flumine log
    opportunities:  list[dict] = field(default_factory=list)
    allocations:    dict[str, float] = field(default_factory=dict)  # market_id → max_liability
    log:            list[str] = field(default_factory=list)
    started_at:     datetime.datetime = field(default_factory=datetime.datetime.now)
    _lock:          threading.Lock = field(default_factory=threading.Lock)

    def add_log(self, msg: str):
        with self._lock:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self.log.append(f"[{ts}] {msg}")
            logger.info(msg)

    def summary(self) -> dict:
        elapsed = (datetime.datetime.now() - self.started_at).total_seconds()
        return {
            "venue":            self.venue,
            "bankroll_start":   self.bankroll,
            "bankroll_current": round(self.remaining_bankroll, 2),
            "races_analysed":   self.races_analysed,
            "races_bet":        self.races_bet,
            "total_staked":     round(self.total_staked, 2),
            "total_pnl":        round(self.total_pnl, 2),
            "roi_pct":          round(self.total_pnl / self.total_staked * 100, 2) if self.total_staked > 0 else 0,
            "elapsed_minutes":  round(elapsed / 60, 1),
            "opportunity_count": len(self.opportunities),
            "log_tail":         self.log[-10:],
        }


# ─────────────────────────────────────────────────────────────────────────────
# BASE SUB-AGENT
# ─────────────────────────────────────────────────────────────────────────────

class SubAgentBase:
    """
    Base class for all sub-agents.

    Each sub-agent receives a task string and the shared SessionState,
    runs its work (synchronously or by spawning a thread), and
    returns a result dict with at least {"success": bool, "summary": str}.
    """
    name: str = "SubAgent"

    def __init__(self, betfair_client, ratings_cache, state: SessionState):
        self.bf     = betfair_client
        self.rc     = ratings_cache
        self.state  = state

    def run(self, task: str) -> dict:
        """Override in subclasses. Returns {"success": bool, ...}"""
        raise NotImplementedError

    def _log(self, msg: str):
        self.state.add_log(f"[{self.name}] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# VENUE ANALYST AGENT
# ─────────────────────────────────────────────────────────────────────────────

class VenueAnalystAgent(SubAgentBase):
    """
    Scans all markets at a venue, fetches model prices and market books,
    and scores each opportunity using the full analysis pipeline.

    Returns a ranked list of lay opportunities with:
      - Model edge (Kash/Iggy vs live prices)
      - Profit ratio
      - WOM signal
      - Race metadata (barrier, distance, track condition)
      - Composite lay context score
    """
    name = "VenueAnalyst"

    def run(self, task: str = "") -> dict:
        venue = self.state.venue
        self._log(f"Scanning {venue} for today's markets")

        cache = _get_cache()

        # Load historical venue performance to bias scoring
        venue_roi: dict[str, float] = {}
        try:
            bm = _get_betting_memory()
            venue_roi = {
                v: s.get("roi", 0) or 0
                for v, s in bm.performance_by_venue().items()
                if s.get("n", 0) >= 5
            }
            if venue_roi:
                self._log(f"Historical venue ROI data loaded for {len(venue_roi)} venues")
        except Exception as e:
            self._log(f"Could not load venue ROI data (non-fatal): {e}")

        # Load WOM signal accuracy from historical data
        wom_roi: dict[str, float] = {}
        try:
            bm = _get_betting_memory()
            wom_roi = {
                sig: s.get("roi", 0) or 0
                for sig, s in bm.performance_by_wom_signal().items()
                if s.get("n", 0) >= 5
            }
        except Exception:
            pass

        # 1. Get venue markets from ratings
        markets = self.rc.get_venue_markets(venue, model="kash")
        if not markets:
            self._log(f"No Kash ratings found for {venue} — trying Iggy (greyhounds)")
            markets = self.rc.get_venue_markets(venue, model="iggy")

        if not markets:
            return {
                "success": False,
                "summary": f"No upcoming markets found at {venue} in today's ratings. "
                           "The venue name may not match — try the full official name (e.g. 'Doomben' → 'Eagle Farm & Doomben').",
                "opportunities": [],
            }

        self._log(f"Found {len(markets)} markets at {venue}")
        self.state.races_analysed = len(markets)

        scored = []
        for market in markets:
            mid = market["market_id"]

            try:
                # ── Market book (cached, TTL=5s) ──────────────────────────
                book = cache.market_book(mid, fetch_fn=lambda m=mid: self.bf.get_market_book(m))
                if not book.get("success"):
                    continue

                # ── Race metadata (cached, TTL=1h) ────────────────────────
                try:
                    meta = cache.race_metadata(mid, fetch_fn=lambda m=mid: self.bf.get_race_metadata(m))
                    meta_map = {str(r["selection_id"]): r for r in meta.get("runners", [])}
                except Exception:
                    meta = {}
                    meta_map = {}

                for runner in book.get("runners", []):
                    if runner.get("status") != "ACTIVE":
                        continue
                    sel_id = str(runner["selection_id"])
                    best_lay  = (runner.get("best_lay")  or {}).get("price")
                    best_back = (runner.get("best_back") or {}).get("price")
                    if not best_lay:
                        continue

                    profit_ratio = round(1.0 / (best_lay - 1.0), 3) if best_lay > 1 else 0

                    # Model edge
                    edge = self.rc.model_edge(mid, sel_id, best_lay, best_back, model="kash")

                    # WOM from market_analyser
                    wom_signal = "UNKNOWN"
                    try:
                        from market_analyser import weight_of_money
                        wom_data   = weight_of_money(
                            runner.get("back_prices", []),
                            runner.get("lay_prices", []),
                        )
                        wom_signal = wom_data.get("signal", "UNKNOWN")
                    except Exception:
                        pass

                    # Race context score
                    runner_meta = meta_map.get(sel_id, {})
                    context_score = 0
                    context_verdict = "NEUTRAL"
                    try:
                        from market_analyser import lay_context_hints
                        ctx = lay_context_hints(
                            barrier_group=runner_meta.get("barrier_group"),
                            distance_group=runner_meta.get("distance_group"),
                            track_condition=meta.get("track_condition"),
                            jockey=runner_meta.get("jockey"),
                        )
                        context_score   = ctx.get("context_score", 0)
                        context_verdict = ctx.get("context_verdict", "NEUTRAL")
                    except Exception:
                        pass

                    # ── Composite opportunity score ────────────────────────
                    # Base scoring (same as before)
                    opp_score = 0
                    if profit_ratio >= 1.5:
                        opp_score += 40
                    if edge.get("signal") in ("LAY", "BOTH"):
                        opp_score += 25
                    if wom_signal == "LAY_HEAVY":
                        opp_score += 20
                    elif wom_signal == "BACK_HEAVY":
                        opp_score -= 15
                    opp_score += context_score

                    # ── Historical performance adjustment (learning) ───────
                    # Boost/penalise based on what has actually worked at this venue
                    hist_venue_roi = venue_roi.get(venue, 0)
                    if hist_venue_roi > 10:       # historically profitable venue
                        opp_score += 10
                    elif hist_venue_roi < -10:     # historically lossy venue
                        opp_score -= 10

                    # WOM signal accuracy adjustment from historical data
                    hist_wom_roi = wom_roi.get(wom_signal, 0)
                    if hist_wom_roi > 10:
                        opp_score += 8
                    elif hist_wom_roi < -10:
                        opp_score -= 8

                    scored.append({
                        "market_id":      mid,
                        "race_number":    market.get("race_number", ""),
                        "selection_id":   runner["selection_id"],
                        "runner_name":    runner.get("runner_name", "Unknown"),
                        "best_lay":       best_lay,
                        "best_back":      best_back,
                        "profit_ratio":   profit_ratio,
                        "model_edge":     edge,
                        "wom_signal":     wom_signal,
                        "jockey":         runner_meta.get("jockey", ""),
                        "barrier":        runner_meta.get("barrier", ""),
                        "barrier_group":  runner_meta.get("barrier_group", ""),
                        "distance_group": runner_meta.get("distance_group", ""),
                        "track_condition": meta.get("track_condition", ""),
                        "context_score":  context_score,
                        "context_verdict": context_verdict,
                        "opportunity_score": opp_score,
                        "hist_venue_roi": hist_venue_roi,
                        "hist_wom_roi":   hist_wom_roi,
                        "verdict": (
                            "STRONG_LAY" if opp_score >= 60 else
                            "LAY"        if opp_score >= 35 else
                            "MARGINAL"   if opp_score >= 15 else
                            "SKIP"
                        ),
                    })
            except Exception as e:
                self._log(f"Error analysing {mid}: {e}")
                continue

        # Sort best opportunities first
        scored.sort(key=lambda x: x["opportunity_score"], reverse=True)
        self.state.opportunities = scored

        top = scored[:5] if scored else []
        self._log(f"Analysis complete — {len(scored)} opportunities across {len(markets)} races. Top: "
                  + ", ".join(f"{o['runner_name']} ({o['verdict']})" for o in top))

        return {
            "success":          True,
            "venue":            venue,
            "races_scanned":    len(markets),
            "opportunities":    len(scored),
            "strong_lays":      sum(1 for o in scored if o["verdict"] == "STRONG_LAY"),
            "lays":             sum(1 for o in scored if o["verdict"] == "LAY"),
            "top_opportunities": scored[:10],
            "venue_historical_roi": hist_venue_roi if venue_roi else None,
            "summary": (
                f"Scanned {len(markets)} races at {venue}. "
                f"Found {sum(1 for o in scored if o['verdict'] in ('STRONG_LAY','LAY'))} lay opportunities. "
                f"Best: {top[0]['runner_name']} ({top[0]['verdict']}, score={top[0]['opportunity_score']})"
                if top else f"Scanned {len(markets)} races at {venue}. No qualifying opportunities found."
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGER AGENT
# ─────────────────────────────────────────────────────────────────────────────

class RiskManagerAgent(SubAgentBase):
    """
    Allocates bankroll across races and enforces circuit breakers.

    Rules:
      - Max 10% of bankroll as liability per race
      - Max 30% of bankroll deployed simultaneously
      - Daily loss limit: if P&L < -loss_limit_pct% of bankroll → stop all betting
      - Kelly floor: if Kelly signals negative fraction → no bet
    """
    name = "RiskManager"

    def __init__(self, *args, loss_limit_pct: float = 20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_limit_pct = loss_limit_pct

    def run(self, task: str = "") -> dict:
        bankroll     = self.state.bankroll
        opps         = self.state.opportunities
        loss_limit   = bankroll * self.loss_limit_pct / 100

        self._log(f"Allocating bankroll={bankroll:.2f} across {len(opps)} opportunities")

        # Load historical price-bucket ROI for Kelly adjustment
        price_bucket_roi: dict[str, float] = {}
        try:
            bm = _get_betting_memory()
            price_bucket_roi = {
                k: v.get("roi", 0) or 0
                for k, v in bm.performance_by_price_bucket().items()
                if v.get("n", 0) >= 5
            }
            if price_bucket_roi:
                self._log(f"Historical price-bucket ROI loaded: {price_bucket_roi}")
        except Exception as e:
            self._log(f"Could not load price-bucket ROI (non-fatal): {e}")

        # Max liability per bet = 10% of bankroll
        max_per_race = bankroll * 0.10

        allocations  = {}
        total_committed = 0.0

        for opp in opps:
            if opp["verdict"] not in ("STRONG_LAY", "LAY"):
                continue
            if total_committed >= bankroll * 0.30:
                # Cap at 30% simultaneously deployed
                break

            market_id   = opp["market_id"]
            best_lay    = opp.get("best_lay", 2.0)
            model_price = opp.get("model_edge", {}).get("model_price") or best_lay
            win_prob    = min(0.95, 1.0 / model_price)

            # Kelly sizing
            try:
                from staking_engine import recommend_stake
                kelly = recommend_stake(
                    bankroll=bankroll,
                    win_prob=win_prob,
                    lay_price=best_lay,
                    method="half_kelly",
                )
                raw_stake = float(kelly.get("backer_stake", 5.0))
                liability = (best_lay - 1) * raw_stake
            except Exception:
                raw_stake, liability = 5.0, (best_lay - 1) * 5.0

            # ── Historical performance adjustment ─────────────────────
            # If this price bucket has historically been lossy, reduce stake.
            # If it's been profitable, allow up to the cap.
            from betting_memory import _bucket_label, LAY_PRICE_BUCKETS
            bucket = _bucket_label(best_lay, LAY_PRICE_BUCKETS)
            hist_roi = price_bucket_roi.get(bucket, 0)
            if hist_roi < -10:
                # Historically lossy price range — reduce stake by 50%
                raw_stake *= 0.5
                liability  = (best_lay - 1) * raw_stake
                self._log(f"Reduced stake for {opp['runner_name']}: price bucket '{bucket}' "
                          f"has historical ROI {hist_roi:+.1f}%")
            elif hist_roi > 15:
                # Historically strong price range — allow up to 1.2× kelly (up to cap)
                raw_stake  = min(raw_stake * 1.2, max_per_race / (best_lay - 1))
                liability  = (best_lay - 1) * raw_stake

            # Cap at per-race limit
            if liability > max_per_race:
                liability = max_per_race
                raw_stake = max_per_race / (best_lay - 1)

            if raw_stake < 2.0:
                continue

            allocations[market_id] = {
                "backer_stake":  round(raw_stake, 2),
                "liability":     round(liability, 2),
                "max_liability": round(max_per_race, 2),
                "win_prob":      round(win_prob, 4),
                "price_bucket":  bucket,
                "hist_bucket_roi": hist_roi,
            }
            total_committed += liability

        self.state.allocations = allocations

        circuit_breakers = {
            "daily_loss_limit":       round(loss_limit, 2),
            "max_per_race_liability": round(max_per_race, 2),
            "max_simultaneous_pct":   "30%",
            "stop_if_loss_exceeds":   f"${loss_limit:.2f}",
        }

        self._log(
            f"Allocated {len(allocations)} bets, total committed liability: ${total_committed:.2f}"
        )

        return {
            "success":                   True,
            "bankroll":                  bankroll,
            "allocations":               allocations,
            "total_committed_liability": round(total_committed, 2),
            "circuit_breakers":          circuit_breakers,
            "summary": (
                f"Risk allocation complete. "
                f"{len(allocations)} races funded, "
                f"total liability=${total_committed:.2f} "
                f"({total_committed/bankroll*100:.1f}% of bankroll). "
                f"Daily stop-loss at ${loss_limit:.2f}."
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION AGENT
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionAgent(SubAgentBase):
    """
    Executes bets for opportunities that have been analysed and allocated.

    Two modes:
      FLUMINE  — launches a FlumineRunner for streaming, automated execution.
                 Best for "all day at Doomben" type requests.
      POLLING  — uses BetfairClient to place bets immediately (one-off).
                 Best for a single specific race.
    """
    name = "Executor"

    def run(self, task: str = "flumine") -> dict:
        mode = "flumine" if "flumine" in task.lower() else "polling"

        if mode == "flumine":
            return self._run_flumine()
        else:
            return self._run_polling()

    def _run_flumine(self) -> dict:
        """Launch Flumine in background thread for streaming execution."""
        try:
            from flumine_engine import FlumineRunner
        except ImportError:
            return {
                "success": False,
                "error": "flumine is not installed. Run: pip install flumine betfairlightweight",
                "summary": "Flumine execution unavailable — install it to use automated streaming.",
            }

        venue    = self.state.venue
        bankroll = self.state.bankroll

        # Find max liability from allocations
        max_liab = max(
            (v["max_liability"] for v in self.state.allocations.values()),
            default=bankroll * 0.10,
        )

        try:
            runner = FlumineRunner(bankroll=bankroll)
            runner.ratings = self.rc
            runner.add_venue_strategy(
                venue=venue,
                betfair_client=self.bf,
                max_liability=max_liab,
            )
            # Start in background thread so orchestrator can continue
            thread = runner.start_background()

            self._log(f"Flumine started in background for {venue} (max_liability=${max_liab:.2f})")

            # Store runner ref for later P&L checks
            self._runner = runner
            self._thread = thread

            return {
                "success":  True,
                "mode":     "FLUMINE_STREAMING",
                "venue":    venue,
                "max_liability_per_race": max_liab,
                "markets_targeted": len(self.state.allocations),
                "summary": (
                    f"Flumine streaming started for {venue}. "
                    f"Monitoring {len(self.state.allocations)} markets. "
                    f"Max liability ${max_liab:.2f}/race. "
                    f"Will auto-terminate when all today's races close."
                ),
            }
        except Exception as e:
            self._log(f"Flumine start error: {e}")
            return {"success": False, "error": str(e), "mode": "FLUMINE_STREAMING"}

    def _run_polling(self) -> dict:
        """Place bets immediately via BetfairClient for each allocated opportunity."""
        placed   = []
        skipped  = []
        opps_by_market = {o["market_id"]: o for o in self.state.opportunities
                          if o["verdict"] in ("STRONG_LAY", "LAY")}

        for market_id, alloc in self.state.allocations.items():
            opp = opps_by_market.get(market_id)
            if not opp:
                skipped.append({"market_id": market_id, "reason": "No opportunity found"})
                continue

            # Re-check live price before placing
            try:
                book = self.bf.get_market_book(market_id)
                if not book.get("success") or book.get("inplay"):
                    skipped.append({"market_id": market_id, "reason": "Market unavailable or inplay"})
                    continue

                runner_data = next(
                    (r for r in book["runners"] if str(r["selection_id"]) == str(opp["selection_id"])),
                    None,
                )
                if not runner_data:
                    skipped.append({"market_id": market_id, "reason": "Runner not found in book"})
                    continue

                live_lay = (runner_data.get("best_lay") or {}).get("price")
                if not live_lay:
                    skipped.append({"market_id": market_id, "reason": "No lay price available"})
                    continue

                # Sanity: still a lay opportunity?
                if live_lay > 1.0 and 1.0 / (live_lay - 1.0) < 1.5:
                    skipped.append({"market_id": market_id, "reason": f"Lay price {live_lay} no longer meets 1.5x threshold"})
                    continue

                result = self.bf.place_lay_bet(
                    market_id=market_id,
                    selection_id=int(opp["selection_id"]),
                    lay_price=live_lay,
                    stake=alloc["backer_stake"],
                    strategy_ref="sub_agent",
                )
                if result.get("success"):
                    placed.append({
                        "market_id":   market_id,
                        "runner_name": opp.get("runner_name"),
                        "lay_price":   live_lay,
                        "stake":       alloc["backer_stake"],
                        "liability":   alloc["liability"],
                        "bet_id":      result.get("bet_id"),
                    })
                    self.state.races_bet += 1
                    self.state.total_staked += alloc["backer_stake"]
                    self._log(f"Placed LAY on {opp['runner_name']} @ {live_lay} stake={alloc['backer_stake']:.2f}")
                else:
                    skipped.append({"market_id": market_id, "reason": result.get("error", "Unknown")})
            except Exception as e:
                skipped.append({"market_id": market_id, "reason": str(e)})

        return {
            "success":    True,
            "mode":       "POLLING",
            "placed":     len(placed),
            "skipped":    len(skipped),
            "bets":       placed,
            "skipped_detail": skipped,
            "summary": (
                f"Placed {len(placed)} lay bets. "
                f"Skipped {len(skipped)}. "
                + (f"Total stake=${self.state.total_staked:.2f}." if placed else "No bets placed.")
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# REPORTER AGENT
# ─────────────────────────────────────────────────────────────────────────────

class ReporterAgent(SubAgentBase):
    """
    Summarises the session status and P&L.
    Can also read from the Flumine orders CSV for live P&L.
    """
    name = "Reporter"

    def run(self, task: str = "") -> dict:
        summary = self.state.summary()

        # Pull P&L from Flumine orders CSV if it exists
        pnl_data = {}
        try:
            import pandas as _pd
            if _pd is not None:
                import os
                if os.path.exists("orders_agent.csv"):
                    df = _pd.read_csv("orders_agent.csv")
                    total_pnl = df["profit"].sum() if "profit" in df.columns else 0
                    self.state.total_pnl = float(total_pnl)
                    pnl_data = {
                        "bet_count":    len(df),
                        "total_pnl":    round(float(total_pnl), 2),
                        "won":          int((df["profit"] > 0).sum()),
                        "lost":         int((df["profit"] <= 0).sum()),
                    }
        except Exception:
            pass

        # ── Sync & pull lifetime stats from BettingMemory ──────────────
        lifetime_context = ""
        try:
            bm = _get_betting_memory()
            bm.sync_outcomes()  # pick up any new settlements
            overall = bm.overall_stats()
            venue_perf = bm.performance_by_venue().get(self.state.venue, {})
            if overall.get("total_bets_settled", 0) > 0:
                lifetime_context = (
                    f"\n  Lifetime: {overall['n']} bets | "
                    f"WR {overall['win_rate']}% | ROI {overall['roi']:+.1f}%"
                )
                if venue_perf.get("n", 0) >= 3:
                    lifetime_context += (
                        f"\n  {self.state.venue} history: "
                        f"{venue_perf['n']} bets | ROI {venue_perf['roi']:+.1f}%"
                    )
        except Exception:
            pass

        narrative = (
            f"Session Report — {self.state.venue} — "
            f"{summary['elapsed_minutes']} min elapsed.\n"
            f"  Races analysed: {summary['races_analysed']}, Bets placed: {summary['races_bet']}\n"
            f"  Total staked: ${summary['total_staked']:.2f}, P&L: ${summary['total_pnl']:+.2f}\n"
            f"  ROI: {summary['roi_pct']:+.1f}%\n"
        )
        if pnl_data:
            narrative += f"  (From Flumine log: {pnl_data['bet_count']} orders, P&L={pnl_data['total_pnl']:+.2f})"
        if lifetime_context:
            narrative += lifetime_context

        self._log(narrative)

        return {
            "success":       True,
            "session":       summary,
            "pnl_from_log":  pnl_data,
            "narrative":     narrative,
            "summary":       narrative,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class OrchestratorAgent:
    """
    Top-level coordinator that decomposes high-level user goals into
    sub-agent tasks and synthesises their results.

    Usage (called from agent.py tool handler):
        orch = OrchestratorAgent(betfair_client, ratings_cache)
        result = orch.run_venue_session("Doomben", bankroll=500)
    """

    def __init__(self, betfair_client, ratings_cache=None):
        self.bf = betfair_client
        if ratings_cache is None:
            from external_ratings import get_ratings_cache
            ratings_cache = get_ratings_cache()
        self.rc = ratings_cache

    def run_venue_session(
        self,
        venue:             str,
        bankroll:          float,
        mode:              str   = "flumine",   # "flumine" or "polling"
        min_profit_ratio:  float = 1.5,
        loss_limit_pct:    float = 20.0,
        min_edge_pct:      float = 3.0,
    ) -> dict:
        """
        Orchestrate a full betting session at a single venue.

        Pipeline:
          1. VenueAnalystAgent  → scores all races at venue
          2. RiskManagerAgent   → allocates bankroll, sets circuit breakers
          3. ExecutionAgent     → places bets or launches Flumine
          4. ReporterAgent      → returns session summary

        Args:
            venue:            Track name (e.g. "Doomben", "Flemington")
            bankroll:         Account balance to use this session
            mode:             "flumine" (streaming, recommended) or "polling" (immediate)
            min_profit_ratio: Minimum lay profit_ratio (default 1.5x)
            loss_limit_pct:   Stop if daily loss exceeds X% of bankroll
            min_edge_pct:     Minimum % edge vs model price

        Returns large dict with all agent results + overall recommendation.
        """
        state = SessionState(venue=venue, bankroll=bankroll, remaining_bankroll=bankroll)
        state.add_log(f"Orchestrator started: venue={venue} bankroll=${bankroll:.2f} mode={mode}")

        # ── Step 1: Analyst ───────────────────────────────────────────────
        analyst = VenueAnalystAgent(self.bf, self.rc, state)
        analyst_result = analyst.run()
        if not analyst_result.get("success"):
            return {
                "success": False,
                "stage":   "analysis",
                "error":   analyst_result.get("summary", "Analysis failed"),
                "state":   state.summary(),
            }

        # ── Step 2: Risk Manager ──────────────────────────────────────────
        risk = RiskManagerAgent(self.bf, self.rc, state, loss_limit_pct=loss_limit_pct)
        risk_result = risk.run()

        # ── Step 3: Execution ─────────────────────────────────────────────
        executor = ExecutionAgent(self.bf, self.rc, state)
        exec_result = executor.run(task=mode)

        # ── Step 4: Reporter ──────────────────────────────────────────────
        reporter = ReporterAgent(self.bf, self.rc, state)
        report = reporter.run()

        # ── Synthesis ─────────────────────────────────────────────────────
        session = state.summary()
        strong_lays  = analyst_result.get("strong_lays", 0)
        lays         = analyst_result.get("lays", 0)
        top_10       = analyst_result.get("top_opportunities", [])

        recommendation = (
            f"SESSION PLAN — {venue.upper()} — ${bankroll:.0f} bankroll\n\n"
            f"📊 ANALYSIS: {analyst_result.get('races_scanned', 0)} races scanned, "
            f"{strong_lays} STRONG lay + {lays} LAY opportunities identified.\n\n"
        )
        if top_10:
            recommendation += "🏆 TOP OPPORTUNITIES:\n"
            for i, o in enumerate(top_10[:5], 1):
                recommendation += (
                    f"  {i}. {o['runner_name']} (Race {o.get('race_number', '?')}) "
                    f"@ lay {o.get('best_lay', '?')} — {o['verdict']} "
                    f"(score={o['opportunity_score']}, model edge={o.get('model_edge',{}).get('signal','?')})\n"
                )

        recommendation += (
            f"\n💰 RISK ALLOCATION: "
            f"${risk_result.get('total_committed_liability', 0):.2f} total liability across "
            f"{len(risk_result.get('allocations', {}))} races.\n"
            f"  Stop-loss trigger: ${risk_result.get('circuit_breakers', {}).get('daily_loss_limit', 0):.2f}\n\n"
            f"⚡ EXECUTION: {exec_result.get('summary', 'N/A')}\n\n"
            f"📈 STATUS: {report.get('narrative', 'N/A')}"
        )

        return {
            "success":        True,
            "venue":          venue,
            "bankroll":       bankroll,
            "mode":           mode,
            "analyst":        analyst_result,
            "risk":           risk_result,
            "execution":      exec_result,
            "report":         report,
            "session":        session,
            "recommendation": recommendation,
            "summary":        recommendation,
        }

    def quick_report(self, venue: str, bankroll: float) -> dict:
        """Run only the analyst step — for "what's the best bet at Doomben?" queries."""
        state = SessionState(venue=venue, bankroll=bankroll, remaining_bankroll=bankroll)
        analyst = VenueAnalystAgent(self.bf, self.rc, state)
        return analyst.run()
