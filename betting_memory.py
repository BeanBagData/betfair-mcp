"""
BettingMemory — Rich outcome tracking and strategy learning engine.

Extends the basic MemoryStore (agent_memory.json) with a separate
betting_history.json that records every bet with full context —
market conditions at time of placement — and updates with outcomes
after settlement (read from bet_log.json / orders CSV).

The engine derives actionable strategy insights from historical data:
  - ROI by lay price bucket     (1.01–1.33 / 1.34–1.67 / 1.68–3.0 / 3.0+)
  - WOM signal accuracy         (did LAY_HEAVY actually predict losses?)
  - Model edge accuracy         (did Kash/Iggy LAY edge correlate with profit?)
  - Venue performance           (which venues have been profitable?)
  - Timing window performance   (OPTIMAL vs MONITOR vs LAST_CHANCE)
  - Best staking method by edge (does Kelly outperform Fixed at low edges?)

These insights are injected into the Gemini system prompt so every
new bet decision is informed by the agent's own track record.
"""

from __future__ import annotations

import json
import logging
import os
import datetime
import statistics
from collections import defaultdict
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

HISTORY_PATH   = "betting_history.json"
BET_LOG_PATH   = "bet_log.json"          # written by agent._log_bet
ORDERS_CSV     = "orders_agent.csv"      # written by Flumine

LAY_PRICE_BUCKETS = [
    (1.01, 1.33, "1.01–1.33 (very short)"),
    (1.34, 1.67, "1.34–1.67 (short, high profit_ratio)"),
    (1.68, 2.50, "1.68–2.50 (mid-range)"),
    (2.51, 4.00, "2.51–4.00 (longer, riskier lay)"),
    (4.01, 99.0, "4.01+   (high risk lay)"),
]

EDGE_BUCKETS = [
    (0,   3,  "0–3%  (marginal)"),
    (3,   7,  "3–7%  (good)"),
    (7,   15, "7–15% (strong)"),
    (15, 100, "15%+  (very strong)"),
]

MIN_BETS_FOR_INSIGHT = 5   # don't surface insights with fewer samples


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bucket_label(value: float, buckets: list[tuple]) -> str:
    for lo, hi, label in buckets:
        if lo <= value <= hi:
            return label
    return "other"


def _safe_load_json(path: str, default) -> Any:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load {path}: {e}")
    return default


def _safe_save_json(path: str, data: Any) -> None:
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Could not save {path}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# BET RECORD
# ─────────────────────────────────────────────────────────────────────────────

class BetRecord:
    """
    Full context snapshot at bet placement + outcome after settlement.

    Fields set at placement:
        bet_id, placed_at, bet_type ('lay'/'back'), market_id, selection_id,
        runner_name, venue, race_name, price (lay or back price), stake,
        liability, strategy_ref, wom_signal, model_signal, model_edge_pct,
        timing_window, profit_ratio, sp_far, edge_vs_sp

    Fields updated after settlement (from bet_log / orders CSV):
        settled_at, won (bool), profit, bsp_actual, result_source
    """

    def __init__(self, **kwargs):
        # Placement fields
        self.bet_id          = kwargs.get("bet_id", "")
        self.placed_at       = kwargs.get("placed_at", datetime.datetime.now().isoformat())
        self.bet_type        = kwargs.get("bet_type", "lay")          # 'lay' or 'back'
        self.market_id       = kwargs.get("market_id", "")
        self.selection_id    = kwargs.get("selection_id", 0)
        self.runner_name     = kwargs.get("runner_name", "Unknown")
        self.venue           = kwargs.get("venue", "")
        self.race_name       = kwargs.get("race_name", "")
        self.price           = kwargs.get("price", 0.0)               # lay or back price
        self.stake           = kwargs.get("stake", 0.0)               # backer stake for lay
        self.liability       = kwargs.get("liability", 0.0)
        self.strategy_ref    = kwargs.get("strategy_ref", "")

        # Market signals at bet time
        self.wom_signal      = kwargs.get("wom_signal", "UNKNOWN")    # BACK_HEAVY/LAY_HEAVY/BALANCED
        self.model_signal    = kwargs.get("model_signal", "UNKNOWN")  # LAY/BACK/NEUTRAL
        self.model_edge_pct  = kwargs.get("model_edge_pct", 0.0)      # % edge vs model price
        self.timing_window   = kwargs.get("timing_window", "UNKNOWN") # OPTIMAL/MONITOR/LAST_CHANCE
        self.profit_ratio    = kwargs.get("profit_ratio", 0.0)        # 1 / (lay_price - 1)
        self.sp_far          = kwargs.get("sp_far", None)             # BSP far estimate at bet time
        self.edge_vs_sp      = kwargs.get("edge_vs_sp", None)         # lay_price - sp_far (positive = value)
        self.opportunity_score = kwargs.get("opportunity_score", 0)   # composite score from analyst

        # Settlement fields (filled in later)
        self.settled_at      = kwargs.get("settled_at", None)
        self.won             = kwargs.get("won", None)                # True = lay won (horse lost)
        self.profit          = kwargs.get("profit", None)
        self.bsp_actual      = kwargs.get("bsp_actual", None)
        self.result_source   = kwargs.get("result_source", None)      # 'bet_log' or 'flumine_csv'

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "BetRecord":
        return cls(**d)

    @property
    def is_settled(self) -> bool:
        return self.won is not None

    @property
    def price_bucket(self) -> str:
        return _bucket_label(self.price, LAY_PRICE_BUCKETS)

    @property
    def edge_bucket(self) -> str:
        return _bucket_label(abs(self.model_edge_pct), EDGE_BUCKETS)


# ─────────────────────────────────────────────────────────────────────────────
# BETTING MEMORY
# ─────────────────────────────────────────────────────────────────────────────

class BettingMemory:
    """
    Persistent bet-outcome store with strategy analytics.

    Designed to be used alongside MemoryStore (which handles entity memory).
    This class owns betting_history.json and provides the strategy-learning layer.

    Typical lifecycle:
        1. Agent calls record_placement() when a bet is placed
        2. Agent calls sync_outcomes() periodically (or on startup) to pull
           settlement data from bet_log.json and orders_agent.csv
        3. Agent calls get_strategy_insights() to get a formatted summary
           of what's working, which is injected into the system prompt
    """

    MAX_RECORDS = 1000   # rolling cap to keep file size manageable

    def __init__(self, path: str = HISTORY_PATH):
        self.path = path
        self._records: dict[str, BetRecord] = {}  # bet_id → BetRecord
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        data = _safe_load_json(self.path, [])
        for d in data:
            r = BetRecord.from_dict(d)
            if r.bet_id:
                self._records[r.bet_id] = r
        logger.info(f"BettingMemory loaded {len(self._records)} records from {self.path}")

    def _save(self):
        records = sorted(self._records.values(), key=lambda r: r.placed_at or "", reverse=True)
        records = records[:self.MAX_RECORDS]
        _safe_save_json(self.path, [r.to_dict() for r in records])

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_placement(
        self,
        bet_id:          str,
        bet_type:        str,
        market_id:       str,
        selection_id:    int,
        runner_name:     str,
        price:           float,
        stake:           float,
        liability:       float,
        venue:           str          = "",
        race_name:       str          = "",
        strategy_ref:    str          = "",
        wom_signal:      str          = "UNKNOWN",
        model_signal:    str          = "UNKNOWN",
        model_edge_pct:  float        = 0.0,
        timing_window:   str          = "UNKNOWN",
        profit_ratio:    float        = 0.0,
        sp_far:          Optional[float] = None,
        edge_vs_sp:      Optional[float] = None,
        opportunity_score: int        = 0,
    ):
        """Record a newly placed bet with full market context."""
        record = BetRecord(
            bet_id=bet_id,
            bet_type=bet_type,
            market_id=market_id,
            selection_id=selection_id,
            runner_name=runner_name,
            price=price,
            stake=stake,
            liability=liability,
            venue=venue,
            race_name=race_name,
            strategy_ref=strategy_ref,
            wom_signal=wom_signal,
            model_signal=model_signal,
            model_edge_pct=model_edge_pct,
            timing_window=timing_window,
            profit_ratio=profit_ratio,
            sp_far=sp_far,
            edge_vs_sp=edge_vs_sp,
            opportunity_score=opportunity_score,
        )
        self._records[bet_id] = record
        self._save()
        logger.info(f"BettingMemory: recorded placement {bet_id} ({runner_name} @ {price})")

    def update_outcome(self, bet_id: str, won: bool, profit: float,
                       settled_at: str = None, bsp_actual: float = None,
                       result_source: str = "bet_log"):
        """Update a bet record with its settlement outcome."""
        if bet_id not in self._records:
            # Create a minimal stub so we don't lose outcome data
            self._records[bet_id] = BetRecord(bet_id=bet_id)
            logger.debug(f"BettingMemory: created stub for unknown bet_id {bet_id}")

        r = self._records[bet_id]
        r.won           = won
        r.profit        = profit
        r.settled_at    = settled_at or datetime.datetime.now().isoformat()
        r.bsp_actual    = bsp_actual
        r.result_source = result_source
        self._save()

    # ── Outcome Sync ──────────────────────────────────────────────────────────

    def sync_outcomes(self) -> int:
        """
        Pull settlement data from bet_log.json and orders_agent.csv,
        updating any unsettled records.

        Returns number of records newly settled.
        """
        updated = 0
        updated += self._sync_from_bet_log()
        updated += self._sync_from_flumine_csv()
        if updated:
            self._save()
        return updated

    def _sync_from_bet_log(self) -> int:
        """Read bet_log.json and settle any matching open records."""
        logs = _safe_load_json(BET_LOG_PATH, [])
        settled = 0
        for entry in logs:
            bet_id = str(entry.get("bet_id", ""))
            if not bet_id or bet_id not in self._records:
                continue
            r = self._records[bet_id]
            if r.is_settled:
                continue
            # bet_log doesn't always have profit — it's a placement log
            # Look for settled status fields
            profit = entry.get("profit")
            won    = entry.get("won")
            if profit is not None and won is not None:
                r.won     = bool(won)
                r.profit  = float(profit)
                r.settled_at    = entry.get("settled_at") or entry.get("timestamp")
                r.result_source = "bet_log"
                settled += 1
        return settled

    def _sync_from_flumine_csv(self) -> int:
        """Read orders_agent.csv (Flumine output) and settle matching records."""
        settled = 0
        try:
            import csv
            if not os.path.exists(ORDERS_CSV):
                return 0
            with open(ORDERS_CSV, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    bet_id = str(row.get("bet_id", row.get("id", "")))
                    if not bet_id:
                        continue

                    # If we don't have this bet recorded, create a stub
                    if bet_id not in self._records:
                        price_str  = row.get("price", row.get("lay_price", "0"))
                        stake_str  = row.get("size", row.get("stake", "0"))
                        try:
                            price  = float(price_str)
                            stake  = float(stake_str)
                        except (ValueError, TypeError):
                            price, stake = 0.0, 0.0
                        self._records[bet_id] = BetRecord(
                            bet_id       = bet_id,
                            market_id    = row.get("market_id", ""),
                            runner_name  = row.get("runner_name", row.get("selection", "Unknown")),
                            price        = price,
                            stake        = stake,
                            result_source= "flumine_csv",
                        )

                    r = self._records[bet_id]
                    if r.is_settled:
                        continue

                    profit_str = row.get("profit", row.get("net_profit", ""))
                    if profit_str in (None, "", "None"):
                        continue
                    try:
                        profit = float(profit_str)
                    except ValueError:
                        continue

                    r.won           = profit > 0
                    r.profit        = profit
                    r.settled_at    = row.get("settled_at", row.get("date", ""))
                    r.bsp_actual    = _try_float(row.get("bsp"))
                    r.result_source = "flumine_csv"
                    settled += 1
        except Exception as e:
            logger.warning(f"Flumine CSV sync error: {e}")
        return settled

    # ── Analytics ─────────────────────────────────────────────────────────────

    def settled_records(self) -> list[BetRecord]:
        return [r for r in self._records.values() if r.is_settled]

    def all_records(self) -> list[BetRecord]:
        return list(self._records.values())

    def _group_by(self, records: list[BetRecord], key_fn) -> dict[str, list[BetRecord]]:
        groups = defaultdict(list)
        for r in records:
            groups[key_fn(r)].append(r)
        return dict(groups)

    def _roi_stats(self, records: list[BetRecord]) -> dict:
        """Compute ROI stats for a group of settled bets."""
        n      = len(records)
        wins   = sum(1 for r in records if r.won)
        profit = sum(r.profit for r in records if r.profit is not None)
        staked = sum(r.stake  for r in records if r.stake)
        return {
            "n":       n,
            "wins":    wins,
            "losses":  n - wins,
            "win_rate": round(wins / n * 100, 1) if n else 0,
            "profit":  round(profit, 2),
            "staked":  round(staked, 2),
            "roi":     round(profit / staked * 100, 1) if staked > 0 else None,
        }

    def performance_by_price_bucket(self) -> dict[str, dict]:
        settled = self.settled_records()
        groups  = self._group_by(settled, lambda r: r.price_bucket)
        return {bucket: self._roi_stats(recs) for bucket, recs in groups.items()}

    def performance_by_wom_signal(self) -> dict[str, dict]:
        settled = self.settled_records()
        groups  = self._group_by(settled, lambda r: r.wom_signal or "UNKNOWN")
        return {signal: self._roi_stats(recs) for signal, recs in groups.items()}

    def performance_by_model_edge(self) -> dict[str, dict]:
        settled = self.settled_records()
        groups  = self._group_by(settled, lambda r: r.edge_bucket)
        return {bucket: self._roi_stats(recs) for bucket, recs in groups.items()}

    def performance_by_venue(self) -> dict[str, dict]:
        settled = self.settled_records()
        groups  = self._group_by(settled, lambda r: r.venue or "Unknown")
        return {v: self._roi_stats(recs) for v, recs in groups.items()}

    def performance_by_timing(self) -> dict[str, dict]:
        settled = self.settled_records()
        groups  = self._group_by(settled, lambda r: r.timing_window or "UNKNOWN")
        return {w: self._roi_stats(recs) for w, recs in groups.items()}

    def performance_by_model_signal(self) -> dict[str, dict]:
        settled = self.settled_records()
        groups  = self._group_by(settled, lambda r: r.model_signal or "UNKNOWN")
        return {s: self._roi_stats(recs) for s, recs in groups.items()}

    def recent_form(self, n: int = 20) -> dict:
        """Last N settled bets — useful for hot/cold streaks."""
        settled = sorted(self.settled_records(),
                         key=lambda r: r.settled_at or "", reverse=True)[:n]
        if not settled:
            return {"n": 0}
        stats = self._roi_stats(settled)
        stats["streak"] = self._current_streak(settled)
        return stats

    def _current_streak(self, records_newest_first: list[BetRecord]) -> str:
        if not records_newest_first:
            return "no bets"
        streak = 0
        outcome = records_newest_first[0].won
        for r in records_newest_first:
            if r.won == outcome:
                streak += 1
            else:
                break
        return f"{'W' if outcome else 'L'}{streak}"

    def overall_stats(self) -> dict:
        settled = self.settled_records()
        total   = len(self._records)
        stats   = self._roi_stats(settled)
        stats.update({
            "total_bets_placed":   total,
            "total_bets_settled":  len(settled),
            "pending_settlement":  total - len(settled),
        })
        return stats

    # ── Insight Generation ────────────────────────────────────────────────────

    def get_strategy_insights(self) -> str:
        """
        Return a formatted string of key strategy insights derived from
        historical bet outcomes. This is injected into the system prompt
        so the model can factor in its own performance history.

        Returns empty string if there are fewer than MIN_BETS_FOR_INSIGHT settled bets.
        """
        settled = self.settled_records()
        if len(settled) < MIN_BETS_FOR_INSIGHT:
            return ""

        lines = ["## MY HISTORICAL PERFORMANCE INSIGHTS (learn from these)\n"]
        lines.append(f"Based on {len(settled)} settled bets (from your own track record):\n")

        # Overall
        overall = self.overall_stats()
        lines.append(
            f"**Overall**: {overall['n']} bets | "
            f"Win rate {overall['win_rate']}% | "
            f"ROI {overall['roi']:+.1f}% | "
            f"Total P&L: ${overall['profit']:+.2f}"
        )

        # Recent form
        recent = self.recent_form(20)
        if recent.get("n", 0) >= 3:
            lines.append(
                f"**Recent form (last {recent['n']})**: "
                f"{recent['win_rate']}% win rate | "
                f"ROI {recent['roi']:+.1f}% | "
                f"Streak: {recent['streak']}"
            )

        # Price bucket performance
        price_perf = self.performance_by_price_bucket()
        price_insights = _extract_best_worst(price_perf)
        if price_insights:
            lines.append(f"\n**Lay Price Analysis** (use this to calibrate price gates):")
            for label, stats in sorted(price_perf.items(), key=lambda x: x[0]):
                if stats["n"] >= MIN_BETS_FOR_INSIGHT:
                    lines.append(
                        f"  {label}: {stats['n']} bets | "
                        f"{stats['win_rate']}% WR | "
                        f"ROI {stats['roi']:+.1f}%"
                        + (" ← BEST" if label == price_insights.get("best") else "")
                        + (" ← WORST" if label == price_insights.get("worst") else "")
                    )

        # WOM signal accuracy
        wom_perf = self.performance_by_wom_signal()
        wom_lines = [
            f"  {sig}: {s['n']} bets | {s['win_rate']}% WR | ROI {s['roi']:+.1f}%"
            for sig, s in sorted(wom_perf.items())
            if s["n"] >= MIN_BETS_FOR_INSIGHT
        ]
        if wom_lines:
            lines.append(f"\n**WOM Signal Accuracy**:")
            lines.extend(wom_lines)
            best_wom = max(wom_perf.items(), key=lambda x: x[1].get("roi") or -999
                           if x[1]["n"] >= MIN_BETS_FOR_INSIGHT else -999, default=None)
            if best_wom:
                lines.append(f"  → Best WOM signal for laying: **{best_wom[0]}**")

        # Model edge performance
        edge_perf = self.performance_by_model_edge()
        edge_lines = [
            f"  {bucket}: {s['n']} bets | {s['win_rate']}% WR | ROI {s['roi']:+.1f}%"
            for bucket, s in sorted(edge_perf.items())
            if s["n"] >= MIN_BETS_FOR_INSIGHT
        ]
        if edge_lines:
            lines.append(f"\n**Model Edge vs ROI** (does bigger edge = more profit?):")
            lines.extend(edge_lines)

        # Model signal (LAY vs BACK vs NEUTRAL)
        signal_perf = self.performance_by_model_signal()
        sig_lines = [
            f"  {sig}: {s['n']} bets | ROI {s['roi']:+.1f}%"
            for sig, s in signal_perf.items()
            if s["n"] >= MIN_BETS_FOR_INSIGHT
        ]
        if sig_lines:
            lines.append(f"\n**Model Signal Performance**:")
            lines.extend(sig_lines)

        # Venue performance
        venue_perf = self.performance_by_venue()
        strong_venues  = [v for v, s in venue_perf.items()
                          if s["n"] >= MIN_BETS_FOR_INSIGHT and (s["roi"] or 0) > 5]
        weak_venues    = [v for v, s in venue_perf.items()
                          if s["n"] >= MIN_BETS_FOR_INSIGHT and (s["roi"] or 0) < -10]
        if strong_venues or weak_venues:
            lines.append(f"\n**Venue Performance**:")
            if strong_venues:
                lines.append(f"  ✅ Profitable venues: {', '.join(strong_venues)}")
            if weak_venues:
                lines.append(f"  ❌ Unprofitable venues (be cautious): {', '.join(weak_venues)}")
            for v, s in sorted(venue_perf.items(), key=lambda x: x[1].get("roi") or 0, reverse=True):
                if s["n"] >= MIN_BETS_FOR_INSIGHT:
                    lines.append(f"  {v}: {s['n']} bets | ROI {s['roi']:+.1f}%")

        # Timing window performance
        timing_perf = self.performance_by_timing()
        timing_lines = [
            f"  {w}: {s['n']} bets | ROI {s['roi']:+.1f}%"
            for w, s in timing_perf.items()
            if s["n"] >= MIN_BETS_FOR_INSIGHT
        ]
        if timing_lines:
            lines.append(f"\n**Timing Window Performance**:")
            lines.extend(timing_lines)
            best_timing = max(
                ((w, s) for w, s in timing_perf.items() if s["n"] >= MIN_BETS_FOR_INSIGHT),
                key=lambda x: x[1].get("roi") or -999, default=None
            )
            if best_timing:
                lines.append(f"  → Best timing window: **{best_timing[0]}**")

        lines.append(
            "\n_Apply these patterns to new betting decisions. "
            "Down-weight signals that historically underperform. "
            "Increase confidence when multiple historically-positive signals align._\n"
        )

        return "\n".join(lines)

    def get_context_block(self) -> str:
        """
        Compact context block for injection into each user message —
        a running summary of session and recent performance.
        """
        settled = self.settled_records()
        if not settled:
            return ""

        recent = self.recent_form(10)
        overall = self.overall_stats()

        lines = ["<betting_performance>"]
        lines.append(f"Lifetime: {overall['n']} bets | WR {overall['win_rate']}% | ROI {overall['roi']:+.1f}%")
        if recent.get("n", 0) >= 3:
            lines.append(
                f"Last {recent['n']} bets: WR {recent['win_rate']}% | "
                f"ROI {recent['roi']:+.1f}% | Streak {recent['streak']}"
            )
        lines.append("</betting_performance>")
        return "\n".join(lines)

    def get_pending_bets(self) -> list[dict]:
        """Return bets placed but not yet settled — for display / tracking."""
        return [r.to_dict() for r in self._records.values() if not r.is_settled]

    def get_recent_bets(self, n: int = 10) -> list[dict]:
        """Return last N settled bets sorted newest first."""
        settled = sorted(self.settled_records(),
                         key=lambda r: r.settled_at or "", reverse=True)
        return [r.to_dict() for r in settled[:n]]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_best_worst(perf: dict[str, dict]) -> dict:
    """Find the best and worst performing buckets by ROI (min sample size enforced)."""
    valid = {k: v for k, v in perf.items()
             if v["n"] >= MIN_BETS_FOR_INSIGHT and v.get("roi") is not None}
    if not valid:
        return {}
    best  = max(valid.items(), key=lambda x: x[1]["roi"])
    worst = min(valid.items(), key=lambda x: x[1]["roi"])
    return {"best": best[0], "worst": worst[0]}


def _try_float(val) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON ACCESSOR
# ─────────────────────────────────────────────────────────────────────────────

_betting_memory_instance: Optional[BettingMemory] = None


def get_betting_memory() -> BettingMemory:
    """Return the process-wide BettingMemory singleton."""
    global _betting_memory_instance
    if _betting_memory_instance is None:
        _betting_memory_instance = BettingMemory()
    return _betting_memory_instance
