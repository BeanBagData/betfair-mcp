"""
Prediction candidate ledger and calibration analytics.

BettingMemory records placed bets. This module records every candidate the
prediction layer evaluates, including skipped runners, so the system can
measure calibration, false positives, and missed winners over time.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

LEDGER_PATH = "candidate_ledger.json"
MAX_RECORDS = 10_000


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class CandidateRecord:
    """One evaluated runner at one scan time."""

    candidate_id: str
    market_id: str
    selection_id: int
    runner_name: str = "Unknown"
    venue: str = ""
    race_name: str = ""
    race_number: str = ""
    scanned_at: str = ""
    decision: str = "SKIPPED"
    verdict: str = "SKIP"
    score: int = 0
    lay_price: float = 0.0
    best_back: Optional[float] = None
    stake: float = 0.0
    liability: float = 0.0
    estimated_win_prob: float = 0.0
    estimated_win_prob_source: str = "UNKNOWN"
    profit_ratio: float = 0.0
    wom_signal: str = "UNKNOWN"
    model_signal: str = "UNKNOWN"
    model_edge_pct: float = 0.0
    sp_near: Optional[float] = None
    sp_far: Optional[float] = None
    edge_net: Optional[float] = None
    context_score: int = 0
    movement_signal: str = "UNKNOWN"
    issues: list[str] | None = None
    components: dict[str, Any] | None = None
    bet_id: str = ""

    # Settlement fields. `won` means the runner won the race, not the lay bet.
    settled_at: Optional[str] = None
    won: Optional[bool] = None
    profit: Optional[float] = None
    bsp_actual: Optional[float] = None
    result_source: Optional[str] = None

    def __post_init__(self):
        if not self.scanned_at:
            self.scanned_at = _now_iso()
        if self.issues is None:
            self.issues = []
        if self.components is None:
            self.components = {}
        self.selection_id = _safe_int(self.selection_id)
        self.score = _safe_int(self.score)
        self.lay_price = _safe_float(self.lay_price, 0.0) or 0.0
        self.best_back = _safe_float(self.best_back)
        self.stake = _safe_float(self.stake, 0.0) or 0.0
        self.liability = _safe_float(self.liability, 0.0) or 0.0
        self.estimated_win_prob = _safe_float(self.estimated_win_prob, 0.0) or 0.0
        self.profit_ratio = _safe_float(self.profit_ratio, 0.0) or 0.0
        self.model_edge_pct = _safe_float(self.model_edge_pct, 0.0) or 0.0
        self.sp_near = _safe_float(self.sp_near)
        self.sp_far = _safe_float(self.sp_far)
        self.edge_net = _safe_float(self.edge_net)
        self.context_score = _safe_int(self.context_score)
        self.profit = _safe_float(self.profit)
        self.bsp_actual = _safe_float(self.bsp_actual)

    @property
    def is_settled(self) -> bool:
        return self.won is not None

    @property
    def score_decile(self) -> str:
        score = max(0, min(100, int(self.score)))
        if score == 100:
            return "100"
        lo = (score // 10) * 10
        return f"{lo}-{lo + 9}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateRecord":
        return cls(**data)


class PredictionLedger:
    """JSON-backed candidate store with calibration summaries."""

    def __init__(self, path: str | os.PathLike = LEDGER_PATH):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._records: dict[str, CandidateRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            for item in data:
                record = CandidateRecord.from_dict(item)
                self._records[record.candidate_id] = record
        except Exception as exc:
            logger.warning("Could not load prediction ledger %s: %s", self.path, exc)

    def _save(self) -> None:
        records = sorted(
            self._records.values(),
            key=lambda r: r.scanned_at or "",
            reverse=True,
        )[:MAX_RECORDS]
        self.path.write_text(json.dumps([r.to_dict() for r in records], indent=2, default=str))

    def record_candidate(self, record: CandidateRecord | dict[str, Any]) -> None:
        if isinstance(record, dict):
            record = CandidateRecord.from_dict(record)
        with self._lock:
            self._records[record.candidate_id] = record
            self._save()

    def record_candidates(self, records: list[CandidateRecord | dict[str, Any]]) -> None:
        with self._lock:
            for record in records:
                if isinstance(record, dict):
                    record = CandidateRecord.from_dict(record)
                self._records[record.candidate_id] = record
            self._save()

    def update_outcome(
        self,
        candidate_id: str,
        *,
        won: bool,
        profit: float,
        settled_at: Optional[str] = None,
        bsp_actual: Optional[float] = None,
        result_source: str = "manual",
    ) -> bool:
        with self._lock:
            record = self._records.get(candidate_id)
            if not record:
                return False
            record.won = bool(won)
            record.profit = float(profit)
            record.settled_at = settled_at or _now_iso()
            record.bsp_actual = _safe_float(bsp_actual)
            record.result_source = result_source
            self._save()
            return True

    def attach_bet(
        self,
        candidate_id: str,
        *,
        bet_id: str,
        stake: float,
        liability: float,
    ) -> bool:
        with self._lock:
            record = self._records.get(candidate_id)
            if not record:
                return False
            record.bet_id = str(bet_id)
            record.stake = float(stake)
            record.liability = float(liability)
            record.decision = "SELECTED"
            self._save()
            return True

    def update_outcome_by_bet_id(
        self,
        bet_id: str,
        *,
        won: bool,
        profit: float,
        settled_at: Optional[str] = None,
        bsp_actual: Optional[float] = None,
        result_source: str = "manual",
    ) -> bool:
        with self._lock:
            for record in self._records.values():
                if str(record.bet_id) == str(bet_id):
                    record.won = bool(won)
                    record.profit = float(profit)
                    record.settled_at = settled_at or _now_iso()
                    record.bsp_actual = _safe_float(bsp_actual)
                    record.result_source = result_source
                    self._save()
                    return True
        return False

    def all_records(self) -> list[CandidateRecord]:
        return list(self._records.values())

    def settled_records(self) -> list[CandidateRecord]:
        return [r for r in self._records.values() if r.is_settled]

    def calibration_by_score_decile(self) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[CandidateRecord]] = {}
        for record in self._records.values():
            groups.setdefault(record.score_decile, []).append(record)

        out: dict[str, dict[str, Any]] = {}
        for decile, records in sorted(groups.items()):
            settled = [r for r in records if r.is_settled]
            runner_wins = sum(1 for r in settled if r.won is True)
            lay_wins = sum(1 for r in settled if r.won is False)
            predicted = [r.estimated_win_prob for r in settled if 0 <= r.estimated_win_prob <= 1]
            profit = sum(r.profit for r in settled if r.profit is not None)
            stake = sum(r.stake for r in settled if r.stake)
            brier = None
            if predicted:
                brier = sum(
                    (r.estimated_win_prob - (1.0 if r.won else 0.0)) ** 2
                    for r in settled
                    if 0 <= r.estimated_win_prob <= 1
                ) / len(predicted)

            out[decile] = {
                "candidates": len(records),
                "settled": len(settled),
                "runner_win_rate": round(runner_wins / len(settled) * 100, 2) if settled else 0.0,
                "lay_win_rate": round(lay_wins / len(settled) * 100, 2) if settled else 0.0,
                "avg_predicted_win_prob": round(sum(predicted) / len(predicted), 4) if predicted else None,
                "profit": round(profit, 2),
                "staked": round(stake, 2),
                "roi": round(profit / stake * 100, 2) if stake else None,
                "brier_score": round(brier, 6) if brier is not None else None,
            }
        return out

    def closing_line_value_stats(self) -> dict[str, Any]:
        records = [
            r for r in self._records.values()
            if r.lay_price and r.bsp_actual and r.lay_price > 1.0 and r.bsp_actual > 1.0
        ]
        if not records:
            return {"n": 0}

        diffs = [round(r.lay_price - (r.bsp_actual or 0), 4) for r in records]
        positive = sum(1 for d in diffs if d > 0)
        return {
            "n": len(records),
            "avg_lay_minus_bsp": round(sum(diffs) / len(diffs), 4),
            "positive_clv_rate": round(positive / len(records) * 100, 2),
        }

    def rejected_winner_analysis(self, limit: int = 10) -> dict[str, Any]:
        skipped_winners = [
            r for r in self._records.values()
            if r.decision.upper() == "SKIPPED" and r.won is True
        ]
        skipped_winners.sort(key=lambda r: r.score, reverse=True)
        return {
            "skipped_winners": len(skipped_winners),
            "examples": [
                {
                    "candidate_id": r.candidate_id,
                    "market_id": r.market_id,
                    "selection_id": r.selection_id,
                    "runner_name": r.runner_name,
                    "score": r.score,
                    "verdict": r.verdict,
                    "estimated_win_prob": r.estimated_win_prob,
                }
                for r in skipped_winners[:limit]
            ],
        }

    def summary(self) -> dict[str, Any]:
        settled = self.settled_records()
        return {
            "total_candidates": len(self._records),
            "settled_candidates": len(settled),
            "calibration_by_score_decile": self.calibration_by_score_decile(),
            "closing_line_value": self.closing_line_value_stats(),
            "rejected_winners": self.rejected_winner_analysis(),
        }


_prediction_ledger: PredictionLedger | None = None
_prediction_ledger_lock = threading.Lock()


def get_prediction_ledger(path: str | os.PathLike = LEDGER_PATH) -> PredictionLedger:
    global _prediction_ledger
    if _prediction_ledger is None:
        with _prediction_ledger_lock:
            if _prediction_ledger is None:
                _prediction_ledger = PredictionLedger(path=path)
    return _prediction_ledger
