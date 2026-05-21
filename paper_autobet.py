"""Daily paper-bet automation.

This module runs a small background scheduler that places at least one
paper lay bet each day when PAPER_MODE is enabled. The selector uses live
Betfair market books and BSP estimates rather than the ratings feed, so it
keeps working even when external model downloads are flaky.
"""

from __future__ import annotations

import datetime as _dt
import argparse
import json
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

from betfair_client import is_paper_mode
from opportunity_scoring import ScoreConfig, score_lay_opportunity
from prediction_ledger import CandidateRecord, get_prediction_ledger
from staking_engine import estimate_edge_from_sp, recommend_stake

logger = logging.getLogger(__name__)

GREYHOUND_EVENT_TYPE_ID = "4339"
DEFAULT_TIMEZONE = "Australia/Melbourne"
DEFAULT_TARGET_TIME = "08:30"
DEFAULT_RETRY_MINUTES = 30
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MIN_LAY_PRICE = 1.02
STATE_PATH = Path("paper_autobet_state.json")
PAPER_SCORE_CONFIG = ScoreConfig(
    min_profit_ratio=0.2,
    min_edge_pct=0.0,
    lay_score=0,
    marginal_score=0,
)


@dataclass(frozen=True)
class PaperBetCandidate:
    candidate_id: str
    market_id: str
    market_name: str
    venue: str
    race_number: int
    selection_id: int
    runner_name: str
    lay_price: float
    best_back: float | None
    sp_near: float | None
    sp_far: float | None
    edge: dict[str, Any]
    score: float
    score_details: dict[str, Any]


def _load_state(path: Path = STATE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return {}


def _save_state(state: dict[str, Any], path: Path = STATE_PATH) -> None:
    try:
        path.write_text(json.dumps(state, indent=2, default=str))
    except Exception as exc:
        logger.warning("Could not save %s: %s", path, exc)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"false", "0", "no", "off", ""}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_scheduler_config_from_env() -> dict[str, Any]:
    """Read scheduler config from environment variables."""
    return {
        "timezone_name": os.environ.get("PAPER_AUTOBET_TZ", DEFAULT_TIMEZONE),
        "target_time": os.environ.get("PAPER_AUTOBET_TIME", DEFAULT_TARGET_TIME),
        "retry_minutes": _env_int("PAPER_AUTOBET_RETRY_MINUTES", DEFAULT_RETRY_MINUTES),
        "enabled": _env_flag("PAPER_AUTOBET_ENABLED", True),
        "max_attempts": _env_int("PAPER_AUTOBET_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS),
        "state_path": Path(os.environ.get("PAPER_AUTOBET_STATE_PATH", str(STATE_PATH))),
    }


def _parse_hhmm(raw: str) -> tuple[int, int]:
    try:
        hour_s, minute_s = raw.split(":", 1)
        hour = int(hour_s)
        minute = int(minute_s)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute
    except Exception:
        pass
    raise ValueError(f"Invalid HH:MM time: {raw!r}")


def _now_in_tz(tz_name: str, now: Optional[_dt.datetime] = None) -> _dt.datetime:
    tz = ZoneInfo(tz_name)
    return (now or _dt.datetime.now(tz)).astimezone(tz)


def _today_target(now: _dt.datetime, target_time: str) -> _dt.datetime:
    hour, minute = _parse_hhmm(target_time)
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _is_greyhound_race_market(market_name: str) -> bool:
    return bool(re.match(r"^R\d+\b", market_name or ""))


def discover_todays_greyhound_markets(
    client,
    now: Optional[_dt.datetime] = None,
    *,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> list[dict[str, Any]]:
    """Return today's Australian greyhound WIN race markets, sorted by start time."""
    local_now = _now_in_tz(timezone_name, now)
    local_end = local_now.replace(hour=23, minute=59, second=59, microsecond=0)
    utc_now = local_now.astimezone(ZoneInfo("UTC"))
    utc_end = local_end.astimezone(ZoneInfo("UTC"))

    params = {
        "filter": {
            "eventTypeIds": [GREYHOUND_EVENT_TYPE_ID],
            "marketCountries": ["AU"],
            "marketTypes": ["WIN"],
            "marketStartTime": {
                "from": utc_now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": utc_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        },
        "marketProjection": ["RUNNER_DESCRIPTION", "EVENT", "MARKET_START_TIME"],
        "maxResults": "200",
        "sort": "FIRST_TO_START",
    }

    result = client._betting_request("listMarketCatalogue", params)
    if not result.get("success"):
        raise RuntimeError(result.get("error", "Unable to list greyhound markets"))

    markets: list[dict[str, Any]] = []
    for market in result.get("data", []):
        market_name = market.get("marketName", "") or ""
        if not _is_greyhound_race_market(market_name):
            continue
        event = market.get("event", {}) or {}
        race_match = re.match(r"^R(\d+)\b", market_name)
        if not race_match:
            continue
        markets.append(
            {
                "market_id": market.get("marketId", ""),
                "market_name": market_name,
                "venue": event.get("venue") or event.get("name") or "",
                "race_number": int(race_match.group(1)),
                "start_time": market.get("marketStartTime"),
            }
        )

    markets.sort(key=lambda m: (m["start_time"] or "", m["race_number"]))
    return markets


def rank_daily_candidates(client, markets: list[dict[str, Any]]) -> list[PaperBetCandidate]:
    """Build and rank daily paper-bet candidates from live market data."""
    candidates: list[PaperBetCandidate] = []

    for market in markets:
        market_id = market["market_id"]
        market_name = market["market_name"]
        venue = market["venue"]
        race_number = market["race_number"]

        book = client.get_market_book(market_id)
        if not book.get("success") or book.get("inplay"):
            continue

        try:
            sp_result = client.get_sp_predictions(market_id)
        except Exception as exc:
            logger.info("SP lookup failed for %s: %s", market_id, exc)
            sp_result = {"success": False, "runners": []}

        sp_map: dict[str, dict[str, Any]] = {}
        for runner in sp_result.get("runners", []):
            sp_map[str(runner.get("selection_id"))] = runner

        for runner in book.get("runners", []):
            if runner.get("status") != "ACTIVE":
                continue

            best_lay = (runner.get("best_lay") or {}).get("price")
            best_back = (runner.get("best_back") or {}).get("price")
            if best_lay is None or float(best_lay) <= DEFAULT_MIN_LAY_PRICE:
                continue

            sel_id = int(runner.get("selection_id"))
            sp_data = sp_map.get(str(sel_id), {})
            sp_near = sp_data.get("sp_near")
            sp_far = sp_data.get("sp_far")
            edge = estimate_edge_from_sp(float(best_lay), sp_near, sp_far)
            score_details = score_lay_opportunity(
                lay_price=float(best_lay),
                best_back=best_back,
                sp_edge=edge,
                config=PAPER_SCORE_CONFIG,
            )
            score = float(score_details.get("score", 0.0))

            candidates.append(
                PaperBetCandidate(
                    candidate_id=f"daily:{market_id}:{sel_id}",
                    market_id=market_id,
                    market_name=market_name,
                    venue=venue,
                    race_number=race_number,
                    selection_id=sel_id,
                    runner_name=str(runner.get("runner_name", "Unknown")),
                    lay_price=float(best_lay),
                    best_back=float(best_back) if best_back is not None else None,
                    sp_near=float(sp_near) if sp_near is not None else None,
                    sp_far=float(sp_far) if sp_far is not None else None,
                    edge=edge,
                    score=score,
                    score_details=score_details,
                )
            )

    candidates.sort(key=lambda c: (c.score, c.edge.get("edge_net") or -999), reverse=True)
    return candidates


def _bankroll_from_client(client) -> float:
    funds = client.get_account_funds()
    if not funds.get("success"):
        return 0.0
    return float(funds.get("available_to_bet", 0.0))


def run_daily_paper_bet(
    client,
    *,
    now: Optional[_dt.datetime] = None,
    timezone_name: str = DEFAULT_TIMEZONE,
    bankroll: Optional[float] = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    fixed_liability: float = 5.0,
) -> dict[str, Any]:
    """Run one daily paper-bet placement attempt."""
    local_now = _now_in_tz(timezone_name, now)

    if not is_paper_mode():
        return {
            "success": False,
            "paper_mode": False,
            "attempted": False,
            "message": "Daily paper-bet automation is disabled when PAPER_MODE=false.",
        }

    if bankroll is None:
        bankroll = _bankroll_from_client(client)

    markets = discover_todays_greyhound_markets(client, local_now, timezone_name=timezone_name)
    candidates = rank_daily_candidates(client, markets)

    if candidates:
        try:
            get_prediction_ledger().record_candidates([
                CandidateRecord(
                    candidate_id=candidate.candidate_id,
                    market_id=candidate.market_id,
                    selection_id=candidate.selection_id,
                    runner_name=candidate.runner_name,
                    venue=candidate.venue,
                    race_name=candidate.market_name,
                    race_number=str(candidate.race_number),
                    decision="SKIPPED",
                    verdict=candidate.score_details.get("verdict", "SKIP"),
                    score=int(candidate.score_details.get("score", candidate.score)),
                    lay_price=candidate.lay_price,
                    best_back=candidate.best_back,
                    estimated_win_prob=candidate.score_details.get("estimated_win_prob", 0.0),
                    estimated_win_prob_source=candidate.score_details.get("estimated_win_prob_source", "UNKNOWN"),
                    profit_ratio=candidate.score_details.get("profit_ratio", 0.0),
                    wom_signal=candidate.score_details.get("wom_signal", "UNKNOWN"),
                    model_signal=candidate.edge.get("verdict", "UNKNOWN"),
                    model_edge_pct=candidate.edge.get("edge_pct", 0.0) or 0.0,
                    sp_near=candidate.sp_near,
                    sp_far=candidate.sp_far,
                    edge_net=candidate.edge.get("edge_net"),
                    issues=candidate.score_details.get("issues", []),
                    components=candidate.score_details.get("components", {}),
                )
                for candidate in candidates
            ])
        except Exception as exc:
            logger.info("Prediction ledger candidate recording failed: %s", exc)

    if not candidates:
        return {
            "success": False,
            "paper_mode": True,
            "attempted": False,
            "date": local_now.date().isoformat(),
            "message": "No active greyhound runners found for today.",
        }

    placed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for candidate in candidates[: max(1, max_attempts)]:
        stake_plan = recommend_stake(
            bankroll=max(bankroll, 0.0),
            win_prob=float(
                candidate.score_details.get("estimated_win_prob")
                or candidate.edge.get("estimated_win_prob_for_kelly")
                or candidate.edge.get("market_win_prob")
                or 1.0 / candidate.lay_price
            ),
            lay_price=candidate.lay_price,
            method="half_kelly",
        )

        if not stake_plan.get("success"):
            stake = round(min(max(fixed_liability / max(candidate.lay_price - 1.0, 0.01), 2.0), max(bankroll * 0.02, 5.0)), 2)
        else:
            stake = float(stake_plan.get("backer_stake", fixed_liability))
            if stake <= 0:
                stake = fixed_liability

        context = {
            "venue": candidate.venue,
            "race_name": candidate.market_name,
            "runner_name": candidate.runner_name,
            "wom_signal": candidate.edge.get("verdict", "UNKNOWN"),
            "model_signal": candidate.edge.get("verdict", "UNKNOWN"),
            "model_edge_pct": candidate.edge.get("edge_pct", 0.0) or 0.0,
            "timing_window": "DAILY_AUTOBET",
            "profit_ratio": round(1.0 / (candidate.lay_price - 1.0), 3) if candidate.lay_price > 1.0 else 0.0,
            "sp_far": candidate.sp_far,
            "edge_vs_sp": candidate.edge.get("edge_raw"),
            "opportunity_score": int(candidate.score_details.get("score", candidate.score)),
            "score_verdict": candidate.score_details.get("verdict"),
            "score_components": candidate.score_details.get("components", {}),
            "estimated_win_prob": candidate.score_details.get("estimated_win_prob"),
            "estimated_win_prob_source": candidate.score_details.get("estimated_win_prob_source"),
            "candidate_id": candidate.candidate_id,
        }

        result = client.place_lay_bet(
            market_id=candidate.market_id,
            selection_id=candidate.selection_id,
            lay_price=candidate.lay_price,
            stake=stake,
            strategy_ref="daily_paper",
            context=context,
        )

        if result.get("success"):
            try:
                get_prediction_ledger().attach_bet(
                    candidate.candidate_id,
                    bet_id=str(result.get("bet_id")),
                    stake=stake,
                    liability=round((candidate.lay_price - 1.0) * stake, 2),
                )
            except Exception as exc:
                logger.info("Prediction ledger bet linkage failed: %s", exc)
            placed.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "market_id": candidate.market_id,
                    "market_name": candidate.market_name,
                    "venue": candidate.venue,
                    "race_number": candidate.race_number,
                    "runner_name": candidate.runner_name,
                    "selection_id": candidate.selection_id,
                    "lay_price": candidate.lay_price,
                    "stake": stake,
                    "bet_id": result.get("bet_id"),
                    "score": candidate.score,
                    "score_details": candidate.score_details,
                    "edge": candidate.edge,
                }
            )
            break

        skipped.append(
            {
                "market_id": candidate.market_id,
                "runner_name": candidate.runner_name,
                "reason": result.get("error") or result.get("message") or "Unknown",
            }
        )

    success = bool(placed)
    summary = (
        f"Placed {len(placed)} paper bet(s) from {len(candidates)} candidate(s)."
        if success
        else f"Scanned {len(candidates)} candidate(s) but placed no paper bets."
    )

    return {
        "success": success,
        "paper_mode": True,
        "attempted": True,
        "date": local_now.date().isoformat(),
        "bankroll": round(float(bankroll), 2),
        "market_count": len(markets),
        "candidate_count": len(candidates),
        "placed": placed,
        "skipped": skipped,
        "summary": summary,
    }


class DailyPaperBetScheduler:
    """Background scheduler that runs `run_daily_paper_bet` once per day."""

    def __init__(
        self,
        client_factory: Callable[[], Any],
        *,
        timezone_name: str = DEFAULT_TIMEZONE,
        target_time: str = DEFAULT_TARGET_TIME,
        retry_minutes: int = DEFAULT_RETRY_MINUTES,
        enabled: Optional[bool] = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        state_path: Path = STATE_PATH,
    ):
        self._client_factory = client_factory
        self._timezone_name = timezone_name
        self._target_time = target_time
        self._retry_minutes = retry_minutes
        self._max_attempts = max_attempts
        if enabled is None:
            enabled = _env_flag("PAPER_AUTOBET_ENABLED", True)
        self._enabled = bool(is_paper_mode() and enabled)
        self._state_path = state_path
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state = _load_state(self._state_path)
        self._last_result: dict[str, Any] | None = self._state.get("last_result")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def last_result(self) -> dict[str, Any] | None:
        return self._last_result

    def status(self) -> dict[str, Any]:
        now = _now_in_tz(self._timezone_name)
        next_run = self._next_run_at(now)
        return {
            "enabled": self._enabled,
            "paper_mode": is_paper_mode(),
            "timezone": self._timezone_name,
            "target_time": self._target_time,
            "retry_minutes": self._retry_minutes,
            "next_run_at": next_run.isoformat() if next_run else None,
            "last_attempt_at": self._state.get("last_attempt_at"),
            "last_success_at": self._state.get("last_success_at"),
            "last_result": self._last_result,
        }

    def start(self) -> bool:
        if not self._enabled or not is_paper_mode():
            logger.info("Daily paper-bet scheduler disabled (enabled=%s, paper_mode=%s)", self._enabled, is_paper_mode())
            return False
        if self._thread and self._thread.is_alive():
            return True

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="daily-paper-bets")
        self._thread.start()
        logger.info("Daily paper-bet scheduler started (tz=%s, target=%s)", self._timezone_name, self._target_time)
        return True

    def stop(self) -> None:
        self._stop_event.set()

    def run_once(self) -> dict[str, Any]:
        return self._run_once_at(None)

    def _run_once_at(self, now: Optional[_dt.datetime]) -> dict[str, Any]:
        local_now = _now_in_tz(self._timezone_name, now)
        client = self._client_factory()
        if hasattr(client, "login"):
            login_result = client.login()
            if not login_result.get("success"):
                result = {
                    "success": False,
                    "paper_mode": is_paper_mode(),
                    "attempted": True,
                    "message": f"Login failed: {login_result.get('message') or login_result.get('error')}",
                }
                self._record_attempt(result, success=False, now=local_now)
                return result

        result = run_daily_paper_bet(
            client,
            now=local_now,
            bankroll=None,
            timezone_name=self._timezone_name,
            max_attempts=self._max_attempts,
        )
        self._record_attempt(result, success=bool(result.get("success")), now=local_now)
        return result

    def run_due_once(self, now: Optional[_dt.datetime] = None) -> dict[str, Any]:
        """Run one automatic attempt only if today's scheduled attempt is due."""
        local_now = _now_in_tz(self._timezone_name, now)

        if not self._enabled or not is_paper_mode():
            return {
                "success": False,
                "paper_mode": is_paper_mode(),
                "attempted": False,
                "reason": "disabled",
                "date": local_now.date().isoformat(),
                "message": "Daily paper-bet automation is disabled.",
            }

        if self._success_recorded_for_date(local_now.date()):
            return {
                "success": True,
                "paper_mode": True,
                "attempted": False,
                "reason": "already_placed_today",
                "date": local_now.date().isoformat(),
                "next_run_at": self._next_run_at(local_now).isoformat(),
                "message": "A daily paper bet has already been placed for this date.",
            }

        next_run = self._next_run_at(local_now)
        if next_run > local_now:
            return {
                "success": True,
                "paper_mode": True,
                "attempted": False,
                "reason": "not_due",
                "date": local_now.date().isoformat(),
                "next_run_at": next_run.isoformat(),
                "message": "Daily paper-bet automation is not due yet.",
            }

        return self._run_once_at(local_now)

    def _record_attempt(
        self,
        result: dict[str, Any],
        *,
        success: bool,
        now: Optional[_dt.datetime] = None,
    ) -> None:
        local_now = _now_in_tz(self._timezone_name, now)
        with self._lock:
            self._state["enabled"] = self._enabled
            self._state["timezone"] = self._timezone_name
            self._state["target_time"] = self._target_time
            self._state["retry_minutes"] = self._retry_minutes
            self._state["last_attempt_at"] = local_now.isoformat()
            self._state["last_result"] = result
            if success:
                self._state["last_success_at"] = local_now.isoformat()
            self._last_result = result
            _save_state(self._state, self._state_path)

    def _success_recorded_for_date(self, local_date: _dt.date) -> bool:
        last_success_at = self._state.get("last_success_at")
        if not last_success_at:
            return False
        try:
            last_success_dt = _dt.datetime.fromisoformat(last_success_at)
            return last_success_dt.astimezone(ZoneInfo(self._timezone_name)).date() == local_date
        except Exception:
            return False

    def _next_run_at(self, now: _dt.datetime) -> _dt.datetime:
        target_today = _today_target(now, self._target_time)
        if self._success_recorded_for_date(now.date()):
            tomorrow = now + _dt.timedelta(days=1)
            return _today_target(tomorrow, self._target_time)

        last_attempt_at = self._state.get("last_attempt_at")
        if last_attempt_at:
            try:
                last_attempt_dt = _dt.datetime.fromisoformat(last_attempt_at).astimezone(ZoneInfo(self._timezone_name))
                if last_attempt_dt.date() == now.date() and last_attempt_dt < now:
                    return max(
                        now,
                        last_attempt_dt + _dt.timedelta(minutes=self._retry_minutes),
                    )
            except Exception:
                pass

        if now < target_today:
            return target_today
        return now

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            now = _now_in_tz(self._timezone_name)
            next_run = self._next_run_at(now)
            wait_seconds = max(0.0, (next_run - now).total_seconds())
            if self._stop_event.wait(wait_seconds):
                return

            try:
                result = self.run_once()
                logger.info("Daily paper-bet scheduler run completed: %s", result.get("summary"))
            except Exception as exc:
                logger.exception("Daily paper-bet scheduler run failed")
                self._record_attempt(
                    {
                        "success": False,
                        "paper_mode": is_paper_mode(),
                        "attempted": True,
                        "message": str(exc),
                    },
                    success=False,
                )


def _load_dotenv_file() -> None:
    """Load the project .env file when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(Path(__file__).resolve().with_name(".env"))


def build_scheduler_from_env() -> DailyPaperBetScheduler:
    """Create a scheduler using environment configuration and BetfairClient."""
    from betfair_client import BetfairClient

    cfg = load_scheduler_config_from_env()
    return DailyPaperBetScheduler(lambda: BetfairClient(), **cfg)


def cli(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Daily Betfair paper-bet automation")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run-due",
        action="store_true",
        help="Run a paper-bet attempt only if today's scheduled run is due.",
    )
    mode.add_argument(
        "--force",
        action="store_true",
        help="Force one paper-bet attempt now, ignoring the daily due check.",
    )
    mode.add_argument(
        "--status",
        action="store_true",
        help="Print scheduler status without placing a paper bet.",
    )
    args = parser.parse_args(argv)

    _load_dotenv_file()
    scheduler = build_scheduler_from_env()

    if args.status:
        result = {"success": True, "scheduler": scheduler.status()}
    elif args.force:
        result = scheduler.run_once()
    else:
        result = scheduler.run_due_once()

    print(json.dumps(result, indent=2, default=str))
    if result.get("success") or not result.get("attempted"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))
