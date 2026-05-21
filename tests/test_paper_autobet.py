"""Tests for the daily paper-bet scheduler."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from paper_autobet import DailyPaperBetScheduler, load_scheduler_config_from_env


class FakeClient:
    def __init__(self):
        self.place_calls = []

    def login(self):
        return {"success": True}

    def get_account_funds(self):
        return {"success": True, "available_to_bet": 100.0}

    def _betting_request(self, endpoint, params):
        assert endpoint == "listMarketCatalogue"
        return {
            "success": True,
            "data": [
                {
                    "marketId": "1.23",
                    "marketName": "R1 350m Mdn",
                    "marketStartTime": "2026-05-20T00:00:00Z",
                    "event": {"venue": "Bendigo"},
                    "runners": [],
                }
            ],
        }

    def get_market_book(self, market_id):
        assert market_id == "1.23"
        return {
            "success": True,
            "inplay": False,
            "runners": [
                {
                    "selection_id": 101,
                    "runner_name": "1. Fast One",
                    "status": "ACTIVE",
                    "best_back": {"price": 2.0},
                    "best_lay": {"price": 2.2},
                },
                {
                    "selection_id": 102,
                    "runner_name": "2. Slow One",
                    "status": "ACTIVE",
                    "best_back": {"price": 4.8},
                    "best_lay": {"price": 5.0},
                },
            ],
        }

    def get_sp_predictions(self, market_id):
        assert market_id == "1.23"
        return {
            "success": True,
            "runners": [
                {"selection_id": 101, "sp_near": 2.8, "sp_far": 3.0},
                {"selection_id": 102, "sp_near": 4.0, "sp_far": 4.1},
            ],
        }

    def place_lay_bet(self, **kwargs):
        self.place_calls.append(kwargs)
        return {
            "success": True,
            "paper": True,
            "bet_id": "paper-abc123",
        }


def test_scheduler_places_best_daily_paper_bet(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPER_MODE", "true")
    monkeypatch.setenv("PAPER_AUTOBET_ENABLED", "true")
    monkeypatch.setenv("PAPER_AUTOBET_TIME", "08:30")
    monkeypatch.setenv("PAPER_AUTOBET_TZ", "Australia/Melbourne")

    client = FakeClient()
    state_path = tmp_path / "paper_autobet_state.json"
    scheduler = DailyPaperBetScheduler(
        lambda: client,
        state_path=state_path,
        enabled=True,
        max_attempts=3,
    )

    result = scheduler.run_once()

    assert result["success"] is True
    assert result["placed"]
    assert client.place_calls[0]["selection_id"] == 102
    assert client.place_calls[0]["market_id"] == "1.23"
    assert client.place_calls[0]["strategy_ref"] == "daily_paper"
    assert "score_components" in client.place_calls[0]["context"]
    assert client.place_calls[0]["context"]["estimated_win_prob_source"] == "sp"
    assert state_path.exists()
    assert scheduler.status()["last_success_at"] is not None


def test_scheduler_records_paper_candidates_in_prediction_ledger(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPER_MODE", "true")

    from prediction_ledger import PredictionLedger
    import paper_autobet

    ledger = PredictionLedger(path=tmp_path / "candidate_ledger.json")
    monkeypatch.setattr(paper_autobet, "get_prediction_ledger", lambda: ledger, raising=False)

    client = FakeClient()
    scheduler = DailyPaperBetScheduler(
        lambda: client,
        state_path=tmp_path / "paper_autobet_state.json",
        enabled=True,
        max_attempts=3,
    )

    result = scheduler.run_once()

    assert result["success"] is True
    records = ledger.all_records()
    assert len(records) == 2
    linked = [record for record in records if record.bet_id == "paper-abc123"]
    assert len(linked) == 1
    assert linked[0].decision == "SELECTED"
    assert linked[0].stake > 0


def test_rank_daily_candidates_exposes_shared_score_details():
    client = FakeClient()
    markets = [
        {
            "market_id": "1.23",
            "market_name": "R1 350m Mdn",
            "venue": "Bendigo",
            "race_number": 1,
        }
    ]

    from paper_autobet import rank_daily_candidates

    candidates = rank_daily_candidates(client, markets)

    assert candidates
    assert candidates[0].score_details["verdict"] in {"LAY", "STRONG_LAY"}
    assert "sp_value" in candidates[0].score_details["components"]


def test_run_due_once_waits_until_target_time(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPER_MODE", "true")

    client = FakeClient()
    scheduler = DailyPaperBetScheduler(
        lambda: client,
        timezone_name="UTC",
        target_time="08:30",
        state_path=tmp_path / "paper_autobet_state.json",
        enabled=True,
    )

    result = scheduler.run_due_once(
        now=datetime(2026, 5, 21, 8, 0, tzinfo=ZoneInfo("UTC"))
    )

    assert result["attempted"] is False
    assert result["reason"] == "not_due"
    assert client.place_calls == []


def test_run_due_once_does_not_duplicate_successful_day(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPER_MODE", "true")

    client = FakeClient()
    scheduler = DailyPaperBetScheduler(
        lambda: client,
        timezone_name="UTC",
        target_time="08:30",
        state_path=tmp_path / "paper_autobet_state.json",
        enabled=True,
    )

    first = scheduler.run_due_once(
        now=datetime(2026, 5, 21, 9, 0, tzinfo=ZoneInfo("UTC"))
    )
    second = scheduler.run_due_once(
        now=datetime(2026, 5, 21, 10, 0, tzinfo=ZoneInfo("UTC"))
    )

    assert first["success"] is True
    assert second["attempted"] is False
    assert second["reason"] == "already_placed_today"
    assert len(client.place_calls) == 1


def test_scheduler_config_reads_env(monkeypatch):
    monkeypatch.setenv("PAPER_AUTOBET_ENABLED", "false")
    monkeypatch.setenv("PAPER_AUTOBET_TIME", "09:15")
    monkeypatch.setenv("PAPER_AUTOBET_TZ", "UTC")
    monkeypatch.setenv("PAPER_AUTOBET_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("PAPER_AUTOBET_RETRY_MINUTES", "12")

    cfg = load_scheduler_config_from_env()

    assert cfg["enabled"] is False
    assert cfg["target_time"] == "09:15"
    assert cfg["timezone_name"] == "UTC"
    assert cfg["max_attempts"] == 4
    assert cfg["retry_minutes"] == 12
