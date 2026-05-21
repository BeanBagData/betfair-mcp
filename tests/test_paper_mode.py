"""Tests for PAPER_MODE — bet methods must not call Betfair when paper-mode is on."""

import json
import os
from unittest.mock import patch

import pytest

from betfair_client import BetfairClient, is_paper_mode


def _make_client(monkeypatch, paper_mode: str):
    monkeypatch.setenv("BETFAIR_USERNAME", "test")
    monkeypatch.setenv("BETFAIR_PASSWORD", "test")
    monkeypatch.setenv("BETFAIR_APP_KEY", "test")
    monkeypatch.setenv("PAPER_MODE", paper_mode)
    return BetfairClient()


def test_is_paper_mode_default_true(monkeypatch):
    monkeypatch.delenv("PAPER_MODE", raising=False)
    assert is_paper_mode() is True


def test_is_paper_mode_false_when_explicitly_disabled(monkeypatch):
    monkeypatch.setenv("PAPER_MODE", "false")
    assert is_paper_mode() is False


def test_is_paper_mode_accepts_common_truthy_values(monkeypatch):
    for value in ("true", "TRUE", "True", "1", "yes"):
        monkeypatch.setenv("PAPER_MODE", value)
        assert is_paper_mode() is True, f"PAPER_MODE={value!r} should be True"


def test_is_paper_mode_accepts_common_falsy_values(monkeypatch):
    for value in ("false", "FALSE", "0", "no"):
        monkeypatch.setenv("PAPER_MODE", value)
        assert is_paper_mode() is False, f"PAPER_MODE={value!r} should be False"


def test_place_lay_bet_does_not_call_betfair_in_paper_mode(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # so bet_log.json writes to tmp dir
    client = _make_client(monkeypatch, "true")

    with patch.object(client, "_betting_request") as mock_request:
        result = client.place_lay_bet(
            market_id="1.1",
            selection_id=42,
            lay_price=3.0,
            stake=10.0,
        )

    mock_request.assert_not_called()
    assert result["success"] is True
    assert result["paper"] is True
    assert result["bet_id"].startswith("paper-")
    assert result["lay_price"] == 3.0
    assert result["stake"] == 10.0
    assert result["liability"] == pytest.approx(20.0, abs=0.01)  # (3.0-1) * 10


def test_place_back_bet_does_not_call_betfair_in_paper_mode(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _make_client(monkeypatch, "true")

    with patch.object(client, "_betting_request") as mock_request:
        result = client.place_back_bet(
            market_id="1.1",
            selection_id=42,
            back_price=3.0,
            stake=10.0,
        )

    mock_request.assert_not_called()
    assert result["success"] is True
    assert result["paper"] is True
    assert result["bet_id"].startswith("paper-")


def test_cancel_order_refuses_real_bet_id_in_paper_mode(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _make_client(monkeypatch, "true")

    with patch.object(client, "_betting_request") as mock_request:
        result = client.cancel_order(market_id="1.1", bet_id="real-betfair-id-12345")

    mock_request.assert_not_called()
    assert result["success"] is False
    assert "paper" in result.get("error", "").lower() or "paper" in result.get("message", "").lower()


def test_cancel_order_removes_paper_bet_from_log(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _make_client(monkeypatch, "true")

    # Place a paper bet first.
    placed = client.place_lay_bet(
        market_id="1.1", selection_id=42, lay_price=3.0, stake=10.0,
    )
    bet_id = placed["bet_id"]
    assert bet_id.startswith("paper-")

    # Now cancel it.
    with patch.object(client, "_betting_request") as mock_request:
        cancel_result = client.cancel_order(market_id="1.1", bet_id=bet_id)

    mock_request.assert_not_called()
    assert cancel_result["success"] is True
    assert cancel_result.get("paper") is True


def test_paper_bet_appended_to_bet_log(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _make_client(monkeypatch, "true")

    client.place_lay_bet(
        market_id="1.1", selection_id=42, lay_price=3.0, stake=10.0,
    )

    log_path = tmp_path / "bet_log.json"
    assert log_path.exists()

    entries = json.loads(log_path.read_text())
    assert isinstance(entries, list)
    assert len(entries) == 1
    assert entries[0]["paper"] is True
    assert entries[0]["market_id"] == "1.1"
    assert entries[0]["bet_id"].startswith("paper-")


def test_paper_bet_uses_configured_log_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    log_path = tmp_path / "shared-state" / "bet_log.json"
    monkeypatch.setenv("PAPER_BET_LOG_PATH", str(log_path))
    client = _make_client(monkeypatch, "true")

    result = client.place_lay_bet(
        market_id="1.1", selection_id=42, lay_price=3.0, stake=10.0,
    )

    assert result["success"] is True
    assert log_path.exists()
    assert not (tmp_path / "bet_log.json").exists()


def test_paper_lay_bet_accepts_context_for_automation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _make_client(monkeypatch, "true")

    result = client.place_lay_bet(
        market_id="1.1",
        selection_id=42,
        lay_price=3.0,
        stake=10.0,
        strategy_ref="daily_paper",
        context={"venue": "Bendigo", "runner_name": "1. Fast One"},
    )

    assert result["success"] is True
    entries = json.loads((tmp_path / "bet_log.json").read_text())
    assert entries[0]["strategy_ref"] == "daily_paper"
    assert entries[0]["context"]["venue"] == "Bendigo"
    assert entries[0]["context"]["runner_name"] == "1. Fast One"


def test_place_lay_bet_calls_betfair_when_paper_mode_off(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    client = _make_client(monkeypatch, "false")

    with patch.object(client, "_betting_request") as mock_request:
        mock_request.return_value = {
            "success": True,
            "data": {
                "status": "SUCCESS",
                "instructionReports": [
                    {"betId": "real-12345", "orderStatus": "EXECUTABLE",
                     "placedDate": "2026-05-19T00:00:00Z",
                     "sizeMatched": 0, "averagePriceMatched": 0},
                ],
            },
        }
        result = client.place_lay_bet(
            market_id="1.1", selection_id=42, lay_price=3.0, stake=10.0,
        )

    mock_request.assert_called_once()
    assert result["success"] is True
    assert result.get("paper") is not True
    assert result["bet_id"] == "real-12345"


def test_cancel_order_refuses_cancel_all_in_paper_mode(monkeypatch, tmp_path):
    """Cancelling without a bet_id (cancel-all) is refused in paper mode."""
    monkeypatch.chdir(tmp_path)
    client = _make_client(monkeypatch, "true")

    with patch.object(client, "_betting_request") as mock_request:
        result = client.cancel_order(market_id="1.1", bet_id=None)

    mock_request.assert_not_called()
    assert result["success"] is False
    assert result["error"] == "PAPER_MODE_CANCEL_ALL_NOT_SUPPORTED"


def test_cancel_order_returns_not_found_for_unknown_paper_bet(monkeypatch, tmp_path):
    """Cancelling a paper-prefixed bet_id that isn't in the ledger returns NOT_FOUND."""
    monkeypatch.chdir(tmp_path)
    client = _make_client(monkeypatch, "true")

    with patch.object(client, "_betting_request") as mock_request:
        result = client.cancel_order(market_id="1.1", bet_id="paper-doesnotexist")

    mock_request.assert_not_called()
    assert result["success"] is False
    assert result["error"] == "PAPER_BET_NOT_FOUND"


def test_paper_bet_returns_failure_when_ledger_write_fails(monkeypatch, tmp_path):
    """If the ledger write raises, the bet method must return success=False."""
    monkeypatch.chdir(tmp_path)
    client = _make_client(monkeypatch, "true")

    with patch("betfair_client._append_paper_bet_log", return_value=False):
        result = client.place_lay_bet(
            market_id="1.1", selection_id=42, lay_price=3.0, stake=10.0,
        )

    assert result["success"] is False
    assert result["paper"] is True
    assert result["error"] == "PAPER_LEDGER_WRITE_FAILED"


def test_flumine_runner_refuses_to_start_in_paper_mode(monkeypatch):
    """The streaming path bypasses BetfairClient, so paper mode must block it up front."""
    monkeypatch.setenv("BETFAIR_USERNAME", "test")
    monkeypatch.setenv("BETFAIR_PASSWORD", "test")
    monkeypatch.setenv("BETFAIR_APP_KEY", "test")
    monkeypatch.setenv("PAPER_MODE", "true")

    from flumine_engine import FluminePaperModeError, FlumineRunner

    with pytest.raises(FluminePaperModeError) as exc_info:
        FlumineRunner()

    assert "PAPER_MODE=true" in str(exc_info.value)
