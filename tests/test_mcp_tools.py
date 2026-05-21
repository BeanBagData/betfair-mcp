"""Tests for the MCP tools added to bring mcp_server up to parity with agent.py."""

import asyncio
import json
import sys

import pytest


@pytest.fixture
def fresh_mcp_server(monkeypatch, fake_creds):
    """Import a fresh copy of mcp_server with credentials set."""
    monkeypatch.delitem(sys.modules, "mcp_server", raising=False)
    import mcp_server
    monkeypatch.delitem(sys.modules, "mcp_server", raising=False)
    return mcp_server


def test_list_venue_markets_tool_registered(fresh_mcp_server):
    tools = asyncio.run(fresh_mcp_server.list_tools())
    names = {t.name for t in tools}
    assert "list_venue_markets" in names


def test_get_external_ratings_tool_registered(fresh_mcp_server):
    tools = asyncio.run(fresh_mcp_server.list_tools())
    names = {t.name for t in tools}
    assert "get_external_ratings" in names


def test_get_session_report_tool_registered(fresh_mcp_server):
    tools = asyncio.run(fresh_mcp_server.list_tools())
    names = {t.name for t in tools}
    assert "get_session_report" in names


def test_daily_paper_bet_status_tool_registered(fresh_mcp_server):
    tools = asyncio.run(fresh_mcp_server.list_tools())
    names = {t.name for t in tools}
    assert "get_daily_paper_bet_status" in names


def test_daily_paper_bet_status_returns_payload(fresh_mcp_server):
    result = asyncio.run(
        fresh_mcp_server.call_tool("get_daily_paper_bet_status", arguments={})
    )
    payload = json.loads(result[0].text)
    assert payload["success"] is True
    assert "scheduler" in payload


def test_get_session_report_returns_no_active_when_orchestrator_unset(
    fresh_mcp_server, monkeypatch, fake_creds
):
    """When no orchestration has run, the tool returns a structured 'no active session' response."""
    monkeypatch.setattr(fresh_mcp_server, "_orchestrator", None, raising=False)

    result = asyncio.run(
        fresh_mcp_server.call_tool("get_session_report", arguments={})
    )
    payload = json.loads(result[0].text)
    assert payload["success"] is False
    assert "no active" in payload["message"].lower()


def test_place_back_bet_tool_registered(fresh_mcp_server):
    tools = asyncio.run(fresh_mcp_server.list_tools())
    names = {t.name for t in tools}
    assert "place_back_bet" in names


def test_place_back_bet_via_mcp_uses_paper_mode(monkeypatch, fake_creds, tmp_path):
    """Placing a back bet through MCP must respect PAPER_MODE."""
    monkeypatch.setenv("PAPER_MODE", "true")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "mcp_server", raising=False)
    import mcp_server
    monkeypatch.delitem(sys.modules, "mcp_server", raising=False)

    result = asyncio.run(
        mcp_server.call_tool(
            "place_back_bet",
            arguments={
                "market_id":    "1.1",
                "selection_id": 42,
                "back_price":   3.0,
                "stake":        10.0,
            },
        )
    )
    payload = json.loads(result[0].text)
    assert payload["success"] is True
    assert payload["paper"] is True
    assert payload["bet_id"].startswith("paper-")


def test_start_stream_session_refuses_paper_mode(monkeypatch, fake_creds):
    """Flumine places live exchange orders, so MCP must block it in PAPER_MODE."""
    monkeypatch.setenv("PAPER_MODE", "true")
    monkeypatch.delitem(sys.modules, "mcp_server", raising=False)
    import mcp_server
    monkeypatch.delitem(sys.modules, "mcp_server", raising=False)

    result = asyncio.run(
        mcp_server.call_tool(
            "start_stream_session",
            arguments={"venue": "Doomben"},
        )
    )
    payload = json.loads(result[0].text)
    assert payload["success"] is False
    assert "PAPER_MODE=true" in payload["error"]


def test_performance_summary_includes_prediction_calibration(fresh_mcp_server, monkeypatch):
    class FakeClient:
        def get_cleared_orders(self, days_back):
            return {"success": True, "orders": [], "days_back": days_back}

    class FakeMemory:
        def sync_outcomes(self):
            return 0

        def overall_stats(self):
            return {}

        def recent_form(self, n):
            return {}

        def performance_by_price_bucket(self):
            return {}

        def performance_by_wom_signal(self):
            return {}

        def performance_by_model_edge(self):
            return {}

        def performance_by_venue(self):
            return {}

        def performance_by_timing(self):
            return {}

        def performance_by_model_signal(self):
            return {}

    class FakeLedger:
        def summary(self):
            return {
                "total_candidates": 3,
                "settled_candidates": 2,
                "calibration_by_score_decile": {"80-89": {"settled": 2}},
            }

    monkeypatch.setattr(fresh_mcp_server, "_get_client", lambda: FakeClient())
    monkeypatch.setattr(fresh_mcp_server, "get_betting_memory", lambda: FakeMemory())
    monkeypatch.setattr(fresh_mcp_server, "get_prediction_ledger", lambda: FakeLedger(), raising=False)

    result = asyncio.run(
        fresh_mcp_server.call_tool("get_performance_summary", arguments={"days_back": 1})
    )
    payload = json.loads(result[0].text)

    assert payload["success"] is True
    assert payload["memory_summary"]["prediction_calibration"]["total_candidates"] == 3


def test_log_bet_outcome_updates_prediction_ledger(fresh_mcp_server, monkeypatch):
    class FakeMemory:
        def update_outcome(self, **kwargs):
            self.kwargs = kwargs

    class FakeLedger:
        def __init__(self):
            self.calls = []

        def update_outcome_by_bet_id(self, bet_id, **kwargs):
            self.calls.append({"bet_id": bet_id, **kwargs})
            return True

    memory = FakeMemory()
    ledger = FakeLedger()
    monkeypatch.setattr(fresh_mcp_server, "get_betting_memory", lambda: memory)
    monkeypatch.setattr(fresh_mcp_server, "get_prediction_ledger", lambda: ledger)

    result = asyncio.run(
        fresh_mcp_server.call_tool(
            "log_bet_outcome",
            arguments={
                "bet_id": "paper-123",
                "won": False,
                "profit": 10.0,
                "bsp_actual": 1.8,
            },
        )
    )
    payload = json.loads(result[0].text)

    assert payload["success"] is True
    assert payload["prediction_ledger_updated"] is True
    assert ledger.calls[0]["bet_id"] == "paper-123"
    assert ledger.calls[0]["won"] is False
    assert ledger.calls[0]["profit"] == 10.0
