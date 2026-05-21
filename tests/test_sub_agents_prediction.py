"""Prediction-quality integration tests for sub_agents.py."""

import pytest

import sub_agents
from opportunity_scoring import ScoreConfig
from prediction_ledger import PredictionLedger
from sub_agents import ExecutionAgent, RiskManagerAgent, SessionState, VenueAnalystAgent


class FakeCache:
    def market_book(self, market_id, fetch_fn):
        return fetch_fn()

    def bsp_predictions(self, market_id, fetch_fn):
        return fetch_fn()

    def race_metadata(self, market_id, fetch_fn):
        return fetch_fn()


class FakeRatings:
    def get_venue_markets(self, venue, model="kash"):
        if model == "kash":
            return [
                {
                    "market_id": "1.23",
                    "race_number": "R1",
                    "venue": venue,
                    "runners": [],
                }
            ]
        return []

    def model_edge(self, market_id, selection_id, current_lay=None, current_back=None, model="kash"):
        if str(selection_id) == "101":
            return {"signal": "LAY", "edge_pct": 4.0, "model_price": 1.85}
        return {"signal": "LAY", "edge_pct": 9.0, "model_price": 1.95}


class FakeClient:
    def __init__(self):
        self.place_calls = []

    def get_market_book(self, market_id):
        return {
            "success": True,
            "market_id": market_id,
            "status": "OPEN",
            "inplay": False,
            "runners": [
                {
                    "selection_id": 101,
                    "runner_name": "Moderate Edge",
                    "status": "ACTIVE",
                    "best_back": {"price": 1.64},
                    "best_lay": {"price": 1.67},
                    "back_prices": [{"price": 1.64, "size": 100}],
                    "lay_prices": [{"price": 1.67, "size": 100}],
                },
                {
                    "selection_id": 102,
                    "runner_name": "BSP Value",
                    "status": "ACTIVE",
                    "best_back": {"price": 1.62},
                    "best_lay": {"price": 1.65},
                    "back_prices": [{"price": 1.62, "size": 100}],
                    "lay_prices": [{"price": 1.65, "size": 500}],
                },
            ],
        }

    def get_sp_predictions(self, market_id):
        return {
            "success": True,
            "market_id": market_id,
            "runners": [
                {"selection_id": 101, "sp_near": 1.75, "sp_far": 1.72},
                {
                    "selection_id": 102,
                    "sp_near": 1.56,
                    "sp_far": 1.50,
                    "edge_analysis": {
                        "verdict": "STRONG_VALUE",
                        "edge_net": 0.04,
                        "estimated_win_prob_for_kelly": 0.6667,
                        "sp_far": 1.50,
                        "tick_delta": 10,
                    },
                },
            ],
        }

    def get_race_metadata(self, market_id):
        return {
            "success": True,
            "track_condition": "Soft6",
            "runners": [
                {"selection_id": 102, "barrier_group": "outside", "distance_group": "sprint"}
            ],
        }

    def place_lay_bet(self, **kwargs):
        self.place_calls.append(kwargs)
        return {"success": True, "bet_id": "paper-123"}


def test_venue_analyst_uses_bsp_edge_and_records_candidates(monkeypatch, tmp_path):
    monkeypatch.setattr(sub_agents, "_get_cache", lambda: FakeCache())
    ledger = PredictionLedger(path=tmp_path / "candidate_ledger.json")
    monkeypatch.setattr(sub_agents, "_get_prediction_ledger", lambda: ledger, raising=False)

    state = SessionState(venue="Bendigo", bankroll=100.0, remaining_bankroll=100.0)
    agent = VenueAnalystAgent(
        FakeClient(),
        FakeRatings(),
        state,
        score_config=ScoreConfig(min_profit_ratio=1.5, min_edge_pct=3.0),
    )

    result = agent.run()

    assert result["success"] is True
    top = result["top_opportunities"][0]
    assert top["selection_id"] == 102
    assert top["runner_name"] == "BSP Value"
    assert top["sp_edge"]["verdict"] == "STRONG_VALUE"
    assert top["estimated_win_prob_source"] == "sp"
    assert len(ledger.all_records()) == 2


def test_venue_analyst_feeds_market_movement_into_score(monkeypatch, tmp_path):
    class MovingClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def get_market_book(self, market_id):
            self.calls += 1
            book = super().get_market_book(market_id)
            runner = book["runners"][1]
            if self.calls >= 2:
                runner["best_back"] = {"price": 1.78}
                runner["best_lay"] = {"price": 1.80}
                runner["back_prices"] = [{"price": 1.78, "size": 100}]
                runner["lay_prices"] = [{"price": 1.80, "size": 500}]
            return book

    monkeypatch.setattr(sub_agents, "_get_cache", lambda: FakeCache())
    ledger = PredictionLedger(path=tmp_path / "candidate_ledger.json")
    monkeypatch.setattr(sub_agents, "_get_prediction_ledger", lambda: ledger, raising=False)
    sub_agents._movement_baselines.clear()

    state = SessionState(venue="Bendigo", bankroll=100.0, remaining_bankroll=100.0)
    agent = VenueAnalystAgent(MovingClient(), FakeRatings(), state)

    agent.run()
    result = agent.run()

    top = result["top_opportunities"][0]
    assert top["selection_id"] == 102
    assert top["movement_signal"]["signal"] == "DRIFTED"
    assert top["score_components"]["movement"] > 0


def test_risk_manager_allocates_by_runner_and_uses_estimated_win_prob(monkeypatch):
    captured_probs = []

    def fake_recommend_stake(**kwargs):
        captured_probs.append(kwargs["win_prob"])
        return {"success": True, "backer_stake": 10.0, "liability": 6.5}

    import staking_engine

    monkeypatch.setattr(staking_engine, "recommend_stake", fake_recommend_stake)

    state = SessionState(venue="Bendigo", bankroll=200.0, remaining_bankroll=200.0)
    state.opportunities = [
        {
            "market_id": "1.23",
            "selection_id": 101,
            "runner_name": "First",
            "verdict": "LAY",
            "best_lay": 1.65,
            "estimated_win_prob": 0.31,
            "model_edge": {"model_price": 2.2},
        },
        {
            "market_id": "1.23",
            "selection_id": 102,
            "runner_name": "Second",
            "verdict": "STRONG_LAY",
            "best_lay": 1.66,
            "estimated_win_prob": 0.27,
            "model_edge": {"model_price": 2.4},
        },
    ]

    result = RiskManagerAgent(None, None, state).run()

    assert result["success"] is True
    assert set(result["allocations"]) == {"1.23:101", "1.23:102"}
    assert captured_probs == [0.31, 0.27]


def test_polling_execution_uses_allocation_runner_key_and_passes_context(monkeypatch, tmp_path):
    client = FakeClient()
    ledger = PredictionLedger(path=tmp_path / "candidate_ledger.json")
    ledger.record_candidate(
        {
            "candidate_id": "scan-1:1.23:102",
            "market_id": "1.23",
            "selection_id": 102,
            "runner_name": "BSP Value",
            "score": 88,
            "verdict": "STRONG_LAY",
            "decision": "SELECTED",
        }
    )
    state = SessionState(venue="Bendigo", bankroll=100.0, remaining_bankroll=100.0)
    state.opportunities = [
        {
            "candidate_id": "scan-1:1.23:102",
            "market_id": "1.23",
            "selection_id": 102,
            "runner_name": "BSP Value",
            "verdict": "STRONG_LAY",
            "best_lay": 1.65,
            "profit_ratio": 1.538,
            "wom_signal": "LAY_HEAVY",
            "model_edge": {"signal": "LAY", "edge_pct": 9.0},
            "sp_edge": {"sp_far": 1.5, "edge_net": 0.04},
            "opportunity_score": 88,
        }
    ]
    state.allocations = {
        "1.23:102": {
            "market_id": "1.23",
            "selection_id": 102,
            "backer_stake": 5.0,
            "liability": 3.25,
        }
    }

    monkeypatch.setattr(sub_agents, "_get_prediction_ledger", lambda: ledger)

    result = ExecutionAgent(client, FakeRatings(), state).run(task="polling")

    assert result["placed"] == 1
    assert client.place_calls[0]["market_id"] == "1.23"
    assert client.place_calls[0]["selection_id"] == 102
    assert client.place_calls[0]["context"]["runner_name"] == "BSP Value"
    assert client.place_calls[0]["context"]["opportunity_score"] == 88
    assert client.place_calls[0]["context"]["candidate_id"] == "scan-1:1.23:102"
    assert ledger.all_records()[0].bet_id == "paper-123"
