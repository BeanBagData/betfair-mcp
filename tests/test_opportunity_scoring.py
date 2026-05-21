"""Tests for unified lay opportunity scoring."""

import pytest

from opportunity_scoring import ScoreConfig, score_lay_opportunity


def test_score_rewards_aligned_bsp_model_wom_context_and_drift():
    result = score_lay_opportunity(
        lay_price=1.65,
        best_back=1.62,
        model_edge={"signal": "LAY", "edge_pct": 8.0, "model_price": 1.9},
        wom_signal="LAY_HEAVY",
        context_score=8,
        sp_edge={
            "verdict": "STRONG_VALUE",
            "edge_net": 0.041,
            "estimated_win_prob_for_kelly": 0.6667,
            "sp_far": 1.5,
            "tick_delta": 12,
        },
        movement_signal={"signal": "DRIFTED", "delta_ticks": 5},
        spread={"assessment": "TIGHT"},
    )

    assert result["verdict"] == "STRONG_LAY"
    assert result["score"] >= 80
    assert result["components"]["sp_value"] > 0
    assert result["components"]["model_edge"] > 0
    assert result["components"]["wom"] > 0
    assert result["components"]["movement"] > 0
    assert result["estimated_win_prob"] == pytest.approx(0.6667, abs=0.0001)
    assert result["estimated_win_prob_source"] == "sp"


def test_configurable_profit_and_edge_gates_affect_verdict():
    relaxed = score_lay_opportunity(
        lay_price=1.65,
        model_edge={"signal": "LAY", "edge_pct": 4.0, "model_price": 1.9},
        wom_signal="LAY_HEAVY",
        config=ScoreConfig(min_profit_ratio=1.5, min_edge_pct=3.0),
    )
    strict = score_lay_opportunity(
        lay_price=1.65,
        model_edge={"signal": "LAY", "edge_pct": 4.0, "model_price": 1.9},
        wom_signal="LAY_HEAVY",
        config=ScoreConfig(min_profit_ratio=2.0, min_edge_pct=7.0),
    )

    assert relaxed["verdict"] in {"LAY", "STRONG_LAY"}
    assert strict["verdict"] in {"MARGINAL", "SKIP"}
    assert "profit_ratio_below_minimum" in strict["issues"]
    assert "model_edge_below_minimum" in strict["issues"]


def test_estimated_win_probability_falls_back_to_model_then_lay_price():
    model_based = score_lay_opportunity(
        lay_price=2.0,
        model_edge={"signal": "LAY", "model_price": 4.0, "edge_pct": 50.0},
    )
    market_based = score_lay_opportunity(lay_price=2.0, model_edge={})

    assert model_based["estimated_win_prob"] == pytest.approx(0.25, abs=0.0001)
    assert model_based["estimated_win_prob_source"] == "model"
    assert market_based["estimated_win_prob"] == pytest.approx(0.5, abs=0.0001)
    assert market_based["estimated_win_prob_source"] == "market"
