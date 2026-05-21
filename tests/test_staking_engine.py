"""Tests for staking_engine.py — Kelly invariants, fixed staking, comparison."""

import pytest

from staking_engine import (
    kelly_fraction,
    kelly_lay_stake,
    recommend_stake,
    compare_staking_methods,
)


# ─────────────────────────────────────────────────────────────────────────────
# kelly_fraction — raw fraction calculator
# ─────────────────────────────────────────────────────────────────────────────

def test_kelly_fraction_returns_zero_when_no_edge():
    # win_prob exactly matches market implied prob (1/lay_price) → no edge.
    # lay_price = 2.0 → implied win_prob = 0.5.
    assert kelly_fraction(win_prob=0.5, lay_price=2.0) == pytest.approx(0.0, abs=1e-6)


def test_kelly_fraction_negative_when_market_overestimates_horse():
    # win_prob < market implied → market thinks horse wins more often than we do
    # → lay is positive EV → Kelly should be POSITIVE for a lay bet.
    # (We are betting the horse loses; if we think it loses more often than the
    # market does, that's lay-side value.)
    assert kelly_fraction(win_prob=0.3, lay_price=2.0) > 0


def test_kelly_fraction_negative_when_we_think_horse_wins_more():
    # win_prob > market implied → market thinks horse loses more often than we do
    # → laying is negative EV → Kelly negative.
    assert kelly_fraction(win_prob=0.7, lay_price=2.0) < 0


def test_kelly_fraction_zero_for_invalid_lay_price():
    assert kelly_fraction(win_prob=0.5, lay_price=1.0) == 0.0
    assert kelly_fraction(win_prob=0.5, lay_price=0.5) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# kelly_lay_stake — full Kelly recommendation
# ─────────────────────────────────────────────────────────────────────────────

def test_kelly_lay_stake_returns_no_bet_for_negative_edge():
    result = kelly_lay_stake(bankroll=1000, win_prob=0.7, lay_price=2.0)
    assert result["success"] is False
    assert "negative kelly" in result.get("reason", "").lower() or \
           "negative kelly" in result.get("message", "").lower()


def test_kelly_lay_stake_rejects_win_prob_out_of_range():
    for bad_p in (0.0, 1.0, -0.1, 1.5):
        result = kelly_lay_stake(bankroll=1000, win_prob=bad_p, lay_price=2.0)
        assert result["success"] is False


def test_kelly_lay_stake_rejects_invalid_lay_price():
    result = kelly_lay_stake(bankroll=1000, win_prob=0.5, lay_price=1.0)
    assert result["success"] is False


def test_kelly_lay_stake_rejects_non_positive_bankroll():
    result = kelly_lay_stake(bankroll=0, win_prob=0.3, lay_price=2.0)
    assert result["success"] is False
    result = kelly_lay_stake(bankroll=-100, win_prob=0.3, lay_price=2.0)
    assert result["success"] is False


def test_kelly_lay_stake_liability_never_exceeds_cap():
    # max_liability_pct=0.15 means liability cannot exceed 15% of bankroll.
    result = kelly_lay_stake(
        bankroll=1000,
        win_prob=0.10,        # huge edge
        lay_price=2.0,
        partial=1.0,
        max_liability_pct=0.15,
    )
    assert result["success"] is True
    assert result["liability"] <= 1000 * 0.15 + 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Kelly variants ordering: quarter < half < full (positive edge case)
# ─────────────────────────────────────────────────────────────────────────────

def test_kelly_variants_order_by_aggression():
    bankroll, win_prob, lay_price = 1000, 0.30, 2.0  # positive lay edge

    full    = kelly_lay_stake(bankroll, win_prob, lay_price, partial=1.0)
    half    = kelly_lay_stake(bankroll, win_prob, lay_price, partial=0.5)
    quarter = kelly_lay_stake(bankroll, win_prob, lay_price, partial=0.25)

    assert all(r["success"] for r in (full, half, quarter))
    assert quarter["backer_stake"] <= half["backer_stake"] <= full["backer_stake"]


# ─────────────────────────────────────────────────────────────────────────────
# recommend_stake — top-level dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def test_recommend_stake_default_is_half_kelly():
    bankroll, win_prob, lay_price = 1000, 0.30, 2.0
    default = recommend_stake(bankroll, win_prob, lay_price)  # method omitted
    half    = recommend_stake(bankroll, win_prob, lay_price, method="half_kelly")
    assert default["backer_stake"] == half["backer_stake"]


def test_recommend_stake_fixed_method_uses_fixed_liability():
    result = recommend_stake(
        bankroll=1000, win_prob=0.30, lay_price=2.0,
        method="fixed", fixed_liability=20.0,
    )
    assert result["success"] is True
    # Fixed liability of $20 at lay 2.0 → backer_stake = liability / (lay-1) = 20.
    assert result["liability"] == pytest.approx(20.0, abs=0.01)


def test_recommend_stake_unknown_method_falls_back_to_half_kelly():
    a = recommend_stake(1000, 0.30, 2.0, method="not_a_real_method")
    b = recommend_stake(1000, 0.30, 2.0, method="half_kelly")
    assert a["backer_stake"] == b["backer_stake"]


# ─────────────────────────────────────────────────────────────────────────────
# compare_staking_methods — returns multiple methods + recommendation
# ─────────────────────────────────────────────────────────────────────────────

def test_compare_staking_methods_returns_multiple_methods():
    result = compare_staking_methods(bankroll=1000, win_prob=0.30, lay_price=2.0)
    assert "comparison" in result
    assert len(result["comparison"]) >= 3  # at least kelly variants + fixed
    assert isinstance(result.get("recommendation", ""), str)
    assert result["recommendation"]  # non-empty


def test_stake_never_exceeds_bankroll_invariant():
    """For any positive-edge inputs, no recommended stake can risk > bankroll."""
    for win_prob in (0.10, 0.20, 0.30, 0.40):
        for lay_price in (1.5, 2.0, 3.0, 5.0):
            for bankroll in (100, 500, 1000, 10_000):
                result = recommend_stake(bankroll, win_prob, lay_price, method="half_kelly")
                if result.get("success"):
                    assert result["liability"] <= bankroll, (
                        f"Liability {result['liability']} > bankroll {bankroll} "
                        f"for win_prob={win_prob}, lay_price={lay_price}"
                    )
