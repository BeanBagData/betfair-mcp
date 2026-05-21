"""Tests for market_analyser.py — WOM, spread, and timing classifiers."""

import pytest

from market_analyser import weight_of_money, market_spread, timing_advice


# ─────────────────────────────────────────────────────────────────────────────
# weight_of_money — classifies BACK_HEAVY / LAY_HEAVY / BALANCED / NO_LIQUIDITY
#
# atb_rungs / atl_rungs are lists of {"price": float, "size": float} dicts.
# The function takes the first `depth` of each (default 5) and computes the
# fraction of total ladder liquidity on each side.
# ─────────────────────────────────────────────────────────────────────────────

def test_wom_back_heavy_when_back_volume_dominates():
    atb = [{"price": 3.0, "size": 1000}]
    atl = [{"price": 3.1, "size": 200}]
    result = weight_of_money(atb_rungs=atb, atl_rungs=atl)
    assert result["signal"] == "BACK_HEAVY"
    assert result["back_wom"] >= 0.60


def test_wom_lay_heavy_when_lay_volume_dominates():
    atb = [{"price": 3.0, "size": 200}]
    atl = [{"price": 3.1, "size": 1000}]
    result = weight_of_money(atb_rungs=atb, atl_rungs=atl)
    assert result["signal"] == "LAY_HEAVY"
    assert result["lay_wom"] >= 0.60


def test_wom_balanced_when_volumes_close():
    atb = [{"price": 3.0, "size": 500}]
    atl = [{"price": 3.1, "size": 500}]
    result = weight_of_money(atb_rungs=atb, atl_rungs=atl)
    assert result["signal"] == "BALANCED"


def test_wom_no_liquidity_when_total_volume_zero():
    result = weight_of_money(atb_rungs=[], atl_rungs=[])
    assert result["signal"] == "NO_LIQUIDITY"


def test_wom_threshold_is_60_percent():
    """Exactly 60% is BACK_HEAVY; 59% is BALANCED."""
    # 600/(600+400) = 0.60 → BACK_HEAVY
    atb = [{"price": 3.0, "size": 600}]
    atl = [{"price": 3.1, "size": 400}]
    assert weight_of_money(atb_rungs=atb, atl_rungs=atl)["signal"] == "BACK_HEAVY"

    # 590/(590+410) = 0.59 → BALANCED
    atb = [{"price": 3.0, "size": 590}]
    atl = [{"price": 3.1, "size": 410}]
    assert weight_of_money(atb_rungs=atb, atl_rungs=atl)["signal"] == "BALANCED"


# ─────────────────────────────────────────────────────────────────────────────
# market_spread — classifies TIGHT / NORMAL / WIDE / NO_MARKET
# ─────────────────────────────────────────────────────────────────────────────

def test_spread_no_market_when_prices_missing():
    assert market_spread(None, 3.0)["assessment"] == "NO_MARKET"
    assert market_spread(3.0, None)["assessment"] == "NO_MARKET"
    assert market_spread(None, None)["assessment"] == "NO_MARKET"


def test_spread_tight_when_best_back_equals_best_lay():
    """Same price both sides → 0 ticks → TIGHT."""
    result = market_spread(best_back_price=3.0, best_lay_price=3.0)
    assert result["assessment"] == "TIGHT"
    assert result["spread_ticks"] == 0


def test_spread_tight_for_one_tick_gap():
    # At price 3.0 a single tick is 0.05 (Betfair ladder).
    result = market_spread(best_back_price=3.0, best_lay_price=3.05)
    assert result["assessment"] == "TIGHT"
    assert result["spread_ticks"] <= 2


def test_spread_returns_input_prices():
    result = market_spread(best_back_price=3.0, best_lay_price=3.1)
    assert result["best_back"] == 3.0
    assert result["best_lay"] == 3.1


# ─────────────────────────────────────────────────────────────────────────────
# timing_advice — classifies TOO_EARLY / MONITOR / OPTIMAL / LAST_CHANCE / INPLAY
# Boundaries (from the source):
#   > 1800s → TOO_EARLY
#   > 600s  → MONITOR
#   > 120s  → OPTIMAL
#   > 0s    → LAST_CHANCE
#   ≤ 0s    → INPLAY
# ─────────────────────────────────────────────────────────────────────────────

def test_timing_too_early_for_far_future():
    assert timing_advice(seconds_to_jump=3600)["window"] == "TOO_EARLY"


def test_timing_monitor_in_middle_window():
    assert timing_advice(seconds_to_jump=900)["window"] == "MONITOR"


def test_timing_optimal_in_2_to_10_minute_window():
    for sec in (130, 300, 599):
        assert timing_advice(seconds_to_jump=sec)["window"] == "OPTIMAL", \
            f"Expected OPTIMAL at {sec}s"


def test_timing_last_chance_under_2_minutes():
    for sec in (1, 60, 119):
        assert timing_advice(seconds_to_jump=sec)["window"] == "LAST_CHANCE", \
            f"Expected LAST_CHANCE at {sec}s"


def test_timing_inplay_at_zero_or_below():
    assert timing_advice(seconds_to_jump=0)["window"] == "INPLAY"
    assert timing_advice(seconds_to_jump=-60)["window"] == "INPLAY"


def test_timing_returns_minutes_helper():
    result = timing_advice(seconds_to_jump=300)
    assert result["minutes_to_jump"] == pytest.approx(5.0, abs=0.1)
