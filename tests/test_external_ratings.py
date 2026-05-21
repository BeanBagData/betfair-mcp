"""Tests for external_ratings.RatingsCache.model_edge — signal classification."""

import pytest

from external_ratings import RatingsCache


@pytest.fixture
def cache_with_model_price(monkeypatch):
    """Return a RatingsCache whose get_model_price is patched to return a fixed value."""

    def _make(model_price):
        cache = RatingsCache()
        monkeypatch.setattr(cache, "get_model_price", lambda *args, **kwargs: model_price)
        return cache

    return _make


def test_lay_signal_when_lay_price_below_model_price(cache_with_model_price):
    cache = cache_with_model_price(model_price=3.0)
    result = cache.model_edge(market_id="1.1", selection_id=42, current_lay=2.5)
    assert result["signal"] == "LAY"
    assert result["lay_value"] is True
    assert result["lay_edge"] == pytest.approx(0.5, abs=0.001)


def test_back_signal_when_back_price_above_model_price(cache_with_model_price):
    cache = cache_with_model_price(model_price=3.0)
    result = cache.model_edge(market_id="1.1", selection_id=42, current_back=3.5)
    assert result["signal"] == "BACK"
    assert result["back_value"] is True
    assert result["back_edge"] == pytest.approx(0.5, abs=0.001)


def test_both_signal_prioritises_lay(cache_with_model_price):
    """If lay < model < back, both signals fire — agent is lay-focused → LAY wins."""
    cache = cache_with_model_price(model_price=3.0)
    result = cache.model_edge(
        market_id="1.1",
        selection_id=42,
        current_lay=2.5,
        current_back=3.5,
    )
    assert result["signal"] == "BOTH"
    assert result["lay_value"] is True
    assert result["back_value"] is True
    # Recommendation text should mention LAY priority.
    assert "lay" in result["recommendation"].lower()


def test_none_signal_when_prices_align_with_model(cache_with_model_price):
    cache = cache_with_model_price(model_price=3.0)
    result = cache.model_edge(
        market_id="1.1",
        selection_id=42,
        current_lay=3.0,
        current_back=3.0,
    )
    assert result["signal"] == "NONE"
    assert result["lay_value"] is False
    assert result["back_value"] is False


def test_no_model_price_signal(cache_with_model_price):
    cache = cache_with_model_price(model_price=None)
    result = cache.model_edge(market_id="1.1", selection_id=42, current_lay=2.5)
    assert result["signal"] == "NO_MODEL_PRICE"
    assert result["edge_pct"] is None


def test_edge_pct_is_relative_to_model_price(cache_with_model_price):
    cache = cache_with_model_price(model_price=4.0)
    result = cache.model_edge(market_id="1.1", selection_id=42, current_lay=3.0)
    # edge = 4.0 - 3.0 = 1.0; edge_pct = 1.0 / 4.0 * 100 = 25.0
    assert result["edge_pct"] == pytest.approx(25.0, abs=0.1)


def test_returns_model_name_uppercased(cache_with_model_price):
    cache = cache_with_model_price(model_price=3.0)
    result = cache.model_edge(market_id="1.1", selection_id=42, current_lay=2.5, model="kash")
    assert result["model"] == "KASH"

    result = cache.model_edge(market_id="1.1", selection_id=42, current_lay=2.5, model="iggy")
    assert result["model"] == "IGGY"
