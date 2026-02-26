"""
External Ratings Module
Downloads and caches Betfair's publicly available model ratings:
  - Kash Ratings Model (Australian thoroughbred horse racing)
  - Iggy Joey Model    (Australian greyhound racing)

Both are free, no authentication required.
Derived from the How-to-Automate tutorial series (scripts 3 & 4).

Usage:
    from external_ratings import RatingsCache
    cache = RatingsCache()
    kash = cache.get_kash()   # today's thoroughbred ratings
    iggy = cache.get_iggy()   # today's greyhound ratings
    edge = cache.model_edge(market_id, selection_id, current_lay, current_back)
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, date
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# URL BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

_BASE = "https://betfair-data-supplier-prod.herokuapp.com/api/widgets"

def _kash_url(dt: date) -> str:
    return f"{_BASE}/kash-ratings-model/datasets?date={dt.strftime('%Y-%m-%d')}&presenter=RatingsPresenter&csv=true"

def _iggy_url(dt: date) -> str:
    return f"{_BASE}/iggy-joey/datasets?date={dt.strftime('%Y-%m-%d')}&presenter=RatingsPresenter&csv=true"


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN MAPPING
# ─────────────────────────────────────────────────────────────────────────────

_RENAME = {
    "meetings.races.bfExchangeMarketId":         "market_id",
    "meetings.races.runners.bfExchangeSelectionId": "selection_id",
    "meetings.races.runners.ratedPrice":         "model_price",
    "meetings.races.number":                     "race_number",
    "meetings.name":                             "venue",
    "meetings.races.runners.name":               "runner_name",
    "meetings.races.runners.runnerId":           "runner_id",
}


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOADER
# ─────────────────────────────────────────────────────────────────────────────

def _download_ratings(url: str, label: str) -> Optional[pd.DataFrame]:
    """
    Download CSV from a Betfair ratings endpoint and return a tidy DataFrame.

    Returns None if the download fails or the CSV is empty.
    """
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
    except Exception as e:
        logger.warning(f"[{label}] Download failed: {e}")
        return None

    if df.empty:
        logger.warning(f"[{label}] Empty CSV returned — no races today?")
        return None

    # Rename known columns; ignore unknowns
    df = df.rename(columns={k: v for k, v in _RENAME.items() if k in df.columns})

    # Ensure key columns are correct types
    for col in ("market_id", "selection_id"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "model_price" in df.columns:
        df["model_price"] = pd.to_numeric(df["model_price"], errors="coerce")

    logger.info(f"[{label}] Downloaded {len(df)} runner rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────────────────────────────────────

class RatingsCache:
    """
    Thread-safe cache for today's Kash and Iggy model ratings.

    Ratings are downloaded once per session (or refreshed explicitly).
    Lookups are O(1) via a pre-built (market_id, selection_id) index.

    Example
    -------
    cache = RatingsCache()
    kash_df = cache.get_kash()   # full DataFrame
    edge    = cache.model_edge("1.234567", "98765432", current_lay=3.5)
    """

    def __init__(self, dt: Optional[date] = None):
        self._date     = dt or date.today()
        self._lock     = threading.Lock()
        self._kash_df: Optional[pd.DataFrame] = None
        self._iggy_df: Optional[pd.DataFrame] = None
        # Indexed: {(market_id, selection_id): model_price}
        self._kash_idx: dict[tuple[str, str], float] = {}
        self._iggy_idx: dict[tuple[str, str], float] = {}

    # ── Public getters ─────────────────────────────────────────────────────

    def get_kash(self, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """Return today's Kash thoroughbred ratings DataFrame (downloads on first call)."""
        with self._lock:
            if self._kash_df is None or force_refresh:
                self._kash_df = _download_ratings(_kash_url(self._date), "Kash")
                if self._kash_df is not None:
                    self._kash_idx = self._build_index(self._kash_df)
        return self._kash_df

    def get_iggy(self, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """Return today's Iggy greyhound ratings DataFrame (downloads on first call)."""
        with self._lock:
            if self._iggy_df is None or force_refresh:
                self._iggy_df = _download_ratings(_iggy_url(self._date), "Iggy")
                if self._iggy_df is not None:
                    self._iggy_idx = self._build_index(self._iggy_df)
        return self._iggy_df

    def get_model_price(
        self,
        market_id: str,
        selection_id: str | int,
        model: str = "kash",
    ) -> Optional[float]:
        """
        Look up the model-rated price for a specific runner.

        Args:
            market_id:    Betfair market ID e.g. "1.234567890"
            selection_id: Runner selection ID (int or str)
            model:        "kash" (thoroughbreds) or "iggy" (greyhounds)

        Returns:
            Rated price (float) or None if not found.
        """
        key = (str(market_id).strip(), str(selection_id).strip())
        if model == "kash":
            self.get_kash()          # ensure loaded
            return self._kash_idx.get(key)
        elif model == "iggy":
            self.get_iggy()
            return self._iggy_idx.get(key)
        return None

    def model_edge(
        self,
        market_id: str,
        selection_id: str | int,
        current_lay:  Optional[float] = None,
        current_back: Optional[float] = None,
        model: str = "kash",
    ) -> dict:
        """
        Compute the edge between the Betfair model price and current market prices.

        Strategy (from How-to-Automate tutorials):
          - If BACK price > model_price → BACK opportunity (market over-values the horse)
          - If LAY  price < model_price → LAY  opportunity (market under-values the horse)
          - If both → LAY takes priority (safer for the lay-focused agent)

        Returns:
            {
              "model_price":    float | None,
              "current_lay":    float | None,
              "current_back":   float | None,
              "lay_edge":       float | None,   # model_price - lay_price (positive = lay value)
              "back_edge":      float | None,   # back_price - model_price (positive = back value)
              "signal":         "LAY" | "BACK" | "BOTH" | "NONE" | "NO_MODEL_PRICE",
              "lay_value":      bool,           # True if lay price < model price
              "back_value":     bool,           # True if back price > model price
              "recommendation": str,            # Human-readable action
              "edge_pct":       float | None,   # Percentage edge on the preferred side
            }
        """
        model_price = self.get_model_price(market_id, selection_id, model)

        if model_price is None:
            return {
                "model_price": None,
                "current_lay": current_lay,
                "current_back": current_back,
                "lay_edge": None,
                "back_edge": None,
                "signal": "NO_MODEL_PRICE",
                "lay_value": False,
                "back_value": False,
                "recommendation": f"No {model.upper()} model price found for this runner.",
                "edge_pct": None,
            }

        lay_edge  = None
        back_edge = None
        lay_value  = False
        back_value = False

        if current_lay is not None:
            lay_edge  = round(model_price - current_lay, 3)
            lay_value = current_lay < model_price          # lay cheaper than model → lay EV

        if current_back is not None:
            back_edge  = round(current_back - model_price, 3)
            back_value = current_back > model_price        # back higher than model → back EV

        if lay_value and back_value:
            signal = "BOTH"
            # Agent is lay-focused → default to LAY when both signals present
            preferred_edge = lay_edge
            recommendation = (
                f"Both LAY and BACK edges detected. "
                f"LAY edge: {lay_edge:+.2f} ticks (lay at {current_lay} vs model {model_price}). "
                f"BACK edge: {back_edge:+.2f} ticks. "
                f"Prioritising LAY per agent rules."
            )
        elif lay_value:
            signal = "LAY"
            preferred_edge = lay_edge
            recommendation = (
                f"LAY opportunity: market lay price {current_lay} < model price {model_price}. "
                f"Edge: {lay_edge:+.2f} ticks. The market underestimates this horse."
            )
        elif back_value:
            signal = "BACK"
            preferred_edge = back_edge
            recommendation = (
                f"BACK opportunity: market back price {current_back} > model price {model_price}. "
                f"Edge: {back_edge:+.2f} ticks. The market overestimates this horse."
            )
        else:
            signal = "NONE"
            preferred_edge = None
            recommendation = (
                f"No edge detected. Market prices are in line with model price {model_price}."
            )

        edge_pct = None
        if preferred_edge is not None and model_price > 0:
            edge_pct = round(abs(preferred_edge) / model_price * 100, 2)

        return {
            "model":          model.upper(),
            "model_price":    model_price,
            "current_lay":    current_lay,
            "current_back":   current_back,
            "lay_edge":       lay_edge,
            "back_edge":      back_edge,
            "signal":         signal,
            "lay_value":      lay_value,
            "back_value":     back_value,
            "recommendation": recommendation,
            "edge_pct":       edge_pct,
        }

    def get_venue_markets(
        self,
        venue: str,
        model: str = "kash",
    ) -> list[dict]:
        """
        Return all markets at a given venue from today's ratings.

        Args:
            venue: Venue/track name (case-insensitive substring match)
            model: "kash" or "iggy"

        Returns:
            List of {market_id, race_number, venue, runners: [{selection_id, runner_name, model_price}]}
            sorted by race_number ascending.
        """
        df = self.get_kash() if model == "kash" else self.get_iggy()
        if df is None or df.empty:
            return []

        venue_lower = venue.lower().strip()
        mask = df.get("venue", pd.Series(dtype=str)).str.lower().str.contains(venue_lower, na=False)
        sub = df[mask]

        if sub.empty:
            return []

        markets = []
        for mid, group in sub.groupby("market_id"):
            runners = []
            for _, row in group.iterrows():
                runners.append({
                    "selection_id": row.get("selection_id", ""),
                    "runner_name":  row.get("runner_name", "Unknown"),
                    "model_price":  row.get("model_price"),
                })
            # Sort runners by model price (favourite first)
            runners.sort(key=lambda r: r.get("model_price") or 9999)

            markets.append({
                "market_id":   str(mid),
                "race_number": group["race_number"].iloc[0] if "race_number" in group.columns else "",
                "venue":       group["venue"].iloc[0]       if "venue" in group.columns else venue,
                "runner_count": len(runners),
                "runners":     runners,
            })

        # Sort by race number
        def _race_key(m):
            try:
                return int(str(m["race_number"]).replace("R", "").strip())
            except (ValueError, TypeError):
                return 99

        markets.sort(key=_race_key)
        return markets

    def to_dict(self, model: str = "kash") -> dict:
        """Return a summary dict for logging/debugging."""
        df = self.get_kash() if model == "kash" else self.get_iggy()
        if df is None:
            return {"loaded": False, "model": model}
        markets = df["market_id"].nunique() if "market_id" in df.columns else 0
        runners = len(df)
        return {
            "loaded":   True,
            "model":    model.upper(),
            "date":     self._date.isoformat(),
            "markets":  markets,
            "runners":  runners,
        }

    # ── Internal ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_index(df: pd.DataFrame) -> dict[tuple[str, str], float]:
        """Build fast (market_id, selection_id) → model_price lookup."""
        idx = {}
        if "market_id" not in df.columns or "selection_id" not in df.columns:
            return idx
        for _, row in df.iterrows():
            key = (str(row["market_id"]).strip(), str(row["selection_id"]).strip())
            price = row.get("model_price")
            if pd.notna(price):
                idx[key] = float(price)
        return idx


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON  (one cache per process)
# ─────────────────────────────────────────────────────────────────────────────

_cache: Optional[RatingsCache] = None


def get_ratings_cache(dt: Optional[date] = None) -> RatingsCache:
    """Return the module-level RatingsCache singleton, creating it if needed."""
    global _cache
    if _cache is None or (dt is not None and _cache._date != dt):
        _cache = RatingsCache(dt or date.today())
    return _cache
