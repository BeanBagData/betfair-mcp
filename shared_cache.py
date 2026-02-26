"""
SharedCache — Thread-safe TTL cache for Betfair API responses.

All sub-agents and the main agent share a single cache instance so
the same market book, ratings, BSP, or metadata is never fetched twice
within its TTL window.

TTL defaults (tuned for racing):
  market_book   5s   — prices move fast, stale after one tick
  wom           10s  — WOM is computed from the book, same TTL + buffer
  bsp           60s  — BSP predictions refresh every ~30s
  metadata      3600s— jockey/barrier/distance don't change
  ratings       3600s— Kash/Iggy refresh once per day; 1h is conservative

Usage:
    cache = SharedCache.instance()              # singleton
    book  = cache.get_or_fetch(
                key=f"market_book:{market_id}",
                ttl=5,
                fetch_fn=lambda: betfair.get_market_book(market_id),
            )
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class _CacheEntry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl: float):
        self.value      = value
        self.expires_at = time.monotonic() + ttl

    @property
    def is_valid(self) -> bool:
        return time.monotonic() < self.expires_at


class SharedCache:
    """
    In-memory TTL cache shared across all agent components.

    Keys use namespaced prefixes:
        market_book:{market_id}
        wom:{market_id}
        bsp:{market_id}
        metadata:{market_id}
        ratings:{model}           ('kash' or 'iggy')
        venue_markets:{venue}
        account_balance
    """

    # ── Default TTLs (seconds) ──────────────────────────────────────────────
    TTL = {
        "market_book":    5,
        "wom":            10,
        "bsp":            60,
        "metadata":       3_600,
        "ratings":        3_600,
        "venue_markets":  300,
        "account_balance": 30,
    }

    _instance: Optional["SharedCache"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._store: dict[str, _CacheEntry] = {}
        self._lock  = threading.RLock()
        self._hits  = 0
        self._misses= 0

    # ── Singleton ──────────────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "SharedCache":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Core get / set ─────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry and entry.is_valid:
                self._hits += 1
                logger.debug(f"Cache HIT  [{key}]  (hits={self._hits})")
                return entry.value
            if entry:
                del self._store[key]
            self._misses += 1
            logger.debug(f"Cache MISS [{key}]  (misses={self._misses})")
            return None

    def set(self, key: str, value: Any, ttl: float) -> None:
        with self._lock:
            self._store[key] = _CacheEntry(value, ttl)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def invalidate_prefix(self, prefix: str) -> int:
        """Remove all entries whose key starts with prefix. Returns count removed."""
        with self._lock:
            to_remove = [k for k in self._store if k.startswith(prefix)]
            for k in to_remove:
                del self._store[k]
            return len(to_remove)

    # ── High-level fetch-or-compute ─────────────────────────────────────────

    def get_or_fetch(
        self,
        key: str,
        fetch_fn: Callable[[], Any],
        ttl: Optional[float] = None,
        skip_cache_if: Callable[[Any], bool] = None,
    ) -> Any:
        """
        Return cached value for key, or call fetch_fn() and cache the result.

        Args:
            key:              Cache key (e.g. "market_book:1.12345")
            fetch_fn:         Zero-arg callable that fetches fresh data
            ttl:              Cache lifetime in seconds; auto-inferred from key prefix if None
            skip_cache_if:    Optional predicate — if True for the fetched value, don't cache it
                              (e.g. skip_cache_if=lambda r: not r.get("success"))
        """
        if ttl is None:
            prefix = key.split(":")[0]
            ttl = self.TTL.get(prefix, 30)

        cached = self.get(key)
        if cached is not None:
            return cached

        result = fetch_fn()

        if skip_cache_if is None or not skip_cache_if(result):
            self.set(key, result, ttl)

        return result

    # ── Convenience wrappers for common patterns ───────────────────────────

    def market_book(self, market_id: str, fetch_fn: Callable) -> Any:
        return self.get_or_fetch(
            key=f"market_book:{market_id}",
            fetch_fn=fetch_fn,
            skip_cache_if=lambda r: not r.get("success"),
        )

    def bsp_predictions(self, market_id: str, fetch_fn: Callable) -> Any:
        return self.get_or_fetch(
            key=f"bsp:{market_id}",
            fetch_fn=fetch_fn,
            skip_cache_if=lambda r: not r.get("success"),
        )

    def race_metadata(self, market_id: str, fetch_fn: Callable) -> Any:
        return self.get_or_fetch(
            key=f"metadata:{market_id}",
            fetch_fn=fetch_fn,
            skip_cache_if=lambda r: isinstance(r, dict) and not r.get("success", True),
        )

    def ratings(self, model: str, fetch_fn: Callable) -> Any:
        return self.get_or_fetch(
            key=f"ratings:{model}",
            fetch_fn=fetch_fn,
        )

    def venue_markets(self, venue: str, fetch_fn: Callable) -> Any:
        return self.get_or_fetch(
            key=f"venue_markets:{venue.lower().replace(' ', '_')}",
            fetch_fn=fetch_fn,
            skip_cache_if=lambda r: not r.get("success"),
        )

    def account_balance(self, fetch_fn: Callable) -> Any:
        return self.get_or_fetch(
            key="account_balance",
            fetch_fn=fetch_fn,
            skip_cache_if=lambda r: not r.get("success"),
        )

    # ── Stats ──────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries":    len(self._store),
                "hits":       self._hits,
                "misses":     self._misses,
                "hit_rate":   round(self._hits / total * 100, 1) if total else 0,
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits  = 0
            self._misses= 0
