"""
app/models/cache.py — In-Memory Prompt/Response Cache

Provides fast, zero-cost deduplication for repeated agent queries.
Stores responses keyed by SHA-256 hashes of the normalized prompt/context.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Callable

logger = logging.getLogger("yogasutra.cache")


class ResponseCache:
    """In-memory LRU/TTL cache for model and tool responses."""

    def __init__(self, max_size: int = 256, default_ttl_sec: float = 3600.0) -> None:
        self.max_size = max_size
        self.default_ttl_sec = default_ttl_sec
        self._cache: dict[str, dict[str, Any]] = {}
        self.hits: int = 0
        self.misses: int = 0

    def _hash_key(self, key_data: Any) -> str:
        """Create a deterministic SHA-256 hash string from arbitrary key data."""
        if isinstance(key_data, str):
            s = key_data
        else:
            try:
                s = json.dumps(key_data, sort_keys=True)
            except Exception:
                s = str(key_data)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def get(self, key_data: Any) -> Any | None:
        """Retrieve cached value if present and not expired."""
        h = self._hash_key(key_data)
        item = self._cache.get(h)
        if not item:
            self.misses += 1
            return None

        # Check expiration
        now = time.monotonic()
        if now > item["expires_at"]:
            del self._cache[h]
            self.misses += 1
            return None

        self.hits += 1
        return item["value"]

    def set(self, key_data: Any, value: Any, ttl_sec: float | None = None) -> None:
        """Store value in cache."""
        if len(self._cache) >= self.max_size:
            # Evict oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        h = self._hash_key(key_data)
        ttl = ttl_sec if ttl_sec is not None else self.default_ttl_sec
        self._cache[h] = {
            "value": value,
            "expires_at": time.monotonic() + ttl,
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0


# Global singleton cache instance
response_cache = ResponseCache()
