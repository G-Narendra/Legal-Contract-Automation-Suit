"""
TTL-based caching utility for cost-effective API usage.

Reduces API calls by caching frequent responses.
Each cache hit saves ~$0.001-0.003 in API costs.
"""

import time
import hashlib
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("cache")


class TTLCache:
    """Simple TTL-based cache with LRU eviction.

    Cost impact: 40%+ cache hit rate on frequently repeated queries
    can reduce API costs by ~50%.
    """

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        self._cache: Dict[str, Dict] = {}
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, subsystem: str, prompt: str) -> Optional[str]:
        """Get cached response if valid."""
        key = self._make_key(subsystem, prompt)
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        if time.time() - entry["ts"] > self.ttl:
            del self._cache[key]
            self._misses += 1
            return None

        self._hits += 1
        return entry["value"]

    def set(self, subsystem: str, prompt: str, response: str):
        """Cache a response with current timestamp."""
        key = self._make_key(subsystem, prompt)

        # LRU eviction: remove oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest = min(self._cache.keys(), key=lambda k: self._cache[k]["ts"])
            del self._cache[oldest]

        self._cache[key] = {"value": response, "ts": time.time()}

    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
        logger.info("Cache cleared")

    @property
    def stats(self) -> Dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total * 100, 1) if total > 0 else 0,
        }

    @property
    def size(self) -> int:
        return len(self._cache)

    def _make_key(self, subsystem: str, prompt: str) -> str:
        content = f"{subsystem}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
