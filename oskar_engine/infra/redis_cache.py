import json
import os
from typing import Any, Optional

import redis


class RedisCache:
    """
    v0.7 Shared Memory Client for OSKAR analytics.
    Replaces in-memory Python dictionaries with scalable Redis storage
    for horizontal scaling of A/B Testing, Warnings, and Trust features.
    """

    def __init__(self):
        # Default to local container or fallback to localhost during dev
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            self.client = redis.from_url(redis_url, decode_responses=True)
            self.client.ping()
            self.connected = True
            print(f"[RedisCache] Connected to {redis_url}")
        except redis.ConnectionError:
            self.connected = False
            print(
                "[RedisCache] Warning: Could not connect to Redis. Falling back to local dictionary."
            )
            self._fallback_cache = {}

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        doc = json.dumps(value)
        if self.connected:
            try:
                self.client.set(key, doc, ex=ttl_seconds)
                return True
            except Exception as e:
                print(f"[RedisCache] SET error: {e}")
                return False
        else:
            self._fallback_cache[key] = doc
            return True

    def get(self, key: str) -> Optional[Any]:
        if self.connected:
            try:
                doc = self.client.get(key)
                return json.loads(doc) if doc else None
            except Exception as e:
                print(f"[RedisCache] GET error: {e}")
                return None
        else:
            doc = self._fallback_cache.get(key)
            return json.loads(doc) if doc else None

    def incr(self, key: str) -> int:
        """Increment a counter and return its new value."""
        if self.connected:
            try:
                return self.client.incr(key)
            except Exception as e:
                print(f"[RedisCache] INCR error: {e}")
                return 0
        else:
            current = int(self._fallback_cache.get(key, 0))
            self._fallback_cache[key] = str(current + 1)
            return current + 1

    def keys(self, pattern: str) -> list[str]:
        if self.connected:
            return self.client.keys(pattern)
        else:
            # Very naive regex/pattern match for the fallback dict
            prefix = pattern.replace("*", "")
            return [k for k in self._fallback_cache.keys() if k.startswith(prefix)]


# Global instance
redis_cache = RedisCache()
