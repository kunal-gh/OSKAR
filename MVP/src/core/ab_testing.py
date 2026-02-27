"""
ab_testing.py — OSKAR v0.5 Platform Layer
------------------------------------------
A/B testing framework for content warning variant efficacy.

OSKAR can show multiple variants of moderation warnings to users and
track which variant is most effective at reducing harmful behavior.

Warning Variants (three-bucket):
  A — "INFORMATIONAL": Neutral fact-check label with source links
  B — "FRICTION":      "Are you sure?" soft delay before posting
  C — "SOCIAL":        "X of your followers reported this as false"

Metrics tracked per variant:
  - impressions:     Times the warning was shown
  - acks:            User acknowledged / dismissed the warning
  - retractions:     User deleted or edited the post after warning
  - ctr:             Click-through rate = acks / impressions
  - retraction_rate: Retractions / impressions

All state is stored in-memory (dict) with optional Redis persistence
if REDIS_URL is set. This is intentionally lightweight for the v0.5 MVP.

Usage:
    from ab_testing import ABTestingEngine
    ab = ABTestingEngine()

    variant = ab.get_variant("user_123")        # → "A" | "B" | "C"
    ab.record_impression("user_123", variant)
    ab.record_feedback("user_123", variant, action="retraction")
    report = ab.get_report()
"""

import hashlib
import time
from typing import Literal

from src.infra.redis_cache import redis_cache

Action = Literal["impression", "ack", "retraction"]
Variant = Literal["A", "B", "C"]

VARIANT_NAMES = {
    "A": "Informational Label",
    "B": "Friction / Soft Delay",
    "C": "Social Proof Warning",
}


class ABTestingEngine:
    """
    v0.7 Redis-backed A/B tester for warning variant effectiveness.
    """

    def __init__(self):
        pass  # Relies on global redis_cache

    def get_variant(self, user_id: str) -> Variant:
        """Deterministically assign a warning variant to a user."""
        uv_key = f"ab:user_variant:{user_id}"
        cached_variant = redis_cache.get(uv_key)
        if cached_variant:
            return cached_variant

        h = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        variant = ["A", "B", "C"][h % 3]
        redis_cache.set(uv_key, variant, ttl_seconds=60 * 60 * 24 * 30)
        return variant

    def record_impression(self, user_id: str, variant: Variant) -> None:
        """Record that a warning was shown to the user."""
        redis_cache.incr(f"ab:stats:{variant}:impressions")

        # Log event (simplified append via incremental key for v0.7)
        log_id = redis_cache.incr("ab:log_counter")
        redis_cache.set(
            f"ab:log:{log_id}",
            {"ts": time.time(), "user_id": user_id, "variant": variant, "action": "impression"},
            ttl_seconds=60 * 60 * 24 * 7,
        )

    def record_feedback(self, user_id: str, variant: Variant, action: Action) -> None:
        if action not in ("impression", "ack", "retraction"):
            return

        if action != "impression":
            redis_cache.incr(f"ab:stats:{variant}:{action}s")

        log_id = redis_cache.incr("ab:log_counter")
        redis_cache.set(
            f"ab:log:{log_id}",
            {"ts": time.time(), "user_id": user_id, "variant": variant, "action": action},
            ttl_seconds=60 * 60 * 24 * 7,
        )

    def get_report(self) -> dict:
        report = {}
        best_variant = "A"
        best_rate = -1.0

        for v in ["A", "B", "C"]:
            # Retrieve string ints from Redis hash mapping and cast to int
            imp = int(redis_cache.get(f"ab:stats:{v}:impressions") or 0)
            acks = int(redis_cache.get(f"ab:stats:{v}:acks") or 0)
            rets = int(redis_cache.get(f"ab:stats:{v}:retractions") or 0)

            ctr = round(acks / imp, 4) if imp > 0 else 0.0
            rr = round(rets / imp, 4) if imp > 0 else 0.0

            report[v] = {
                "name": VARIANT_NAMES[v],
                "impressions": imp,
                "acks": acks,
                "retractions": rets,
                "ctr": ctr,
                "retraction_rate": rr,
            }
            if rr > best_rate:
                best_rate = rr
                best_variant = v

        total_logs = int(redis_cache.get("ab:log_counter") or 0)
        return {
            "variants": report,
            "total_events": total_logs,
            "best_variant": best_variant,
        }

    def reset(self) -> None:
        # Cannot easily wipe all keys natively through generic redis wrapper without flushdb,
        # but we can reset the primary counters to 0 for the test suite.
        redis_cache.set("ab:log_counter", 0)
        for v in ["A", "B", "C"]:
            redis_cache.set(f"ab:stats:{v}:impressions", 0)
            redis_cache.set(f"ab:stats:{v}:acks", 0)
            redis_cache.set(f"ab:stats:{v}:retractions", 0)
