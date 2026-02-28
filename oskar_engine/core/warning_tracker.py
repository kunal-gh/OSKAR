"""
warning_tracker.py — OSKAR v0.5 Platform Layer
-----------------------------------------------
A thin log wrapper that records every warning event (impression, ack,
retraction) alongside the OSKAR risk context at the time of display.

The WarningTracker is the authoritative event store; it delegates
variant assignment to ABTestingEngine.

Events are kept in-memory as an ordered list. In production, these
would be flushed to PostgreSQL or a time-series database for analysis.

Usage:
    from warning_tracker import WarningTracker
    tracker = WarningTracker()

    # When showing a warning:
    event = tracker.log_impression("user_123", risk_score=0.74)
    # event = {"event_id": "...", "variant": "B", "ts": ...}

    # When user retracts:
    tracker.log_feedback(event["event_id"], action="retraction")
"""

import time
import uuid
from typing import Optional

from src.core.ab_testing import ABTestingEngine
from src.infra.redis_cache import redis_cache


class WarningTracker:
    """
    v0.7 Redis-backed Warning Tracker.
    Records warning impressions and user feedback events.
    """

    def __init__(self):
        self.ab = ABTestingEngine()

    def log_impression(
        self,
        user_id: str,
        risk_score: float = 0.0,
        route: str = "soft_warning",
        content_hash: Optional[str] = None,
    ) -> dict:
        variant = self.ab.get_variant(user_id)
        event_id = str(uuid.uuid4())
        ts = time.time()

        event = {
            "event_id": event_id,
            "user_id": user_id,
            "variant": variant,
            "risk_score": round(risk_score, 4),
            "route": route,
            "content_hash": content_hash,
            "ts": ts,
            "action": "impression",
        }

        redis_cache.set(f"warning:event:{event_id}", event, ttl_seconds=60 * 60 * 24 * 7)
        self.ab.record_impression(user_id, variant)
        return event

    def log_feedback(self, event_id: str, action: str) -> dict:  # "ack" | "retraction"
        key = f"warning:event:{event_id}"
        original = redis_cache.get(key)

        if not original:
            return {"error": f"Event '{event_id}' not found"}

        feedback = {
            **original,
            "action": action,
            "ts": time.time(),
            "latency_ms": round((time.time() - original["ts"]) * 1000, 1),
        }

        redis_cache.set(f"warning:event:{event_id}_{action}", feedback, ttl_seconds=60 * 60 * 24 * 7)
        self.ab.record_feedback(original["user_id"], original["variant"], action)
        return feedback

    def get_report(self) -> dict:
        return self.ab.get_report()

    def get_recent_events(self, limit: int = 50) -> list:
        # Redis implementation accesses keys matching pattern
        # Since fallback provides all keys, we hydrate and sort
        keys = redis_cache.keys("warning:event:*")
        events = []
        for k in keys:
            doc = redis_cache.get(k)
            if doc:
                events.append(doc)

        events = sorted(events, key=lambda e: e["ts"], reverse=True)
        return events[:limit]
