"""
test_ab_testing.py — OSKAR v0.5
Tests ABTestingEngine and WarningTracker: init, variant assignment,
impression recording, feedback, and report schema.
"""

import pytest
from ab_testing import ABTestingEngine, VARIANT_NAMES
from warning_tracker import WarningTracker
from redis_cache import redis_cache


@pytest.fixture(autouse=True)
def clear_redis_cache():
    """Ensure a clean Redis state (or local dict) before every test."""
    redis_cache._fallback_cache.clear()
    if redis_cache.connected:
        redis_cache.client.flushdb()


# ─── ABTestingEngine Tests ─────────────────────────────────────────

def test_ab_engine_initialization():
    """ABTestingEngine initializes with three variants."""
    ab = ABTestingEngine()
    report = ab.get_report()
    assert "A" in report["variants"]
    assert "B" in report["variants"]
    assert "C" in report["variants"]


def test_ab_variant_assignment_deterministic():
    """Same user_id must always get the same variant."""
    ab = ABTestingEngine()
    v1 = ab.get_variant("user_alice")
    v2 = ab.get_variant("user_alice")
    assert v1 == v2
    assert v1 in ("A", "B", "C")


def test_ab_variant_assignment_covers_all():
    """Different users should be spread across A, B, C."""
    ab = ABTestingEngine()
    assigned = {ab.get_variant(f"user_{i}") for i in range(30)}
    # With 30 users and hash-mod-3, all three variants must appear
    assert assigned == {"A", "B", "C"}


def test_ab_impression_recording():
    """Recording impressions should increment the counter."""
    ab = ABTestingEngine()
    ab.record_impression("u1", "A")
    ab.record_impression("u2", "A")
    report = ab.get_report()
    assert report["variants"]["A"]["impressions"] == 2


def test_ab_feedback_recording():
    """Recording a retraction should increment the retraction counter."""
    ab = ABTestingEngine()
    ab.record_impression("u1", "B")
    ab.record_feedback("u1", "B", "retraction")
    report = ab.get_report()
    assert report["variants"]["B"]["retractions"] == 1


def test_ab_ctr_calculation():
    """CTR = acks / impressions."""
    ab = ABTestingEngine()
    for _ in range(4):
        ab.record_impression("uk", "C")
    ab.record_feedback("uk", "C", "ack")
    ab.record_feedback("uk", "C", "ack")
    report = ab.get_report()
    assert report["variants"]["C"]["ctr"] == 0.5


def test_ab_report_schema():
    """Report must have correct top-level keys."""
    ab = ABTestingEngine()
    report = ab.get_report()
    assert "variants" in report
    assert "total_events" in report
    assert "best_variant" in report
    assert report["best_variant"] in ("A", "B", "C")


def test_ab_reset():
    """reset() clears all counters and logs."""
    ab = ABTestingEngine()
    ab.record_impression("u1", "A")
    ab.record_feedback("u1", "A", "retraction")
    ab.reset()
    report = ab.get_report()
    assert report["variants"]["A"]["impressions"] == 0
    assert report["total_events"] == 0


# ─── WarningTracker Tests ─────────────────────────────────────────

def test_warning_tracker_log_impression():
    """log_impression must return an event with expected keys."""
    tracker = WarningTracker()
    event = tracker.log_impression("u1", risk_score=0.74, route="soft_warning")
    assert "event_id" in event
    assert "variant" in event
    assert event["variant"] in ("A", "B", "C")
    assert event["risk_score"] == 0.74


def test_warning_tracker_log_feedback():
    """log_feedback should record a retraction and be retrievable."""
    tracker = WarningTracker()
    event = tracker.log_impression("u2", risk_score=0.85, route="human_review")
    feedback = tracker.log_feedback(event["event_id"], action="retraction")
    assert feedback["action"] == "retraction"
    assert "latency_ms" in feedback


def test_warning_tracker_invalid_event():
    """log_feedback with bad event_id should return error dict."""
    tracker = WarningTracker()
    result = tracker.log_feedback("nonexistent-id", action="ack")
    assert "error" in result


def test_warning_tracker_report():
    """get_report() after impressions should show correct totals."""
    tracker = WarningTracker()
    for i in range(6):
        ev = tracker.log_impression(f"user_{i}", risk_score=0.6)
        if i % 2 == 0:
            tracker.log_feedback(ev["event_id"], action="retraction")
    report = tracker.get_report()
    total_retractions = sum(
        v["retractions"] for v in report["variants"].values()
    )
    assert total_retractions == 3


def test_warning_tracker_recent_events():
    """get_recent_events returns events in correct count."""
    tracker = WarningTracker()
    for i in range(10):
        tracker.log_impression(f"u{i}", risk_score=float(i) / 10)
    events = tracker.get_recent_events(limit=5)
    assert len(events) == 5
