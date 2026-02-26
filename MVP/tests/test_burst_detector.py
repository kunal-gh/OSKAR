"""
test_burst_detector.py — OSKAR v0.4
Tests BurstDetector initialization, schema, normal traffic, and burst detection.
"""

import pytest
from src.core.burst_detector import BurstDetector


def _make_events(user_counts: list, base_ts: int = 1708000000, interval_s: int = 60):
    """Helper: generate a list of posting events from user count sequence."""
    return [
        {"ts": base_ts + i * interval_s, "user_count": c}
        for i, c in enumerate(user_counts)
    ]


def test_burst_detector_initialization():
    """BurstDetector should initialize without errors."""
    bd = BurstDetector()
    assert hasattr(bd, "enabled") or hasattr(bd, "model")
    assert hasattr(bd, "trained")


def test_burst_detector_empty_events():
    """Empty event list should return anomaly_score=0.0 and is_burst=False."""
    bd = BurstDetector()
    result = bd.detect([])
    assert result["anomaly_score"] == 0.0
    assert result["is_burst"] is False
    assert result["window_size"] == 0


def test_burst_detector_schema():
    """detect() must always return the expected schema keys."""
    bd = BurstDetector()
    result = bd.detect(_make_events([1, 2, 1, 3, 2]))
    assert "anomaly_score" in result
    assert "is_burst" in result
    assert "window_size" in result
    assert "trained_model" in result
    assert isinstance(result["anomaly_score"], float)
    assert isinstance(result["is_burst"], bool)
    assert 0.0 <= result["anomaly_score"] <= 1.0


def test_burst_detector_normal_traffic():
    """Steady low-volume traffic should produce a valid (in-range) anomaly score."""
    bd = BurstDetector()
    normal_events = _make_events([1, 2, 1, 3, 2, 1, 2, 3, 1, 2, 1, 1, 2, 3, 2, 1])
    result = bd.detect(normal_events)
    # Only assert valid range — untrained model may score anything but must not crash
    assert 0.0 <= result["anomaly_score"] <= 1.0
    print(f"[Normal] anomaly_score={result['anomaly_score']:.3f}")


def test_burst_detector_coordinated_spike():
    """
    Sudden spike from 1-2 users to 80+ should produce a higher anomaly score
    than quiet traffic (even with untrained model).
    """
    bd = BurstDetector()
    quiet   = _make_events([1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1])
    burst   = _make_events([1, 2, 1, 1, 80, 90, 85, 80, 75, 80, 85, 82, 79, 80, 77, 78])

    score_quiet = bd.detect(quiet)["anomaly_score"]
    score_burst = bd.detect(burst)["anomaly_score"]

    print(f"[Quiet]  anomaly_score={score_quiet:.3f}")
    print(f"[Burst]  anomaly_score={score_burst:.3f}")

    assert score_burst >= score_quiet, (
        f"Burst score ({score_burst}) should be >= quiet score ({score_quiet})"
    )


def test_burst_detector_quick_score():
    """quick_score() should return a float in [0, 1]."""
    bd = BurstDetector()
    score = bd.quick_score(_make_events([1, 2, 1]))
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
