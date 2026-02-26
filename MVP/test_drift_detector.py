"""
test_drift_detector.py — OSKAR v0.6
Tests DriftDetector: baseline formation, sliding window updates, and drift threshold alerting.
"""

import pytest
import numpy as np
from drift_detector import DriftDetector


class MockDriftEmbedder:
    """Mock embedder returning directional vectors."""
    def encode(self, texts):
        res = []
        for text in texts:
            if "baseline" in text.lower():
                # Baseline vectors point roughly along X-axis
                vec = np.array([1.0, 0.1, 0.0], dtype=np.float32)
            elif "drifted" in text.lower():
                # Drifted vectors point strictly along Y-axis (orthogonal distance = 1.0)
                vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                # Default
                vec = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            
            # Normalize and pad to 768
            norm = np.linalg.norm(vec)
            full_vec = np.zeros(768, dtype=np.float32)
            full_vec[:3] = vec / norm
            res.append(full_vec)
            
        return np.array(res)


def test_drift_detector_initialization():
    dd = DriftDetector(embedder=None, window_size=10, baseline_size=20)
    assert dd.baseline_size == 20
    assert dd.window_size == 10
    assert dd.baseline_centroid is None
    assert dd.current_drift_score == 0.0


def test_baseline_formation():
    """Baseline must form exactly when count reaches baseline_size."""
    dd = DriftDetector(embedder=MockDriftEmbedder(), window_size=5, baseline_size=5)
    
    for i in range(4):
        res = dd.track("Baseline text")
        assert res["baseline_ready"] is False
        
    # The 5th text locks in the baseline
    res = dd.track("Baseline text")
    assert res["baseline_ready"] is True
    assert dd.baseline_centroid is not None
    assert dd.baseline_count == 5


def test_drift_detection_logic():
    """Drift score should spike when text semantics fundamentally change."""
    dd = DriftDetector(embedder=MockDriftEmbedder(), window_size=5, baseline_size=5)
    
    # Fill the baseline and the recent window with identical vectors
    dd.force_baseline(["Baseline 1", "Baseline 2", "Baseline 3", "Baseline 4", "Baseline 5"])
    for i in range(11):
        res = dd.track("Baseline text steady state")
        
    initial_drift = res["drift_score"]
    assert initial_drift < 0.05  # Essentially 0 (vectors are identical)
    assert res["is_drifting"] is False
    
    # Introduce entirely orthongonal (drifted) semantic text
    # We must push enough to flush the old baseline text out of the recent window of size 5
    for i in range(5):
        drift_res = dd.track("Drifted completely new topic")
        
    # The recent window is now full of drifted text. Cosine distance to baseline should be ~1.0
    final_drift = drift_res["drift_score"]
    assert final_drift > 0.9
    assert drift_res["is_drifting"] is True


def test_empty_text_safeguard():
    """Empty strings should not calculate distance or crash the detector."""
    dd = DriftDetector(embedder=MockDriftEmbedder(), window_size=2, baseline_size=2)
    res = dd.track("")
    assert res["is_drifting"] is False
    assert dd.baseline_count == 0
