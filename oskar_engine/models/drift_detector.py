"""
drift_detector.py — OSKAR v0.6 Enterprise Governance
-----------------------------------------------------
Detects semantic data drift by comparing the distribution of incoming
embeddings against a baseline distribution.

In production, drift detection ensures that if user behavior/language changes
significantly (e.g., new slang, evasion tactics, shifting topics), the Ops
team is alerted that the underlying NLP models might be degrading in accuracy.

Implementation (v0.6):
- Maintains a reference "baseline" centroid (mean of first N embeddings).
- Maintains a "sliding window" centroid (mean of last M embeddings).
- Calculates the cosine distance between baseline and sliding window.
- Updates a Prometheus Gauge (`oskar_data_drift_score`) with the distance.
- If distance > threshold, the metric flags for operator review.
"""

from collections import deque

import numpy as np
from prometheus_client import Gauge

# Metric to expose to Prometheus/Grafana
DRIFT_GAUGE = Gauge(
    "oskar_data_drift_score", "Cosine distance between recent embedding window and baseline"
)

# Configuration
WINDOW_SIZE = 100  # Number of recent embeddings to keep in sliding window
BASELINE_SIZE = 500  # Number of initial embeddings to form the stable baseline
EMBEDDING_DIM = 768


class DriftDetector:
    """
    Tracks semantic data drift over time using embedding centroids.
    """

    def __init__(
        self, embedder=None, window_size: int = WINDOW_SIZE, baseline_size: int = BASELINE_SIZE
    ):
        self.embedder = embedder
        self.window_size = window_size
        self.baseline_size = baseline_size

        # State
        self.baseline_count = 0
        self.baseline_sum = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        self.baseline_centroid = None

        self.recent_embeddings = deque(maxlen=window_size)
        self.recent_centroid = None

        # The current drift score [0.0 to 2.0]
        self.current_drift_score = 0.0

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return vec

    def _get_embedding(self, text: str) -> np.ndarray:
        if self.embedder:
            try:
                emb = self.embedder.encode([text])[0]
                return self._normalize(emb.astype(np.float32))
            except Exception:
                pass
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    def track(self, text: str) -> dict:
        """
        Ingest new text, update distributions, and calculate drift.
        """
        if not text.strip():
            return {"drift_score": self.current_drift_score, "is_drifting": False}

        emb = self._get_embedding(text)
        if np.all(emb == 0):
            return {"drift_score": self.current_drift_score, "is_drifting": False}

        # 1. Build baseline if not complete
        if self.baseline_count < self.baseline_size:
            self.baseline_sum += emb
            self.baseline_count += 1
            if self.baseline_count == self.baseline_size:
                # Lock in the baseline centroid
                self.baseline_centroid = self._normalize(self.baseline_sum / self.baseline_size)

        # 2. Update sliding window
        self.recent_embeddings.append(emb)

        # 3. Calculate drift if baseline forms and we have some recent data
        if self.baseline_centroid is not None and len(self.recent_embeddings) >= min(
            self.window_size, 2
        ):
            # Calculate sliding window centroid
            window_sum = np.sum(list(self.recent_embeddings), axis=0)
            self.recent_centroid = self._normalize(window_sum / len(self.recent_embeddings))

            # Cosine similarity (inner product of normalized vectors) -> Cosine distance
            cos_sim = np.dot(self.baseline_centroid, self.recent_centroid)
            # Clip to valid range to avoid floating point errors
            cos_sim = max(-1.0, min(1.0, float(cos_sim)))

            # Distance: 0 = identical, 1 = orthogonal, 2 = exactly opposite
            distance = 1.0 - cos_sim
            self.current_drift_score = round(distance, 4)

            # Update Prometheus metric
            DRIFT_GAUGE.set(self.current_drift_score)

        is_drifting = self.current_drift_score > 0.3  # Arbitrary threshold for v0.6

        return {
            "drift_score": self.current_drift_score,
            "is_drifting": is_drifting,
            "baseline_ready": self.baseline_centroid is not None,
        }

    def force_baseline(self, texts: list[str]) -> None:
        """Helper to quickly seed the baseline in tests/startup."""
        for text in texts:
            self.track(text)
