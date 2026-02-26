"""
test_narrative_tracker.py — OSKAR v0.6
Tests NarrativeTracker: clustering logic, centroid updates, and polarization.
"""

import pytest
import numpy as np
from narrative_tracker import NarrativeTracker


class DummyEmbedder:
    """Mock embedder returning deterministic, distinct vectors for testing."""
    def encode(self, texts):
        res = []
        for text in texts:
            # We want distinct vectors for distinct texts
            # Use random seed based on text hash to generate a pseudo-random unit vector
            rng = np.random.default_rng(seed=abs(hash(text)) % (2**32))
            vec = rng.standard_normal(768, dtype=np.float32)
            # Normalize
            vec = vec / np.linalg.norm(vec)
            res.append(vec)
        return np.array(res)


def test_narrative_tracker_initialization():
    nt = NarrativeTracker(embedder=None)
    assert nt.threshold == 0.65
    assert nt.index.ntotal == 0


def test_narrative_tracker_fallback_no_embedder():
    nt = NarrativeTracker(embedder=None)
    res = nt.track("Hello world", "userA")
    assert res["narrative_cluster_id"] == "nar_unknown"
    assert res["polarization_score"] == 0.0
    assert res["is_new_narrative"] is False


def test_narrative_tracker_first_post_creates_cluster():
    nt = NarrativeTracker(embedder=DummyEmbedder())
    res = nt.track("First post about subject A", "u1")
    assert res["narrative_cluster_id"].startswith("nar_")
    assert res["is_new_narrative"] is True
    assert res["polarization_score"] == 0.0  # Not enough history
    assert nt.index.ntotal == 1


def test_narrative_tracker_identical_posts_cluster_together():
    nt = NarrativeTracker(embedder=DummyEmbedder())
    text1 = "Identical message"
    text2 = "Identical message"

    res1 = nt.track(text1, "u1")
    res2 = nt.track(text2, "u2")

    assert res1["is_new_narrative"] is True
    assert res2["is_new_narrative"] is False
    assert res1["narrative_cluster_id"] == res2["narrative_cluster_id"]
    assert nt.index.ntotal == 1


def test_narrative_tracker_different_posts_create_new_clusters():
    nt = NarrativeTracker(embedder=DummyEmbedder(), threshold=0.1) # Strict threshold
    res1 = nt.track("This is narrative A", "u1")
    res2 = nt.track("Completely different topic B", "u2")

    assert res1["narrative_cluster_id"] != res2["narrative_cluster_id"]
    assert res2["is_new_narrative"] is True
    assert nt.index.ntotal == 2


def test_polarization_score_calculation():
    """
    Test the entropy-based polarization calculation.
    High polarization: user only posts in ONE narrative cluster.
    Low polarization: user posts evenly across MANY clusters.
    """
    nt = NarrativeTracker(embedder=DummyEmbedder(), threshold=0.1)
    
    # User 1: High polarization (all posts in one cluster)
    for _ in range(5):
        nt.track("A specific conspiracy theory", "user_highly_polarized")

    # User 2: Low polarization (posts across 5 different topics)
    nt.track("Topic Alpha", "user_unpolarized")
    nt.track("Topic Beta1", "user_unpolarized")
    nt.track("Topic Gamma", "user_unpolarized")
    nt.track("Topic Delta", "user_unpolarized")
    nt.track("Topic Omega", "user_unpolarized")

    # Final track to get the score
    res_high = nt.track("A specific conspiracy theory", "user_highly_polarized")
    res_low = nt.track("New Topic Zeta", "user_unpolarized")

    # High polarization should be near 1.0 (they have 6 posts all in 1 cluster)
    # Actually, if unique_clusters == 1, our logic caps at total_posts / 10.0, max 1.0.
    # 6 posts -> 0.6 max for this test logic
    assert res_high["polarization_score"] == 0.6
    
    # Low polarization should be exactly 0.0 (entropy is max since evenly distributed across 6 clusters)
    assert res_low["polarization_score"] == 0.0
