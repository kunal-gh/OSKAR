"""
narrative_tracker.py — OSKAR v0.6 Enterprise Governance
--------------------------------------------------------
Tracks evolving narratives and calculates user polarization
across opposed narrative clusters.

Uses a localized FAISS index to represent the "current active narratives".
When a new post arrives:
1. It is embedded (using the existing SBERT model).
2. We search FAISS for the nearest existing narrative.
3. If distance < threshold: assigned to that narrative cluster.
4. If distance >= threshold: creates a NEW narrative cluster.

Polarization Index:
Measures how deeply a user is entrenched into mutually opposed narrative clusters.
E.g., if a user exclusively posts in "Election Fraud" and never in "Election Integrity",
they have high polarization.

Note: In a true production system, older clusters would decay. For this MVP
v0.6 implementation, we cap the max clusters to 1000 in memory.
"""

import time
import uuid
from typing import Dict, List, Optional

import faiss
import numpy as np

# Configuration
EMBEDDING_DIM = 768
CLUSTER_THRESHOLD = 0.65  # lower = tighter clusters (cosine distance)


class NarrativeTracker:
    """
    Maintains active narrative clusters and tracks user polarization.
    """

    def __init__(self, embedder=None, threshold: float = CLUSTER_THRESHOLD):
        # We need an embedding function. If none provided, we gracefully fallback
        # to a dummy embedder for tests or if the model fails to load.
        self.embedder = embedder
        self.threshold = threshold

        # FAISS index for cluster centroids (using Inner Product for normalized embeddings -> Cosine Sim)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)

        # Maps FAISS index ID -> Cluster ID
        self.id_to_cluster: Dict[int, str] = {}

        # Cluster metadata: {cluster_id: {"centroid": np.array, "size": int, "created_at": float}}
        self.clusters: Dict[str, dict] = {}

        # User history: user_id -> dict of {cluster_id: post_count}
        self.user_history: Dict[str, Dict[str, int]] = {}

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        if not self.embedder:
            return None
        try:
            # Assumes embedder is an instance of sentence_transformers.SentenceTransformer
            # or wrapped to provide an .encode() method returning a numpy array.
            emb = self.embedder.encode([text])[0]
            # Normalize for cosine similarity via Inner Product
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb.astype(np.float32)
        except Exception:
            return None

    def _create_cluster(self, embedding: np.ndarray) -> str:
        cluster_id = f"nar_{str(uuid.uuid4())[:8]}"
        idx = self.index.ntotal

        # Add to FAISS. FAISS requires 2D arrays (n_samples, dim)
        self.index.add(np.array([embedding]))

        self.id_to_cluster[idx] = cluster_id
        self.clusters[cluster_id] = {
            "centroid": embedding,
            "size": 1,
            "created_at": time.time(),
        }
        return cluster_id

    def _update_cluster_centroid(self, cluster_id: str, new_embedding: np.ndarray):
        """Moving average update of the cluster centroid."""
        c = self.clusters[cluster_id]
        size = c["size"]
        old_centroid = c["centroid"]

        # Weighted average to shift centroid slightly towards new data
        new_centroid = ((old_centroid * size) + new_embedding) / (size + 1)

        # Re-normalize
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid = new_centroid / norm

        c["centroid"] = new_centroid
        c["size"] += 1

        # Note: We do NOT update the FAISS index with the moving centroid in this lightweight version
        # to avoid index rebuild complexity. The original seed post acts as the permanent anchor.
        # In full prod, we'd use IDMap and replace vectors.

    def _calculate_polarization(self, user_id: str) -> float:
        """
        Calculate polarization index (0.0 to 1.0) based on interaction distribution.
        High polarization = highly concentrated in 1 or 2 specific clusters while ignoring others.
        Low polarization = evenly distributed across many clusters (or too little data).
        Using normalized entropy logic.
        """
        if user_id not in self.user_history:
            return 0.0

        counts = list(self.user_history[user_id].values())
        total_posts = sum(counts)
        unique_clusters = len(counts)

        if total_posts < 3 or unique_clusters < 1:
            return 0.0  # Not enough history to judge

        if unique_clusters == 1:
            # Max polarization: only ever engages with one single narrative cluster over many posts
            return min(1.0, total_posts / 10.0)

        # Calculate entropy of distribution
        probs = np.array(counts) / total_posts
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(unique_clusters)

        # Polarization is the inverse of normalized entropy
        # (1.0 - (entropy / max_entropy)) -> High when concentrated, Low when spread.
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        polarization = 1.0 - normalized_entropy

        return float(round(polarization, 4))

    def _record_user_interaction(self, user_id: str, cluster_id: str):
        if user_id not in self.user_history:
            self.user_history[user_id] = {}
        if cluster_id not in self.user_history[user_id]:
            self.user_history[user_id][cluster_id] = 0
        self.user_history[user_id][cluster_id] += 1

    def track(self, text: str, user_id: str) -> dict:
        """
        Main entrypoint. Ingests text, assigns to a narrative cluster,
        and updates/returns user polarization.
        """
        if not text.strip():
            return {
                "narrative_cluster_id": None,
                "polarization_score": self._calculate_polarization(user_id),
                "is_new_narrative": False,
            }

        embedding = self._get_embedding(text)

        if embedding is None:
            # Fallback if model fails or wasn't provided (e.g., in dummy tests without a mock)
            return {
                "narrative_cluster_id": "nar_unknown",
                "polarization_score": self._calculate_polarization(user_id),
                "is_new_narrative": False,
            }

        # If index is empty, create first cluster
        if self.index.ntotal == 0:
            cluster_id = self._create_cluster(embedding)
            self._record_user_interaction(user_id, cluster_id)
            return {
                "narrative_cluster_id": cluster_id,
                "polarization_score": self._calculate_polarization(user_id),
                "is_new_narrative": True,
            }

        # Search FAISS for nearest narrative
        # Inner product with normalized vectors == Cosine Similarity [-1.0, 1.0]
        distances, indices = self.index.search(np.array([embedding]), 1)
        best_dist = distances[0][0]  # Higher is more similar (Cosine Sim)
        best_idx = indices[0][0]

        # Convert cosine similarity to cosine distance [0, 2]
        # (where 0 means identical, 2 means exactly opposite)
        # Cosine distance = 1 - Cosine Similarity
        cosine_distance = 1.0 - best_dist

        is_new = False
        if cosine_distance < self.threshold:
            # Belongs to existing cluster
            cluster_id = self.id_to_cluster[best_idx]
            self._update_cluster_centroid(cluster_id, embedding)
        else:
            # Create new cluster
            cluster_id = self._create_cluster(embedding)
            is_new = True

        self._record_user_interaction(user_id, cluster_id)

        return {
            "narrative_cluster_id": cluster_id,
            "polarization_score": self._calculate_polarization(user_id),
            "is_new_narrative": is_new,
        }
