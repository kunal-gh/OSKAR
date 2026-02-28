import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
COLLECTION_NAME = "oskar_knowledge"
TOP_K_DEFAULT = 5


class EvidenceRetrieval:
    """
    SBERT + Qdrant + Neo4j evidence retrieval module (v1.0 Microservices).

    Replaces FAISS with Qdrant for persistent, multi-node vector search.
    """

    def __init__(self, use_neo4j: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(SBERT_MODEL, device=self.device)
        self.dim = 768
        self.neo4j_layer = None

        # Initialize Qdrant
        try:
            self.qdrant = QdrantClient(host=QDRANT_HOST, port=6333)
            self._ensure_collection()
            print(f"[EvidenceRetrieval] Connected to Qdrant at {QDRANT_HOST}")
        except Exception as e:
            logging.error(f"[EvidenceRetrieval] Qdrant connection failed: {e}")
            self.qdrant = None

        # v0.3: Try Neo4j
        if use_neo4j:
            try:
                from neo4j_knowledge_graph import KnowledgeGraph

                self.neo4j_layer = KnowledgeGraph(auto_seed=True)
                if not self.neo4j_layer.connected:
                    self.neo4j_layer = None
            except Exception as e:
                print(f"[EvidenceRetrieval] Neo4j layer init skipped: {e}")

    def _ensure_collection(self):
        if not self.qdrant:
            return
        collections = self.qdrant.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(size=self.dim, distance=qmodels.Distance.COSINE),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_evidence(self, texts: List[str]):
        """Add documents to the Qdrant collection."""
        if not texts or not self.qdrant:
            return

        embeddings = self.encoder.encode(texts, convert_to_numpy=True).tolist()

        points = [
            qmodels.PointStruct(
                id=hash(text) % (2**63),
                vector=emb,
                payload={"text": text},  # Simple hash for demo
            )
            for text, emb in zip(texts, embeddings)
        ]

        self.qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    def retrieve(self, query: str, top_k: int = TOP_K_DEFAULT) -> List[Dict]:
        if not self.qdrant:
            return []

        q_emb = self.encoder.encode([query], convert_to_numpy=True)[0].tolist()

        hits = self.qdrant.search(collection_name=COLLECTION_NAME, query_vector=q_emb, limit=top_k)

        return [{"id": hit.id, "text": hit.payload["text"], "score": hit.score} for hit in hits]

    def verify_claim(self, claim: str) -> dict:
        """
        Compare claim against Qdrant knowledge base AND Neo4j graph triples (v1.0).

        Qdrant cosine similarity thresholds (0–1):
          ≥ 0.80 → supported
          ≤ 0.50 → uncertain
          between → refuted

        If Neo4j triples are found, their count boosts confidence:
          +0.05 per matching triple (capped at +0.20)
        """
        results = self.retrieve(claim, top_k=1)
        graph_triples = []

        # ── Qdrant verdict ──────────────────────────────────────────────
        if not results:
            verdict, conf = "uncertain", 0.0
            best_text = None
        else:
            best = results[0]
            sim = best["score"]
            best_text = best["text"][:300]

            if sim >= 0.80:
                verdict, conf = "supported", float(sim)
            elif sim <= 0.50:
                verdict, conf = "uncertain", float(1.0 - sim)
            else:
                verdict, conf = "refuted", float(1.0 - sim)

        # ── Neo4j graph augmentation ──────────────────────────────────
        if self.neo4j_layer and self.neo4j_layer.connected:
            try:
                graph_triples = self.neo4j_layer.query_context(claim)
                if graph_triples:
                    # Each matching triple nudges confidence toward supported
                    boost = min(0.20, len(graph_triples) * 0.05)
                    if verdict in ("supported", "uncertain"):
                        conf = min(1.0, conf + boost)
                    # If graph directly contradicts the claim text, boost refuted signal
                    # (a full NLI check is Phase 14; for now, trust Qdrant verdict)
            except Exception as e:
                print(f"[Neo4j] Query failed: {e}")

        return {
            "verdict": verdict,
            "confidence": round(conf, 4),
            "evidence": best_text,
            "graph_triples": graph_triples,
        }
