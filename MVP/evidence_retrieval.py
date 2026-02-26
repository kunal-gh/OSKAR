import json
import os
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import Optional

INDEX_DIR = "knowledge_base"
INDEX_PATH = os.path.join(INDEX_DIR, "wiki.faiss")
TEXTS_PATH = os.path.join(INDEX_DIR, "wiki_texts.json")
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K_DEFAULT = 5


class EvidenceRetrieval:
    """
    SBERT + FAISS + Neo4j evidence retrieval module (v0.3 Graph-RAG).

    On startup:
      - If pre-built FAISS index exists on disk → loads it instantly (< 100ms)
      - Otherwise → starts with an empty in-memory index (dev/test mode)
      - Attempts to connect to Neo4j (bolt://localhost:7687)
        → If Neo4j is available, graph triples augment FAISS results
        → If Neo4j is unavailable, silently falls back to FAISS-only

    Output schema for verify_claim:
      { "verdict": "supported|refuted|uncertain", "confidence": float,
        "evidence": str|None, "graph_triples": list[dict] }
    """

    def __init__(self, use_neo4j: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(SBERT_MODEL, device=self.device)
        self.dim = 768
        self._texts: list[str] = []
        self.neo4j_layer = None

        if os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH):
            print(f"[EvidenceRetrieval] Loading pre-built FAISS index from {INDEX_PATH}...")
            self.index = faiss.read_index(INDEX_PATH)
            with open(TEXTS_PATH, "r", encoding="utf-8") as f:
                self._texts = json.load(f)
            print(f"[EvidenceRetrieval] Loaded {self.index.ntotal} passages.")
        else:
            print("[EvidenceRetrieval] No pre-built index found. Starting with empty in-memory index.")
            self.index = faiss.IndexFlatIP(self.dim)

        # v0.3: Try Neo4j — auto-falls back to FAISS-only if unavailable
        if use_neo4j:
            try:
                from neo4j_knowledge_graph import KnowledgeGraph
                self.neo4j_layer = KnowledgeGraph(auto_seed=True)
                if not self.neo4j_layer.connected:
                    self.neo4j_layer = None
            except Exception as e:
                print(f"[EvidenceRetrieval] Neo4j layer init skipped: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_evidence(self, texts: list[str], metadata: list[dict] = None):
        """Add documents to the in-memory index (dev/test use)."""
        if not texts:
            return
        embs = self.embed(texts)
        faiss.normalize_L2(embs)
        self.index.add(embs)
        self._texts.extend(texts)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(texts, convert_to_numpy=True).astype("float32")

    def retrieve(self, query: str, top_k: int = TOP_K_DEFAULT) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        q_emb = self.embed([query])
        faiss.normalize_L2(q_emb)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._texts):
                results.append({
                    "id": int(idx),
                    "text": self._texts[idx],
                    "score": float(score),  # cosine similarity (0–1)
                })
        return results

    def verify_claim(self, claim: str) -> dict:
        """
        Compare claim against FAISS knowledge base AND Neo4j graph triples (v0.3).

        FAISS cosine similarity thresholds (0–1):
          ≥ 0.80 → supported
          ≤ 0.50 → uncertain
          between → refuted

        If Neo4j triples are found, their count boosts confidence:
          +0.05 per matching triple (capped at +0.20)
        """
        results = self.retrieve(claim, top_k=1)
        graph_triples = []

        # ── FAISS verdict ──────────────────────────────────────────────
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
                    # (a full NLI check is Phase 14; for now, trust FAISS verdict)
            except Exception as e:
                print(f"[Neo4j] Query failed: {e}")

        return {
            "verdict": verdict,
            "confidence": round(conf, 4),
            "evidence": best_text,
            "graph_triples": graph_triples,
        }

