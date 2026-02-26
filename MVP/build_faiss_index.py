"""
Script to build a real Wikipedia-based FAISS knowledge index.
Downloads a small but quality subset via HuggingFace datasets (Wikipedia 20220301.en, first 50K passages).
Builds a 768-dim SBERT FAISS index and saves it to disk.

Run this ONCE: python build_faiss_index.py
The index is saved to knowledge_base/wiki.faiss and knowledge_base/wiki_texts.json
"""
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

INDEX_DIR = "knowledge_base"
INDEX_PATH = os.path.join(INDEX_DIR, "wiki.faiss")
TEXTS_PATH = os.path.join(INDEX_DIR, "wiki_texts.json")
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
NUM_PASSAGES = 5_000   # Fast but real: 10x better than hardcoded seeds
BATCH_SIZE = 256

def build_index():
    os.makedirs(INDEX_DIR, exist_ok=True)
    print(f"Loading Wikipedia dataset (first {NUM_PASSAGES} passages)...")
    # Using the squad dataset which has Wikipedia contexts — fast to download
    dataset = load_dataset("squad", split="train", trust_remote_code=True)

    # Deduplicate contexts (SQuAD has repeated passages per question)
    seen = set()
    passages = []
    for item in dataset:
        ctx = item["context"].strip()
        if ctx not in seen:
            seen.add(ctx)
            passages.append(ctx)
        if len(passages) >= NUM_PASSAGES:
            break

    print(f"  Got {len(passages)} unique passages.")

    print("Loading SBERT encoder...")
    encoder = SentenceTransformer(MODEL_NAME)
    dim = 768

    print("Encoding passages in batches...")
    all_embeddings = []
    for i in range(0, len(passages), BATCH_SIZE):
        batch = passages[i:i + BATCH_SIZE]
        embs = encoder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(embs)
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  Encoded {min(i + BATCH_SIZE, len(passages))}/{len(passages)}")

    all_embeddings = np.vstack(all_embeddings).astype("float32")

    print("Building FAISS index (IndexFlatIP with normalized vectors for cosine sim)...")
    faiss.normalize_L2(all_embeddings)
    index = faiss.IndexFlatIP(dim)  # Inner product == cosine sim on normalized vecs
    index.add(all_embeddings)

    print(f"Saving FAISS index to {INDEX_PATH} and texts to {TEXTS_PATH}...")
    faiss.write_index(index, INDEX_PATH)
    with open(TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False)

    print(f"\n✅ Done! Index contains {index.ntotal} vectors.")
    print(f"   Index file: {INDEX_PATH} ({os.path.getsize(INDEX_PATH) / 1e6:.1f} MB)")

if __name__ == "__main__":
    build_index()
