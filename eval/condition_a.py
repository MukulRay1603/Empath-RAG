"""
eval/condition_a.py
Condition A — BM25 sparse retrieval baseline (no emotion components).
Loads all chunk texts from SQLite, builds a BM25 index, retrieves top_k
chunks for a query using keyword overlap only.
Used by run_ablation.py as the Condition A retrieval function.
"""

import sqlite3
import json
from rank_bm25 import BM25Okapi

DB_PATH = "data/indexes/metadata.db"

def load_bm25_index(db_path: str = DB_PATH):
    """Load all chunk texts from SQLite and build BM25 index. Returns (bm25, id_list, text_list)."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
    conn.close()
    ids   = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, ids, texts

def retrieve_bm25(query: str, bm25, ids, texts, top_k: int = 5):
    """Retrieve top_k chunks using BM25. Returns list of text strings."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    import numpy as np
    top_indices = scores.argsort()[::-1][:top_k]
    return [texts[i] for i in top_indices]

if __name__ == "__main__":
    print("Building BM25 index from SQLite (this takes ~60-90s for 1.67M chunks)...")
    bm25, ids, texts = load_bm25_index()
    print(f"BM25 index built: {len(texts):,} documents")
    # Quick sanity check
    results = retrieve_bm25("I feel hopeless and overwhelmed", bm25, ids, texts, top_k=3)
    print(f"Sample retrieval (3 chunks):")
    for i, r in enumerate(results):
        print(f"  [{i+1}] {r[:100]}...")
