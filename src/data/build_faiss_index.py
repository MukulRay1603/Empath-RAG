import os
import json
import sqlite3

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm

# Import from sibling module — run from repo root as: python -m src.data.build_faiss_index
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocess import clean_text

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 256
STRIDE = 32
MAX_CHUNKS = 8


def chunk_text(text, tokenizer, chunk_size=CHUNK_SIZE, stride=STRIDE, max_chunks=MAX_CHUNKS):
    tokens = tokenizer.encode(text)
    if len(tokens) < 64:
        return [text]
    chunks = []
    start = 0
    while start < len(tokens) and len(chunks) < max_chunks:
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
        start += chunk_size - stride
    return chunks


def load_reddit_posts(data_dir="data/raw/reddit_mental_health"):
    all_posts = []
    if not os.path.exists(data_dir):
        print(f"WARNING: {data_dir} does not exist yet. Run dataset download first.")
        return all_posts
    for fname in os.listdir(data_dir):
        if fname.endswith(".csv") or fname.endswith(".json"):
            fpath = os.path.join(data_dir, fname)
            try:
                df = pd.read_csv(fpath, on_bad_lines="skip")
                if "post" in df.columns:
                    all_posts.extend(df["post"].dropna().tolist())
                elif "body" in df.columns:
                    all_posts.extend(df["body"].dropna().tolist())
                elif "selftext" in df.columns:
                    all_posts.extend(df["selftext"].dropna().tolist())
            except Exception as e:
                print(f"Skipping {fname}: {e}")
    return all_posts


def build_index(
    reddit_dir="data/raw/reddit_mental_health",
    index_path="data/indexes/faiss_flat.index",
    db_path="data/indexes/metadata.db",
):
    os.makedirs("data/indexes", exist_ok=True)

    all_posts = load_reddit_posts(reddit_dir)
    print(f"Raw posts loaded: {len(all_posts)}")

    encoder = SentenceTransformer(MODEL_NAME)
    tok = AutoTokenizer.from_pretrained("roberta-base")

    chunks = []
    for post in tqdm(all_posts, desc="Chunking"):
        cleaned = clean_text(post)
        if not cleaned:
            continue
        chunks.extend(chunk_text(cleaned, tok))

    print(f"Total chunks: {len(chunks)}")
    if not chunks:
        print("No chunks to index. Exiting.")
        return

    embeddings = encoder.encode(
        chunks, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]  # 768
    if len(chunks) > 100_000:
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, 100)
        index.train(embeddings)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        text TEXT,
        emotion_label INTEGER DEFAULT -1,
        safety_score REAL DEFAULT 0.7,
        source TEXT
    )""")
    for i, chunk in enumerate(chunks):
        c.execute(
            "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?)",
            (i, chunk, -1, 0.7, "reddit"),
        )
    conn.commit()
    conn.close()

    print(f"Index built: {index.ntotal} vectors | SQLite: {len(chunks)} rows")
    print(f"Index saved: {index_path}")
    print(f"Metadata DB: {db_path}")


if __name__ == "__main__":
    build_index()
