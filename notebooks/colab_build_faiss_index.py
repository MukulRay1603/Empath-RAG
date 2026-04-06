# EmpathRAG — FAISS Index Builder
# Run on Google Colab Pro (A100).
# Estimated time: 30-60 minutes for full Reddit Mental Health corpus.
#
# SETUP INSTRUCTIONS:
# 1. Upload the entire data/raw/reddit_mental_health/ folder to Google Drive
#    at: My Drive/empathrag/data/raw/reddit_mental_health/
# 2. Set Colab runtime to A100 GPU
# 3. Run all cells in order
# 4. Download faiss_flat.index and metadata.db from Drive when done

# ── Cell 1: Install ──────────────────────────────────────────────────────────
# !pip install sentence-transformers faiss-cpu tqdm -q
# !pip install transformers -q

# ── Cell 2: Mount Drive ──────────────────────────────────────────────────────
# from google.colab import drive
# drive.mount("/content/drive")

# BASE = "/content/drive/MyDrive/empathrag"
# REDDIT_DIR = f"{BASE}/data/raw/reddit_mental_health"
# INDEX_PATH = f"{BASE}/data/indexes/faiss_flat.index"
# DB_PATH    = f"{BASE}/data/indexes/metadata.db"
# import os
# os.makedirs(f"{BASE}/data/indexes", exist_ok=True)

# ── Cell 3: Build index ──────────────────────────────────────────────────────
import os
import re
import sqlite3
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
# When running locally for testing, override these paths:
REDDIT_DIR = "data/raw/reddit_mental_health"
INDEX_PATH = "data/indexes/faiss_flat.index"
DB_PATH    = "data/indexes/metadata.db"

MODEL_NAME  = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE  = 256
STRIDE      = 32
MAX_CHUNKS  = 8
# A100: batch_size=128 is safe. RTX 3060 6GB laptop: use 16.
BATCH_SIZE  = 128


# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r"u/\w+", "", text)
    text = re.sub(r"r/\w+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\[deleted\]|\[removed\]", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text, tokenizer, chunk_size=CHUNK_SIZE, stride=STRIDE, max_chunks=MAX_CHUNKS):
    tokens = tokenizer.encode(text)
    if len(tokens) < 64:
        return [text]
    chunks = []
    start = 0
    while start < len(tokens) and len(chunks) < max_chunks:
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end], skip_special_tokens=True))
        start += chunk_size - stride
    return chunks


# ── Load posts ────────────────────────────────────────────────────────────────
def load_reddit_posts(data_dir):
    all_posts = []
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    print(f"Loading from {len(files)} CSV files...")
    for fname in tqdm(files, desc="Reading CSVs"):
        fpath = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(fpath, on_bad_lines="skip", usecols=lambda c: c in ["post", "body", "selftext"])
            for col in ["post", "body", "selftext"]:
                if col in df.columns:
                    all_posts.extend(df[col].dropna().tolist())
                    break
        except Exception as e:
            print(f"  Skipping {fname}: {e}")
    print(f"Total raw posts loaded: {len(all_posts):,}")
    return all_posts


# ── Main ──────────────────────────────────────────────────────────────────────
def build_index():
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    # Load and chunk
    tok = AutoTokenizer.from_pretrained("roberta-base")
    all_posts = load_reddit_posts(REDDIT_DIR)

    chunks = []
    for post in tqdm(all_posts, desc="Chunking"):
        cleaned = clean_text(str(post))
        if not cleaned:
            continue
        chunks.extend(chunk_text(cleaned, tok))

    print(f"Total chunks to encode: {len(chunks):,}")

    # Encode — use GPU automatically if available
    encoder = SentenceTransformer(MODEL_NAME)
    print(f"Encoding on: {encoder.device}")
    embeddings = encoder.encode(
        chunks,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    # Build FAISS index
    dim = embeddings.shape[1]  # 768
    if len(chunks) > 100_000:
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, 100)
        print("Training IVFFlat index...")
        index.train(embeddings)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index saved: {index.ntotal:,} vectors → {INDEX_PATH}")

    # SQLite sidecar
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        text TEXT,
        emotion_label INTEGER DEFAULT -1,
        safety_score REAL DEFAULT 0.7,
        source TEXT
    )""")
    # Insert in batches to avoid memory spike
    ISERT_BATCH = 10_000
    for i in range(0, len(chunks), ISERT_BATCH):
        batch = chunks[i:i+ISERT_BATCH]
        c.executemany(
            "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?)",
            [(i+j, text, -1, 0.7, "reddit") for j, text in enumerate(batch)]
        )
        conn.commit()
    conn.close()
    print(f"SQLite DB saved: {len(chunks):,} rows → {DB_PATH}")
    print("Done. Download faiss_flat.index and metadata.db from Drive.")


if __name__ == "__main__":
    build_index()
