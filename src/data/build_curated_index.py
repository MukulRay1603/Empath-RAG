"""
Build a separate FAISS + SQLite index for the curated EmpathRAG corpus.

Run from repo root:
    python -m src.data.build_curated_index
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .curated_resources import ingestion_rows, validate_file


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_INPUT = "data/curated/resources_seed.jsonl"
DEFAULT_INDEX = "data/curated/indexes/faiss_curated.index"
DEFAULT_DB = "data/curated/indexes/metadata_curated.db"


def build_curated_index(
    input_path: str = DEFAULT_INPUT,
    index_path: str = DEFAULT_INDEX,
    db_path: str = DEFAULT_DB,
    model_name: str = MODEL_NAME,
) -> None:
    rows, _ = validate_file(input_path, strict=True)
    usable = ingestion_rows(rows)
    if not usable:
        raise ValueError("No usable curated rows found after validation/filtering.")

    texts = [row["text"] for row in usable]
    print(f"Curated rows loaded: {len(rows)}")
    print(f"Rows entering retrieval index: {len(usable)}")

    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)

    _write_metadata(db_path, usable)

    print(f"Curated FAISS index saved: {index_path}")
    print(f"Curated metadata DB saved: {db_path}")
    print(f"Vectors indexed: {index.ntotal}")


def _write_metadata(db_path: str, rows: list[dict]) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            resource_id TEXT UNIQUE NOT NULL,
            text TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_name TEXT NOT NULL,
            source_type TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            topic TEXT NOT NULL,
            audience TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            usage_mode TEXT NOT NULL,
            summary TEXT NOT NULL,
            last_checked TEXT NOT NULL,
            notes TEXT NOT NULL
        )
        """
    )
    c.execute("CREATE INDEX idx_chunks_topic ON chunks(topic)")
    c.execute("CREATE INDEX idx_chunks_risk ON chunks(risk_level)")
    c.execute("CREATE INDEX idx_chunks_usage ON chunks(usage_mode)")

    for idx, row in enumerate(rows):
        c.execute(
            """
            INSERT INTO chunks (
                id, resource_id, text, source_id, source_name, source_type,
                title, url, topic, audience, risk_level, usage_mode, summary,
                last_checked, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                idx,
                row["id"],
                row["text"],
                row["source_id"],
                row["source_name"],
                row["source_type"],
                row["title"],
                row["url"],
                row["topic"],
                row["audience"],
                row["risk_level"],
                row["usage_mode"],
                row["summary"],
                row["last_checked"],
                row["notes"],
            ),
        )

    conn.commit()
    conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build curated EmpathRAG FAISS index.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    build_curated_index(
        input_path=args.input,
        index_path=args.index,
        db_path=args.db,
        model_name=args.model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
