# FAISS Index Build — Colab Instructions

## What this does
Encodes all Reddit Mental Health posts into 768-dim vectors and builds a FAISS index.
Expected time on A100: 30-60 minutes. Do NOT run locally on RTX 3060 — it would take 6-12 hours.

## Steps

1. Upload `data/raw/reddit_mental_health/` (108 CSV files, ~3.1GB) to Google Drive at:
   `My Drive/empathrag/data/raw/reddit_mental_health/`

2. In `colab_build_faiss_index.py`, uncomment the Cell 2 block (Drive mount + path config)
   and comment out the local path config block at the top of Cell 3.

3. Open Colab → Runtime → Change runtime type → A100 GPU

4. Run all cells in order.

5. When complete, download from Drive:
   - `My Drive/empathrag/data/indexes/faiss_flat.index`
   - `My Drive/empathrag/data/indexes/metadata.db`
   Place them in your local `data/indexes/` folder.

## Expected output
- Total chunks: ~1-5 million (depends on corpus)
- FAISS index: IndexIVFFlat if >100K chunks, IndexFlatL2 if smaller
- SQLite DB: same number of rows as chunks, emotion_label=-1 (filled on Day 10)
