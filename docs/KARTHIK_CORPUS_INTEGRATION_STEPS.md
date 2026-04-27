# Karthik Corpus Integration Steps

Use this after Karthik sends the curated corpus delivery.

## Expected Files

Place the files here:

```text
data/curated/resources_seed.jsonl
data/curated/source_inventory.csv
data/curated/excluded_sources.csv
data/curated/raw_pages/
```

The only required file for indexing is `resources_seed.jsonl`. The others are
for review, reproducibility, and the class/research writeup.

## Validate

```bash
python -m src.data.curated_resources data/curated/resources_seed.jsonl
```

If validation fails, fix the reported line numbers before indexing.

## Build Curated Index

```bash
python -m src.data.build_curated_index
```

This creates:

```text
data/curated/indexes/faiss_curated.index
data/curated/indexes/metadata_curated.db
```

These artifacts are intentionally separate from the existing Reddit research
index. Do not delete or overwrite `data/indexes/faiss_flat.index` or
`data/indexes/metadata.db`.

## Run Retrieval Audit

```bash
python eval/run_curated_retrieval_audit.py
```

Review `eval/curated_retrieval_audit.json` and confirm top sources are safe,
official, and relevant.

## Run Demo

Default demo behavior uses `auto` retrieval mode:

- If curated index exists, use `curated_support`.
- If curated index is missing, fall back to `reddit_research`.

```bash
python demo/app.py
```

To force a mode:

```bash
set EMPATHRAG_RETRIEVAL_CORPUS=curated_support
python demo/app.py
```

or:

```bash
set EMPATHRAG_RETRIEVAL_CORPUS=reddit_research
python demo/app.py
```

## Presentation Safety Note

For the MSML demo, describe Reddit as the original research corpus and curated
resources as the safer V2 student-support corpus. The curated corpus is the
preferred path for any UMD counseling conversation or future publication work.
