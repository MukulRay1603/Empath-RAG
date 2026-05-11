"""Train lightweight EmpathRAG Core route and safety-tier classifiers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.ml_router import DEFAULT_MODEL_DIR, save_models, train_classifier  # noqa: E402


DEFAULT_DATASET = ROOT / "eval" / "empathrag_core_supervised.csv"


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-dir", type=Path, default=ROOT / DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    rows = [row for row in read_rows(args.dataset) if row.get("split") == "train"]
    if len(rows) < 10:
        raise SystemExit("Need at least 10 training rows. Run eval/prepare_karthik_dataset.py first.")

    texts = [row["query_text"] for row in rows]
    route_labels = [row["route_label"] for row in rows]
    tier_labels = [row["safety_tier"] for row in rows]
    route_model = train_classifier(texts, route_labels)
    tier_model = train_classifier(texts, tier_labels)
    save_models(route_model, tier_model, args.model_dir)
    print(f"Trained router on {len(rows)} rows")
    print(f"Saved models to {args.model_dir}")


if __name__ == "__main__":
    main()
