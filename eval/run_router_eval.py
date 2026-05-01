"""Evaluate rule routing vs lightweight ML routing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.ml_router import DEFAULT_MODEL_DIR, MLRouter  # noqa: E402
from pipeline.v2_schema import SafetyTier, classify_route  # noqa: E402


DEFAULT_DATASET = ROOT / "eval" / "empathrag_core_supervised.csv"


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-dir", type=Path, default=ROOT / DEFAULT_MODEL_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "router_eval_results.json")
    args = parser.parse_args()

    rows = [row for row in read_rows(args.dataset) if row.get("split") == args.split]
    router = MLRouter(args.model_dir)
    cases = []
    rule_route_correct = 0
    ml_route_correct = 0
    ml_tier_correct = 0

    for row in rows:
        expected_route = row["route_label"]
        expected_tier = row["safety_tier"]
        rule_route = classify_route(row["query_text"], SafetyTier(expected_tier), row.get("audience_mode") or "student").route.value
        pred = router.predict(row["query_text"], rule_route, expected_tier)
        rule_route_correct += int(rule_route == expected_route)
        ml_route_correct += int(pred.route_label == expected_route)
        ml_tier_correct += int(pred.safety_tier == expected_tier)
        cases.append(
            {
                "query_id": row["query_id"],
                "query_text": row["query_text"],
                "expected_route": expected_route,
                "rule_route": rule_route,
                "ml_route": pred.route_label,
                "expected_tier": expected_tier,
                "ml_tier": pred.safety_tier,
                "route_confidence": pred.route_confidence,
                "tier_confidence": pred.tier_confidence,
                "used_ml": pred.used_ml,
                "reason": pred.reason,
            }
        )

    total = len(rows)
    result = {
        "summary": {
            "rows": total,
            "model_available": router.available,
            "rule_route_accuracy": rule_route_correct / total if total else None,
            "ml_route_accuracy": ml_route_correct / total if total else None,
            "ml_tier_accuracy": ml_tier_correct / total if total else None,
        },
        "cases": cases,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
