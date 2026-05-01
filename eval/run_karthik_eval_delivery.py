"""
Run Karthik's V1 eval delivery against the V2.5 fast backend.

This is an experiment harness, not a clinical validation. It uses a fresh
session for each single-turn eval query, then separately scores the risky /
ambiguous cases. The output is intended for iteration and presentation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))

import app  # noqa: E402


DEFAULT_DELIVERY = ROOT / "Data_Karthik" / "empathrag_eval_delivery_v1"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def split_semicolon(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip() and item.strip().lower() != "none"]


def source_org_hit(expected_source_names: list[str], retrieved_sources: list[dict]) -> bool:
    if not expected_source_names:
        return True
    actual = [str(source.get("source_name", "")) for source in retrieved_sources]
    return any(any(expected in source or source in expected for source in actual) for expected in expected_source_names)


def score_eval_queries(delivery_dir: Path) -> dict:
    rows = read_csv(delivery_dir / "eval_queries.csv")
    cases = []
    intercept_correct = 0
    source_hits = 0
    latencies = []

    for row in rows:
        pipeline = app.FastDemoPipeline(app.CURATED_DB_PATH, "curated_support", 5)
        t0 = time.perf_counter()
        result = pipeline.run(row["query_text"])
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        latencies.append(latency_ms)

        expected_intercept = row["should_intercept"].strip().lower() == "yes"
        actual_intercept = bool(result.get("crisis"))
        intercept_match = expected_intercept == actual_intercept
        intercept_correct += int(intercept_match)

        expected_sources = split_semicolon(row.get("expected_source_names", ""))
        hit = source_org_hit(expected_sources, result.get("retrieved_sources", []))
        source_hits += int(hit)

        cases.append(
            {
                "query_id": row["query_id"],
                "scenario_category": row["scenario_category"],
                "risk_category": row["risk_category"],
                "query_text": row["query_text"],
                "expected_intercept": expected_intercept,
                "actual_intercept": actual_intercept,
                "intercept_match": intercept_match,
                "expected_source_names": expected_sources,
                "actual_source_names": [source.get("source_name", "") for source in result.get("retrieved_sources", [])],
                "source_org_hit": hit,
                "route": result.get("route_label", ""),
                "safety_tier": result.get("safety_tier", ""),
                "output_guard": result.get("output_guard", {}),
                "latency_ms": latency_ms,
            }
        )

    total = len(rows)
    return {
        "summary": {
            "rows": total,
            "intercept_accuracy": intercept_correct / total if total else None,
            "source_org_hit_rate": source_hits / total if total else None,
            "average_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        },
        "cases": cases,
    }


def score_risky_cases(delivery_dir: Path) -> dict:
    rows = read_csv(delivery_dir / "risky_or_ambiguous_cases.csv")
    cases = []
    intercept_correct = 0

    for row in rows:
        pipeline = app.FastDemoPipeline(app.CURATED_DB_PATH, "curated_support", 5)
        result = pipeline.run(row["query_text"])
        expected_intercept = row["should_intercept"].strip().lower() == "yes"
        actual_intercept = bool(result.get("crisis"))
        intercept_match = expected_intercept == actual_intercept
        intercept_correct += int(intercept_match)
        cases.append(
            {
                "case_id": row["case_id"],
                "correct_risk_category": row["correct_risk_category"],
                "query_text": row["query_text"],
                "expected_intercept": expected_intercept,
                "actual_intercept": actual_intercept,
                "intercept_match": intercept_match,
                "route": result.get("route_label", ""),
                "safety_tier": result.get("safety_tier", ""),
                "expected_handling": row["expected_handling"],
            }
        )

    total = len(rows)
    return {
        "summary": {
            "rows": total,
            "intercept_accuracy": intercept_correct / total if total else None,
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery-dir", type=Path, default=DEFAULT_DELIVERY)
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "karthik_eval_results.json")
    args = parser.parse_args()

    result = {
        "delivery_dir": str(args.delivery_dir),
        "eval_queries": score_eval_queries(args.delivery_dir),
        "risky_or_ambiguous_cases": score_risky_cases(args.delivery_dir),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "eval_queries": result["eval_queries"]["summary"],
            "risky_or_ambiguous_cases": result["risky_or_ambiguous_cases"]["summary"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
