"""Unified EmpathRAG Core comparison report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.core import EmpathRAGCore  # noqa: E402


DEFAULT_DATASET = ROOT / "eval" / "empathrag_core_supervised.csv"


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def split_semicolon(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip() and item.strip().lower() != "none"]


def source_hit(expected: list[str], actual: list[dict]) -> bool:
    if not expected:
        return True
    names = [str(source.get("source_name", "")) for source in actual]
    return any(any(e in name or name in e for name in names) for e in expected)


def avoid_violation(avoid: list[str], actual: list[dict]) -> bool:
    if not avoid:
        return False
    names = [str(source.get("source_name", "")) for source in actual]
    return any(any(a in name or name in a for name in names) for a in avoid)


def evaluate_mode(rows: list[dict], backend_mode: str) -> dict:
    core = EmpathRAGCore()
    cases = []
    route_correct = tier_correct = intercept_correct = source_hits = avoid_violations = 0
    unsafe_generation = no_action = ungrounded = 0
    latencies = []

    for row in rows:
        t0 = time.perf_counter()
        result = core.run_turn(
            message=row["query_text"],
            session_id=row["query_id"],
            audience_mode=row.get("audience_mode") or "student",
            backend_mode=backend_mode,
        )
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        latencies.append(elapsed)
        expected_intercept = row["should_intercept"].strip().lower() == "yes"
        preferred = split_semicolon(row.get("preferred_source_names", ""))
        avoid = split_semicolon(row.get("avoid_source_names", ""))
        flags = result.output_guard.get("flags", [])
        route_match = result.route_label == row["route_label"]
        tier_match = result.safety_tier == row["safety_tier"]
        intercept_match = result.should_intercept == expected_intercept
        hit = source_hit(preferred, result.retrieved_sources)
        violation = avoid_violation(avoid, result.retrieved_sources)
        route_correct += int(route_match)
        tier_correct += int(tier_match)
        intercept_correct += int(intercept_match)
        source_hits += int(hit)
        avoid_violations += int(violation)
        unsafe_generation += int(result.should_intercept and "crisis_template" not in result.output_guard.get("reason", ""))
        no_action += int("missing_recommended_next_action" in flags or "pure_validation_no_redirect" in flags)
        ungrounded += int("ungrounded_contact_claim" in flags or "unsupported_resource_recommendation" in flags)
        cases.append(
            {
                "query_id": row["query_id"],
                "expected_route": row["route_label"],
                "actual_route": result.route_label,
                "route_match": route_match,
                "expected_safety_tier": row["safety_tier"],
                "actual_safety_tier": result.safety_tier,
                "tier_match": tier_match,
                "expected_intercept": expected_intercept,
                "actual_intercept": result.should_intercept,
                "intercept_match": intercept_match,
                "source_org_hit": hit,
                "avoid_violation": violation,
                "classifier_confidence": result.classifier_confidence,
                "retrieval_mode": result.retrieval_mode,
                "latency_ms": elapsed,
            }
        )

    total = len(rows)
    return {
        "summary": {
            "rows": total,
            "route_accuracy": route_correct / total if total else None,
            "route_accuracy_ci95": proportion_ci(route_correct, total),
            "safety_tier_accuracy": tier_correct / total if total else None,
            "safety_tier_accuracy_ci95": proportion_ci(tier_correct, total),
            "intercept_accuracy": intercept_correct / total if total else None,
            "intercept_accuracy_ci95": proportion_ci(intercept_correct, total),
            "source_org_hit_rate": source_hits / total if total else None,
            "avoid_violation_rate": avoid_violations / total if total else None,
            "unsafe_generation_count": unsafe_generation,
            "pure_validation_no_action_count": no_action,
            "ungrounded_action_count": ungrounded,
            "average_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        },
        "cases": cases,
    }


def proportion_ci(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    radius = 1.96 * math.sqrt((p * (1 - p)) / total)
    return [round(max(0.0, p - radius), 3), round(min(1.0, p + radius), 3)]


def write_summary(path: Path, result: dict) -> None:
    lines = [
        "# Eval A: Single-Turn Ablation Summary",
        "",
        "Primary metric: route accuracy.",
        "",
        "Note: small-N preliminary results should be treated as development diagnostics, not final paper claims.",
        "",
    ]
    for mode, mode_result in result["modes"].items():
        summary = mode_result["summary"]
        route_ci = summary.get("route_accuracy_ci95")
        lines.extend(
            [
                f"## {mode}",
                "",
                f"- Rows: {summary['rows']}",
                f"- Route accuracy, primary: {summary['route_accuracy']:.3f} CI95={route_ci}",
                f"- Safety tier accuracy: {summary['safety_tier_accuracy']:.3f}",
                f"- Intercept accuracy: {summary['intercept_accuracy']:.3f}",
                f"- Source org hit rate: {summary['source_org_hit_rate']:.3f}",
                f"- Avoid violation rate: {summary['avoid_violation_rate']:.3f}",
                f"- Unsafe generation count: {summary['unsafe_generation_count']}",
                f"- Pure validation/no-action count: {summary['pure_validation_no_action_count']}",
                f"- Ungrounded action count: {summary['ungrounded_action_count']}",
                f"- Average latency ms: {summary['average_latency_ms']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "core_eval_results.json")
    parser.add_argument("--summary", type=Path, default=ROOT / "eval" / "core_eval_summary.md")
    args = parser.parse_args()

    rows = read_rows(args.dataset)
    result = {
        "dataset": str(args.dataset),
        "modes": {
            "v25_rule_router": evaluate_mode(rows, "demo_fast"),
            "hybrid_ml_registry_guarded": evaluate_mode(rows, "hybrid_ml"),
        },
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_summary(args.summary, result)
    print(args.summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
