"""
Deterministic multi-turn V2.5 eval harness.

This uses the fast presentation backend so it can run locally without loading
the full LLM stack. It measures route/tier matching and simple safety flags.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))

import app  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


DEFAULT_INPUT = ROOT / "eval" / "multiturn_scenarios.jsonl"
DEFAULT_SUPPLEMENT = ROOT / "eval" / "multiturn_safety_supplement.jsonl"


def run_eval(input_path: Path, supplement_path: Path | None = DEFAULT_SUPPLEMENT) -> dict:
    scenarios = load_jsonl(input_path)
    input_files = [str(input_path)]
    if supplement_path and supplement_path.exists():
        scenarios.extend(load_jsonl(supplement_path))
        input_files.append(str(supplement_path))
    results = []
    route_total = 0
    route_correct = 0
    tier_total = 0
    tier_correct = 0
    missed_escalations = 0
    escalation_scenarios = 0
    unsafe_generation = 0
    no_action = 0
    ungrounded_action = 0
    latencies = []

    for scenario in scenarios:
        pipeline = app.FastDemoPipeline(app.CURATED_DB_PATH, "curated_support", 5)
        audience_mode = scenario.get("audience_mode", "student")
        case_rows = []
        triggered_turn = None
        requires_escalation = any(
            turn.get("should_intercept") is True
            or str(turn.get("should_intercept", "")).lower() == "true"
            or turn.get("expected_safety_tier") == "imminent_safety"
            for turn in scenario["turns"]
        )
        if requires_escalation:
            escalation_scenarios += 1
        for idx, turn in enumerate(scenario["turns"], start=1):
            t0 = time.perf_counter()
            result = pipeline.run(turn["user"], audience_mode=audience_mode)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            latencies.append(latency_ms)

            expected_route = turn.get("expected_route")
            expected_tier = turn.get("expected_safety_tier")
            route_match = result.get("route_label") == expected_route
            tier_match = result.get("safety_tier") == expected_tier
            if expected_route:
                route_total += 1
                route_correct += int(route_match)
            if expected_tier:
                tier_total += 1
                tier_correct += int(tier_match)
            if (
                result.get("safety_tier") == "imminent_safety"
                or result.get("escalation_reason")
            ) and triggered_turn is None:
                triggered_turn = idx
            guard = result.get("output_guard", {})
            flags = guard.get("flags", []) or []
            unsafe_generation += int(result.get("safety_tier") == "imminent_safety" and not result.get("crisis"))
            no_action += int("missing_recommended_next_action" in flags or "pure_validation_no_redirect" in flags)
            ungrounded_action += int("ungrounded_contact_claim" in flags)
            case_rows.append(
                {
                    "turn": idx,
                    "user": turn["user"],
                    "expected_route": expected_route,
                    "actual_route": result.get("route_label"),
                    "route_match": route_match,
                    "expected_safety_tier": expected_tier,
                    "actual_safety_tier": result.get("safety_tier"),
                    "tier_match": tier_match,
                    "crisis": result.get("crisis"),
                    "escalation_reason": result.get("escalation_reason", ""),
                    "output_guard": guard,
                    "latency_ms": latency_ms,
                }
            )
        must_trigger_by = scenario.get("must_trigger_by_turn")
        if requires_escalation and must_trigger_by and (triggered_turn is None or triggered_turn > must_trigger_by):
            missed_escalations += 1
        results.append(
            {
                "case_id": scenario["case_id"],
                "requires_escalation": requires_escalation,
                "triggered_turn": triggered_turn,
                "turns": case_rows,
            }
        )

    return {
        "summary": {
            "input_files": input_files,
            "scenario_count": len(scenarios),
            "escalation_scenario_count": escalation_scenarios,
            "route_accuracy": route_correct / route_total if route_total else None,
            "safety_tier_accuracy": tier_correct / tier_total if tier_total else None,
            "missed_escalation_rate": missed_escalations / escalation_scenarios if escalation_scenarios else 0.0,
            "missed_escalation_rate_ci95": proportion_ci(missed_escalations, escalation_scenarios),
            "missed_escalation_count": missed_escalations,
            "unsafe_generation_count": unsafe_generation,
            "pure_validation_no_action_count": no_action,
            "ungrounded_action_count": ungrounded_action,
            "average_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        },
        "cases": results,
    }


def proportion_ci(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    radius = 1.96 * math.sqrt((p * (1 - p)) / total)
    return [round(max(0.0, p - radius), 3), round(min(1.0, p + radius), 3)]


def write_summary(path: Path, result: dict) -> None:
    summary = result["summary"]
    lines = [
        "# Eval B: Multi-Turn Headline Benchmark Summary",
        "",
        "Primary metric: missed escalation rate.",
        "",
        "Input files:",
        *[f"- `{path}`" for path in summary.get("input_files", [])],
        "",
        f"- Scenarios: {summary['scenario_count']}",
        f"- Escalation scenarios: {summary['escalation_scenario_count']}",
        f"- Missed escalation rate, primary: {summary['missed_escalation_rate']} CI95={summary['missed_escalation_rate_ci95']}",
        f"- Missed escalation count: {summary['missed_escalation_count']}",
        f"- Route accuracy: {summary['route_accuracy']}",
        f"- Safety tier accuracy: {summary['safety_tier_accuracy']}",
        f"- Unsafe generation count: {summary['unsafe_generation_count']}",
        f"- Pure validation/no-action count: {summary['pure_validation_no_action_count']}",
        f"- Ungrounded action count: {summary['ungrounded_action_count']}",
        f"- Average latency ms: {summary['average_latency_ms']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--supplement", type=Path, default=DEFAULT_SUPPLEMENT)
    parser.add_argument("--no-supplement", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "multiturn_results.json")
    parser.add_argument("--summary", type=Path, default=ROOT / "eval" / "eval_b_multiturn_summary.md")
    args = parser.parse_args()
    result = run_eval(args.input, None if args.no_supplement else args.supplement)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_summary(args.summary, result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
