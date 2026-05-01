"""
Deterministic multi-turn V2.5 eval harness.

This uses the fast presentation backend so it can run locally without loading
the full LLM stack. It measures route/tier matching and simple safety flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))

import app  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_eval(input_path: Path) -> dict:
    scenarios = load_jsonl(input_path)
    results = []
    route_total = 0
    route_correct = 0
    tier_total = 0
    tier_correct = 0
    missed_escalations = 0
    unsafe_generation = 0
    no_action = 0
    ungrounded_action = 0
    latencies = []

    for scenario in scenarios:
        pipeline = app.FastDemoPipeline(app.CURATED_DB_PATH, "curated_support", 5)
        audience_mode = scenario.get("audience_mode", "student")
        case_rows = []
        triggered_turn = None
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
        if must_trigger_by and (triggered_turn is None or triggered_turn > must_trigger_by):
            missed_escalations += 1
        results.append({"case_id": scenario["case_id"], "triggered_turn": triggered_turn, "turns": case_rows})

    return {
        "summary": {
            "scenario_count": len(scenarios),
            "route_accuracy": route_correct / route_total if route_total else None,
            "safety_tier_accuracy": tier_correct / tier_total if tier_total else None,
            "missed_escalation_count": missed_escalations,
            "unsafe_generation_count": unsafe_generation,
            "pure_validation_no_action_count": no_action,
            "ungrounded_action_count": ungrounded_action,
            "average_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        },
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "eval" / "multiturn_scenarios.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "multiturn_results.json")
    args = parser.parse_args()
    result = run_eval(args.input)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
