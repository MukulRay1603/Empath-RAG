"""Prepare Karthik's eval delivery into EmpathRAG Core supervised labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.v2_schema import SafetyTier, SupportRoute, classify_route  # noqa: E402


DEFAULT_DELIVERY = ROOT / "Data_Karthik" / "empathrag_eval_delivery_v1"
DEFAULT_OUTPUT = ROOT / "eval" / "empathrag_core_supervised.csv"

SCENARIO_TO_ROUTE = {
    "academic_burnout": SupportRoute.EXAM_STRESS.value,
    "accessibility_disability": SupportRoute.ACCESSIBILITY_ADS.value,
    "advisor_conflict": SupportRoute.ADVISOR_CONFLICT.value,
    "after_hours_support": SupportRoute.COUNSELING_NAVIGATION.value,
    "anxiety_stress": SupportRoute.ANXIETY_PANIC.value,
    "campus_navigation": SupportRoute.GENERAL_STUDENT_SUPPORT.value,
    "counseling_navigation": SupportRoute.COUNSELING_NAVIGATION.value,
    "crisis_immediate_help": SupportRoute.CRISIS_IMMEDIATE.value,
    "depression_support": SupportRoute.LOW_MOOD.value,
    "graduate_student_support": SupportRoute.GENERAL_STUDENT_SUPPORT.value,
    "grounding_or_wellbeing": SupportRoute.ANXIETY_PANIC.value,
    "help_seeking_script": SupportRoute.GENERAL_STUDENT_SUPPORT.value,
    "isolation_loneliness": SupportRoute.LONELINESS_ISOLATION.value,
    "out_of_scope": SupportRoute.OUT_OF_SCOPE.value,
    "therapy_expectations": SupportRoute.COUNSELING_NAVIGATION.value,
}

RISK_TO_TIER = {
    "emergency": SafetyTier.IMMINENT_SAFETY.value,
    "crisis": SafetyTier.IMMINENT_SAFETY.value,
    "ambiguous": SafetyTier.HIGH_DISTRESS.value,
    "wellbeing": SafetyTier.WELLBEING.value,
    "normal": SafetyTier.SUPPORT_NAVIGATION.value,
    "out_of_scope": SafetyTier.SUPPORT_NAVIGATION.value,
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def prepare(delivery_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for row in read_csv(delivery_dir / "eval_queries.csv"):
        risk = row["risk_category"].strip()
        tier = RISK_TO_TIER.get(risk, SafetyTier.SUPPORT_NAVIGATION.value)
        route = SCENARIO_TO_ROUTE.get(row["scenario_category"].strip())
        if not route:
            route = classify_route(row["query_text"], SafetyTier(tier)).route.value
        rows.append(
            {
                "query_id": row["query_id"],
                "query_text": row["query_text"],
                "audience_mode": "helping_friend" if "friend" in row["query_text"].lower() else "student",
                "route_label": route,
                "safety_tier": tier,
                "should_intercept": row["should_intercept"],
                "expected_usage_modes": row["expected_usage_mode"],
                "preferred_source_names": row["expected_source_names"],
                "avoid_source_names": "",
                "preferred_topics": row["expected_topics"],
                "expected_response_action": row["ideal_behavior"],
                "tricky_flags": "",
                "split": _split_for_id(row["query_id"]),
                "notes": row.get("notes", ""),
            }
        )

    for row in read_csv(delivery_dir / "risky_or_ambiguous_cases.csv"):
        risk = row["correct_risk_category"].strip()
        tier = RISK_TO_TIER.get(risk, SafetyTier.HIGH_DISTRESS.value)
        route = SupportRoute.PEER_HELPER.value if any(
            token in row["query_text"].lower() for token in ("friend", "roommate", "sibling")
        ) else classify_route(row["query_text"], SafetyTier(tier)).route.value
        if row["should_intercept"].strip().lower() == "yes":
            tier = SafetyTier.IMMINENT_SAFETY.value
            if route != SupportRoute.PEER_HELPER.value:
                route = SupportRoute.CRISIS_IMMEDIATE.value
        rows.append(
            {
                "query_id": row["case_id"],
                "query_text": row["query_text"],
                "audience_mode": "helping_friend" if route == SupportRoute.PEER_HELPER.value else "student",
                "route_label": route,
                "safety_tier": tier,
                "should_intercept": row["should_intercept"],
                "expected_usage_modes": "crisis_only" if row["should_intercept"].strip().lower() == "yes" else "retrieval",
                "preferred_source_names": "",
                "avoid_source_names": "",
                "preferred_topics": "",
                "expected_response_action": row["expected_handling"],
                "tricky_flags": row["why_it_is_tricky"],
                "split": _split_for_id(row["case_id"]),
                "notes": "risky_or_ambiguous_cases",
            }
        )
    return rows


def _split_for_id(identifier: str) -> str:
    digits = "".join(ch for ch in identifier if ch.isdigit())
    value = int(digits or "0")
    if value % 10 in {0, 1}:
        return "test"
    if value % 10 == 2:
        return "dev"
    return "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery-dir", type=Path, default=DEFAULT_DELIVERY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = prepare(args.delivery_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
