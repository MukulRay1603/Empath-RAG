"""Validate Karthik's EmpathRAG evaluation dataset delivery.

Run from repo root:
    python eval/validate_eval_delivery.py path/to/empathrag_eval_delivery_v1
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_FILES = {
    "README_eval_notes.md",
    "eval_queries.csv",
    "source_target_map.csv",
    "risky_or_ambiguous_cases.csv",
}

EVAL_QUERY_COLUMNS = [
    "query_id",
    "query_text",
    "scenario_category",
    "risk_category",
    "expected_usage_mode",
    "expected_topics",
    "expected_source_types",
    "expected_source_names",
    "should_intercept",
    "ideal_behavior",
    "notes",
]

SOURCE_TARGET_COLUMNS = [
    "need_id",
    "user_need",
    "preferred_topics",
    "preferred_source_names",
    "avoid_source_names",
    "notes",
]

RISKY_CASE_COLUMNS = [
    "case_id",
    "query_text",
    "why_it_is_tricky",
    "correct_risk_category",
    "should_intercept",
    "expected_handling",
]

SCENARIO_CATEGORIES = {
    "counseling_navigation",
    "after_hours_support",
    "crisis_immediate_help",
    "anxiety_stress",
    "depression_support",
    "academic_burnout",
    "advisor_conflict",
    "graduate_student_support",
    "accessibility_disability",
    "isolation_loneliness",
    "therapy_expectations",
    "help_seeking_script",
    "grounding_or_wellbeing",
    "campus_navigation",
    "out_of_scope",
}

RISK_CATEGORIES = {"normal", "wellbeing", "crisis", "emergency", "ambiguous", "out_of_scope"}
USAGE_MODES = {"retrieval", "wellbeing_only", "crisis_only", "none"}
YES_NO = {"yes", "no"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate EmpathRAG eval delivery.")
    parser.add_argument("delivery_dir", type=Path)
    args = parser.parse_args()

    issues = validate_delivery(args.delivery_dir)
    if issues:
        print(f"Validation failed with {len(issues)} issue(s):")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Validation passed.")
    return 0


def validate_delivery(delivery_dir: Path) -> list[str]:
    issues: list[str] = []
    if not delivery_dir.exists():
        return [f"delivery directory not found: {delivery_dir}"]

    present = {path.name for path in delivery_dir.iterdir() if path.is_file()}
    missing = REQUIRED_FILES - present
    for name in sorted(missing):
        issues.append(f"missing required file: {name}")
    if missing:
        return issues

    eval_rows = _read_csv(delivery_dir / "eval_queries.csv", EVAL_QUERY_COLUMNS, issues)
    source_rows = _read_csv(delivery_dir / "source_target_map.csv", SOURCE_TARGET_COLUMNS, issues)
    risky_rows = _read_csv(delivery_dir / "risky_or_ambiguous_cases.csv", RISKY_CASE_COLUMNS, issues)

    _check_unique(eval_rows, "query_id", issues)
    _check_unique(source_rows, "need_id", issues)
    _check_unique(risky_rows, "case_id", issues)

    if eval_rows and not (50 <= len(eval_rows) <= 70):
        issues.append(f"eval_queries.csv should contain 50-70 rows; found {len(eval_rows)}")
    if source_rows and not (15 <= len(source_rows) <= 25):
        issues.append(f"source_target_map.csv should contain 15-25 rows; found {len(source_rows)}")
    if risky_rows and not (15 <= len(risky_rows) <= 25):
        issues.append(f"risky_or_ambiguous_cases.csv should contain 15-25 rows; found {len(risky_rows)}")

    for row in eval_rows:
        row_id = row["query_id"]
        _check_allowed(row, "scenario_category", SCENARIO_CATEGORIES, row_id, issues)
        _check_allowed(row, "risk_category", RISK_CATEGORIES, row_id, issues)
        _check_allowed(row, "expected_usage_mode", USAGE_MODES, row_id, issues)
        _check_allowed(row, "should_intercept", YES_NO, row_id, issues)
        _check_risk_consistency(row, row_id, issues)
        if not row["query_text"].strip():
            issues.append(f"{row_id}: query_text is empty")
        if not row["ideal_behavior"].strip():
            issues.append(f"{row_id}: ideal_behavior is empty")

    for row in risky_rows:
        row_id = row["case_id"]
        _check_allowed(row, "correct_risk_category", RISK_CATEGORIES, row_id, issues)
        _check_allowed(row, "should_intercept", YES_NO, row_id, issues)

    return issues


def _read_csv(path: Path, expected_columns: list[str], issues: list[str]) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            actual = reader.fieldnames or []
            if actual != expected_columns:
                issues.append(
                    f"{path.name}: columns must be {expected_columns}; found {actual}"
                )
                return []
            return list(reader)
    except Exception as exc:
        issues.append(f"{path.name}: failed to read CSV: {exc}")
        return []


def _check_unique(rows: list[dict[str, str]], field: str, issues: list[str]) -> None:
    seen: set[str] = set()
    for row in rows:
        value = row.get(field, "").strip()
        if not value:
            issues.append(f"{field}: empty ID")
        elif value in seen:
            issues.append(f"{field}: duplicate ID {value}")
        seen.add(value)


def _check_allowed(
    row: dict[str, str],
    field: str,
    allowed: set[str],
    row_id: str,
    issues: list[str],
) -> None:
    value = row.get(field, "").strip()
    if value not in allowed:
        issues.append(f"{row_id}: {field}={value!r} must be one of {sorted(allowed)}")


def _check_risk_consistency(row: dict[str, str], row_id: str, issues: list[str]) -> None:
    risk = row["risk_category"].strip()
    usage = row["expected_usage_mode"].strip()
    intercept = row["should_intercept"].strip()
    if risk in {"crisis", "emergency"} and intercept != "yes":
        issues.append(f"{row_id}: crisis/emergency rows should use should_intercept=yes")
    if risk == "emergency" and usage != "crisis_only":
        issues.append(f"{row_id}: emergency rows should use expected_usage_mode=crisis_only")
    if risk == "normal" and intercept != "no":
        issues.append(f"{row_id}: normal rows should use should_intercept=no")
    if risk == "wellbeing" and usage not in {"wellbeing_only", "retrieval"}:
        issues.append(f"{row_id}: wellbeing rows should use wellbeing_only or retrieval")
    if risk == "out_of_scope" and usage != "none":
        issues.append(f"{row_id}: out_of_scope rows should use expected_usage_mode=none")


if __name__ == "__main__":
    raise SystemExit(main())
