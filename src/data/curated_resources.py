"""
Utilities for EmpathRAG curated resource corpora.

The curated corpus is a JSONL file prepared from official/student-support
resources. It intentionally stays separate from the Reddit research corpus.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_FIELDS = (
    "id",
    "source_id",
    "source_name",
    "source_type",
    "title",
    "url",
    "topic",
    "audience",
    "risk_level",
    "usage_mode",
    "text",
    "summary",
    "last_checked",
    "notes",
)

SOURCE_TYPES = {
    "university_resource",
    "crisis_resource",
    "government_public_health",
    "student_support",
    "clinician_review_candidate",
}

TOPICS = {
    "crisis_immediate_help",
    "counseling_services",
    "after_hours_support",
    "academic_burnout",
    "advisor_conflict",
    "isolation_loneliness",
    "anxiety_stress",
    "depression_support",
    "accessibility_disability",
    "graduate_student_support",
    "help_seeking_script",
    "grounding_exercise",
    "campus_navigation",
    "therapy_expectations",
    "peer_support",
    "emergency_services",
}

AUDIENCES = {
    "umd_student",
    "graduate_student",
    "student_general",
    "crisis_support",
    "supporter_or_friend",
}

RISK_LEVELS = {"safe", "wellbeing", "crisis_resource", "exclude"}
USAGE_MODES = {"retrieval", "wellbeing_only", "crisis_only", "metadata_only"}


@dataclass(frozen=True)
class ValidationIssue:
    line_no: int
    row_id: str
    message: str


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    path = Path(path)
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_no} must be a JSON object.")
        row["_line_no"] = line_no
        rows.append(row)
    return rows


def validate_rows(rows: Iterable[dict]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()

    for row in rows:
        line_no = int(row.get("_line_no", 0))
        row_id = str(row.get("id", "")).strip()

        for field in REQUIRED_FIELDS:
            if not str(row.get(field, "")).strip():
                issues.append(ValidationIssue(line_no, row_id, f"missing field: {field}"))

        if row_id in seen_ids:
            issues.append(ValidationIssue(line_no, row_id, "duplicate id"))
        if row_id:
            seen_ids.add(row_id)

        _check_allowed(issues, row, line_no, row_id, "source_type", SOURCE_TYPES)
        _check_allowed(issues, row, line_no, row_id, "topic", TOPICS)
        _check_allowed(issues, row, line_no, row_id, "audience", AUDIENCES)
        _check_allowed(issues, row, line_no, row_id, "risk_level", RISK_LEVELS)
        _check_allowed(issues, row, line_no, row_id, "usage_mode", USAGE_MODES)

        text = str(row.get("text", "")).strip()
        word_count = len(text.split())
        if text and not (40 <= word_count <= 300):
            issues.append(
                ValidationIssue(
                    line_no,
                    row_id,
                    f"text length {word_count} words outside review band 40-300",
                )
            )
        if row.get("risk_level") == "exclude" and row.get("usage_mode") != "metadata_only":
            issues.append(
                ValidationIssue(
                    line_no,
                    row_id,
                    "exclude rows must use usage_mode=metadata_only or be removed",
                )
            )

    return issues


def ingestion_rows(rows: Iterable[dict]) -> list[dict]:
    """Rows safe to embed into the curated retrieval index."""
    usable = []
    for row in rows:
        if row.get("risk_level") == "exclude":
            continue
        if row.get("usage_mode") == "metadata_only":
            continue
        usable.append({k: v for k, v in row.items() if not k.startswith("_")})
    return usable


def validate_file(path: str | Path, strict: bool = True) -> tuple[list[dict], list[ValidationIssue]]:
    rows = load_jsonl(path)
    issues = validate_rows(rows)
    if strict and issues:
        messages = "\n".join(
            f"line {i.line_no} ({i.row_id or 'no id'}): {i.message}" for i in issues
        )
        raise ValueError(f"Curated corpus validation failed:\n{messages}")
    return rows, issues


def _check_allowed(
    issues: list[ValidationIssue],
    row: dict,
    line_no: int,
    row_id: str,
    field: str,
    allowed: set[str],
) -> None:
    value = row.get(field)
    if value and value not in allowed:
        issues.append(
            ValidationIssue(
                line_no,
                row_id,
                f"{field}={value!r} is not one of {sorted(allowed)}",
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate EmpathRAG curated JSONL corpus.")
    parser.add_argument("path", help="Path to resources_seed.jsonl")
    parser.add_argument("--non-strict", action="store_true", help="Print issues but exit 0.")
    args = parser.parse_args()

    rows, issues = validate_file(args.path, strict=False)
    usable = ingestion_rows(rows)
    print(f"Rows: {len(rows)}")
    print(f"Usable retrieval rows: {len(usable)}")

    if issues:
        print(f"Issues: {len(issues)}")
        for issue in issues:
            print(f"- line {issue.line_no} ({issue.row_id or 'no id'}): {issue.message}")
        return 0 if args.non_strict else 1

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
