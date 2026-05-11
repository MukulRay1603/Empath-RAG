"""Validate and ingest Karthik's EmpathRAG Core dataset delivery.

This script is intentionally conservative: it validates the delivery folder,
copies the single-turn labels into the supervised router dataset, copies the
multi-turn scenarios into the eval harness, and writes a small report. Resource
profile additions are validated and reported, but not auto-merged into the
runtime registry because contact details need a human provenance review.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.v2_schema import SafetyTier, SupportRoute  # noqa: E402


DEFAULT_DELIVERY = ROOT / "Data_Karthik" / "empathrag_core_dataset_v2"
DEFAULT_SINGLE_OUTPUT = ROOT / "eval" / "empathrag_core_supervised.csv"
DEFAULT_MULTITURN_OUTPUT = ROOT / "eval" / "multiturn_scenarios.jsonl"
DEFAULT_REPORT_JSON = ROOT / "eval" / "core_dataset_v2_ingest_report.json"
DEFAULT_REPORT_MD = ROOT / "eval" / "core_dataset_v2_ingest_report.md"

REQUIRED_FILES = {
    "README_dataset_notes.md",
    "single_turn_labeled.csv",
    "multi_turn_scenarios.jsonl",
    "source_target_map.csv",
    "risky_ambiguous_cases.csv",
    "resource_profile_additions.csv",
}

SINGLE_TURN_COLUMNS = [
    "query_id",
    "query_text",
    "audience_mode",
    "route_label",
    "safety_tier",
    "should_intercept",
    "expected_usage_modes",
    "preferred_source_names",
    "avoid_source_names",
    "preferred_topics",
    "expected_response_action",
    "tricky_flags",
    "split",
    "notes",
]

RESOURCE_ADDITION_COLUMNS = [
    "resource_name",
    "resource_type",
    "official_url",
    "source_authority",
    "route_labels",
    "safety_tiers",
    "usage_modes",
    "audience",
    "contact_mode",
    "contact_value",
    "hours",
    "location",
    "confidentiality_status",
    "last_verified",
    "notes",
]

VALID_ROUTES = {item.value for item in SupportRoute}
VALID_TIERS = {item.value for item in SafetyTier}
VALID_AUDIENCE = {"student", "helping_friend"}
VALID_SPLITS = {"train", "dev", "test"}
VALID_USAGE_MODES = {"retrieval", "wellbeing_only", "crisis_only"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_bool(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "1", "y"}:
        return "true"
    if normalized in {"false", "no", "0", "n"}:
        return "false"
    return normalized


def split_tokens(value: str) -> list[str]:
    return [token.strip() for token in str(value).replace(";", ",").split(",") if token.strip()]


def validate_delivery(delivery_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "delivery_dir": str(delivery_dir),
        "status": "pass",
        "errors": [],
        "warnings": [],
        "counts": {},
        "label_distribution": {"route_label": {}, "safety_tier": {}, "split": {}},
        "next_steps": [],
    }

    if not delivery_dir.exists():
        report["errors"].append(f"Delivery directory not found: {delivery_dir}")
        report["status"] = "fail"
        return report

    missing = sorted(name for name in REQUIRED_FILES if not (delivery_dir / name).exists())
    if missing:
        report["errors"].append(f"Missing required files: {', '.join(missing)}")

    single_path = delivery_dir / "single_turn_labeled.csv"
    if single_path.exists():
        rows = read_csv(single_path)
        report["counts"]["single_turn_rows"] = len(rows)
        _validate_single_turn_rows(rows, report)

    multiturn_path = delivery_dir / "multi_turn_scenarios.jsonl"
    if multiturn_path.exists():
        scenarios = _read_jsonl(multiturn_path, report)
        report["counts"]["multi_turn_scenarios"] = len(scenarios)
        _validate_multiturn_scenarios(scenarios, report)

    risky_path = delivery_dir / "risky_ambiguous_cases.csv"
    if risky_path.exists():
        report["counts"]["risky_ambiguous_rows"] = len(read_csv(risky_path))

    resource_path = delivery_dir / "resource_profile_additions.csv"
    if resource_path.exists():
        additions = read_csv(resource_path)
        report["counts"]["resource_profile_additions"] = len(additions)
        _validate_resource_additions(additions, report)

    if report["errors"]:
        report["status"] = "fail"
    elif report["warnings"]:
        report["status"] = "pass_with_warnings"

    report["next_steps"] = _next_steps(report)
    return report


def _validate_single_turn_rows(rows: list[dict[str, str]], report: dict[str, Any]) -> None:
    if not rows:
        report["errors"].append("single_turn_labeled.csv has no rows.")
        return

    columns = set(rows[0].keys())
    missing_columns = [col for col in SINGLE_TURN_COLUMNS if col not in columns]
    if missing_columns:
        report["errors"].append(f"single_turn_labeled.csv missing columns: {', '.join(missing_columns)}")

    seen_ids: set[str] = set()
    for idx, row in enumerate(rows, start=2):
        query_id = row.get("query_id", "").strip()
        if not query_id:
            report["errors"].append(f"Row {idx}: query_id is blank.")
        elif query_id in seen_ids:
            report["errors"].append(f"Row {idx}: duplicate query_id {query_id}.")
        seen_ids.add(query_id)

        if not row.get("query_text", "").strip():
            report["errors"].append(f"Row {idx}: query_text is blank.")

        _validate_choice(row, "audience_mode", VALID_AUDIENCE, idx, report)
        _validate_choice(row, "route_label", VALID_ROUTES, idx, report)
        _validate_choice(row, "safety_tier", VALID_TIERS, idx, report)
        _validate_choice(row, "split", VALID_SPLITS, idx, report)

        should_intercept = normalize_bool(row.get("should_intercept", ""))
        if should_intercept not in {"true", "false"}:
            report["errors"].append(f"Row {idx}: should_intercept must be true/false/yes/no.")
        row["should_intercept"] = should_intercept

        for usage_mode in split_tokens(row.get("expected_usage_modes", "")):
            if usage_mode not in VALID_USAGE_MODES:
                report["warnings"].append(f"Row {idx}: unknown expected_usage_modes value '{usage_mode}'.")

        if row.get("safety_tier") == SafetyTier.IMMINENT_SAFETY.value and should_intercept != "true":
            report["warnings"].append(f"Row {idx}: imminent_safety row is not marked should_intercept=true.")
        if should_intercept == "true" and row.get("safety_tier") != SafetyTier.IMMINENT_SAFETY.value:
            report["warnings"].append(f"Row {idx}: intercept row should usually use imminent_safety tier.")

        for field in ("route_label", "safety_tier", "split"):
            value = row.get(field, "").strip()
            bucket = report["label_distribution"][field]
            bucket[value] = bucket.get(value, 0) + 1


def _validate_multiturn_scenarios(scenarios: list[dict[str, Any]], report: dict[str, Any]) -> None:
    case_ids: set[str] = set()
    for idx, scenario in enumerate(scenarios, start=1):
        case_id = str(scenario.get("case_id", "")).strip()
        if not case_id:
            report["errors"].append(f"Scenario line {idx}: case_id is blank.")
        elif case_id in case_ids:
            report["errors"].append(f"Scenario line {idx}: duplicate case_id {case_id}.")
        case_ids.add(case_id)

        audience_mode = str(scenario.get("audience_mode", "student")).strip()
        if audience_mode not in VALID_AUDIENCE:
            report["errors"].append(f"Scenario {case_id or idx}: invalid audience_mode '{audience_mode}'.")

        turns = scenario.get("turns")
        if not isinstance(turns, list) or not turns:
            report["errors"].append(f"Scenario {case_id or idx}: turns must be a non-empty list.")
            continue
        if len(turns) < 3:
            report["warnings"].append(f"Scenario {case_id or idx}: fewer than 3 turns limits trajectory evaluation.")
        for turn_idx, turn in enumerate(turns, start=1):
            prefix = f"Scenario {case_id or idx} turn {turn_idx}"
            if not str(turn.get("user", "")).strip():
                report["errors"].append(f"{prefix}: user text is blank.")
            route = str(turn.get("expected_route", "")).strip()
            tier = str(turn.get("expected_safety_tier", "")).strip()
            if route not in VALID_ROUTES:
                report["errors"].append(f"{prefix}: invalid expected_route '{route}'.")
            if tier not in VALID_TIERS:
                report["errors"].append(f"{prefix}: invalid expected_safety_tier '{tier}'.")
            intercept = normalize_bool(str(turn.get("should_intercept", "")))
            if intercept not in {"true", "false"}:
                report["errors"].append(f"{prefix}: should_intercept must be true/false.")


def _validate_resource_additions(rows: list[dict[str, str]], report: dict[str, Any]) -> None:
    if not rows:
        report["warnings"].append("resource_profile_additions.csv is empty.")
        return
    columns = set(rows[0].keys())
    missing_columns = [col for col in RESOURCE_ADDITION_COLUMNS if col not in columns]
    if missing_columns:
        report["errors"].append(f"resource_profile_additions.csv missing columns: {', '.join(missing_columns)}")

    for idx, row in enumerate(rows, start=2):
        if not row.get("resource_name", "").strip():
            report["errors"].append(f"Resource row {idx}: resource_name is blank.")
        official_url = row.get("official_url", "").strip()
        if not official_url.startswith(("https://", "http://", "internal://")):
            report["errors"].append(f"Resource row {idx}: official_url must be a URL or internal:// provenance.")
        for route in split_tokens(row.get("route_labels", "")):
            if route not in VALID_ROUTES:
                report["warnings"].append(f"Resource row {idx}: unknown route_labels value '{route}'.")
        for tier in split_tokens(row.get("safety_tiers", "")):
            if tier not in VALID_TIERS:
                report["warnings"].append(f"Resource row {idx}: unknown safety_tiers value '{tier}'.")
        for field in ("contact_value", "hours", "location"):
            if not row.get(field, "").strip():
                report["warnings"].append(f"Resource row {idx}: {field} is blank; use 'unknown' if not verified.")


def _validate_choice(
    row: dict[str, str],
    field: str,
    valid_values: set[str],
    row_number: int,
    report: dict[str, Any],
) -> None:
    value = row.get(field, "").strip()
    if value not in valid_values:
        report["errors"].append(f"Row {row_number}: invalid {field} '{value}'.")


def _read_jsonl(path: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                report["errors"].append(f"{path.name} line {line_no}: invalid JSON: {exc}")
                continue
            if not isinstance(value, dict):
                report["errors"].append(f"{path.name} line {line_no}: expected JSON object.")
                continue
            rows.append(value)
    return rows


def _next_steps(report: dict[str, Any]) -> list[str]:
    if report["status"] == "fail":
        return [
            "Send the report errors back to Karthik before training.",
            "Do not overwrite eval/empathrag_core_supervised.csv with a failed delivery.",
        ]
    return [
        "Run eval/train_ml_router.py to train local TF-IDF route and tier classifiers.",
        "Run eval/run_router_eval.py for Eval A single-turn router metrics.",
        "Run eval/run_multiturn_eval.py for Eval B trajectory metrics.",
        "Manually review resource_profile_additions.csv before merging new services into data/curated/service_graph.jsonl.",
    ]


def write_report(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# EmpathRAG Core Dataset V2 Ingest Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Delivery directory: `{report['delivery_dir']}`",
        "",
        "## Counts",
    ]
    for key, value in report.get("counts", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Label Distribution"])
    for field, buckets in report.get("label_distribution", {}).items():
        lines.append(f"### {field}")
        if buckets:
            for key, value in sorted(buckets.items()):
                lines.append(f"- `{key}`: {value}")
        else:
            lines.append("- No rows counted.")
    lines.extend(["", "## Errors"])
    lines.extend([f"- {item}" for item in report["errors"]] or ["- None"])
    lines.extend(["", "## Warnings"])
    lines.extend([f"- {item}" for item in report["warnings"]] or ["- None"])
    lines.extend(["", "## Next Steps"])
    lines.extend([f"- {item}" for item in report["next_steps"]])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ingest(args: argparse.Namespace) -> dict[str, Any]:
    report = validate_delivery(args.delivery_dir)
    write_report(report, args.report_json, args.report_md)
    if report["status"] == "fail":
        raise SystemExit(f"Dataset delivery failed validation. See {args.report_md}")

    single_rows = read_csv(args.delivery_dir / "single_turn_labeled.csv")
    for row in single_rows:
        row["should_intercept"] = normalize_bool(row.get("should_intercept", ""))
    write_csv(args.output, single_rows, SINGLE_TURN_COLUMNS)

    if not args.skip_multiturn_copy:
        args.multiturn_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.delivery_dir / "multi_turn_scenarios.jsonl", args.multiturn_output)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-dir", type=Path, default=DEFAULT_DELIVERY)
    parser.add_argument("--output", type=Path, default=DEFAULT_SINGLE_OUTPUT)
    parser.add_argument("--multiturn-output", type=Path, default=DEFAULT_MULTITURN_OUTPUT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--skip-multiturn-copy", action="store_true")
    args = parser.parse_args()

    report = ingest(args)
    print(json.dumps({"status": report["status"], "counts": report["counts"]}, indent=2))
    print(f"Wrote supervised labels to {args.output}")
    if not args.skip_multiturn_copy:
        print(f"Wrote multi-turn scenarios to {args.multiturn_output}")
    print(f"Wrote ingest report to {args.report_md}")


if __name__ == "__main__":
    main()
