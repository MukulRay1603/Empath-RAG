"""Clean and import Karthik's V2 curated corpus candidate.

This script keeps Karthik's raw delivery untouched and writes the cleaned local
candidate into data/curated/, which is ignored by git.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.curated_resources import validate_file


DEFAULT_INPUT_DIR = Path("Data_Karthik/v2")
DEFAULT_OUTPUT_DIR = Path("data/curated")

DROP_ROW_IDS = {
    # Broken phone fragment remains in V2 and is redundant with umd_counseling_026.
    "umd_counseling_005",
    # Popup residue cleanup leaves this too short; other 988 rows cover it.
    "988_lifeline_003",
}

UNUSED_INCLUDE_SOURCE_IDS = {
    "src_058",
    "src_066",
    "src_067",
    "src_068",
    "src_069",
    "src_072",
}

POPUP_PATTERNS = (
    r"You are opening a new tab\.",
    r"You are leaving 988lifeline\.org for another website\.",
    r"Their content and privacy policies apply\.",
    r"Would you like to continue'?",
    r"If you reject, you will still be able to access the website and chat service\.",
    r"Learn more",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean Karthik V2 curated corpus.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output = output_dir / "raw_pages"
    shutil.copytree(input_dir / "raw_pages", raw_output, dirs_exist_ok=True)

    rows = _load_rows(input_dir / "resources_seed.jsonl")
    cleaned_rows = []
    dropped = []
    for row in rows:
        if row["id"] in DROP_ROW_IDS:
            dropped.append(row["id"])
            continue
        row = dict(row)
        row["text"] = _clean_text(row["text"])
        row["summary"] = _clean_text(row["summary"])
        if row["id"] in {
            "988_lifeline_009",
            "988_lifeline_021",
            "nimh_new_021",
            "nimh_new_022",
            "jed_new_001",
        }:
            row["notes"] = row["notes"] + " Local V2 import removed popup/link residue."
        cleaned_rows.append(row)

    _write_jsonl(output_dir / "resources_seed.jsonl", cleaned_rows)
    _write_inventory(
        input_dir / "source_inventory.csv",
        output_dir / "source_inventory.csv",
        used_source_ids={row["source_id"] for row in cleaned_rows},
    )
    shutil.copy2(input_dir / "excluded_sources.csv", output_dir / "excluded_sources.csv")
    shutil.copy2(input_dir / "README_corpus_notes.md", output_dir / "README_corpus_notes.md")

    _, issues = validate_file(output_dir / "resources_seed.jsonl", strict=False)
    if issues:
        for issue in issues:
            print(f"issue line {issue.line_no} ({issue.row_id}): {issue.message}")
        raise SystemExit(1)

    print(f"Input rows: {len(rows)}")
    print(f"Output rows: {len(cleaned_rows)}")
    print(f"Dropped rows: {', '.join(dropped) if dropped else 'none'}")
    print(f"Wrote cleaned corpus to: {output_dir}")
    return 0


def _load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_inventory(input_path: Path, output_path: Path, used_source_ids: set[str]) -> None:
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("source_inventory.csv is empty")

    fieldnames = list(rows[0].keys())
    for row in rows:
        source_id = row.get("source_id", "")
        if (
            source_id in UNUSED_INCLUDE_SOURCE_IDS
            and source_id not in used_source_ids
            and row.get("include_status") == "include"
        ):
            row["include_status"] = "partial"
            row["reason"] = "Reviewed source; no chunks included in cleaned local corpus"

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _clean_text(text: str) -> str:
    cleaned = text
    for pattern in POPUP_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bchat at\s+\.", "chat through the source website.", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*[:).,-]+\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


if __name__ == "__main__":
    raise SystemExit(main())
