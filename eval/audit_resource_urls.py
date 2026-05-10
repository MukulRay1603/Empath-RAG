"""Audit every URL surfaced by EmpathRAG and report which are live, redirected,
or broken. Goal: catch 404s before students see them in the demo.

Checks:
* All `source_url` values in data/curated/service_graph.jsonl
* All `url` values in the curated corpus (data/curated/indexes/metadata_curated.db)
  if the index has been built locally.

Writes a Markdown report to eval/audit_resource_urls_<timestamp>.md with one
row per URL: HTTP status, final URL after redirects, error if any.

Usage:
    .\\venv\\Scripts\\python.exe eval\\audit_resource_urls.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE_GRAPH = ROOT / "data" / "curated" / "service_graph.jsonl"
METADATA_DB = ROOT / "data" / "curated" / "indexes" / "metadata_curated.db"

USER_AGENT = (
    "EmpathRAG-URL-Auditor/0.1 (academic prototype; "
    "+https://github.com/MukulRay1603/Empath-RAG)"
)


def head_or_get(url: str, timeout_s: float = 8.0) -> dict:
    """Return {status, final_url, error} for a URL. Tries HEAD then falls back
    to a tiny GET so servers that don't implement HEAD still report cleanly."""
    import urllib.request
    import urllib.error

    if not url or url in {"N/A", "unknown"}:
        return {"status": 0, "final_url": "", "error": "missing_url"}

    def _request(method: str) -> dict:
        req = urllib.request.Request(
            url,
            method=method,
            headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                final = resp.geturl()
                code = getattr(resp, "status", 200)
                return {"status": code, "final_url": final, "error": ""}
        except urllib.error.HTTPError as e:
            return {"status": e.code, "final_url": url, "error": f"http_{e.code}"}
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return {"status": 0, "final_url": url, "error": f"{type(e).__name__}"}

    r = _request("HEAD")
    # Some servers return 405 / 403 for HEAD or block UAs they don't recognize.
    # Try a GET as fallback before declaring the URL broken.
    if r["status"] in (0, 403, 405, 501) or r["status"] >= 500:
        r2 = _request("GET")
        if r2["status"] and 200 <= r2["status"] < 400:
            return r2
        if r["status"] == 0 and r2["status"] != 0:
            return r2
    return r


def load_service_graph_urls() -> list[dict]:
    if not SERVICE_GRAPH.exists():
        return []
    rows: list[dict] = []
    with SERVICE_GRAPH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({
                "kind": "service_graph",
                "service_id": obj.get("service_id", ""),
                "resource_name": obj.get("resource_name", ""),
                "url": obj.get("source_url", ""),
                "last_verified": obj.get("last_verified", ""),
            })
    return rows


def load_corpus_urls() -> list[dict]:
    if not METADATA_DB.exists():
        return []
    conn = sqlite3.connect(METADATA_DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT DISTINCT source_name, url, last_checked FROM chunks WHERE url IS NOT NULL AND url != ''"
        ).fetchall()
    except sqlite3.OperationalError as exc:
        conn.close()
        return [{
            "kind": "corpus",
            "service_id": "",
            "resource_name": "(corpus DB schema mismatch)",
            "url": "",
            "last_verified": str(exc),
        }]
    conn.close()
    return [
        {
            "kind": "corpus",
            "service_id": "",
            "resource_name": r["source_name"] or "",
            "url": r["url"] or "",
            "last_verified": r["last_checked"] or "",
        }
        for r in rows
    ]


def dedupe(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        u = r.get("url", "")
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(r)
    return out


def main() -> int:
    rows = dedupe(load_service_graph_urls() + load_corpus_urls())
    print(f"[audit] {len(rows)} unique URLs to check\n")

    results: list[dict] = []
    for i, row in enumerate(rows, 1):
        url = row["url"]
        r = head_or_get(url)
        result = {**row, **r}
        results.append(result)
        status = r["status"] or "?"
        flag = "ok " if (isinstance(status, int) and 200 <= status < 400) else "FAIL"
        # Mark obvious redirects (final_url differs from start)
        redir = " (redirect)" if r["final_url"] and r["final_url"] != url and 200 <= (status if isinstance(status, int) else 0) < 400 else ""
        print(f"  [{flag}] {status:>3} {url[:90]}{redir}")
        if r["error"]:
            print(f"        error: {r['error']}")
        time.sleep(0.15)  # be polite to servers

    broken = [r for r in results if not (isinstance(r["status"], int) and 200 <= r["status"] < 400)]
    print(f"\n[summary] {len(results) - len(broken)}/{len(results)} live · {len(broken)} broken or unreachable\n")
    if broken:
        print("Broken URLs:")
        for b in broken:
            print(f"  - [{b['kind']}] {b['resource_name'] or b['service_id']} -> {b['url']} ({b['error'] or b['status']})")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = ROOT / "eval" / f"audit_resource_urls_{ts}.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"# Resource URL audit — {ts}\n\n")
        f.write(f"Total unique URLs: {len(results)}. Live: {len(results) - len(broken)}. Broken: {len(broken)}.\n\n")
        if broken:
            f.write("## Broken / unreachable\n\n")
            f.write("| Source | Service ID | URL | Status | Error |\n")
            f.write("|---|---|---|---|---|\n")
            for b in broken:
                f.write(f"| {b['resource_name']} | `{b['service_id']}` | <{b['url']}> | {b['status']} | {b['error']} |\n")
            f.write("\n")
        f.write("## All URLs\n\n")
        f.write("| Status | Source | URL | Last verified |\n")
        f.write("|---|---|---|---|\n")
        for r in sorted(results, key=lambda x: (isinstance(x["status"], int) and not (200 <= x["status"] < 400), x["resource_name"] or "")):
            f.write(f"| {r['status']} | {r['resource_name'] or r['service_id']} | <{r['url']}> | {r['last_verified']} |\n")
    print(f"\n[report] {report_path}")
    return 0 if not broken else 1


if __name__ == "__main__":
    sys.exit(main())
