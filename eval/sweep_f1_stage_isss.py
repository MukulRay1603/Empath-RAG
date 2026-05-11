"""F-1 stage-vs-ISSS contract sweep.

Verifies the V3 stage-aware planner rule for international students:

* LISTEN  — international concern is acknowledged generally; ISSS is NOT named,
            and factual F-1 mechanics (RCL, OPT, CPT, SEVIS, I-20, deportation,
            reinstatement) are NOT discussed.
* PERMISSION — ISSS may be soft-named as a resource hint; factual mechanics
            still NOT discussed.
* OFFER   — ISSS is named directly and the relevant factual mechanic for the
            sub-topic (work authorization, visa status, academic standing,
            deportation fear) SHOULD appear.

Each of the 4 F-1 sub-topics is exercised at all three stages with the
rephraser ON. Failures are reported and exit code is non-zero.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except Exception:
    pass

os.environ["EMPATHRAG_REPHRASER_ENABLED"] = "1"

from src.pipeline.core import EmpathRAGCore  # noqa: E402
from src.pipeline.response_planner import classify_intl_topic  # noqa: E402


# Sub-topic prompts crafted in three flavors:
# - vague_listen: F-1 keywords present but NO question mark and NO explicit
#   ask; should land at LISTEN on turn 1.
# - explicit_offer: same concern with an explicit ask; should land at OFFER.
F1_CASES = {
    "work_authorization": {
        "vague_listen": "I've been stressed about my OPT timing for weeks.",
        "explicit_offer": "Can I work off-campus before I finish my OPT paperwork? What should I do?",
    },
    "visa_status": {
        "vague_listen": "My F-1 status feels like it's on shaky ground lately.",
        "explicit_offer": "Will my F-1 status get cancelled if I drop a class? What should I do?",
    },
    "academic_standing": {
        "vague_listen": "I'm on academic probation as an F-1 student and it's been weighing on me.",
        "explicit_offer": "I'm on academic probation and I'm scared about my SEVIS. What should I do?",
    },
    "deportation_fear": {
        "vague_listen": "The fear of being deported has been with me every day.",
        "explicit_offer": "If I fail this semester am I going to get deported? What should I do?",
    },
}

ISSS_TOKENS = ("ISSS", "International Student", "umd_isss")
FACTUAL_TOKENS = (
    "RCL", "reduced course load", "SEVIS", "I-20", "OPT", "CPT",
    "reinstatement", "out of status", "fall out of status",
    "Optional Practical Training", "Curricular Practical Training",
    "prior authorization", "deport",
)


def has_any(text: str, needles: tuple[str, ...]) -> list[str]:
    found = []
    for n in needles:
        if n.lower() in text.lower():
            found.append(n)
    return found


def run_stage(core: EmpathRAGCore, sid: str, prompt: str, turn_index: int) -> dict:
    t0 = time.perf_counter()
    result = core.run_turn(message=prompt, session_id=sid, turn_index=turn_index)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return {
        "prompt": prompt,
        "turn_index": turn_index,
        "actual_stage": result.conversation_stage,
        "actual_route": result.route_label,
        "intl_concern": result.international_concern,
        "intl_topic": result.intl_topic,
        "rephraser_provider": result.rephraser_provider,
        "rephraser_used_llm": result.rephraser_used_llm,
        "response": result.response,
        "isss_tokens": has_any(result.response, ISSS_TOKENS),
        "factual_tokens": has_any(result.response, FACTUAL_TOKENS),
        "latency_ms": round(elapsed, 1),
    }


def evaluate(
    stage: str,
    isss_tokens: list[str],
    factual_tokens: list[str],
    user_message: str,
) -> tuple[bool, list[str]]:
    """Drop factual tokens that the user already used in their message — those
    aren't drift, they're correct mirroring per the system-prompt contract."""
    user_lower = user_message.lower()
    introduced_factual = [t for t in factual_tokens if t.lower() not in user_lower]
    flags: list[str] = []
    if stage == "listen":
        if isss_tokens:
            flags.append(f"isss_named_in_listen:{isss_tokens}")
        if introduced_factual:
            flags.append(f"factual_introduced_in_listen:{introduced_factual}")
    elif stage == "permission":
        if introduced_factual:
            flags.append(f"factual_introduced_in_permission:{introduced_factual}")
    elif stage == "offer":
        if not isss_tokens:
            flags.append("isss_missing_in_offer")
        if not factual_tokens:
            flags.append("no_factual_terms_in_offer")
    return (len(flags) == 0, flags)


def main() -> int:
    from src.pipeline.rephraser import GroqProvider, AnthropicProvider
    g = GroqProvider()
    a = AnthropicProvider()
    print(f"[provider probe] groq={g.available()} anthropic={a.available()}")
    if not g.available() and not a.available():
        print("[fatal] no LLM providers configured.")
        return 2

    rows: list[dict] = []
    fail_count = 0

    for topic, prompts in F1_CASES.items():
        print(f"\n[case] {topic}")
        # Use ONE shared session per topic so session_intl_flag persists across
        # turn 1 (LISTEN) and turn 2 (PERMISSION).
        sid_listen_perm = f"f1_{topic}_lp_{int(time.time()*1000)}"
        listen_row = run_stage(sid=sid_listen_perm, core=CORE, prompt=prompts["vague_listen"], turn_index=1)
        permission_row = run_stage(sid=sid_listen_perm, core=CORE, prompt=prompts["vague_listen"], turn_index=2)
        # Fresh session for OFFER so we test the always-direct path on its own.
        sid_offer = f"f1_{topic}_o_{int(time.time()*1000)}"
        offer_row = run_stage(sid=sid_offer, core=CORE, prompt=prompts["explicit_offer"], turn_index=1)

        for label, row in [("listen", listen_row), ("permission", permission_row), ("offer", offer_row)]:
            row["topic"] = topic
            row["expected_stage"] = label
            ok, flags = evaluate(
                row["actual_stage"], row["isss_tokens"], row["factual_tokens"], row["prompt"]
            )
            row["contract_ok"] = ok
            row["contract_flags"] = flags
            if not ok:
                fail_count += 1
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {label} actual_stage={row['actual_stage']} "
                  f"intl_topic={row['intl_topic']!r} "
                  f"isss={row['isss_tokens'] or '-'} "
                  f"factual={row['factual_tokens'] or '-'} "
                  f"flags={flags or '-'}")
            rows.append(row)

    total = len(rows)
    print(f"\n[summary] {total - fail_count}/{total} contract OK")

    # Markdown report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = ROOT / "eval" / f"sweep_f1_stage_isss_{ts}.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"# F-1 stage × ISSS contract sweep — {ts}\n\n")
        f.write(f"Total cells: {total}. PASS: {total - fail_count}. FAIL: {fail_count}.\n\n")
        for r in rows:
            tag = "PASS" if r["contract_ok"] else "FAIL"
            f.write(f"## [{tag}] {r['topic']} / {r['expected_stage']}\n\n")
            f.write(f"- actual stage: `{r['actual_stage']}` (turn {r['turn_index']})\n")
            f.write(f"- actual route: `{r['actual_route']}`\n")
            f.write(f"- intl_concern: {r['intl_concern']}  intl_topic: `{r['intl_topic']}`\n")
            f.write(f"- provider: `{r['rephraser_provider']}`  used_llm={r['rephraser_used_llm']}\n")
            f.write(f"- ISSS tokens found: {r['isss_tokens'] or '-'}\n")
            f.write(f"- factual tokens found: {r['factual_tokens'] or '-'}\n")
            f.write(f"- contract flags: {r['contract_flags'] or '-'}\n\n")
            f.write(f"**prompt:** {r['prompt']}\n\n")
            f.write("**response:**\n\n```\n" + r["response"] + "\n```\n\n")
            f.write("---\n\n")
    print(f"[report] {report_path}")
    return 0 if fail_count == 0 else 1


CORE = EmpathRAGCore()

if __name__ == "__main__":
    sys.exit(main())
