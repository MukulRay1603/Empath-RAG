# EmpathRAG — V4 full project audit (2026-05-10)

Replaces `OPUS_FULL_PROJECT_AUDIT_2026_05_06.md` (now 4 days stale; predates V3 listening + V4 plan-and-rephrase + V4.1 fixes + V4.2 route+history+sycophancy+decay+incomplete + V4.3 ablation+injection+baseline + V4.4 polish). Read this first.

Branch: `codex/v2.5-support-navigator`. Last commit: `c8cd802`.

---

## TL;DR

EmpathRAG is a **guarded conversational RAG support navigator for UMD students**. Class project for MSML641, currently in the final-prep phase for class demo + counseling-center pitch. The architecture has seven safety layers; ablation evaluation shows the layers protect different failure modes (Stage-1 lexical precheck is load-bearing on missed-escalation specifically; the others gate drift, sycophancy, scope drift, fabrication). Same-model unguarded-vs-guarded comparison: **9/28 missed escalation vs 0/28** with non-overlapping CIs.

The framing is defensible: prototype, support navigator, not therapy / not diagnosis / not emergency service. The implementation honors that framing across documentation, copy, and architecture.

## Architecture (V4)

```
Student message
     │
     ├──▶ [1] Stage-1 lexical safety precheck      safety_policy.py
     │         crisis / wellbeing / pass · always runs first
     │
     ├──▶ [2] Optional model guardrail (DeBERTa NLI + IG)  guardrail_ig.py
     │         off in live demo for latency (available for eval)
     │
     ├──▶ [3] Hybrid route + tier classifier       ml_router.py + v2_schema.py
     │         Rule baseline + TF-IDF / logistic ML
     │
     ├──▶ [4] Resource registry filter             service_graph.py
     │         34 verified UMD/national service objects
     │         Filtered by route × safety_tier × usage_mode
     │
     ├──▶ [5] Stage-aware response planner         response_planner.py
     │         LISTEN → PERMISSION → OFFER → CLARIFY
     │         F-1 sub-topic engine with decay
     │         Minimal-affirm + incomplete-message handlers
     │         Authority-misconduct route (new in V4.2)
     │         Length-cap clarify (new in V4.3)
     │
     ├──▶ [6] Plan-and-rephrase                    rephraser.py + llm_safety.py
     │         Groq Llama 3.3 70B → Anthropic Haiku → fallback
     │         Conversation history threaded as context (V4.2)
     │         Single-retry transient-error backoff (V4.4)
     │         Token batching (~60% fewer Gradio yields)
     │         Post-rephrase safety verification:
     │           - scope drift  - V1 regression  - sycophancy capitulation
     │           - ungrounded phones  - fabricated resources
     │           - length sanity
     │
     └──▶ [7] Output guard                         output_guard.py
               OFFER-stage gate: missing-action / pure-validation /
               dependency / harmful agreement / ungrounded contact
```

Crisis prompts intercept at [1] and skip [6]+[7] entirely — the LLM is never invoked for crisis content.

## Evaluation summary (commit `c8cd802`)

| Eval | Headline |
|---|---|
| Eval B (rephraser ON, 74 scenarios, 28 escalation) | **0/28 missed**, 0 unsafe, 0 ungrounded, ~570 ms avg latency |
| Per-layer ablation (rephraser ON) | baseline 0/28; **no_stage1_precheck 22/28**; others 0/28 — Stage-1 load-bearing on this metric |
| Unguarded Llama 3.3 70B (same model, no pipeline) | **9/28 missed (32%)**, CI95 [0.148, 0.494], 2 harm-endorsement turns |
| Drift sweep (29 cells × 14 routes × 3 stages) | 27-29/29 clean (1-2 stochastic LLM flags within tolerance) |
| F-1 stage × ISSS contract (12 cells) | **12/12 pass** |
| Sycophancy probes (25 cells) | **25/25 clean** |
| Prompt-injection probes (16 cells × 9 attack categories) | **16/16 clean** |
| Fairness spot-check (18 paired prompts × 8 axes) | **18/18 no divergence** |
| Resource URL audit (63 URLs) | 60 live (3 SAMHSA TLS quirks not real outages) |
| Regression tests | **21/21 pass** |
| Length cap | 1998 chars → normal; 13500 chars → clarify with length-specific template |

All numbers reproducible from this commit via `docs/research/REPRODUCIBILITY.md`.

## What shipped since the previous audit (2026-05-06 → 2026-05-10)

| Push | Date | Headline |
|---|---|---|
| V4 (97ee6bf) | 05-07 | Plan-and-rephrase layer with Groq + Anthropic |
| V4 final (655c300) | 05-08 | Real streaming, support plan, voice, ISSS doc schema, About removed, sweeps shipped |
| V4.1 (9ad0a4c) | 05-08 | Topbar fix + auto-scroll + minimal-affirm repetition + PDF export + counselor-assist note |
| V4.2 part 1 (8fdff5c) | 05-08 | ISSS URL 404 fix + URL audit + Karthik V4 data brief |
| V4.2 part 2 (fea6929) | 05-09 | Authority-misconduct route + conversation history in rephraser |
| V4.2 part 3 (d808a62) | 05-09 | Sycophancy guard + F-1 decay + incomplete-message handler |
| V4.3 (847587d) | 05-10 | Prompt-injection audit + input length cap + bait-and-switch restored + per-layer ablation + unguarded baseline |
| V4.4 Tier 1 (4f20fa7) | 05-10 | Regex hardening + clarify enum + stat caveats + PDF unicode + retry/backoff + reproducibility doc |
| V4.4 Tier 2 (c8cd802) | 05-10 | Safety-pipeline UI viz + paper results section + V4 demo script + mobile CSS + HIPAA + privacy docs |

## Key code locations

- `src/pipeline/core.py` — `EmpathRAGCore.run_turn` / `run_turn_streaming` + `_plan_turn` / `_finalize_turn` + ablation `disable_layers` + session state (6 dicts: tier_history, locked_sessions, session_intl_flag, session_turns_since_intl, session_last_specific_route, session_message_history) + minimal-affirm + incomplete-message handlers
- `src/pipeline/rephraser.py` — `ResponseRephraser` + Groq/Anthropic/Mock/Deterministic providers + streaming SSE + retry/backoff + history threading + `verify_rephrased_safety` integration
- `src/pipeline/llm_safety.py` — post-rephrase trust boundary: scope drift, V1 regression, ungrounded phones, fabricated resources, sycophancy capitulation
- `src/pipeline/response_planner.py` — stage-aware planner (LISTEN/PERMISSION/OFFER/CLARIFY) + 14 route templates + F-1 sub-topic engine + authority-misconduct template + INTERNATIONAL_SOURCE_HINT
- `src/pipeline/v2_schema.py` — 14 routes (incl. AUTHORITY_MISCONDUCT) × 4 tiers + word-boundary-bounded authority detection
- `src/pipeline/safety_policy.py` — Stage-1 lexical precheck (unchanged since V3)
- `src/pipeline/output_guard.py` — OFFER-stage gate
- `src/pipeline/ml_router.py` — TF-IDF + logistic
- `src/pipeline/service_graph.py` — registry loader + matcher (34 entries) + documents field
- `src/pipeline/support_plan.py` — MD + PDF export with DejaVu Unicode font
- `src/pipeline/voice.py` — Groq Whisper turbo wrapper
- `demo/app.py` — Gradio UI: FastDemoPipeline, streaming respond(), safety-pipeline visualization, support-plan toggle, voice toggle, mobile CSS

## Key docs

- `README.md` — V4-current
- `docs/research/PAPER_FRAMING.md` — full paper framing incl. Evaluation Results section (V4 numbers)
- `docs/research/REPRODUCIBILITY.md` — how anyone forks and reproduces
- `docs/research/HIPAA_FERPA_GAP_ANALYSIS.md` — deployment-readiness gaps
- `docs/research/PRIVACY_AND_DATA_FLOW.md` — student/clinician-readable privacy summary
- `docs/research/COUNSELOR_ASSIST_DIRECTION.md` — post-demo deployment vision
- `docs/demo/MSML_CLASS_DEMO_SCRIPT_V4.md` — live demo runbook with 6 scripted scenarios
- `docs/planning/MASTER_CHECKLIST.md` — phase tracker
- `docs/team/karthik/KARTHIK_DATA_REQUEST_V4.md` — V3 data ask
- `docs/team/ISSS_DOCUMENT_VETTING.md` — 6 ISSS URLs to vet by hand

## What is honestly missing

For class demo (~1-3 hours of remaining work, your time):
- Live demo screenshots / video
- 6 ISSS document URLs vetted and pasted into `data/curated/service_graph.jsonl`
- One final visual run-through

For the paper:
- RoBERTa route classifier trained (Phase 2 backlog)
- Real anonymized student turns (Karthik H deliverable, when ready)
- Error analysis section (next doc to write)

For UMD CC pilot:
- CAPS clinician walkthrough
- BAA-signed LLM provider replacement (current Groq usage is non-HIPAA)
- Auth + server-side persistence + encryption
- Custom web frontend (Gradio is the wrong tool for deployment)
- Cultural cross-cutting layered (queer, undocumented, parenting, Black, first-gen)

For production deployment:
- See `HIPAA_FERPA_GAP_ANALYSIS.md` Tier 2 list

## Risk register

| Risk | Severity | Mitigation in place |
|---|---|---|
| Groq API outage during demo | High | Anthropic Haiku 4.5 fallback wired; deterministic-template mode is always available; single-retry transient backoff added in V4.4 |
| Prompt injection during live demo | Medium | 16/16 injection probes clean — planner-as-trust-boundary holds across 9 attack categories |
| ISSS URL 404 during live click | High (was) → Low | Replaced with `https://isss.umd.edu/` (auto-redirects to current canonical); audit script catches future breakage |
| Crisis intercept fails on a live crisis prompt | Critical | Stage-1 lexical precheck; deterministic crisis template never goes through LLM; ablation eval shows S1 catches 22/28 escalations alone |
| Student asks "are you HIPAA-compliant?" | Medium | Explicit gap doc; demo script Q&A covers this |
| Mobile viewport renders weirdly | Low | New 420px + 700px breakpoints added in V4.4; verified compile-clean (visual check pending your eyes) |
| Long-message paste crashes UI | Low | Length cap at 2000 chars with clarify-template fallback (V4.3) |
| Stochastic LLM drift mid-demo | Low | Drift sweep typically 27-29/29; failures are noise-level filler-preamble / ai-tell, not safety-relevant |

## What can still go wrong

The single biggest demo-day risk: a question we haven't anticipated about a sensitive scenario combination we haven't tested. The architecture is layered to fail safely (deterministic template fallback), but a clever adversarial probe a student tries on stage could surface a subtle failure mode. The honest answer is: "noted, that's the kind of failure mode we'd want to add to Eval B; the architecture catches most of these by design but no system is exhaustive."

Second-biggest: the F-1 deportation case is emotionally loaded. If a real F-1 student in the audience experiences distress watching the demo, we should pause, acknowledge, and refer to UMD CC / ISSS contacts in real life — not continue the scripted demo.

## Recent fixes worth highlighting in the grading conversation

- **Per-layer ablation evaluation** with the headline that Stage-1 lexical precheck is load-bearing on missed-escalation specifically; the others protect orthogonal failure modes.
- **Same-model unguarded baseline** (Llama 3.3 70B raw) showing 9/28 missed vs our 0/28 — same model, only the architecture differs.
- **Bait-and-switch limitation restored** to the docs, including in `PAPER_FRAMING` as documented V1 motivation for the redesign.
- **Authority-misconduct route** added with OCRSM + Student Conduct after real-conversation review surfaced the failure mode (counselor allegedly suggesting harm was being routed to academic_setback).
- **Sycophancy guard** with 25/25 resistance under explicit "agree with me" pressure.
- **Prompt-injection sweep** 16/16 across 9 attack categories.
