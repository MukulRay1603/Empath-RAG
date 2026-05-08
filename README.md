# EmpathRAG

<div align="center">

**A guarded conversational RAG support navigator for UMD students**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![University](https://img.shields.io/badge/UMD-MSML641-E03A3E?style=flat-square)](https://umd.edu)

*MSML641 · Applied Machine Learning · University of Maryland*

</div>

---

EmpathRAG is a **support-navigation prototype** that helps a UMD student name what kind of support they need, retrieves grounded campus resources, gives one practical next step, and escalates safety risk to human support.

It is **not** therapy, diagnosis, counseling, an emergency service, or a clinical product. It does not replace the UMD Counseling Center, ISSS, ADS, the Graduate School Ombuds, or any clinical care.

> EmpathRAG started as emotion-aware open RAG (V1). Evaluation exposed the structural risks of open empathetic generation: missed escalation, sycophantic validation, dependency reinforcement, ungrounded advice, and weak distinction between ordinary stress and safety risk. **EmpathRAG Core (V2.5) is the guarded redesign**; V4 adds a controlled-paraphrasing layer that gives the system natural warmth without surrendering the safety floor.

---

## What the system does

1. **Listens first.** Reflects what the student said back, in their own words, before suggesting any resource.
2. **Surfaces specific UMD resources only when the conversation calls for them** — never in the first message of a low-risk talk.
3. **Routes 14 student-support topics** (academic setback, exam stress, advisor conflict, accessibility/ADS, basic needs, peer-helper, F-1 visa worry, counseling navigation, anxiety, low mood, loneliness, CARE/violence, general support, out-of-scope) to a small registry of verified UMD and national resources.
4. **For F-1 students**, separates emotional support from immigration questions and routes the latter to ISSS — preserving F-1 mechanics (RCL, SEVIS, OPT, CPT, I-20, reinstatement) verbatim from authoritative templates.
5. **For crisis prompts**, intercepts before generation and redirects to 988 / UMD Counseling Center / emergency support. Crisis prompts never flow through the language model.

## What the system does not do

- It does not diagnose anxiety, depression, PTSD, or any condition.
- It does not prescribe medication or treatment.
- It does not provide clinical judgment.
- It does not promise unconditional availability ("I'm always here") or undermine the student's existing support relationships.
- It does not store conversations server-side beyond what the student explicitly downloads in the Support Plan export.

---

## Architecture

```
Student message
     │
     ├──▶ [1] Stage-1 lexical safety precheck       safety_policy.py
     │         crisis / wellbeing / pass · always runs first
     │
     ├──▶ [2] Optional model guardrail (DeBERTa NLI + IG)   guardrail_ig.py
     │         off by default in the live demo for latency
     │
     ├──▶ [3] Hybrid route + tier classifier        ml_router.py + v2_schema.py
     │         Rule baseline + TF-IDF / logistic ML
     │
     ├──▶ [4] Resource registry filter              service_graph.py
     │         Verified UMD/national service objects
     │         Filtered by route × safety tier × usage_mode
     │
     ├──▶ [5] Stage-aware response planner          response_planner.py
     │         LISTEN → PERMISSION → OFFER
     │         F-1 sub-topic engine + session memory
     │
     ├──▶ [6] Plan-and-rephrase                     rephraser.py + llm_safety.py
     │         Groq Llama 3.3 70B → Anthropic Claude Haiku → fallback
     │         Deterministic planner authors WHAT to say;
     │         the LLM only paraphrases under a strict contract.
     │         Post-rephrase verification (scope drift, V1 regression,
     │         ungrounded phones, fabricated resources).
     │
     └──▶ [7] Output guard                          output_guard.py
               Catches missing-action / pure-validation /
               dependency / harmful agreement at OFFER stage.
```

Crisis prompts bypass step 6 entirely — the deterministic crisis template is rendered, then served. The LLM never sees a crisis-tier message.

### Plan-and-rephrase: controlled paraphrasing

The deterministic planner is the trust boundary. The LLM only paraphrases planner-authored text under a strict system prompt contract:

- Match the input's length closely.
- Mirror specific user phrases (events, fears, named people, time anchors) over abstractions.
- Never introduce a UMD resource by name unless the input already names it.
- Never reframe the user's emotion or use clinical labels.
- Never promise unbounded availability or use AI-tells.
- Never minimize stated fears with pre-text like "don't worry" before a contradiction.

After paraphrasing, `verify_rephrased_safety` checks scope drift, V1 regression, ungrounded phone numbers, fabricated resource names, and length sanity. Failures fall through to the next provider, then to the deterministic template — never silently to a degraded path.

See [`docs/research/PAPER_FRAMING.md`](docs/research/PAPER_FRAMING.md) (Phase 8: Controlled Paraphrasing) for the full architectural argument.

---

## Evaluation

| Eval | Scenarios | Metric | Result |
|---|---|---|---|
| **Eval A** — single-turn ablation | 360 prompts × 14 routes × 4 tiers | Hybrid rule + ML route accuracy | **0.86** |
| | | Intercept accuracy | **0.99** |
| | | Source organization hit rate | **1.00** |
| **Eval B** — multi-turn safety | 74 scenarios (50 Karthik + 24 supplement), 28 escalation | Missed escalation rate | **0/28** |
| | | Unsafe generation count | **0** |
| | | Ungrounded action count | **0** |
| | | (rephraser ON, hard gate) | identical to deterministic baseline |
| **Drift sweep** | 29 cells × 14 routes × 3 stages, rephraser ON | Cells with drift flags | **0** |
| | | Mean rephrased-to-template length ratio | 0.97 |
| **F-1 stage × ISSS contract** | 12 cells × 4 sub-topics × 3 stages | Contract violations | **0/12** |
| | | ISSS at LISTEN | never named |
| | | F-1 mechanics at LISTEN/PERMISSION | not introduced |
| **Fairness spot-check** | 18 paired prompts × 8 demographic axes | Routing divergences | **0** |

> These numbers are on synthetic development data. They are useful for the prototype framing — not clinical or deployment claims. A real fairness audit needs orders of magnitude more data and domain-expert test generation.

Reproduce locally:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\venv\Scripts\python.exe eval\run_empathrag_core_eval.py        # Eval A
.\venv\Scripts\python.exe eval\run_multiturn_eval.py             # Eval B
.\venv\Scripts\python.exe eval\sweep_rephraser_drift.py          # 29-cell drift sweep
.\venv\Scripts\python.exe eval\sweep_f1_stage_isss.py            # F-1 contract
.\venv\Scripts\python.exe eval\sweep_fairness_spot_check.py      # demographic perturbation
```

The sweep scripts emit timestamped Markdown reports under `eval/` (gitignored).

---

## Demo

Local demo (Gradio):

```powershell
$env:EMPATHRAG_DEMO_BACKEND='fast'
$env:EMPATHRAG_RETRIEVAL_CORPUS='curated_support'
$env:EMPATHRAG_TOP_K='5'
$env:EMPATHRAG_REPHRASER_ENABLED='1'      # set to '0' for deterministic-only
$env:PYTHONIOENCODING='utf-8'
.\venv\Scripts\python.exe -u demo\app.py
```

Open **http://127.0.0.1:7860/**. The demo includes:

- Chat surface with stage-aware planner + LLM streaming
- Inspect drawer (Support card · Diagnostics)
- Mode bar to flip between deterministic templates and LLM-rephrased Groq generation (live A/B for inspection)
- Provisional voice input (Groq Whisper turbo) — toggle reveals an inline recorder; transcript drops into the composer for review before send
- Support Plan / clinician-handoff export — Markdown download summarizing the conversation with route, recommended action, and resources mentioned

API keys (place in `.env` at repo root):

```
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-api03-...     # optional fallback
```

Without keys the system runs in deterministic-template mode (no rephrasing, no voice).

---

## Repo structure

```
Empath-RAG/
├── src/pipeline/
│   ├── core.py                # EmpathRAGCore.run_turn / run_turn_streaming
│   ├── rephraser.py           # Plan-and-rephrase orchestrator + Groq/Anthropic providers
│   ├── llm_safety.py          # Post-rephrase verification (trust boundary)
│   ├── response_planner.py    # Stage-aware planner (LISTEN/PERMISSION/OFFER) + F-1 logic
│   ├── safety_policy.py       # Stage-1 lexical safety precheck
│   ├── output_guard.py        # OFFER-stage groundedness + anti-sycophancy
│   ├── ml_router.py           # TF-IDF + logistic route/tier classifier
│   ├── service_graph.py       # Resource registry loader + matcher
│   ├── support_plan.py        # Markdown clinician-handoff export
│   ├── voice.py               # Groq Whisper turbo transcription
│   └── v2_schema.py           # 14 routes × 4 safety tiers
├── demo/
│   └── app.py                 # Gradio UI · streaming · voice toggle · support plan
├── eval/
│   ├── run_empathrag_core_eval.py     # Eval A (single-turn)
│   ├── run_multiturn_eval.py          # Eval B (multi-turn safety)
│   ├── sweep_rephraser_drift.py       # 14-route × 3-stage drift audit
│   ├── sweep_f1_stage_isss.py         # F-1 stage × ISSS contract
│   ├── sweep_fairness_spot_check.py   # Demographic perturbation
│   ├── multiturn_scenarios.jsonl      # Karthik dataset (tracked)
│   └── multiturn_safety_supplement.jsonl
├── data/curated/
│   └── service_graph.jsonl            # 32 verified UMD/national service objects
├── docs/
│   ├── architecture/EMPATHRAG_CORE_ARCHITECTURE.md
│   ├── research/PAPER_FRAMING.md      # Includes Phase 8: Controlled Paraphrasing
│   ├── planning/MASTER_CHECKLIST.md
│   ├── audits/                        # OPUS_FULL_PROJECT_AUDIT, MVP_USABILITY_AUDIT, etc.
│   ├── demo/                          # MSML class demo script
│   └── team/                          # Karthik handoffs · ISSS document vetting
└── tests/
    └── test_v25_support_navigator.py  # 21 regression tests
```

---

## Setup

**Requirements:** Python 3.12 · Windows or Linux · ~5 GB disk for venv + curated index

```powershell
# 1. Clone
git clone https://github.com/MukulRay1603/Empath-RAG.git
cd Empath-RAG

# 2. Virtual environment
python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate    # Linux/macOS

# 3. Dependencies
pip install -r requirements.txt

# 4. .env file at repo root (see Demo section)
```

Curated corpus + ML router artifacts are intentionally untracked; the system gracefully falls back when they are missing. To rebuild from a Karthik V2 delivery, see [`docs/team/karthik/CORPUS_INTEGRATION_STEPS.md`](docs/team/karthik/CORPUS_INTEGRATION_STEPS.md).

---

## What is missing for an actual deployment

This is a research and class-demo prototype. To pilot at UMD, the open work is:

- **Clinician sanity check.** A walkthrough with a UMD CAPS clinician + 1-page rubric review.
- **Resource freshness pipeline.** Automated weekly URL + contact validator with `last_verified` badges.
- **Closed pilot.** ~10–20 UMD students with anonymous per-turn feedback.
- **Privacy-by-default surface.** Opt-in logging only, visible data-handling page.
- **Accessibility audit.** WCAG / screen reader / keyboard.
- **Multilingual reflection** for F-1 students (Hindi / Mandarin / Spanish / Korean openers).
- **Anonymous research telemetry** so iteration is data-driven, not anecdotal.
- **6 ISSS document URLs** to vet — see [`docs/team/ISSS_DOCUMENT_VETTING.md`](docs/team/ISSS_DOCUMENT_VETTING.md).

---

## Documentation

- [`docs/README.md`](docs/README.md) — index
- [`docs/architecture/EMPATHRAG_CORE_ARCHITECTURE.md`](docs/architecture/EMPATHRAG_CORE_ARCHITECTURE.md) — runtime design
- [`docs/research/PAPER_FRAMING.md`](docs/research/PAPER_FRAMING.md) — research story, claims, baselines, Phase 8 (Controlled Paraphrasing)
- [`docs/planning/MASTER_CHECKLIST.md`](docs/planning/MASTER_CHECKLIST.md) — sprint state
- [`docs/audits/OPUS_FULL_PROJECT_AUDIT_2026_05_06.md`](docs/audits/OPUS_FULL_PROJECT_AUDIT_2026_05_06.md) — full project handoff

---

## Contributors

- **Mukul Rayana** — UMD MSML, project lead.
- **Karthik** — UMD MSML, dataset and curated-corpus delivery.

This prototype is built openly for academic use. It is not a UMD product or service.

If you are a UMD CAPS clinician, ISSS advisor, or ADS coordinator and would like to give feedback or suggest scope changes, please open a GitHub issue.

---

## License

MIT — see [`LICENSE`](LICENSE).

Dataset licenses vary by source; see corpus notes under `docs/team/karthik/`.
