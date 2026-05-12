# Reproducibility

What a third-party reviewer or fork can and cannot reproduce from this repo,
and the exact commands to do so.

## TL;DR

| Evaluation | Inputs in repo? | Reproducible? |
|---|---|---|
| Eval B (74 multi-turn scenarios) | ✓ `eval/multiturn_scenarios.jsonl` + `eval/multiturn_safety_supplement.jsonl` | ✓ Full reproduction |
| Per-layer ablation (5 variants × Eval B) | ✓ Same inputs | ✓ Full reproduction |
| Unguarded Llama baseline (Eval B) | ✓ Same inputs | ✓ Full reproduction (requires Groq API key) |
| Drift sweep (29 cells × 14 routes × 3 stages) | ✓ Probes hard-coded in script | ✓ Full reproduction (requires Groq) |
| F-1 stage × ISSS contract (12 cells) | ✓ Probes hard-coded | ✓ Full reproduction (requires Groq) |
| Sycophancy probes (25 cells) | ✓ Probes hard-coded | ✓ Full reproduction (requires Groq) |
| Prompt-injection probes (16 cells) | ✓ Probes hard-coded | ✓ Full reproduction (requires Groq) |
| Fairness spot-check (18 paired prompts) | ✓ Probes hard-coded | ✓ Full reproduction (requires Groq) |
| Eval A (360 single-turn) | ⚠ Requires the curated UMD evaluation dataset (intentionally untracked, internal coursework deliverable) | ⚠ Inputs needed |
| ML router training | ✗ Trained artifacts under `models/router/` are intentionally untracked | ⚠ Re-train from the labeled dataset via `eval/train_ml_router.py` |
| Curated retrieval index | ✗ FAISS index under `data/curated/indexes/` is intentionally untracked | ⚠ Re-build via `src/data/build_curated_index.py` from `data/curated/resources_seed.jsonl` |

## Commands

After cloning + venv setup + `.env` with `GROQ_API_KEY` (and optionally `ANTHROPIC_API_KEY`):

```powershell
# Headline safety eval: 0 / 28 missed escalations (rephraser ON)
EMPATHRAG_REPHRASER_ENABLED=1 ./venv/Scripts/python.exe eval/run_multiturn_eval.py

# Same-model unguarded baseline: 9 / 28 missed escalations
./venv/Scripts/python.exe eval/run_unguarded_baseline.py

# Per-layer ablation: ~20 min, ~$1 in Groq calls
EMPATHRAG_REPHRASER_ENABLED=1 ./venv/Scripts/python.exe eval/run_ablation_eval.py

# Targeted failure-mode sweeps (each ~3-5 min, ~$0.05)
./venv/Scripts/python.exe eval/sweep_rephraser_drift.py
./venv/Scripts/python.exe eval/sweep_f1_stage_isss.py
./venv/Scripts/python.exe eval/sweep_sycophancy_probes.py
./venv/Scripts/python.exe eval/sweep_prompt_injection.py
./venv/Scripts/python.exe eval/sweep_fairness_spot_check.py

# Resource URL health
./venv/Scripts/python.exe eval/audit_resource_urls.py

# Regression suite
./venv/Scripts/python.exe -m pytest tests/test_v25_support_navigator.py -v
```

## Expected results at commit `847587d` (or newer)

| Eval | Headline |
|---|---|
| Eval B (rephraser ON) | 0 / 28 missed escalation, 0 unsafe, 0 ungrounded, ~570 ms avg latency |
| Ablation (rephraser ON) | baseline 0 / 28; no_stage1_precheck 22 / 28; others 0 / 28 |
| Unguarded Llama 3.3 70B | 9 / 28 missed escalation (CI95 [0.148, 0.494]), 2 harm-endorsement turns |
| Drift sweep | 27-29 / 29 clean (1-2 stochastic LLM-level flags within tolerance) |
| F-1 stage × ISSS contract | 12 / 12 pass |
| Sycophancy probes | 25 / 25 clean |
| Prompt-injection probes | 16 / 16 clean |
| Fairness spot-check | 18 / 18 no divergence |
| Resource URL audit | 60 / 63 live (3 SAMHSA TLS quirks are not real outages) |
| Regression tests | 21 / 21 pass |

Numbers from stochastic-LLM evaluations will vary turn-to-turn within the
tolerances above. Deterministic-template evaluations are stable across runs.

## What's intentionally untracked

- **Internal dataset folders** — teammate dataset deliveries are not the repo's to redistribute.
- **`data/curated/indexes/`** — FAISS index + SQLite metadata, regenerable from `resources_seed.jsonl`.
- **`models/router/`** — ML router artifacts (TF-IDF vectorizer + logistic regression), regenerable via `eval/train_ml_router.py` once the labeled dataset is present.
- **`.env`** — API keys.
- **Generated eval reports** under `eval/sweep_*.md`, `eval/ablation_*.md`, `eval/unguarded_*.md`, `eval/audit_*.md` — regenerable from the scripts.

See `.gitignore` for the exhaustive list and the rationale per pattern.

## Versioning and dataset provenance

This evaluation set is based on the curated UMD Student Support Conversational Dataset (216/72/72 train/dev/test split for Eval A; 50 + 24 multi-turn = 74 scenarios for Eval B). The planned next dataset pull extends coverage with authority-misconduct scenarios, sycophancy probes, topic-shift scenarios, incomplete-message scenarios, and real anonymized student turns.

## Honest caveats

- **n = 28 escalation scenarios** is small enough that absolute claims like "0% missed escalation in deployment" are not warranted by this evaluation alone. The meaningful claim is the comparison against the unguarded baseline (0/28 vs 9/28, non-overlapping CIs).
- **Synthetic data** structurally differs from real student phrasing. Numbers here are prototype evidence, not deployment claims.
- **Stochastic LLM components** mean drift / sycophancy / injection sweeps will vary by 1-2 cells run-to-run. The patterns are stable; individual cells are not.
- **Provider availability** affects all rephraser-ON results. Groq Llama 3.3 70B is the primary path; Anthropic Haiku 4.5 is the fallback. With both unavailable, the system falls through to deterministic templates and the rephraser metrics become identical to the deterministic baseline.
