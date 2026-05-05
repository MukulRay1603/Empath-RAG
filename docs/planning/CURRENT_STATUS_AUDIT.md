# EmpathRAG Core Current Status Audit

Date: 2026-05-05  
Branch: `codex/v2.5-support-navigator`  
Status: Core demo and evaluation path are working; repository git state is clean after the latest checkpoint.

## One-Line Summary

EmpathRAG Core is now a guarded conversational RAG system for student-support navigation. It keeps the original chatbot/RAG proposal alive, but adds safety triage, route classification, resource-registry filtering, constrained response planning, output guards, and multi-turn escalation evaluation.

It should still be framed as a prototype support navigator, not therapy, diagnosis, counseling, crisis prevention, or emergency care.

## What It Does Now

Runtime flow:

```text
message + session
-> Stage-1 lexical safety precheck
-> optional DeBERTa/IG guardrail hook, disabled by default for demo reliability
-> rule route/tier baseline
-> local TF-IDF/logistic ML router when artifacts exist
-> trajectory escalation tracker
-> UMD resource registry + curated retrieval filtering
-> constrained response planner
-> output-side anti-sycophancy/groundedness guard
-> Gradio UI with route, tier, confidence, latency, sources, and next action
```

Live demo:

```text
http://127.0.0.1:7860/
```

Demo mode is deterministic/fast and presentation-safe. The trained local router artifacts live under `models/router/` and are ignored by git. If those artifacts are missing, the app falls back to deterministic routing.

## Current Implemented Components

- Unified Core runtime: `src/pipeline/core.py`
- Canonical route/tier schema: `src/pipeline/v2_schema.py`
- Safety triage policy: `src/pipeline/safety_policy.py`
- Lightweight ML router: `src/pipeline/ml_router.py`
- Resource registry loader: `src/pipeline/service_graph.py`
- Response planner: `src/pipeline/response_planner.py`
- Output guard: `src/pipeline/output_guard.py`
- Gradio demo: `demo/app.py`
- Dataset V2 ingest gate: `eval/ingest_core_dataset_v2.py`
- Eval A harness: `eval/run_empathrag_core_eval.py`
- Eval B harness: `eval/run_multiturn_eval.py`
- Eval B safety supplement: `eval/multiturn_safety_supplement.jsonl`
- Regression tests: `tests/test_v25_support_navigator.py`

## Dataset State

Karthik delivery:

```text
Data_Karthik/empathrag_core_dataset_v2/
```

Ingest result:

- Status: `pass_with_warnings`
- Single-turn rows: 360
- Multi-turn scenarios: 50
- Risky/ambiguous cases: 22
- Resource profile additions: 11
- Train/dev/test split: 216/72/72

The warning was `expected_usage_modes=none` for 35 rows. All 35 are `out_of_scope`, so this is acceptable.

We added a separate tracked supplement:

```text
eval/multiturn_safety_supplement.jsonl
```

This adds 24 curated multi-turn safety scenarios without modifying Karthik's original delivery.

## Current Metrics

Eval A: single-turn ablation on 360 synthetic prompts.

- Rule route accuracy: `0.389`
- Hybrid Core route accuracy: `0.856`
- Hybrid source organization hit rate: `1.000`
- Intercept accuracy: `0.994`
- Unsafe generation count: `0`
- Pure validation/no-action count: `0`
- Ungrounded action count: `0`

Router test split:

- Rows: 72
- Rule route accuracy: `0.389`
- ML route accuracy: `0.903`
- ML tier accuracy: `0.889`

Eval B: multi-turn benchmark with Karthik scenarios plus Core safety supplement.

- Total scenarios: `74`
- True escalation scenarios: `28`
- Missed escalation count: `0`
- Missed escalation rate: `0.0`
- Unsafe generation count: `0`
- Pure validation/no-action count: `0`
- Ungrounded action count: `0`

Important caveat: these are synthetic, development-stage results. They are strong enough for class presentation and preliminary research framing, not final clinical or deployment claims.

## What The Demo Can Show

Recommended live demo order:

1. Failed exam / academic setback.
2. ADS accommodations.
3. Basic needs / food insecurity.
4. Peer-helper imminent safety case.
5. Out-of-scope legal/medical request.
6. Academic idiom false-positive resistance.

This shows usefulness, routing, source grounding, peer-helper support, scope discipline, and safety escalation.

## What Is Strong Now

- The project has one consolidated product direction: EmpathRAG Core.
- V1/V2/V2.5 are baselines/checkpoints, not competing products.
- Karthik's single-turn dataset makes the ML router contribution visible.
- Eval B now has enough true escalation cases for a much stronger demo story.
- Stage-1 hard safety is explicit and visible in UI/eval metadata.
- Crisis/imminent safety bypasses normal generation.
- Out-of-scope route avoids support-source retrieval.
- Resource registry has verified UMD service objects.
- The demo is visually polished and transparent.

## What Is Still Weak

- Safety tier accuracy is still noisy, especially between `wellbeing`, `support_navigation`, and `high_distress`.
- RoBERTa route classifier is still a stretch goal, not implemented.
- DeBERTa/IG is available as an optional hook but should not be loaded live during the class demo.
- Resource profile additions from Karthik should be manually reviewed before registry merge.
- The project is not ready for public student-facing deployment.
- Voice input, bounded memory, and Hugging Face deployment are product-roadmap items, not class-demo blockers.

## Immediate Next Steps

- Build final presentation slides around Eval A and Eval B.
- Screenshot the six scripted demo cases.
- Keep local demo stable and fast.
- Preserve V1 BERTScore/Wilcoxon/adversarial results as baseline evidence.
- Decide whether to add voice input and Support Plan Memory after the class presentation.

## Git Hygiene

Current git branch is clean after the latest pushed checkpoint.

Do not commit:

- `Data_Karthik/`
- `data/curated/` generated corpus/index files
- `models/router/`
- generated eval outputs such as `eval/core_eval_results.json`, `eval/multiturn_results.json`, and reports
- `venv/`
- `.env`

Tracked eval inputs/scripts are intentional:

- `eval/multiturn_scenarios.jsonl`
- `eval/multiturn_safety_supplement.jsonl`
- `eval/fixtures/core_dataset_v2_sample/`
- eval runner scripts

Latest important checkpoints:

- `e143b4a Prepare Core dataset intake and resource registry`
- `f046303 Ingest Core dataset and harden router policy`
- `433900d Add Eval B safety supplement`
