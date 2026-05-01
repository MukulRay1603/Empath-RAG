# EmpathRAG Core Architecture

EmpathRAG Core is a guarded conversational RAG system for emotional/student support navigation. It is not a therapist, diagnostic system, emergency service, or clinical product.

## Flow

1. Intake: user message, session state, and mode (`student` or `helping_friend`).
2. Hard safety precheck: deterministic safety policy scans the current message.
3. Hybrid classifier: lightweight TF-IDF + logistic regression predicts route and safety tier when confidence is sufficient.
4. Hard safety override: crisis/imminent rules override ML predictions.
5. Service graph and curated retrieval: filter by route/tier/usage mode before source cards are shown.
6. Response planner: validation, reframe, recommended next action, source option, backup option, follow-up question.
7. Output guard: catches pure validation, unsafe agreement, dependency language, and ungrounded contact/resource claims.
8. UI/eval metadata: route, tier, classifier confidence, retrieval mode, output guard, sources, trajectory.

## Four-Mode Ladder

- `imminent_safety`: normal generation blocked; crisis/human handoff only.
- `high_distress`: short support, grounding, and urgent support options.
- `support_navigation`: practical next step with source-grounded campus route.
- `wellbeing`: low-risk coping support plus campus option where useful.

## Current Routes

- `academic_setback`
- `exam_stress`
- `accessibility_ads`
- `advisor_conflict`
- `counseling_navigation`
- `basic_needs`
- `care_violence_confidential`
- `peer_helper`
- `loneliness_isolation`
- `anxiety_panic`
- `low_mood`
- `crisis_immediate`
- `general_student_support`
- `out_of_scope`

## Service Graph

Minimal graph file:

- `data/curated/service_graph.jsonl`

Loader:

- `src/pipeline/service_graph.py`

The graph only uses verified source URLs from the current corpus. Missing phone numbers, hours, locations, and eligibility rules are marked `unknown`.

## Output Guard

File:

- `src/pipeline/output_guard.py`

Current checks:

- crisis responses must not continue normal academic coaching
- non-crisis responses must include a recommended next action
- pure validation with no redirect is flagged
- dependency-forming language is flagged
- harmful/sycophantic agreement is flagged
- self-degrading compliance is flagged
- ungrounded contact claims are flagged

## Demo Backend

The class demo should use:

```powershell
$env:EMPATHRAG_DEMO_BACKEND='fast'
$env:EMPATHRAG_RETRIEVAL_CORPUS='curated_support'
.\venv\Scripts\python.exe -u demo\app.py
```

The demo uses EmpathRAG Core in `hybrid_ml` mode. If local ML router artifacts are missing, it falls back to the deterministic route rules. The full local LLM backend remains experimental because local model loading can stall.

## ML Router

Files:

- `src/pipeline/ml_router.py`
- `eval/prepare_karthik_dataset.py`
- `eval/train_ml_router.py`
- `eval/run_router_eval.py`

The current model uses TF-IDF n-grams plus logistic regression. It is intentionally lightweight and auditable. Hard safety checks always override it.

## Unified Evaluation

Run:

```powershell
.\venv\Scripts\python.exe -B eval\run_empathrag_core_eval.py
```

Current local checkpoint metrics on the 92-row prepared Karthik dataset:

- Rule route accuracy: 0.935
- Hybrid ML route accuracy: 0.978
- Safety tier accuracy: 0.902
- Intercept accuracy: 1.000
- Source organization hit rate: 0.913
- Unsafe generation count: 0
