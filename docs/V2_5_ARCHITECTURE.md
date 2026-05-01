# EmpathRAG V2.5 Architecture

EmpathRAG V2.5 is a student-support navigator, not a therapist, diagnostic system, emergency service, or clinical product.

## Flow

1. Intake: user message, session state, and mode (`student` or `helping_friend`).
2. Hard safety precheck: deterministic safety policy scans the current message.
3. Safety tier: map to one of four operational tiers.
4. Route classification: assign a support route such as academic setback, ADS, advisor conflict, peer helper, or crisis.
5. Service graph and curated retrieval: filter by route/tier/usage mode before source cards are shown.
6. Response template: short validation, reframe, recommended next action, source option, backup option.
7. Output guard: catches pure validation, unsafe agreement, dependency language, and ungrounded contact claims.
8. UI: shows route, tier, output guard status, sources, and recommended next action.

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

The fast backend is deterministic and presentation-safe. The full real backend remains experimental because local model loading can stall.
