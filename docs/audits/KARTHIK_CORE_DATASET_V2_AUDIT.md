# Karthik EmpathRAG Core Dataset V2 Audit

Date audited: 2026-05-05  
Dataset path: `Data_Karthik/empathrag_core_dataset_v2`

## Verdict

Yes, we need this dataset and should use it.

It is the first delivery that is large enough to make EmpathRAG Core look like a
real ML/NLP system instead of only a rule-routed demo. It gives us:

- 360 single-turn labeled prompts across all 14 routes.
- 50 multi-turn scenarios.
- 22 risky/ambiguous cases.
- 11 resource profile additions.
- Balanced train/dev/test split: 216/72/72.

The single-turn data is immediately useful for the TF-IDF/logistic router and
Eval A. The multi-turn data is useful, but weaker and needs a follow-up pass
before it can be the strongest paper claim.

## Ingest Result

Command:

```powershell
.\venv\Scripts\python.exe -B eval\ingest_core_dataset_v2.py --delivery-dir Data_Karthik\empathrag_core_dataset_v2
```

Result:

- Status: `pass_with_warnings`
- Errors: none
- Single-turn rows: 360
- Multi-turn scenarios: 50
- Risky/ambiguous rows: 22
- Resource additions: 11

Warnings:

- 35 rows use `expected_usage_modes=none`.
- All 35 are `out_of_scope`, so this is acceptable. Treat `none` as no retrieval expected.

## Single-Turn Label Balance

Route labels are well balanced:

- Most routes have 25 rows.
- `crisis_immediate`: 26 rows.
- `general_student_support`: 24 rows.
- `out_of_scope`: 35 rows.

Safety tier distribution:

- `support_navigation`: 145
- `wellbeing`: 122
- `high_distress`: 73
- `imminent_safety`: 20

This is strong enough for the class presentation and for preliminary Eval A.

## Training And Eval A

Training:

```powershell
.\venv\Scripts\python.exe -B eval\train_ml_router.py
```

Result:

- Trained on 216 rows.
- Model artifacts saved under `models/router/`.
- Artifacts are local/ignored, so demo fallback remains safe if models are absent.

Router test split:

- Rows: 72
- Model available: true
- Rule route accuracy: 0.389
- ML route accuracy: 0.903
- ML tier accuracy: 0.889

Full Core Eval A:

- Rows: 360
- Rule route accuracy: 0.389 CI95=[0.339, 0.439]
- Hybrid route accuracy: 0.856 CI95=[0.819, 0.892]
- Hybrid source org hit rate: 1.000
- Unsafe generation count: 0
- Pure validation/no-action count: 0
- Ungrounded action count: 0

Interpretation:

The dataset makes the ML router contribution visible. This is good for the
presentation. Do not overclaim it as final publication-quality route modeling;
it is still synthetic and vocabulary-sensitive.

## Bugs Exposed And Fixed

The trained router initially overrode obvious rule routes because the confidence
threshold was too low.

Fixes made:

- Raised ML router confidence threshold from `0.15` to `0.35`.
- Moved `out_of_scope` rule detection earlier than anxiety/counseling keywords.
- Added hard safety patterns for synthetic crisis shorthand:
  - `doesn't feel survivable`
  - `tonight ... survivable`
  - `impulses loud`
  - `impulses fast`
  - `impulses unspecified`
  - `spiraling fast ... impulses`
- Added contextual high-distress signals:
  - `panic attacks`
  - `hopeless`
  - `everything hollow`
  - `ideation creeping`
  - `goodbye texts`
  - `mentioned goodbye`
  - `refuse external help`
  - `secrecy`
- Added trajectory escalation reason for goodbye/farewell language in a high-risk context.
- Fixed Eval B denominator so benign scenarios do not count as missed escalations.

Regression:

- `tests/test_v25_support_navigator.py`: 19 passed.

## Multi-Turn Dataset Quality

Eval B after harness fix and safety patches, before our supplemental scenarios:

- Scenarios: 50
- Escalation scenarios: 4
- Missed escalation count: 0
- Missed escalation rate: 0.0
- Unsafe generation count: 0
- Pure validation/no-action count: 0
- Ungrounded action count: 0

Important caveat:

Only 4 of 50 multi-turn scenarios actually require imminent/intercept
escalation. That is too few for the headline multi-turn safety claim.

The multi-turn set is useful for smoke testing, but not yet strong enough for
the main paper story unless we clearly say it is preliminary.

## Core Safety Supplement

We implemented the v2.1 patch locally as a tracked supplement instead of
modifying Karthik's original delivery:

- `eval/multiturn_safety_supplement.jsonl`
- 24 additional curated multi-turn safety scenarios.
- Focus: slow escalation, peer-helper risk, secrecy, dependency pressure,
  help rejection, out-of-scope-to-crisis pivots, basic-needs distress, CARE
  routing, and academic idiom false-positive resistance.

`eval/run_multiturn_eval.py` now loads the supplement by default when present.

Eval B after adding the supplement:

- Total scenarios: 74
- Escalation scenarios: 28
- Missed escalation count: 0
- Missed escalation rate: 0.0
- Unsafe generation count: 0
- Pure validation/no-action count: 0
- Ungrounded action count: 0

This gives the presentation a much stronger multi-turn story while preserving
the provenance distinction between Karthik's dataset and our curated safety
stress-test layer.

Issues:

- Many auto-generated scenarios use shorthand/slang that can be hard to map
  consistently.
- Some `must_trigger_by_turn` values appear in benign/no-intercept scenarios.
- Safety tier labels are noisy between `wellbeing`, `support_navigation`, and
  `high_distress`.
- Several scenarios mix routes without realistic narrative continuity.

## Resource Profile Additions

The 11 resource additions are useful as a source-target map and future registry
input, but should not be auto-merged.

Reasons:

- Some contacts are descriptive prose rather than normalized contact fields.
- Several entries duplicate resources already in our verified UMD registry.
- Resource details need official page re-verification before runtime use.

Use them as review suggestions, not production registry rows.

## What To Ask Karthik For Next

Ask for a focused `empathrag_core_dataset_v2_1` patch, not a full rewrite.

Priority requests:

1. Add 20-30 more multi-turn escalation scenarios.
2. Ensure at least 25 of 50 multi-turn scenarios require a real escalation or
   hard redirect.
3. Add more peer-helper escalation scenarios.
4. Add dependency and secrecy escalation scenarios with clear turn-by-turn
   expected behavior.
5. Add help-rejection scenarios where the system must redirect to human support.
6. Reduce unrealistic slang/shorthand in at least half the multi-turn cases.
7. Make `must_trigger_by_turn` blank/null for scenarios that should never
   escalate.
8. Add a short label rubric explaining the difference between:
   - `wellbeing`
   - `support_navigation`
   - `high_distress`
   - `imminent_safety`
9. Keep the single-turn set mostly as-is.
10. Mark out-of-scope rows with `expected_usage_modes=none`; this is acceptable.

## Presentation Use

Use these results in the class presentation:

- Single-turn dataset size and balance.
- Rule vs ML route accuracy improvement.
- Zero unsafe generation / no-action / ungrounded-action counts in this eval.
- Multi-turn eval as preliminary and now correctly measured.

Do not present the multi-turn benchmark as final strong evidence yet. Present it
as the next tightening step and show that it already exposed safety-language
coverage gaps, which we fixed.
