# EmpathRAG V2 Project Memory

This note preserves the current project state, decisions, audit findings, and next steps so the work can continue even if chat context is lost.

## 2026-05-05 Current Core Snapshot

Current branch:

```text
codex/v2.5-support-navigator
```

Current framing:

EmpathRAG Core is one consolidated guarded conversational RAG system for
student-support navigation. V1/V2/V2.5 are checkpoints/baselines, not separate
products.

Current working pieces:

- Core runtime: `src/pipeline/core.py`
- Stage-1 lexical safety precheck: `src/pipeline/safety_policy.py`
- TF-IDF/logistic router: `src/pipeline/ml_router.py`
- Route/tier schema: `src/pipeline/v2_schema.py`
- Resource registry: `data/curated/service_graph.jsonl`
- Response planner: `src/pipeline/response_planner.py`
- Output guard: `src/pipeline/output_guard.py`
- Demo: `demo/app.py`
- Dataset ingest: `eval/ingest_core_dataset_v2.py`
- Eval A: `eval/run_empathrag_core_eval.py`
- Eval B: `eval/run_multiturn_eval.py`
- Safety supplement: `eval/multiturn_safety_supplement.jsonl`

Karthik's `empathrag_core_dataset_v2` was received and ingested:

- 360 single-turn prompts
- 50 multi-turn scenarios
- 22 risky/ambiguous cases
- 11 resource additions
- 216/72/72 train/dev/test split

Current Eval A:

- Rule route accuracy: `0.389`
- Hybrid Core route accuracy: `0.856`
- Source organization hit rate: `1.000`
- Unsafe generation count: `0`

Current Eval B with safety supplement:

- 74 scenarios
- 28 escalation scenarios
- Missed escalation count: `0`
- Unsafe generation count: `0`
- Pure validation/no-action count: `0`

Important caveat:

These are synthetic development results. They are good for class presentation and preliminary research framing, but not clinical/deployment claims.

Latest pushed checkpoints:

- `f046303 Ingest Core dataset and harden router policy`
- `433900d Add Eval B safety supplement`

The older notes below are archival and may mention previous branch names or early corpus-only work.

## Current Goal

EmpathRAG is a mental-health-adjacent RAG project for student support. The near-term goal is a clear, working MSML class demo within roughly 10 days. The longer-term goal is a safer, research-oriented version that could eventually be evaluated for usefulness to UMD student mental-health support contexts.

Important framing:

- This should not present itself as therapy, diagnosis, clinical treatment, or emergency response.
- The system should provide support-oriented information, campus-resource navigation, grounding/wellbeing help, and crisis redirection.
- Crisis and emergency cases must be intercepted by safety logic, not handled as ordinary retrieval generation.
- Existing v1 functionality should remain intact as a fallback demo path.

## Branch And Repo State

- Repository path: `E:\Projects\EmpathRAG\Empath-RAG`
- Current branch: `codex-v2-safety-hardening`
- Remote tracking branch: `origin/codex-v2-safety-hardening`
- `main` should remain untouched for now.
- Karthik's delivered data is currently under `Data_Karthik/` and is untracked.

Important commits already made:

- `81deeef Start v2 safety hardening`
- `fadd796 Add curated corpus integration scaffold`

## Existing V1 Status

V1 is still usable as a class-demo fallback.

The existing Reddit/research retrieval path remains available through:

```powershell
$env:EMPATHRAG_RETRIEVAL_CORPUS='reddit_research'
.\venv\Scripts\python.exe demo\app.py
```

Known smoke-test state:

- `smoke_test_pipeline.py` previously ran with 4/5 passing.
- The known failing case is a neutral literature-review prompt misclassified as `hopeful`.
- This appears to be an existing classifier weakness, not caused by the curated-corpus scaffold.

## V2 Work Already Implemented

### Safety Triage

File:

- `src/pipeline/safety_policy.py`

Implemented:

- `SafetyTriagePolicy`
- `SafetyLevel`
  - `pass`
  - `wellbeing_support`
  - `crisis`
  - `emergency`
- `SafetyDecision`
- Explicit lexical backups for imminent or crisis-risk language
- Fail-closed direction for safety-sensitive paths

Previous adversarial evaluation after triage:

- Triage accuracy: `0.90`
- Crisis recall: `0.95`
- False-positive rate: `0.20`

### Pipeline Hardening

File:

- `src/pipeline/pipeline.py`

Implemented:

- Default `use_real_guardrail=True`
- Default `allow_stub_guardrail=False`
- Real guardrail failure should fail closed unless explicitly overridden
- `retrieval_corpus` support:
  - `reddit_research`
  - `curated_support`
  - `auto`
- `auto` uses curated retrieval if curated FAISS index and metadata DB exist; otherwise it falls back to Reddit retrieval.
- Result metadata now includes:
  - `retrieved_sources`
  - `retrieval_corpus`
- `retrieved_chunks` remains a list of strings for compatibility.

### Curated Corpus Validator

File:

- `src/data/curated_resources.py`

Purpose:

- Validate curated `resources_seed.jsonl`.
- Enforce required fields and controlled labels.

Required fields:

- `id`
- `source_id`
- `source_name`
- `source_type`
- `title`
- `url`
- `topic`
- `audience`
- `risk_level`
- `usage_mode`
- `text`
- `summary`
- `last_checked`
- `notes`

Allowed `source_type` values:

- `university_resource`
- `crisis_resource`
- `government_public_health`
- `student_support`
- `clinician_review_candidate`

Allowed `risk_level` values:

- `safe`
- `wellbeing`
- `crisis_resource`
- `exclude`

Allowed `usage_mode` values:

- `retrieval`
- `wellbeing_only`
- `crisis_only`
- `metadata_only`

Useful command:

```powershell
.\venv\Scripts\python.exe -m src.data.curated_resources Data_Karthik\resources_seed.jsonl --non-strict
```

### Curated Index Builder

File:

- `src/data/build_curated_index.py`

Purpose:

- Build a FAISS index and SQLite metadata DB from curated JSONL.
- Keeps curated resources separate from the original Reddit index.
- Uses `sentence-transformers/all-mpnet-base-v2`.

Expected future command after cleaned data arrives:

```powershell
.\venv\Scripts\python.exe -m src.data.build_curated_index --input data\curated\resources_seed.jsonl --index data\curated\indexes\faiss_curated.index --db data\curated\indexes\metadata_curated.db
```

### Curated Retrieval Audit

File:

- `eval/run_curated_retrieval_audit.py`

Purpose:

- Run a small retrieval audit against curated prompts.
- Writes ignored audit output to `eval/curated_retrieval_audit.json`.

Command:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\venv\Scripts\python.exe eval\run_curated_retrieval_audit.py
```

### Demo Updates

File:

- `demo/app.py`

Implemented:

- `EMPATHRAG_RETRIEVAL_CORPUS` environment variable
- Defaults to `auto`
- Demo displays:
  - retrieval corpus
  - safety level
  - safety reason
  - top source metadata
- Sharing/logging disabled by default through:
  - `EMPATHRAG_SHARE`
  - `EMPATHRAG_LOG_TURNS`

## Documentation Already Added

Files:

- `docs/research/SAFETY_AND_DATASET_PLAN.md`
- `docs/team/karthik/CURATED_CORPUS_HANDOFF.md`
- `docs/team/karthik/CURATED_CORPUS_HANDOFF.pdf`
- `docs/team/karthik/CORPUS_INTEGRATION_STEPS.md`
- `data/curated/resources_seed.example.jsonl`

Desktop copies were also previously saved:

- `C:\Users\mukul\OneDrive\Desktop\TEAMMATE_CURATED_CORPUS_HANDOFF.md`
- `C:\Users\mukul\OneDrive\Desktop\TEAMMATE_CURATED_CORPUS_HANDOFF.pdf`

## Karthik Data Location

Folder:

- `Data_Karthik/`

Files received:

- `resources_seed.jsonl`
- `source_inventory.csv`
- `excluded_sources.csv`
- `raw_pages/`

Karthik's summary claimed:

- Total sources reviewed: `36`
- Total chunks included: `177`
- Total chunks excluded: `3`

Actual audit found:

- `resources_seed.jsonl` rows: `167`
- Unique IDs: `167`
- `source_inventory.csv` rows: `46`
- `excluded_sources.csv` rows: `3`
- JSONL validator passes structurally
- All real checked URLs returned live responses during the spot-check

## Karthik Corpus Actual Distribution

Actual source-type counts:

- `university_resource`: `76`
- `student_support`: `40`
- `government_public_health`: `38`
- `crisis_resource`: `13`

Actual topic counts:

- `counseling_services`: `40`
- `accessibility_disability`: `38`
- `crisis_immediate_help`: `17`
- `graduate_student_support`: `16`
- `help_seeking_script`: `10`
- `anxiety_stress`: `9`
- `grounding_exercise`: `8`
- `advisor_conflict`: `8`
- `academic_burnout`: `7`
- `depression_support`: `5`
- `campus_navigation`: `4`
- `isolation_loneliness`: `3`
- `therapy_expectations`: `1`
- `emergency_services`: `1`

Actual risk distribution:

- `safe`: `137`
- `crisis_resource`: `20`
- `wellbeing`: `10`

Actual usage distribution:

- `retrieval`: `137`
- `crisis_only`: `20`
- `wellbeing_only`: `10`

Actual source counts:

- `EmpathRAG Curated`: `40`
- `UMD Accessibility & Disability Service`: `38`
- `SAMHSA`: `27`
- `UMD Counseling Center`: `25`
- `988 Suicide & Crisis Lifeline`: `13`
- `NIMH`: `10`
- `UMD Graduate School`: `7`
- `UMD Graduate School Ombuds`: `5`
- `CDC`: `1`
- `UMD Dean of Students`: `1`

Chunk length stats:

- Minimum words: `80`
- Median words: `132`
- Maximum words: `248`
- Mean words: about `133.2`

## Karthik Corpus Audit Findings

Technical compatibility:

- Good.
- The file validates structurally.
- It can be indexed.

Source coverage:

- Good start.
- UMD counseling, UMD ADS, 988, graduate support, NIMH, SAMHSA, CDC, and curated support are represented.

Safety/data quality:

- Medium.
- It is not ready for publication or student-facing deployment as-is.
- It may be usable for a class demo only after filtering or careful selection.

Major issues:

- Summary says 177 rows but actual file has 167 rows.
- Around 40 rows have `url: N/A`.
- `README_corpus_notes.md` is missing.
- `source_inventory.csv` marks everything as `include`, which is inaccurate.
- Some source IDs in inventory are not used in JSONL.
- SAMHSA contains duplicated chunks and scrape noise.
- CDC/NIMH/SAMHSA/UMD chunks contain webpage boilerplate.
- Some chunks are broken or incomplete.

Specific broken chunks:

- `umd_counseling_026`: references a phone number that is missing.
- `umd_ads_030`: references an email that is missing.
- `umd_grad_extra_003`: mixes unrelated content and should be split or removed.

Duplicate SAMHSA regions:

- `samhsa_002` through `samhsa_011`
- `samhsa_017` through `samhsa_026`

Boilerplate examples to remove:

- `Skip directly to site content`
- `An official website of the United States government`
- `.gov means it is official`
- `Secure .gov websites use HTTPS`
- `Sign up for Email Updates`

Unhelpful SAMHSA material to remove:

- Medicaid/CHIP
- Block Grants
- Fentanyl Awareness pages
- Tribal Behavioral Health Agenda
- Technical specification manuals
- Disclaimers
- Website navigation
- Link lists

## Important Retrieval Observation

The `EmpathRAG Curated` rows often retrieve very well because they are tailored to student phrasing. However, these rows currently have weak provenance, often `url: N/A`.

Decision:

- Do not discard them automatically.
- Require clear `internal://empathrag-curated/...` provenance if they are hand-authored or synthesized.
- Mark them as requiring human review in `notes`.
- For research or institutional use, separate official-source rows from synthesized support rows in evaluation and reporting.

## Current Integration Decision

Do not integrate Karthik's current delivery as the final curated corpus yet.

Ask Karthik for a cleaned `curated_corpus_delivery_v2/` with:

- fixed summary counts
- cleaned SAMHSA chunks
- removed duplicates
- stripped boilerplate
- fixed broken chunks
- no `url: N/A`
- accurate `source_inventory.csv`
- added `README_corpus_notes.md`

The cleanup request is saved in:

- `docs/team/karthik/CORPUS_CLEANUP_REQUEST.md`

## Safe Integration Plan Once Cleaned Corpus Arrives

1. Place cleaned file at:

```text
data/curated/resources_seed.jsonl
```

2. Validate schema:

```powershell
.\venv\Scripts\python.exe -m src.data.curated_resources data\curated\resources_seed.jsonl --non-strict
```

3. Run additional duplicate and boilerplate checks.

4. Build curated FAISS index:

```powershell
.\venv\Scripts\python.exe -m src.data.build_curated_index --input data\curated\resources_seed.jsonl --index data\curated\indexes\faiss_curated.index --db data\curated\indexes\metadata_curated.db
```

5. Run curated retrieval audit:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\venv\Scripts\python.exe eval\run_curated_retrieval_audit.py
```

6. Run smoke test:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\venv\Scripts\python.exe smoke_test_pipeline.py
```

7. Launch demo with curated retrieval:

```powershell
$env:EMPATHRAG_RETRIEVAL_CORPUS='curated_support'
.\venv\Scripts\python.exe demo\app.py
```

## Next Engineering Tasks

High priority:

- Add a stronger corpus audit script that checks duplicates, boilerplate, `url: N/A`, broken contact references, risk/usage mismatch, and source inventory mismatch.
- Add a curated corpus import command that copies a delivery folder into `data/curated/` only if validation passes.
- Add retrieval gating so `crisis_only` rows are not used in normal retrieval, and `wellbeing_only` rows are retrieved only for wellbeing prompts.
- Improve the neutral-prompt classification issue from the smoke test.
- Add an MSML demo mode with stable, polished prompts and clear source display.

Medium priority:

- Add a small curated retrieval gold set for evaluation.
- Add source diversity controls so retrieval does not overuse internal curated rows.
- Add citation formatting in the demo.
- Add a demo-safe disclaimer that is concise and not alarming.
- Add result logging only when explicitly enabled and with no sensitive raw user text by default.

Research/publication priority:

- Define evaluation protocol.
- Separate official-resource retrieval from synthesized-support retrieval.
- Add human-review labels.
- Create annotation guidelines.
- Add safety benchmark prompts.
- Document corpus construction and exclusion criteria.
- Consider IRB or institutional guidance before any student-facing deployment or user study.

## Demo Strategy

For the MSML class presentation:

- Keep v1 available as fallback.
- Use V2 if the curated corpus passes cleanup and retrieval spot checks.
- Show safety triage and source-aware retrieval rather than claiming clinical capability.
- Use prepared prompts:
  - stress/anxiety about exams
  - navigating counseling resources
  - disability accommodations
  - advisor conflict or graduate support
  - crisis prompt to show safe redirection

Avoid:

- claiming diagnosis
- claiming therapy replacement
- using private student data
- live-testing highly sensitive prompts without a safety explanation

## Git Hygiene

Current branch should keep V2 work isolated.

Before committing new docs/code:

```powershell
git status -sb
```

Do not commit:

- `Data_Karthik/` unless explicitly deciding to version candidate corpus material
- generated FAISS indexes
- generated metadata DBs
- raw sensitive or large scraped pages unless intentionally approved

Existing `.gitignore` already ignores curated seed data, raw pages, indexes, and audit output.

## Short Mental Model

EmpathRAG V2 is moving from a Reddit-based research prototype toward a safer campus-resource RAG system.

The main challenge is not only model quality. It is safety, provenance, retrieval gating, corpus cleanliness, evaluation design, and honest product framing.

The current code scaffold is in a good direction. The current Karthik corpus is structurally useful but needs cleanup before integration.

## 2026-04-30 Karthik V2 Local Cleanup

Karthik delivered a revised corpus under:

```text
Data_Karthik/v2/
```

Raw V2 status:

- Expected files present.
- `resources_seed.jsonl` validates.
- 179 rows.
- No `url: N/A`.
- No exact duplicate text groups.
- Risk/usage labels are internally consistent.
- Remaining issues were minor: one broken UMD counseling row, one too-short 988 row after popup cleanup, six unused inventory rows marked `include`, and some popup/link residue.

Local cleanup script added:

```text
scripts/clean_karthik_v2_corpus.py
```

Local cleaned corpus generated under:

```text
data/curated/
```

Cleaned local corpus status:

- 177 rows.
- Dropped `umd_counseling_005`.
- Dropped `988_lifeline_003`.
- Cleaned popup/link residue patterns.
- Updated unused include inventory rows to `partial`.
- Validation passed.
- Built curated index with 177 vectors at:
  - `data/curated/indexes/faiss_curated.index`
  - `data/curated/indexes/metadata_curated.db`

Pipeline update:

- `src/pipeline/pipeline.py` now respects curated `usage_mode`.
- Normal prompts retrieve only `retrieval`.
- Wellbeing-support prompts retrieve `retrieval` plus `wellbeing_only`.
- Crisis/emergency retrieval, if called directly, retrieves only `crisis_only`.
- Full pipeline crisis cases still intercept before normal retrieval/generation.
- Crisis intercepts can retrieve curated crisis-resource source cards for the demo side panel without invoking normal generation.
- Curated retrieval now limits repeated source names in the top results so one source is less likely to dominate the source panel.

Karthik should now be assigned higher-value work rather than this cleanup:

- Expand official UMD/college support sources.
- Build a small evaluation/gold query set.
- Add human review annotations.
- Help document source licenses and corpus construction decisions.

Karthik's next concrete assignment is documented in:

```text
docs/team/karthik/EVAL_DATASET_TASK.md
```

Validator for Karthik's next delivery:

```text
eval/validate_eval_delivery.py
```

Expected future command:

```powershell
.\venv\Scripts\python.exe eval\validate_eval_delivery.py path\to\empathrag_eval_delivery_v1
```

## 2026-04-30 Demo Polish

The Gradio app was redesigned for the MSML presentation:

- Minimal presentation-grade header.
- V2 curated-mode badges.
- Concise scope statement: not therapy, diagnosis, or emergency care.
- Prepared prompt buttons:
  - Start counseling
  - ADS accommodations
  - Advisor conflict
  - Grounding help
  - Crisis redirect
- Redesigned emotion timeline panel.
- Redesigned safety guardrail panel.
- Redesigned retrieval source cards with source, topic, risk level, usage mode, and source links.
- Demo generation length is configurable with `EMPATHRAG_MAX_TOKENS` and defaults to `140`.
- Demo top-k is configurable with `EMPATHRAG_TOP_K` and defaults to `5`.

Presentation runbook:

```text
docs/demo/MSML_CLASS_DEMO_SCRIPT.md
```
