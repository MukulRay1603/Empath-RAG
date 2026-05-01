# Karthik V2 Corpus Audit

Audit date: 2026-04-30

Delivery path:

```text
Data_Karthik/v2/
```

## Verdict

Karthik's V2 delivery is much better than the first version and is structurally compatible with the EmpathRAG curated corpus pipeline.

It is not yet publication-ready, but it is close enough to use as the candidate V2 corpus after one small cleanup pass and after we add retrieval gating in EmpathRAG.

## Files Received

Expected files are present:

```text
Data_Karthik/v2/
  README_corpus_notes.md
  source_inventory.csv
  excluded_sources.csv
  resources_seed.jsonl
  raw_pages/
```

## Schema Validation

Command run:

```powershell
.\venv\Scripts\python.exe -m src.data.curated_resources Data_Karthik\v2\resources_seed.jsonl --non-strict
```

Result:

```text
Rows: 179
Usable retrieval rows: 179
Validation passed.
```

Important improvements:

- JSONL is valid.
- Row count is now clear: 179 rows.
- IDs are unique.
- No `url: N/A` values remain in `resources_seed.jsonl`.
- Risk and usage labels are internally consistent.
- Exact duplicate text groups are gone.
- `README_corpus_notes.md` is included.
- SAMHSA was reduced heavily from the noisy V1 delivery.

## Actual Corpus Counts

Source type counts:

- `university_resource`: 88
- `government_public_health`: 39
- `student_support`: 39
- `crisis_resource`: 13

Source counts:

- UMD Accessibility & Disability Service: 51
- NAMI: 33
- NIMH: 30
- UMD Counseling Center: 24
- 988 Suicide & Crisis Lifeline: 13
- UMD Graduate School: 7
- CDC: 7
- JED Foundation: 6
- UMD Graduate School Ombuds: 5
- SAMHSA: 2
- UMD Dean of Students: 1

Topic counts:

- `accessibility_disability`: 49
- `counseling_services`: 35
- `crisis_immediate_help`: 29
- `anxiety_stress`: 28
- `depression_support`: 12
- `campus_navigation`: 7
- `graduate_student_support`: 5
- `advisor_conflict`: 5
- `help_seeking_script`: 4
- `isolation_loneliness`: 3
- `grounding_exercise`: 1
- `therapy_expectations`: 1

Risk distribution:

- `safe`: 121
- `crisis_resource`: 39
- `wellbeing`: 19

Usage distribution:

- `retrieval`: 121
- `crisis_only`: 39
- `wellbeing_only`: 19

## Remaining Issues

### 1. One Broken UMD Counseling Chunk Remains

`umd_counseling_026` was fixed correctly.

However, `umd_counseling_005` still has a broken fragment:

```text
Crisis response is available by phone outside of business hours by calling Who is eligible for Counseling Center services'
```

This should be fixed or removed before integration.

Recommended fix:

- Either replace it with clean source text including the correct phone number, or
- remove `umd_counseling_005`, since `umd_counseling_026` already provides clean crisis-contact coverage.

### 2. Some Link/Popup Residue Still Exists

Several 988/NIMH/JED chunks still include fragments like:

- `You are opening a new tab`
- `You are leaving 988lifeline.org for another website`
- `Their content and privacy policies apply`
- `Would you like to continue`
- incomplete link labels such as `chat at .`
- leading punctuation such as `: This lifeline...` or `): Provides information...`

Examples observed:

- `988_lifeline_003`
- `988_lifeline_009`
- `988_lifeline_021`
- `nimh_new_021`
- `nimh_new_022`
- `jed_new_001`

These do not necessarily make the corpus unusable, but they should be cleaned before research/publication use.

### 3. `source_inventory.csv` Still Has Include Rows With No JSONL Chunks

Inventory has 69 sources. JSONL uses 47 source IDs.

Most unused inventory rows are correctly marked `exclude` or `partial`, but these six are marked `include` despite producing no JSONL rows:

- `src_058` - NAMI Getting Help
- `src_066` - Counseling Crisis Services
- `src_067` - Counseling Self-Help Resources
- `src_068` - 988 Chat and Text
- `src_069` - 988 Current Events
- `src_072` - Counseling About Us

Recommended fix:

- If no chunks are included from these sources, mark them `partial`, `needs_review`, or `exclude`.
- If chunks should exist, add the missing rows to `resources_seed.jsonl`.

### 4. Retrieval Gating Is Now Required On Our Side

The corpus labels are consistent, but EmpathRAG currently does not fully use those labels during retrieval.

Current behavior observed in retrieval spot-checks:

- Normal anxiety/counseling/advisor prompts can retrieve `crisis_only` rows in the top results.
- A crisis prompt can retrieve a `safe` depression-support row at rank 1 if retrieval is called directly.

This is a system-side issue, not mainly a Karthik corpus issue.

Required engineering change before making curated retrieval the default:

- Normal prompts should retrieve only `usage_mode = retrieval`.
- Wellbeing prompts may retrieve `retrieval` plus `wellbeing_only`.
- Crisis prompts should be intercepted before normal generation and may use `crisis_only` only for crisis-resource display.
- `crisis_only` rows should not be included as ordinary emotional-grounding context for non-crisis generation.

### 5. Publication And Licensing Caveat

The corpus now includes NAMI and JED Foundation content. These are useful student-support sources, but they are not UMD/government public-domain sources.

For class demo:

- Acceptable as a local candidate corpus with citations and careful framing.

For publication or institutional deployment:

- Track source licenses and terms more carefully.
- Prefer short excerpts, citations, and source links.
- Consider whether non-government/non-UMD content should be separated from official campus resources.
- Consider permissions or a documented fair-use rationale before distributing the corpus.

## Retrieval Spot-Check Summary

A temporary audit index was built:

```text
data/curated/indexes/faiss_karthik_v2_audit.index
data/curated/indexes/metadata_karthik_v2_audit.db
```

Index build result:

```text
Vectors indexed: 179
```

Spot-check results:

- Anxiety/exam prompt retrieved UMD workshops, NAMI anxiety, UMD groups, and wellbeing chunks. Useful overall, but one `crisis_only` JED row appeared in top results.
- UMD counseling intake prompt retrieved good UMD Counseling chunks, but also retrieved broken `umd_counseling_005`.
- Disability accommodation prompt retrieved strong UMD ADS chunks, including graduate assistantship accommodation content.
- Advisor-conflict prompt retrieved strong UMD Graduate School Ombuds chunks.
- Crisis prompt retrieved a mix of safe depression and crisis resources if retrieval is called directly; in the full pipeline, safety triage should intercept before normal retrieval/generation.

## Integration Recommendation

Initial recommendation before local cleanup was not to make the raw Karthik V2
delivery the default curated V2 index yet.

Local follow-up completed:

1. Added a reproducible local cleanup/import script:

```text
scripts/clean_karthik_v2_corpus.py
```

2. Generated a cleaned local corpus under:

```text
data/curated/
```

3. Dropped two rows from the local cleaned corpus:

- `umd_counseling_005` because it retained a broken crisis-phone fragment and was redundant with `umd_counseling_026`.
- `988_lifeline_003` because removing popup residue made it too short to keep as a standalone chunk.

4. Corrected the six unused `include` inventory rows to `partial` in the local cleaned `source_inventory.csv`.

5. Rebuilt the curated FAISS index:

```text
data/curated/indexes/faiss_curated.index
data/curated/indexes/metadata_curated.db
```

Cleaned local index result:

```text
Rows: 177
Validation passed.
Vectors indexed: 177
```

6. Added code-side retrieval gating so `usage_mode` is respected.

Updated recommendation:

- The cleaned local corpus is acceptable as the current V2 class-demo candidate.
- The raw Karthik V2 folder should remain as source input, not the direct demo corpus.
- For publication or UMD-facing use, continue human review and source-license review.

## Minimum Changes Needed Before Demo Integration

Corpus-side:

- Fix or remove `umd_counseling_005`.
- Clean popup/link residue from 988/NIMH/JED rows.
- Correct the six unused `include` rows in `source_inventory.csv`.

Code-side:

- Respect `usage_mode` during curated retrieval.
- Keep crisis resources out of normal prompt context.
- Consider showing crisis resources through a separate safety-response path, not through normal generation context.

Status:

- `usage_mode` retrieval gating has been implemented in `src/pipeline/pipeline.py`.
- Normal prompts now use `retrieval` rows only.
- Wellbeing-support prompts may use `retrieval` plus `wellbeing_only`.
- Crisis and emergency retrieval, if called directly, is restricted to `crisis_only`.
- Full pipeline crisis cases still intercept before ordinary retrieval/generation.

## Overall Assessment

This is a substantial improvement over the first delivery.

The V2 corpus is now structurally sound and much closer to usable. The biggest remaining risk is not the schema; it is retrieval behavior and a few remaining noisy chunks.

For the MSML class project, this can likely be integrated after a focused cleanup and retrieval-gating patch.

For publication or UMD-facing use, it still needs human review, source-license review, more rigorous evaluation, and a clearer separation between official UMD resources, public-health resources, and third-party nonprofit material.
