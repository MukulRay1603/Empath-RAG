# EmpathRAG Curated Corpus Cleanup Request

Hi Karthik,

Thank you for sending the first curated corpus delivery. The structure is useful and it is close to what we need, but because this project is mental-health-adjacent and student-facing, we need one cleanup pass before integrating it into EmpathRAG V2.

Please use this document as the checklist for the revised delivery.

## Current Audit Result

- The JSONL file is structurally valid.
- All 167 current rows have unique IDs.
- The schema mostly matches the EmpathRAG curated resource format.
- The corpus has useful UMD, crisis-resource, accessibility, counseling, and student-support coverage.
- The current delivery should be treated as a candidate corpus, not a final integration corpus.

## Main Issues To Fix

### 1. Fix The Summary Counts

The summary says there are 177 included chunks, but `resources_seed.jsonl` contains 167 rows.

Please do one of the following:

- Update the summary to match the actual final row count, or
- Send the missing 10 rows if they were accidentally omitted.

Also update all source, topic, domain, risk-level, and usage-mode counts so they match the final `resources_seed.jsonl` exactly.

### 2. Clean The SAMHSA Chunks

Many SAMHSA chunks currently contain navigation text, link-list content, policy/program listings, or content that is not useful for student-facing support retrieval.

Please remove chunks that reference unrelated or low-value material such as:

- Medicaid or CHIP
- Block Grants
- Fentanyl Awareness pages
- Tribal Behavioral Health Agenda
- Technical specification manuals
- Disclaimers
- General SAMHSA website navigation
- Long link lists or page menus
- Substance-use program pages that are not directly useful for student mental-health support

Also remove duplicate SAMHSA chunks. The audit found duplicate content around:

- `samhsa_002` through `samhsa_011`
- `samhsa_017` through `samhsa_026`

When in doubt, remove the chunk. We want fewer clean chunks rather than more noisy chunks.

### 3. Strip Website Boilerplate

Several CDC, NIMH, SAMHSA, and UMD chunks still contain scraped website wrapper text.

Please remove text such as:

- `Skip directly to site content`
- `An official website of the United States government`
- `.gov means it is official`
- `Secure .gov websites use HTTPS`
- `Sign up for Email Updates`
- Page navigation menus
- Footer links
- Repeated site-wide headers

Every final chunk should contain only meaningful support, resource, or informational text.

### 4. Fix Broken Or Incomplete Chunks

The following rows need specific attention:

- `umd_counseling_026`: references a phone number, but the phone number is missing from the chunk.
- `umd_ads_030`: references an email address, but the email address is missing from the chunk.
- `umd_grad_extra_003`: mixes unrelated material into one chunk, including study tips, tutoring, graduate social life, office location, and address details.

For each of these:

- Add the missing information from the source page if it is official and current,
- Split the content into cleaner topic-specific chunks, or
- Remove the chunk if it cannot be fixed cleanly.

### 5. Resolve All `url: N/A` Rows

Approximately 40 rows currently have `url` set to `N/A`.

For a mental-health-adjacent system, provenance needs to be clear and traceable. No final row should have `url: N/A`.

For every affected row, please do one of the following:

- Add the real source URL if the text was adapted from an official source.
- If the text is hand-authored or synthesized, use:

```text
internal://empathrag-curated/<short-topic-or-id>
```

For internal rows, also make sure the row includes:

- `source_name`: `EmpathRAG Curated`
- `source_type`: `student_support` or `clinician_review_candidate`, whichever is more appropriate
- `notes`: clear note saying the content is hand-authored or synthesized and requires human review before deployment

### 6. Update `source_inventory.csv`

Currently, every row in `source_inventory.csv` is marked `include`, but that does not match the actual corpus.

Please update each source using one of these statuses:

- `include`: source was reviewed and usable chunks are included
- `partial`: source was partly usable, but some content was excluded
- `exclude`: source was unusable, broken, irrelevant, or returned a 404
- `needs_review`: source requires human review before it can be trusted

Any page that returned a 404 should be marked `exclude`.

Any page that was reviewed but not chunked should be marked `partial`, `exclude`, or `needs_review`, depending on why it was not used.

### 7. Add `README_corpus_notes.md`

Please include a `README_corpus_notes.md` file in the revised delivery.

It should include:

- Corpus creator
- Date
- Total sources reviewed
- Total chunks included
- Total chunks excluded
- Main source domains
- Topic distribution
- Risk-level distribution
- Usage-mode distribution
- Known limitations
- Sources needing review
- Pages that were hard to scrape
- Content you were unsure about
- Suggested next sources
- Any hand-authored or synthesized content that requires human review

### 8. Use The Required Row Labels

Please make sure the final rows use these labels consistently.

Risk levels:

- `safe`: normal support and informational resources
- `wellbeing`: grounding, coping, reflection, and low-risk wellbeing exercises
- `crisis_resource`: crisis-line, emergency, or urgent-help resource content
- `exclude`: content reviewed but not suitable for retrieval

Usage modes:

- `retrieval`: normal support retrieval
- `wellbeing_only`: grounding or wellbeing exercise retrieval only
- `crisis_only`: crisis-resource retrieval only
- `metadata_only`: source metadata retained but not used as normal retrieval text

Important safety rule:

- Crisis resources should be marked `crisis_resource` plus `crisis_only`.
- Normal support resources should be marked `safe` plus `retrieval`.
- Wellbeing exercises should be marked `wellbeing` plus `wellbeing_only`.

### 9. Final Quality Checklist

Before resending, please confirm all of the following:

- `resources_seed.jsonl` has the correct final row count.
- Every JSONL line is valid JSON.
- Every row has a unique `id`.
- Every row has a real URL or an `internal://empathrag-curated/...` provenance value.
- No row has `url: N/A`.
- SAMHSA navigation and link-list chunks are removed.
- Exact duplicate chunks are removed.
- CDC, NIMH, SAMHSA, and UMD boilerplate is stripped.
- Incomplete contact, phone, and email chunks are fixed or removed.
- `source_inventory.csv` statuses are accurate.
- `README_corpus_notes.md` is included.
- Crisis resources are marked `crisis_resource` plus `crisis_only`.
- Normal support resources are marked `safe` plus `retrieval`.
- Wellbeing exercises are marked `wellbeing` plus `wellbeing_only`.
- Any synthesized or hand-authored rows are clearly marked for human review.

## Revised Delivery Format

Please send the cleaned folder using this structure:

```text
curated_corpus_delivery_v2/
  README_corpus_notes.md
  source_inventory.csv
  excluded_sources.csv
  resources_seed.jsonl
  raw_pages/
```

The most important file is:

```text
resources_seed.jsonl
```

Once we receive the cleaned version, we will:

1. Validate the JSONL schema.
2. Run duplicate and boilerplate checks.
3. Build the curated FAISS index.
4. Run retrieval spot checks.
5. Integrate it into the EmpathRAG V2 demo if it passes.

Thanks again. The current structure is solid; this pass is mainly about making the corpus safer, cleaner, traceable, and defensible for student-facing and research-oriented use.
