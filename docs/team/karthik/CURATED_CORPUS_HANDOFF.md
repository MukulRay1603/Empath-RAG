# EmpathRAG V2 Curated Corpus Handoff

This document is the teammate handoff for building the first curated support
corpus for EmpathRAG V2. The goal is to produce clean, source-cited,
student-support content that can be directly ingested into a separate curated
FAISS + SQLite retrieval index.

## 1. Objective

EmpathRAG currently has a large Reddit-based retrieval corpus. That corpus is
useful for research comparison and ablation, but it should not be the primary
student-facing support source for a mental-health-adjacent demo.

Your task is to create a safer curated corpus from official and reputable public
resources. This corpus should help EmpathRAG retrieve grounding context for:

- anxiety before exams, thesis defense, or presentations
- advisor conflict and academic frustration
- burnout, isolation, loneliness, and imposter feelings
- depression/help-seeking language
- campus counseling navigation
- after-hours and crisis support
- disability/accessibility support
- graduate student stress, funding stress, and academic pressure
- grounding exercises and help-seeking scripts

Prioritize correctness, source quality, and safety over volume.

## 2. Deliverables

Please deliver a folder like this:

```text
curated_corpus_delivery/
  README_corpus_notes.md
  source_inventory.csv
  excluded_sources.csv
  resources_seed.jsonl
  raw_pages/
    src_001.txt
    src_002.txt
```

Required files:

- `source_inventory.csv`: every source/page reviewed.
- `resources_seed.jsonl`: final clean chunks for ingestion.
- `README_corpus_notes.md`: summary of what was collected and any concerns.

Optional but helpful:

- `raw_pages/`: saved raw text snapshots from pages.
- `excluded_sources.csv`: pages reviewed but rejected.

## 3. Source Priority

Use official/public sources first.

Priority order:

1. UMD Counseling Center
2. UMD after-hours or crisis support pages
3. UMD Graduate School resources
4. UMD Accessibility and Disability Service
5. UMD Ombuds, conflict-resolution, or student support resources
6. 988 Lifeline
7. NIMH
8. SAMHSA
9. CDC mental health or suicide prevention resources
10. Other reputable nonprofit/clinical education resources only if they fill a
    real coverage gap

Do not use:

- Reddit, Quora, forums, or social media
- random blogs
- commercial therapy marketing pages
- personal stories with graphic crisis details
- pages that describe self-harm methods
- content that gives diagnosis, treatment, or medication instructions
- login-gated or private pages

## 4. Source Inventory Format

Create `source_inventory.csv`.

Required columns:

```csv
source_id,source_name,source_type,title,url,domain,date_accessed,include_status,reason,license_or_terms_notes
```

Allowed `include_status` values:

```text
include
partial
exclude
needs_review
```

Example:

```csv
src_001,UMD Counseling Center,university_resource,Counseling Services,https://counseling.umd.edu/,counseling.umd.edu,2026-04-27,include,Official student counseling resource,Public webpage
```

Use `needs_review` when the page seems useful but contains sensitive, ambiguous,
or policy-heavy content that should be checked before inclusion.

## 5. Main Corpus Format

Create `resources_seed.jsonl`.

Use JSONL format: one valid JSON object per line. Do not wrap the file in a JSON
array.

Required schema:

```json
{
  "id": "umd_counseling_001",
  "source_id": "src_001",
  "source_name": "UMD Counseling Center",
  "source_type": "university_resource",
  "title": "Counseling Services",
  "url": "https://example.edu/page",
  "topic": "counseling_services",
  "audience": "umd_student",
  "risk_level": "safe",
  "usage_mode": "retrieval",
  "text": "Clean paragraph-sized text chunk suitable for retrieval.",
  "summary": "One sentence summary of the chunk.",
  "last_checked": "2026-04-27",
  "notes": "Why this chunk is useful."
}
```

Every field is required. Every `id` must be unique.

## 6. Allowed Field Values

### `source_type`

Use one of:

```text
university_resource
crisis_resource
government_public_health
student_support
clinician_review_candidate
```

### `topic`

Use one primary topic per chunk:

```text
crisis_immediate_help
counseling_services
after_hours_support
academic_burnout
advisor_conflict
isolation_loneliness
anxiety_stress
depression_support
accessibility_disability
graduate_student_support
help_seeking_script
grounding_exercise
campus_navigation
therapy_expectations
peer_support
emergency_services
```

If a chunk fits multiple topics, choose the most specific one.

### `audience`

Use one of:

```text
umd_student
graduate_student
student_general
crisis_support
supporter_or_friend
```

### `risk_level`

Use one of:

```text
safe
wellbeing
crisis_resource
exclude
```

Meanings:

- `safe`: normal support/resource text.
- `wellbeing`: distress/help-seeking content, but not crisis.
- `crisis_resource`: crisis resource or urgent-support content.
- `exclude`: do not put this in the final retrieval corpus.

Prefer putting excluded content in `excluded_sources.csv` instead of
`resources_seed.jsonl`.

### `usage_mode`

Use one of:

```text
retrieval
wellbeing_only
crisis_only
metadata_only
```

Meanings:

- `retrieval`: safe for normal RAG retrieval.
- `wellbeing_only`: use only when the user is distressed/help-seeking.
- `crisis_only`: use only when triage detects crisis or emergency.
- `metadata_only`: useful as source/contact metadata, not generation context.

Examples:

- UMD counseling overview: `retrieval`
- 988 immediate crisis guidance: `crisis_only`
- grounding exercise: `wellbeing_only`
- phone-number-only contact page: `metadata_only`

## 7. Chunking Rules

Do not dump full webpages.

Each chunk should be:

- 80-250 words
- focused on one useful idea
- understandable without the full page
- source-cited with URL
- safe, factual, and student-appropriate

Remove:

- nav menus
- footers
- cookie banners
- sidebars
- repeated boilerplate
- irrelevant event calendars
- legal disclaimers with no support value
- raw HTML artifacts

If a useful page has 1,000 words, split it into 4-8 focused chunks.

## 8. Safety Filtering

Do not include content that:

- describes self-harm methods in detail
- gives instructions for suicide or self-harm
- includes graphic personal crisis stories
- makes diagnosis claims
- claims to replace therapy, counseling, or emergency care
- gives medication instructions
- sounds judgmental, stigmatizing, or moralizing
- is outdated, unofficial, or unsupported

For crisis resources, include only safe action-oriented guidance:

- call or text 988
- contact emergency services
- contact campus crisis or after-hours support
- seek immediate support
- stay with another person
- ask someone nearby for help

## 9. Recommended Corpus Size

First usable version:

```text
Minimum: 80 chunks
Good target: 150-250 chunks
Excellent first pass: 300-500 chunks
```

Suggested distribution:

```text
UMD-specific resources: 40-80 chunks
Crisis resources: 20-40 chunks
Government/public health: 40-80 chunks
Graduate/student academic support: 40-80 chunks
Grounding/help-seeking snippets: 20-60 chunks
```

A clean 150-chunk corpus is better than a noisy 1,000-chunk scrape.

## 10. Collection and Scraping Method

Preferred workflow:

1. Manually collect official source URLs.
2. Record every reviewed URL in `source_inventory.csv`.
3. Use simple scraping only for public pages.
4. Save raw extracted text in `raw_pages/` when possible.
5. Manually clean and chunk the useful text.
6. Annotate every chunk with topic, audience, risk level, and usage mode.
7. Validate JSONL before handoff.

Recommended Python tools:

```text
requests
beautifulsoup4
trafilatura
pandas
```

Basic scraping example:

```python
import requests
from bs4 import BeautifulSoup

url = "https://example.edu/page"
html = requests.get(url, timeout=20).text
soup = BeautifulSoup(html, "html.parser")

for tag in soup(["script", "style", "nav", "footer", "header"]):
    tag.decompose()

text = soup.get_text("\n")
lines = [line.strip() for line in text.splitlines() if line.strip()]
clean_text = "\n".join(lines)
```

Do not trust scraper output directly. Every final chunk should be manually
reviewed.

Respect:

- public pages only
- no login-gated content
- no private student information
- no heavy request volume
- robots/terms restrictions when applicable

## 11. Manual Annotation Process

For each chunk, answer:

1. What student problem does this help with?
2. Is this UMD-specific, general student support, or crisis support?
3. Is it safe for normal retrieval?
4. Should it appear only in wellbeing/crisis contexts?
5. Does it contain actionable resource/navigation information?
6. Is the URL official and useful?

Then fill:

```json
"topic": "...",
"audience": "...",
"risk_level": "...",
"usage_mode": "..."
```

LLMs may help draft labels, but a human should approve every final row.

## 12. Compatibility With EmpathRAG

EmpathRAG currently uses a FAISS index plus SQLite metadata.

The current v1 SQLite metadata roughly contains:

```text
id
text
emotion_label
safety_score
source
```

For v2, we will build a separate curated index from `resources_seed.jsonl`.

Field mapping:

```text
id -> metadata ID
text -> embedded retrieval chunk
source_name -> source display name
url -> citation URL
title -> citation title
topic -> retrieval/evaluation filter
risk_level -> safety filter
usage_mode -> routing filter
summary -> compact display/eval text
```

Do not build FAISS unless asked. Deliver clean structured data; integration will
happen in EmpathRAG.

## 13. Current Reddit Corpus Policy

Do not mix Reddit into the curated corpus.

Current Reddit corpus role:

```text
research baseline
ablation comparison
emotion/retrieval experiment
not primary student-facing support source
```

New curated corpus role:

```text
safer student-support retrieval
MSML class demo
future UMD counseling stakeholder discussion
publication-oriented system improvement
```

If you find useful Reddit-like examples, save them separately only if needed for
future research evaluation. Do not include them in `resources_seed.jsonl`.

## 14. Quality Checklist

Before handoff, verify:

- [ ] Every JSONL line is valid JSON.
- [ ] Every row has a unique `id`.
- [ ] Every row has a matching `source_id` in `source_inventory.csv`.
- [ ] Every row has a working `url`.
- [ ] Every row has `source_name`.
- [ ] Every row has one allowed `source_type`.
- [ ] Every row has one allowed `topic`.
- [ ] Every row has one allowed `audience`.
- [ ] Every row has one allowed `risk_level`.
- [ ] Every row has one allowed `usage_mode`.
- [ ] Text chunks are usually 80-250 words.
- [ ] No nav/footer/cookie/sidebar junk remains.
- [ ] No Reddit/social media/random blog content appears.
- [ ] No graphic self-harm details appear.
- [ ] No diagnosis/treatment/medication instructions appear.
- [ ] Crisis resources are marked `crisis_resource` and `crisis_only`.
- [ ] UMD-specific resources are prioritized.
- [ ] `README_corpus_notes.md` is included.

## 15. README Notes Template

In `README_corpus_notes.md`, include:

```text
Corpus creator:
Date:
Total sources reviewed:
Total chunks included:
Total chunks excluded:
Main source domains:
Known limitations:
Sources needing review:
Pages that were hard to scrape:
Content you were unsure about:
Suggested next sources:
```

## 16. Good Example Row

```json
{
  "id": "umd_counseling_services_001",
  "source_id": "src_001",
  "source_name": "UMD Counseling Center",
  "source_type": "university_resource",
  "title": "Counseling Services",
  "url": "https://counseling.umd.edu/",
  "topic": "counseling_services",
  "audience": "umd_student",
  "risk_level": "safe",
  "usage_mode": "retrieval",
  "text": "The University of Maryland Counseling Center provides support services for students dealing with emotional, academic, social, or personal concerns. Students can use counseling resources to talk through stress, relationship concerns, adjustment difficulties, anxiety, depression, and other challenges that may affect wellbeing or academic life. The Counseling Center can also help students understand what type of support may fit their situation and connect them with appropriate campus or community resources.",
  "summary": "Overview of UMD Counseling Center support services for student wellbeing.",
  "last_checked": "2026-04-27",
  "notes": "Useful for routing students toward official campus counseling support."
}
```

## 17. Bad Example Row

Do not include:

```json
{
  "text": "I saw someone on Reddit say they fixed depression by ignoring everyone and taking random supplements..."
}
```

Why this is bad:

- unofficial
- anecdotal
- potentially unsafe
- not student-resource grounded
- no source metadata
- not appropriate for mental-health support retrieval

## 18. Validation Commands

If using Python, validate JSONL with:

```python
import json
from pathlib import Path

path = Path("resources_seed.jsonl")
seen = set()

for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    row = json.loads(line)
    assert row["id"] not in seen, f"duplicate id on line {line_no}"
    seen.add(row["id"])
    for field in [
        "id", "source_id", "source_name", "source_type", "title", "url",
        "topic", "audience", "risk_level", "usage_mode", "text",
        "summary", "last_checked", "notes"
    ]:
        assert row.get(field), f"missing {field} on line {line_no}"

print(f"Valid rows: {len(seen)}")
```

## 19. Final Handoff Standard

The handoff is ready when:

- `resources_seed.jsonl` is valid JSONL.
- `source_inventory.csv` explains every source.
- UMD resources are clearly prioritized.
- Crisis resources are separated by `risk_level` and `usage_mode`.
- No unsafe or noisy web content is included.
- The data can be ingested without further scraping.

Once delivered, EmpathRAG can build:

- curated FAISS index
- curated SQLite metadata
- source-cited retrieval
- safer MSML class demo
- stronger research direction for publication
