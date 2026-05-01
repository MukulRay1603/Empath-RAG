# Karthik Task: EmpathRAG Core Dataset V2

We are consolidating the project into one system: **EmpathRAG Core**, a guarded conversational RAG assistant for emotional/student support navigation. UMD remains the main case study, but the schema should be reusable for other campus/community profiles later.

Please create a new folder:

```text
empathrag_core_dataset_v2/
```

## Required Files

```text
README_dataset_notes.md
single_turn_labeled.csv
multi_turn_scenarios.jsonl
source_target_map.csv
risky_ambiguous_cases.csv
resource_profile_additions.csv
```

## 1. `single_turn_labeled.csv`

Target size: **300-500 synthetic prompts**.

Required columns:

```text
query_id
query_text
audience_mode
route_label
safety_tier
should_intercept
expected_usage_modes
preferred_source_names
avoid_source_names
preferred_topics
expected_response_action
tricky_flags
split
notes
```

Valid `audience_mode`:

```text
student
helping_friend
```

Valid `route_label`:

```text
academic_setback
exam_stress
accessibility_ads
advisor_conflict
counseling_navigation
basic_needs
care_violence_confidential
peer_helper
loneliness_isolation
anxiety_panic
low_mood
crisis_immediate
general_student_support
out_of_scope
```

Valid `safety_tier`:

```text
imminent_safety
high_distress
support_navigation
wellbeing
```

Valid `split`:

```text
train
dev
test
```

Suggested distribution:

- 60% train
- 20% dev
- 20% test
- At least 20 examples per major route where possible.
- At least 50 risky or ambiguous prompts.
- At least 40 helping-friend prompts.
- At least 30 out-of-scope prompts.

Rules:

- Use synthetic examples only.
- Do not use real student posts, Reddit, TikTok, Discord, private chats, clinical notes, or scraped personal stories.
- Do not include self-harm method details.
- For crisis examples, signal risk without operational/graphic details.
- Keep prompts realistic: short, messy, and student-like.

## 2. `multi_turn_scenarios.jsonl`

Target size: **50 scenarios**, each 3-6 turns.

Schema:

```json
{
  "case_id": "slow_escalation_001",
  "audience_mode": "student",
  "turns": [
    {
      "user": "...",
      "expected_route": "...",
      "expected_safety_tier": "...",
      "should_intercept": false
    }
  ],
  "must_trigger_by_turn": 4,
  "must_not": ["dependency_reinforcement", "pure_validation", "method_leakage"],
  "notes": "..."
}
```

Include:

- slow escalation
- dependency language
- helping a friend
- ambiguous idioms
- ordinary academic stress
- basic needs
- ADS/accommodations
- advisor conflict
- low mood without immediate risk
- explicit imminent safety

## 3. `source_target_map.csv`

Required columns:

```text
need_id
user_need
preferred_topics
preferred_source_names
avoid_source_names
notes
```

Purpose:

This file defines what source families are appropriate for each need. It helps evaluate source appropriateness, not just route accuracy.

## 4. `risky_ambiguous_cases.csv`

Required columns:

```text
case_id
query_text
why_it_is_tricky
correct_risk_category
should_intercept
expected_handling
```

Include cases like:

- academic idioms that sound dangerous but are not literal
- third-person risk from a friend/roommate
- vague hopelessness without immediate plan
- method-seeking phrased indirectly, but without method details
- secrecy/dependency language
- urgent distress that should not be handled as ordinary chat

## 5. `resource_profile_additions.csv`

Collect official resource metadata only.

Required columns:

```text
resource_name
resource_type
official_url
source_authority
route_labels
safety_tiers
usage_modes
audience
contact_mode
contact_value
hours
location
confidentiality_status
last_verified
notes
```

Rules:

- Use `unknown` for missing contact, hours, location, or confidentiality.
- Do not invent phone numbers.
- Do not invent hours.
- Do not invent eligibility.
- Prefer official university, government, or reputable nonprofit sources.
- UMD is the main case study for now.
- Optional: add general resource categories that make the schema reusable elsewhere, but do not collect multiple campuses yet unless asked.

## README Requirements

`README_dataset_notes.md` must include:

- creator
- date
- row counts by file
- route distribution
- safety tier distribution
- intercept distribution
- source collection rules
- privacy statement
- known limitations
- anything uncertain

## Final Quality Checklist

Before sending back:

- Every CSV opens cleanly as UTF-8.
- All required columns are present.
- IDs are unique.
- All route labels are from the allowed list.
- All safety tiers are from the allowed list.
- No real student/private data.
- No method details.
- Crisis examples are safe but still labelable.
- `split` is populated for every single-turn row.
- Sources are official or clearly marked as unknown/needs review.
