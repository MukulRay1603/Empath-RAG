# EmpathRAG Core Dataset Request For Karthik

Hi Karthik,

We have now locked the project direction as **EmpathRAG Core**.

EmpathRAG Core is a guarded conversational RAG system for UMD-style support navigation. It is not therapy, diagnosis, counseling, clinical care, crisis prevention, or an emergency service. The research goal is to compare open emotion-aware RAG against a guarded system that uses safety triage, resource-grounded retrieval, output checks, and multi-turn escalation.

Your dataset is now the main blocking item for training and evaluating the stronger route classifier.

## Deadline

Please send the full delivery by **end-of-day Wednesday, May 6, 2026**.

If the full set is not ready by then, please send a clean partial delivery instead:

- at least 150 single-turn prompts
- at least 15 multi-turn scenarios
- complete labels for route, safety tier, intercept, and source expectations
- a short note explaining what remains incomplete

We can use the partial set for a fallback small-N ablation and continue expanding later.

## Folder To Send Back

Please create:

```text
empathrag_core_dataset_v2/
```

Required files:

```text
README_dataset_notes.md
single_turn_labeled.csv
multi_turn_scenarios.jsonl
source_target_map.csv
risky_ambiguous_cases.csv
resource_profile_additions.csv
```

## File 1: `single_turn_labeled.csv`

Target: **300-500 synthetic prompts**.

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

- 60% train, 20% dev, 20% test
- at least 20 examples per major route where possible
- at least 50 risky or ambiguous prompts
- at least 40 helping-friend prompts
- at least 30 out-of-scope prompts

## File 2: `multi_turn_scenarios.jsonl`

Target: **30 scenarios minimum**, ideally **50 scenarios**.

Each scenario should have 3-6 turns.

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

Include scenarios for:

- slow escalation
- dependency language
- helping a friend or roommate
- ambiguous academic idioms
- ordinary academic stress
- food/basic needs
- ADS/accommodations
- advisor conflict
- low mood without immediate danger
- imminent safety handoff
- sycophancy traps
- help-rejection and secrecy requests

## File 3: `source_target_map.csv`

Required columns:

```text
need_id
user_need
preferred_topics
preferred_source_names
avoid_source_names
notes
```

Purpose: define which source families are appropriate for each user need so we can evaluate source appropriateness, not just classifier accuracy.

## File 4: `risky_ambiguous_cases.csv`

Required columns:

```text
case_id
query_text
why_it_is_tricky
correct_risk_category
should_intercept
expected_handling
```

Include:

- academic idioms that sound dangerous but are not literal
- third-person risk from a friend or roommate
- vague hopelessness without immediate danger
- indirect method-seeking pressure, without method details
- secrecy/dependency language
- urgent distress that should not be handled like ordinary chat

## File 5: `resource_profile_additions.csv`

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
- Do not invent eligibility rules.
- Prefer official university, government, or reputable nonprofit pages.
- UMD remains the main profile for now.

## Strict Data Rules

- Use synthetic examples only.
- Do not use real student posts, Reddit, TikTok, Discord, private chats, clinical notes, or scraped personal stories.
- Do not include self-harm method details.
- Crisis examples should signal risk without operational or graphic details.
- Keep prompts realistic: short, messy, and student-like.
- Every CSV must open cleanly as UTF-8.
- IDs must be unique.
- All route labels and safety tiers must match the allowed lists exactly.

## README Requirements

`README_dataset_notes.md` should include:

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

## Optional Repo Setup Check

If you are running the repo locally:

- Pull latest `codex/v2.5-support-navigator`.
- Verify tokenizer/model dependencies.
- Install `protobuf` if tokenizer loading complains.
- Try running `src/models/guardrail_ig.py`.
- If it fails, note the exact error in `README_dataset_notes.md`.

Thanks. This dataset is the key thing that lets us train and evaluate the stronger EmpathRAG Core route classifier without using real student data.
