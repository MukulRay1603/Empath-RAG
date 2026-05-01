# EmpathRAG v2 Safety and Dataset Plan

EmpathRAG v1 is a research prototype. EmpathRAG v2 should be treated as a
mental-health-adjacent student support system, not as a general chatbot. The
goal is to make the system useful for research publication and eventually
credible enough to discuss with university counseling stakeholders.

## Safety Position

EmpathRAG must not diagnose, provide therapy, or replace emergency care. Its
job is to:

- Reflect student emotion accurately and gently.
- Retrieve safe, relevant support context.
- Encourage appropriate help-seeking when risk is elevated.
- Escalate clearly when language suggests self-harm, imminent danger, or an
  attempt.
- Produce auditable safety metadata for each turn.

The safety layer should be evaluated as a triage system with multiple levels:

- `pass`: no safety signal beyond normal supportive response.
- `wellbeing_support`: elevated distress or help-seeking, but no clear crisis.
- `crisis`: self-harm or suicidal ideation indicators.
- `emergency`: attempt, plan, imminent timing, method, or inability to stay safe.

Binary crisis detection alone is not enough. In v1 adversarial results, the
guardrail showed high recall on direct crisis phrasing but very high
false-positive rates on academic stress and indirect help-seeking prompts. v2
should report calibration curves, per-category recall, per-category false
positive rates, and threshold tradeoffs.

## Dataset Direction

The retrieval corpus should move away from raw Reddit as the primary support
source. Reddit can remain a research comparison corpus, but it is noisy,
unmoderated, and may include unsafe, stigmatizing, or contagion-prone text.

Preferred v2 corpus tiers:

1. University-facing resources
   - UMD Counseling Center public pages.
   - UMD crisis resources and after-hours support pages.
   - Accessibility and Disability Service public guidance.
   - Graduate School wellbeing, ombuds, and academic support resources.

2. Clinician-reviewed public educational content
   - 988 Lifeline public guidance.
   - NIMH educational pages.
   - SAMHSA public resources.
   - CDC suicide prevention public resources.

3. Structured coping and navigation snippets
   - Short grounding exercises.
   - Help-seeking scripts.
   - Advisor conflict navigation.
   - Academic burnout and isolation support.
   - Campus resource routing templates.

4. Research-only comparison corpora
   - Reddit Mental Health.
   - Empathetic Dialogues.
   - GoEmotions.
   - Suicide Detection.

The production-facing retrieval index should use tiers 1-3. Tier 4 should be
used for training, ablation, and benchmarking only unless a clinician-reviewed
filter approves individual snippets.

## Data Governance

Before any real student deployment or UMD stakeholder pilot:

- Get IRB guidance before collecting student conversations.
- Do not log raw user text by default.
- If logging is approved, store minimum necessary data, encrypt at rest, and set
  a retention period.
- Separate research IDs from user identity.
- Add a visible consent statement for studies.
- Create a deletion pathway for participants.
- Document dataset licenses and redistribution limits.

## Evaluation Gaps To Close

Safety:

- Crisis recall by category: direct, euphemistic, negated, third-person,
  historical, sarcastic, academic idiom, imminent attempt.
- False-positive rate by benign category: academic stress, joking/hyperbole,
  help-seeking, resource questions, quoted text.
- Multi-turn escalation tests where risk appears after neutral openers.
- Calibration plots and threshold selection rationale.

Retrieval:

- Manual safety audit of top retrieved chunks.
- Source whitelist and source citation metadata.
- Chunk-level labels for safe, caution, unsafe, and crisis-resource-only.
- Tests that crisis-like retrieved text is not used to intensify responses.

Generation:

- Human ratings for empathy, helpfulness, specificity, safety, and overreach.
- Clinician or counselor review for high-risk response templates.
- Tests that the model does not claim to be a therapist, diagnose, or promise
  confidentiality.

Research:

- Pre-register the evaluation protocol if aiming for publication.
- Report negative results and failure modes.
- Compare against simple baselines honestly: keyword filter, dense RAG, no-RAG,
  and a safety-template-only system.

## Near-Term V2 Implementation Checklist

- Fail closed if the real guardrail checkpoint is missing.
- Add triage levels and return safety metadata on every turn.
- Disable public demo sharing by default.
- Disable raw text logging by default.
- Split demo session state per user.
- Add a curated resource-ingestion path separate from Reddit ingestion.
- Add retrieval source metadata and citations.
- Add a dataset card and model card.
- Add red-team tests to CI that do not require Mistral generation.
