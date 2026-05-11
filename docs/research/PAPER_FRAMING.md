# Paper Framing

Working title:

EmpathRAG Core: Multi-Turn Safety Evaluation and Guarded Conversational Retrieval for UMD Support Navigation

## Core Story

EmpathRAG V1 was an emotion-aware RAG system. It is useful as a baseline, and its original evaluation work should stay in the paper. But broad empathetic generation creates structural risks for mental-health-adjacent student support:

- generic validation
- ungrounded advice
- weak escalation behavior
- poor distinction between ordinary stress and safety risk
- insufficient source/resource transparency

EmpathRAG Core is the guarded redesign:

- campus-specific support navigation
- four-mode safety ladder
- hybrid ML + rule route classification
- resource registry / service-object filtering
- usage-mode gated retrieval
- trajectory escalation
- output-side guardrail
- Integrated Gradients explanation for safety decisions
- transparent source cards

## Contributions

1. Hybrid lexical + ML safety architecture with explicit mode-tiered escalation.
2. Multi-turn safety evaluation framework comparing open V1 behavior against guarded Core behavior.
3. Integrated Gradients explainability for safety decisions from the DeBERTa guardrail.

## Research Question

Can a guarded conversational RAG architecture reduce missed escalation, inappropriate validation, ungrounded support actions, and unsafe source use compared with open emotion-aware RAG, especially in multi-turn student-support scenarios?

## Baselines

- V1 EmpathRAG with broad/legacy retrieval
- EmpathRAG Core guarded conversational RAG

Optional ablations:

- Core without output guard
- Core without trajectory escalation
- Core without resource registry filtering
- Core rule-only router vs Core hybrid ML router

## Future Work Boundary

NLI-style output groundedness is intentionally cut from the 10-day class sprint.
The current output guard is rule-based and checks missing action, pure
validation, dependency language, harmful agreement, and unsupported resource
claims. A later research version can add NLI between response claims and the
top retrieved service objects, but the live demo should not depend on loading a
heavy NLI model.

## Evaluation Design

### Section 4: V1 Baseline Evaluation

Keep the original V1 results as baseline rigor:

- BERTScore / reference-response evaluation.
- Wilcoxon analysis for emotion-conditioned retrieval.
- Single-turn adversarial safety comparison.

These results establish that V1 was measured seriously before the redesign.

### Eval A: Single-Turn Ablation

Compare:

- rule router
- TF-IDF/logistic router
- RoBERTa route classifier, once Karthik's route-labeled dataset exists
- full hybrid Core system

### Eval B: Multi-Turn Headline Benchmark

Compare V1 open RAG against Core guarded RAG on multi-turn cases:

- slow escalation
- dependency formation
- help rejection
- peer-helper/friend risk
- ambiguous academic idioms
- sycophancy traps
- method-seeking pressure without method details

This is the main paper hook.

## Metrics

- route accuracy
- safety tier accuracy
- missed escalation
- unsafe/method leakage count
- pure validation / no redirect rate
- ungrounded action rate
- retrieval source appropriateness
- latency
- actionability score

## Allowed Claims

- prototype
- safety-aware support navigation
- source-grounded routing
- transparent route/resource surfacing
- multi-turn safety evaluation

## Claims To Avoid

- clinically validated
- treats depression or anxiety
- prevents suicide or reduces fatal incidents
- replaces counseling
- emergency system
- autonomous intervention

## Documented limitations (carried forward from V1 evaluation)

These are real failure modes that survived V1's adversarial evaluation. The Core architecture mitigates each — but the underlying weakness is real and is part of the *motivation* for the redesign, not contradicted by it.

- **Bait-and-switch in the NLI guardrail (40% recall).** Positive openers followed by crisis content fool the V1 DeBERTa NLI guardrail into misclassifying the turn as safe. Single most dangerous documented V1 failure mode. **Mitigation in Core:** trajectory escalation tracker + Stage-1 lexical precheck before NLI + locked-session state when 3 consecutive high-risk turns are observed. The underlying NLI weakness still exists; we just stop relying on it alone.
- **Domain-transfer false positives.** Academic hyperbole ("this thesis is killing me") fires the V1 NLI guardrail at high confidence. Trained on r/SuicideWatch; never saw graduate-student idiom. **Mitigation in Core:** Stage-1 lexical precheck runs first; the model guardrail is a second opinion, not the primary gate. False positives flow into the same Core support-navigation path as legitimate stress, not into emergency intercept.
- **Synthetic-data evaluation ceiling.** Karthik's curated dataset (216/72/72 split, 74 multi-turn scenarios) is structurally different from real student phrasing — code-switching, slang, emoji, abbreviations. **Why this matters for the paper:** report claims should be qualified as "on this evaluation set" rather than absolute. The V3 data request to Karthik asks for real anonymized student turns as the next-step evaluation pull.
- **n = 74 multi-turn statistical power.** CI95 on missed-escalation rate is wide; some headline numbers will need a larger sample. The paper should report CIs, not just point estimates.
- **Route classifier 0.86 ceiling.** Hybrid rule + ML. Remaining 14% are mostly emotional prompts falling to `general_student_support`. RoBERTa fine-tuning planned. **Affects paper claims:** route accuracy isn't the headline metric (safety is), but downstream resource specificity depends on it.
- **No deployed runtime faithfulness metric.** RAGAS / DeepEval produced degenerate scores under a small local judge model. **Architectural compensation:** the resource registry filter + post-rephrase verification act as structural faithfulness guards. NLI-based runtime faithfulness explicitly cut from the class sprint per Future Work Boundary.

## Current Limitation

The current class demo uses EmpathRAG Core with a lightweight local ML router. If model artifacts are missing, the system falls back to deterministic routing. Research claims still require Karthik's larger dataset, human review, and careful comparison against V1.

## Phase 8: Controlled Paraphrasing (plan-and-rephrase)

### Why a paraphrasing layer at all

V1 used open emotion-aware RAG, where a generative model authored entire responses conditioned on retrieved content. This gave flexibility but exposed the structural risks the V2 thesis names: ungrounded advice, sycophantic validation, weak escalation, and resource fabrication.

Core's first iteration cut the LLM out of the response path entirely and used deterministic templates. This eliminated the V1 risks but produced responses that felt mechanical, repetitive, and often less validating than the user's distress warranted. The concern is real: a clinically-defensive system that students disengage from is no safer than no system at all.

Phase 8 introduces **plan-and-rephrase** as the architectural compromise. The deterministic planner remains the trust boundary: it decides *what* to say (route, stage, recommended action, named resources, F-1 sub-topic). An LLM only paraphrases the planner's authored response into warmer prose that mirrors the user's words. The LLM cannot invent advice, resources, phone numbers, or claims.

### The system prompt as trust boundary

The contract between the planner and the paraphrasing LLM is the system prompt (`src/pipeline/rephraser.py:SYSTEM_PROMPT`). It mandates:

- Keep the same meaning, structure, and recommendations as the input.
- Match the input's length closely.
- Mirror specific phrases from the user's message — including specific events, fears, named people, and time anchors — rather than abstracting to generic framings.
- Start with the planner's first idea, not a filler preamble.
- Output only the rephrased response. No preamble, quotes, or explanation.

It explicitly forbids:

- New advice, resources, phone numbers, or claims not in the input.
- Introducing a UMD resource by name (ISSS, ADS, Counseling Center, Ombuds, etc.) if the input did not already name it.
- Reframing the user's emotion ("you're catastrophizing", "you have anxiety").
- AI-tells (em-dashes, "I understand", "let me reframe", "I'm here to listen", "as your therapist").
- Promises of availability ("I'm always here").
- Diagnosis, prescription, clinical positioning.
- Toxic positivity ("everything happens for a reason", "look on the bright side").
- Minimization of stated fears ("don't worry", "it's not that bad", "it's a bit more nuanced").

### Provider chain with fail-loud semantics

The orchestrator (`ResponseRephraser`) tries providers in order:

1. **GroqProvider** — Llama 3.3 70B versatile via Groq's OpenAI-compatible API. Primary path; ~600ms per turn.
2. **AnthropicProvider** — Claude Haiku 4.5 via the Messages API. Fallback if Groq fails or rejects on safety grounds.
3. **MockProvider** — testing only.

`DeterministicProvider` is intentionally **not** in the iterating list. If it were, it would always succeed and mask any real LLM failure as if the user had selected deterministic mode. Instead, when every LLM provider fails, the orchestrator returns `provider_name="deterministic_fallback"` with the actual last error surfaced in diagnostics, so failures are not silent.

### Post-rephrase safety verification

Every paraphrased candidate is checked by `verify_rephrased_safety` before being surfaced. Checks:

- **Length sanity** — over 2× template length suggests the LLM started generating new content.
- **Scope drift patterns** — clinical positioning, diagnosis claims, prescription language.
- **V1-regression patterns** — toxic positivity, minimization, promises of availability.
- **Phone-number grounding** — any 10-digit US number that isn't 988 must appear in the original template or retrieved sources.
- **Resource-name fabrication** — branded UMD names (Counseling Center, ADS, ISSS, Ombuds, etc.) must be grounded in the template or retrieved sources.

A failed verification falls through to the next provider, then to the deterministic template. The user always sees a safe response.

### Streaming with safety preserved

For UX, the demo streams Groq's tokens directly to the chat surface as they arrive (Server-Sent Events). The post-rephrase safety check runs on the accumulated text after the stream ends; if the check fails, the stream's contents are replaced with the next provider's candidate or the deterministic template. The output guard runs at the OFFER stage on the final text. This preserves first-token latency (~300-500ms) while keeping the safety floor intact.

### Empirical validation

- **Eval B with rephraser ON** (74 multi-turn scenarios, 28 escalation): missed_escalation = 0, unsafe_generation = 0, ungrounded_action = 0. Pure-validation count actually drops vs. deterministic-only (13 → 8) because natural-language paraphrase contains more action verbs than the rigid templates.
- **Drift sweep** (29 cells, 14 routes × 3 stages × the F-1 sub-topic suite): zero drift flags across length blowup, AI-tells, scope drift, minimization, resource fabrication. Mean rephrased-to-template length ratio 0.97 (output approximately the same length as the planner's authored template, sometimes slightly shorter).
- **F-1 stage × ISSS contract** (12 cells): ISSS named at PERMISSION + OFFER but never at LISTEN; F-1 mechanics (RCL, SEVIS, OPT, CPT, I-20, reinstatement) appear only at OFFER; mirroring of user-supplied terms allowed at all stages.

### What this layer does and does not contribute

It contributes:

- A reproducible architectural pattern for adding LLM warmth to a deterministic safety pipeline without surrendering the safety floor.
- A measurable, observable trust boundary (system prompt + post-rephrase verification) that can be audited and tightened against drift.
- An honest fallback path (`deterministic_fallback` with `last_error`) that does not silently degrade to looking like deterministic mode.

It does not contribute:

- Better routing accuracy (route classification is upstream of the rephraser).
- Better safety classification (the four-tier ladder runs before the rephraser).
- Reliable handling of out-of-scope inputs by itself (the rephraser will faithfully paraphrase whatever the planner authored, which depends on correct routing).

### Honest naming

We call this **controlled paraphrasing** rather than "guarded generation" or "constrained generation." The model is not generating. It is paraphrasing a fixed, planner-authored response under a strict system-prompt contract, with a runtime safety check on every output. The distinction matters: it is what allows us to say the LLM cannot invent advice without overclaiming.
