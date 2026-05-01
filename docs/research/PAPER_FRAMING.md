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

## Current Limitation

The current class demo uses EmpathRAG Core with a lightweight local ML router. If model artifacts are missing, the system falls back to deterministic routing. Research claims still require Karthik's larger dataset, human review, and careful comparison against V1.
