# Paper Framing

Working title:

EmpathRAG Core: Guarded Conversational Retrieval for Emotional Support Navigation

## Core Story

EmpathRAG V1 was an emotion-aware RAG system. It is useful as a baseline, but broad empathetic generation creates structural risks for mental-health-adjacent student support:

- generic validation
- ungrounded advice
- weak escalation behavior
- poor distinction between ordinary stress and safety risk
- insufficient source/resource transparency

EmpathRAG Core pivots to a safer architecture:

- campus-specific support navigation
- four-mode safety ladder
- hybrid ML + rule route classification
- service graph filtering
- usage-mode gated retrieval
- trajectory escalation
- output-side guardrail
- transparent source cards

## Research Question

Can a hybrid ML/rule router with graph-grounded retrieval and hard safety gates reduce inappropriate validation, ungrounded actions, and missed escalation while improving route accuracy and actionability compared with ungated RAG or generic LLM responses?

## Baselines

- V1 EmpathRAG with broad/legacy retrieval
- EmpathRAG Core guarded conversational RAG

Optional ablations:

- Core without output guard
- Core without trajectory escalation
- Core without service graph filtering
- Core rule-only router vs Core hybrid ML router

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
