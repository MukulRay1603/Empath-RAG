# Paper Framing

Working title:

Trajectory-Safe, Graph-Grounded Student Support Navigation for Campus Mental Health Resources

## Core Story

EmpathRAG V1 was an emotion-aware RAG system. It is useful as a baseline, but broad empathetic generation creates structural risks for mental-health-adjacent student support:

- generic validation
- ungrounded advice
- weak escalation behavior
- poor distinction between ordinary stress and safety risk
- insufficient source/resource transparency

EmpathRAG V2.5 pivots to a safer architecture:

- campus-specific support navigation
- four-mode safety ladder
- route classification
- service graph filtering
- usage-mode gated retrieval
- trajectory escalation
- output-side guardrail
- transparent source cards

## Research Question

Can a trajectory-aware, campus-specific service graph with hard safety gates reduce inappropriate validation, ungrounded actions, and missed escalation while improving route accuracy and actionability compared with ungated RAG or generic LLM responses?

## Baselines

- V1 EmpathRAG with broad/legacy retrieval
- V2.5 curated support navigator

Optional ablations:

- V2.5 without output guard
- V2.5 without trajectory escalation
- V2.5 without service graph filtering

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

The current V2.5 class demo uses a deterministic fast backend. That is appropriate for reliability and transparent behavior, but research claims will require a stronger evaluation dataset, human review, and careful comparison against V1.
