# EmpathRAG Core Master Checklist

Date: 2026-05-01  
Active branch: `codex/v2.5-support-navigator`  
Status: one consolidated product direction; V1/V2/V2.5 are baselines/checkpoints only.

## Core Thesis

EmpathRAG started as emotion-aware RAG for student mental-health-adjacent support. Evaluation exposed structural risks in open empathetic generation: missed escalation, sycophantic validation, dependency reinforcement, and ungrounded advice.

EmpathRAG Core is the guarded redesign: a UMD-specific conversational RAG system for support navigation, not therapy, diagnosis, counseling, crisis prevention, or clinical care.

## Named Contributions

1. Hybrid lexical + ML safety architecture with four-mode escalation.
2. Multi-turn safety evaluation framework comparing V1 open RAG against Core guarded RAG.
3. Integrated Gradients explainability for safety decisions from the DeBERTa guardrail.

Supporting engineering contributions:

- UMD-specific resource registry with service objects and verified provenance.
- Output-side groundedness and anti-sycophancy guard.
- Peer-helper mode for "I am worried about someone else" scenarios.

## Phase 1: Resource Registry

- [x] Preserve curated UMD corpus and source metadata.
- [x] Add profile path scaffold under `data/profiles/umd/`.
- [ ] Rename paper/docs wording from "service graph" to "resource registry", "service objects", or "service-tagged resource schema".
- [ ] Expand UMD service objects toward roughly 80 verified entries if time allows.
- [ ] Add/verify UMD-specific high-value resources:
  - [ ] Counseling Center
  - [ ] After-hours crisis support
  - [ ] ADS
  - [ ] Graduate Ombuds
  - [ ] Graduate School support
  - [ ] Campus Pantry / basic-needs support if verified
  - [ ] CARE to Stop Violence if verified
  - [ ] MHEART if verified
- [ ] Use `unknown` for missing hours/location/contact; never invent details.

## Phase 2: Core Pipeline

- [x] Add unified `EmpathRAGCore.run_turn(...)` runtime interface.
- [x] Keep fast demo startup reliable.
- [x] Add TF-IDF/logistic route and tier classifier scaffold.
- [x] Keep deterministic fallback when model artifacts are missing.
- [x] Add output metadata for route, tier, confidence, retrieval mode, output guard, and latency.
- [ ] Make hard lexical safety precheck an explicit Stage 1 before any ML classifier.
- [ ] Optionally call the real DeBERTa safety guardrail for deeper eval and explainability.
- [ ] Keep Integrated Gradients available for safety decisions without blocking the live demo.
- [ ] Add RoBERTa route classifier only after labeled route dataset exists.

Router ablation plan:

- [x] Rule router baseline.
- [x] TF-IDF/logistic router baseline.
- [ ] RoBERTa route classifier trained on Karthik's labeled synthetic data.
- [ ] Full hybrid system: lexical safety + ML routing + resource registry retrieval + output guard.

## Phase 3: Safety And Output Gates

- [x] Four safety tiers:
  - `imminent_safety`
  - `high_distress`
  - `support_navigation`
  - `wellbeing`
- [x] Deterministic crisis/imminent response path.
- [x] Rule-based output guard catches pure validation, harmful agreement, dependency language, and unsupported resource claims.
- [ ] Add explicit `out_of_scope` handling in response planner and UI.
- [ ] Add NLI-style groundedness check for response claims against top retrieved service objects when feasible.
- [ ] Ensure crisis/imminent prompts never enter normal generation.

## Phase 4: Response Layer And Demo

- [x] Constrained response planner for non-crisis responses.
- [x] Recommended next action included in Core result metadata.
- [x] Gradio demo shows route, tier, confidence, retrieval mode, source cards, and output guard.
- [ ] Make peer-helper mode a first-class visible feature, not only a route label.
- [ ] Add peer-helper-specific wording:
  - what to say
  - what not to say
  - when to escalate
  - do not handle immediate safety alone
- [ ] Polish 5-6 scripted demo prompts.
- [ ] Keep demo framed as support navigation, not clinical help.

## Phase 5A: Single-Turn Ablation

Goal: broad comparison across router/system variants.

Variants:

- [ ] Rule router
- [ ] TF-IDF/logistic router
- [ ] RoBERTa route classifier
- [ ] Full hybrid Core system

Metrics:

- route accuracy
- safety tier accuracy
- intercept accuracy
- source appropriateness / source hit rate
- avoid-list violation rate
- unsafe generation count
- pure validation / no-action count
- ungrounded action count
- latency

Target dataset:

- 300-500 synthetic single-turn prompts.
- 14 route labels.
- 4 safety tiers.
- no real/private student data.
- no Reddit/TikTok/Discord scraping.
- no method details.

## Phase 5B: Multi-Turn Headline Evaluation

This is the paper hook.

Goal: compare V1 open/emotion-aware RAG against EmpathRAG Core on vulnerability-amplifying conversations.

Scenario categories:

- slow escalation
- dependency formation
- help rejection
- secrecy requests
- peer-helper/friend risk
- ambiguous academic idioms
- sycophancy traps
- method-seeking pressure without method details
- ordinary academic stress

Metrics:

- missed escalation count
- turn-by-turn route correctness
- turn-by-turn tier correctness
- intercept by required turn
- dependency reinforcement
- pure validation without redirect
- method leakage
- unsafe source use
- latency

Minimum target:

- 30 multi-turn scenarios for class demo/paper draft.
- 50 multi-turn scenarios for stronger research version.

## Phase 6: V1 Results Preservation

Do not orphan the original V1 work. It becomes baseline rigor.

- [ ] Keep BERTScore and Wilcoxon results in the paper as V1 baseline evaluation.
- [ ] Preserve adversarial safety evaluation from V1.
- [ ] Use V1 results to motivate the Core redesign.
- [ ] Present V1 as useful but structurally risky, not as "bad work."

Paper structure:

- Section 4: V1 baseline evaluation.
- Section 5: what V1 missed.
- Section 6: EmpathRAG Core architecture.
- Section 7: Eval A single-turn ablation and Eval B multi-turn V1 vs Core benchmark.

## Phase 7: Karthik Dataset Deadline

Deadline: end of Day 5 from the current sprint.

Blocking deliverables:

- [ ] `single_turn_labeled.csv`: 300-500 synthetic prompts.
- [ ] `multi_turn_scenarios.jsonl`: at least 30 scenarios, target 50.
- [ ] `source_target_map.csv`.
- [ ] `risky_ambiguous_cases.csv`.
- [ ] `resource_profile_additions.csv`.
- [ ] `README_dataset_notes.md`.

Setup/verification tasks for Karthik if he runs the repo locally:

- [ ] Pull latest `codex/v2.5-support-navigator`.
- [ ] Verify tokenizer/model dependencies.
- [ ] Install missing protobuf dependency if tokenizer loading requires it.
- [ ] Verify `src/models/guardrail_ig.py` runs locally.

Fallback if dataset slips:

- [ ] Build a smaller hand-labeled set of about 150 prompts with Karthik.
- [ ] Train/evaluate TF-IDF and, if feasible, a small RoBERTa route classifier.
- [ ] Report small-N limitation honestly.
- [ ] Do not block the demo on RoBERTa route training.

## Claims To Avoid

- Clinically validated.
- Treats depression or anxiety.
- Provides therapy, counseling, diagnosis, or intervention.
- Prevents suicide or reduces fatal incidents.
- Replaces professional or campus support.
- Safe for real student deployment.

## Immediate Next Build Tasks

- [ ] Update docs to consistently use EmpathRAG Core and resource registry wording.
- [ ] Add explicit Stage-1 lexical safety precheck metadata to Core output.
- [ ] Add optional DeBERTa + IG hook to Core, disabled or skipped for fast demo when needed.
- [ ] Improve peer-helper UI and tests.
- [ ] Split eval reports into Eval A and Eval B summaries.
- [ ] Keep demo polished and deterministic for the class presentation.
