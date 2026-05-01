# EmpathRAG Current Status Audit

Date: 2026-05-01  
Branch: `codex/v2.5-support-navigator`  
Audience: project team, research planning, MSML demo planning  
Status: EmpathRAG Core consolidation in progress

## 1. One-Line Summary

EmpathRAG Core is a guarded conversational RAG system for UMD-style support navigation. It is not a therapist, diagnostician, counselor, crisis-prevention system, or emergency service; it should route student concerns to appropriate support paths, show grounded sources, and fail closed when safety risk appears.

The current research hook is not "we made a chatbot." The hook is: V1 open empathetic RAG is evaluated as a baseline, its multi-turn safety failure modes motivate the redesign, and Core is evaluated as a guarded alternative.

## 2. Overall Idea

The original idea was a mental-health-adjacent RAG system that can respond empathetically to student distress while grounding its answers in relevant resources.

The refined Core idea is more precise:

- Detect the type of student-support need.
- Detect whether the message is ordinary support, wellbeing/grounding, crisis, or emergency.
- Retrieve only resources allowed for that safety mode.
- Produce short, practical, source-grounded responses.
- Show the user why the system retrieved each source.
- Avoid pretending to provide therapy, diagnosis, crisis counseling, or clinical authority.

This framing is much stronger for class presentation and research because it turns the project from a generic chatbot into a safety-aware, source-grounded support navigator.

## 2A. Merged Reviewer Consensus

Keep these decisions locked:

- One active product direction: EmpathRAG Core.
- V1/V2/V2.5 remain checkpoints and baselines, not separate products.
- UMD specificity remains the main demo moat.
- Use synthetic labeled data only for route/safety training and evaluation.
- Do not scrape Reddit, TikTok, Discord, private chats, or real student stories for the new dataset.
- Keep V1 BERTScore/Wilcoxon/adversarial work as baseline evaluation.
- Make multi-turn V1 vs Core safety evaluation the headline result.
- Keep Integrated Gradients explainability as a named contribution.
- Use "resource registry" or "service objects" in papers/presentations, not "service graph" unless graph edges/traversal are added.
- Keep the TF-IDF/logistic router as a fast scaffold and baseline; train a RoBERTa route classifier only after Karthik delivers labeled route data.

## 3. What We Have Right Now

### Demo Application

File: `demo/app.py`

The Gradio demo currently runs at:

`http://127.0.0.1:7860`

Current demo mode:

- `EMPATHRAG_DEMO_BACKEND=fast`
- `EMPATHRAG_RETRIEVAL_CORPUS=curated_support`
- logging off by default
- 177 cleaned curated support chunks available locally

The demo now has:

- Dark/turquoise V2 visual design.
- Session ID display.
- Conversation panel.
- Emotion timeline panel.
- Safety guardrail panel.
- Retrieval/source panel.
- Visible support route panel.
- Curated source cards with title, source, topic, risk level, usage mode, source type, and reason retrieved.
- Prompt buttons for common presentation cases.

### Fast Presentation Backend

The fast backend was added because the full local model path can stall during model loading and is not reliable enough for a live class demo.

The fast backend does not use the full LLM stack. It demonstrates the intended V2 behavior using:

- curated corpus metadata from SQLite
- deterministic safety triage
- route-specific response templates
- usage-mode gated retrieval
- source explanations

Current supported routes include:

- `academic setback`
- `stress overload`
- `low mood`
- `accessibility`
- `advisor conflict`
- `counseling navigation`
- `anxiety`
- `immediate safety`
- general `student-support`

Recently verified examples:

- Prompt: `Life is depressing, I failed my exam!`
  - Route: `academic setback`
  - Behavior: gives a specific next-step plan instead of generic counseling text.

- Prompt: `I need ADS accommodations for exams`
  - Route: `accessibility`
  - Behavior: routes to ADS/accommodations support rather than misreading it as exam failure.

- Prompt: `I do not think I can stay safe tonight.`
  - Route: `immediate safety`
  - Behavior: normal generation is intercepted and crisis-only resources are shown.

### Real Pipeline Backend

Files:

- `src/pipeline/pipeline.py`
- `src/pipeline/safety_policy.py`

The real pipeline still exists and has been upgraded with V2 safety logic:

- retrieval corpus modes:
  - `reddit_research`
  - `curated_support`
  - `auto`
- curated corpus selection when curated index exists
- fail-closed safety policy
- safety levels:
  - `pass`
  - `wellbeing_support`
  - `crisis`
  - `emergency`
- usage-mode gated retrieval:
  - normal support: `retrieval`
  - wellbeing support: `retrieval` + `wellbeing_only`
  - crisis/emergency: `crisis_only`
- retrieved source metadata returned to the UI
- normal generation blocked for crisis/emergency cases

The real model path is currently not presentation-safe because model loading/cache/network behavior can hang or take too long locally.

## 4. Dataset / Corpus Status

### Karthik V2 Delivery

Folder:

`Data_Karthik/v2`

Included files:

- `README_corpus_notes.md`
- `source_inventory.csv`
- `excluded_sources.csv`
- `resources_seed.jsonl`
- `raw_pages/`

The revised corpus was much better than the first version. We then performed a local cleanup pass and built the active curated corpus.

### Active Cleaned Corpus

Active local corpus:

`data/curated/resources_seed.jsonl`

Current row count:

177 chunks

Active curated index/database:

- `data/curated/indexes/faiss_curated.index`
- `data/curated/indexes/metadata_curated.db`

Important note:

`data/curated/` is generated/local data and is ignored by git. The source delivery from Karthik lives under `Data_Karthik/`, which is currently untracked.

### Corpus Topics Covered

The corpus is centered around:

- UMD Counseling Center
- UMD Accessibility & Disability Service
- UMD Graduate School
- UMD Graduate School Ombuds
- 988 Suicide & Crisis Lifeline
- SAMHSA
- NIMH
- CDC
- curated/internal support content

Covered support areas include:

- counseling services
- crisis/immediate help
- accessibility/disability support
- graduate student support
- advisor conflict
- academic burnout
- anxiety/stress
- depression support
- grounding exercises
- campus navigation
- help-seeking scripts

### Remaining Corpus Concerns

The corpus is usable for a class demo, but not yet publication-ready.

Known concerns:

- Some sources are still broad clinical/public-health resources rather than student-specific support.
- Some retrieval topics are too coarse.
- Some rows may still surface imperfectly for vague prompts.
- Source relevance needs systematic evaluation.
- Human review is needed before any real student-facing use.
- Provenance and licensing should be reviewed before publication.

## 5. What We Planned

### Short-Term Class Demo Plan

Goal:

Show a polished, reliable, meaningful prototype within roughly 10 days.

Class demo strategy:

- Use the fast curated backend for reliability.
- Present the system as a safety-aware student-support router.
- Avoid claiming it is a therapist or clinically validated tool.
- Show 4-5 scripted scenarios:
  - normal counseling navigation
  - failed exam / academic setback
  - ADS accommodations
  - advisor conflict
  - crisis redirect
- Emphasize that source grounding, safety gating, and transparent routing are the meaningful parts.

Why this is the right demo strategy:

The full local model path is too risky for live presentation. A fast, honest, curated support-router demo is better than a slow LLM demo that hangs.

### Medium-Term V2 Plan

Goal:

Make the project technically stronger and more defensible.

Planned V2 work:

- Improve route detection.
- Improve retrieval ranking.
- Add route confidence and source confidence.
- Add a clearer distinction between support routing, wellbeing exercises, and crisis-only behavior.
- Add better evaluation scripts.
- Integrate Karthik's evaluation dataset once ready.
- Add latency measurements.
- Add regression tests for safety and retrieval behavior.
- Improve source card quality.
- Make demo outputs less generic and more task-oriented.

### Research-Oriented Plan

Goal:

Move from class prototype toward publishable research.

Research direction:

- Frame the project as safety-aware RAG for student-support navigation.
- Compare retrieval modes and safety policies.
- Evaluate source relevance, safety behavior, crisis routing, and response helpfulness.
- Use expert/human review for mental-health-adjacent safety.
- Avoid claiming clinical effectiveness unless evaluated by qualified reviewers.

Possible research questions:

- Does usage-mode gated retrieval reduce unsafe or inappropriate resource surfacing?
- Does explicit support-route classification improve perceived helpfulness?
- Does fail-closed crisis routing reduce unsafe generation?
- How does curated campus/public-health retrieval compare to broad Reddit-style retrieval?
- What failure modes appear under ambiguous distress prompts?

## 6. Critical Ways This Can Break

### 1. Full Model Backend Can Hang

The real local model path can stall during model loading, cache access, or network calls. This is the biggest live-demo risk.

Mitigation:

- Use `EMPATHRAG_DEMO_BACKEND=fast` for class demo.
- Treat full backend as experimental until latency is fixed.

### 2. Generic Responses Make the Project Look Shallow

If the system says the same generic counseling text for every prompt, it does not feel meaningful.

Mitigation:

- Use explicit support routes.
- Show route-specific next steps.
- Show source explanations.
- Add more route templates where needed.

### 3. Retrieval Can Surface Plausible But Wrong Sources

Example risks:

- PTSD resources for ordinary stress.
- admissions/funding resources for exam distress.
- clinical resources when campus support would be better.

Mitigation:

- Add source ranking penalties and route-specific preferences.
- Use Karthik's eval dataset to measure retrieval quality.
- Add regression tests for common prompts.

### 4. Crisis Handling Must Not Depend Only On The LLM

For mental-health-adjacent use, vague or missed crisis prompts are unacceptable.

Mitigation:

- Keep lexical safety backups.
- Fail closed.
- Route crisis/emergency prompts to crisis-only resources.
- Block normal generation when crisis is detected.

### 5. The Corpus Is Not Yet Publication-Ready

The corpus is good enough for a prototype, but not yet defensible as final research infrastructure.

Mitigation:

- Complete human review.
- Add source inventory quality labels.
- Confirm source licenses and provenance.
- Add a data card.
- Add limitations.

### 6. The UI Can Overpromise

A beautiful interface can accidentally imply clinical trust or deployment readiness.

Mitigation:

- Keep clear disclaimers.
- Present as a prototype and support-navigation tool.
- Avoid therapy/diagnosis language.

### 7. No Evaluation Dataset Means No Research Claim

Without a clean eval set, we can demo behavior but cannot make strong empirical claims.

Mitigation:

- Karthik is working on the eval dataset.
- We need scenario labels, expected route, expected safety level, acceptable source domains, and notes.

## 7. What Is Already Good

Strong current parts:

- The project idea is meaningful and timely.
- V2 framing is much stronger than V1 generic mental-health chatbot framing.
- Curated corpus integration exists.
- Safety-mode gated retrieval exists.
- Crisis-only retrieval exists.
- Fail-closed policy exists.
- Fast demo backend is reliable enough for live presentation.
- UI now shows system internals in a presentation-friendly way.
- The work is moving toward transparent routing rather than hidden chatbot behavior.

## 8. What Is Still Weak

Weak current parts:

- Real backend latency/reliability is not solved.
- Fast backend is deterministic and should be described honestly.
- Retrieval ranking is still heuristic.
- Route classification is still heuristic.
- Corpus quality needs human review.
- Evaluation is not complete.
- No publication-level experimental results yet.
- No clinical/expert validation.
- No deployment/privacy/security review.

## 9. What We Need From Karthik Next

Karthik's current main task should be the evaluation dataset, not more random scraping.

Needed eval dataset fields:

- `case_id`
- `user_prompt`
- `expected_route`
- `expected_safety_level`
- `expected_usage_mode`
- `acceptable_topics`
- `acceptable_source_names`
- `unacceptable_sources`
- `needs_crisis_intercept`
- `notes`

Suggested scenario types:

- academic setback
- exam stress
- low mood
- counseling navigation
- ADS accommodations
- advisor conflict
- isolation/loneliness
- panic/grounding
- vague distress
- explicit crisis
- emergency/imminent risk
- out-of-scope prompts

The eval set should include easy, ambiguous, and adversarial prompts.

## 10. Immediate Next Steps

### Priority 1: Stabilize The Demo

- Keep the fast backend as the default.
- Hard-test the scripted demo prompts.
- Avoid switching to real backend during live presentation.
- Make sure `127.0.0.1:7860` is restarted after code changes.

### Priority 2: Improve Meaningfulness

- Add route-confidence display.
- Add source-confidence display.
- Add a "recommended next action" card.
- Add cleaner scripted responses for each route.
- Reduce generic language.

### Priority 3: Add Regression Tests

Test prompts should verify:

- failed exam routes to academic setback
- ADS exam accommodations route to accessibility
- advisor conflict routes to advisor conflict
- grounding request routes to wellbeing/grounding
- crisis prompt routes to immediate safety
- normal prompts do not retrieve crisis-only resources
- crisis prompts do not retrieve normal-only support first

### Priority 4: Prepare MSML Presentation

Presentation should emphasize:

- Problem: students often do not know which support path to use.
- Risk: normal chatbots can hallucinate or mishandle crisis language.
- Solution: safety-aware RAG with support-route classification and gated retrieval.
- Demo: show route, safety mode, sources, and response.
- Limitation: prototype only, not therapy or emergency care.

### Priority 5: Research Planning

Need decisions on:

- What exact research claim we want to make.
- What baselines to compare against.
- What metrics to use.
- Whether human review is feasible.
- Whether UMD Counseling involvement is advisory, evaluative, or only aspirational.

## 11. Suggested Research Framing

Possible title direction:

Safety-Aware Retrieval-Augmented Student Support Navigation for Campus Mental Health Resources

Suggested claim:

This project explores whether a curated, safety-gated RAG pipeline can provide more transparent and safer student-support navigation than ungated retrieval or generic LLM responses.

Avoid claiming:

- It treats mental health conditions.
- It diagnoses students.
- It replaces counseling.
- It is clinically validated.
- It is ready for real deployment.

## 12. Current Honest Status

The project is demo-viable and conceptually promising.

It is not yet research-complete.

The most important current win is the V2 shift from "empathetic chatbot" to "safety-aware student-support router." That is the direction that can make the project meaningful, defensible, and useful.

For the MSML class demo, the current fast curated app is the right path.

For research/publication, the next hard work is evaluation, source quality, safety validation, and real backend reliability.

## 13. V2.5 Checkpoint Update

V2 was checkpointed locally before V2.5 work:

- V1 baseline branch: `checkpoint/v1-baseline`
- V2 curated-support branch: `checkpoint/v2-curated-support`
- V2.5 working branch: `codex/v2.5-support-navigator`

V2.5 adds the next architecture layer without replacing V1 or V2:

- canonical route/tier schema in `src/pipeline/v2_schema.py`
- four-mode ladder: `imminent_safety`, `high_distress`, `support_navigation`, `wellbeing`
- minimal service graph in `data/curated/service_graph.jsonl`
- service graph loader in `src/pipeline/service_graph.py`
- output-side guard in `src/pipeline/output_guard.py`
- peer-helper mode in the demo UI
- basic-needs routing with explicit non-hallucination around Pantry/Thrive details
- academic setback response with professor/TA email script
- regression tests in `tests/test_v25_support_navigator.py`
- multi-turn eval harness in `eval/run_multiturn_eval.py`

The project should now be framed as:

V1 baseline -> V2 curated safety-gated support navigator -> V2.5 graph-grounded, route/tier-explicit navigator with output guard and multi-turn eval scaffolding.

## 14. EmpathRAG Core Consolidation Update

The project is now being consolidated into one active system: **EmpathRAG Core**.

EmpathRAG Core keeps the chatbot/RAG framing from the original proposal, but makes it guarded and source-grounded:

- hard safety precheck
- hybrid ML + rule route/risk classifier
- graph-grounded retrieval
- constrained response planner
- output-side anti-sycophancy/groundedness guard
- multi-turn trajectory escalation
- unified evaluation reports

Current local metrics on the prepared 92-row Karthik dataset:

- Rule route accuracy: 0.935
- Hybrid ML route accuracy: 0.978
- Safety tier accuracy: 0.902
- Intercept accuracy: 1.000
- Source organization hit rate: 0.913
- Unsafe generation count: 0

Karthik's next task is documented in:

- `docs/team/karthik/CORE_DATASET_V2_REQUEST.md`
