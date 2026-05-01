# EmpathRAG V2 Demo Readiness And Risk Audit

Date: 2026-04-30

Purpose: checklist for getting EmpathRAG V2 ready for the MSML class demo while preserving the longer research/publication path.

## Current Status

V1 remains demo-ready as fallback.

V2 now has:

- cleaned curated corpus candidate under `data/curated/`
- curated FAISS index built with 177 vectors
- curated source metadata in SQLite
- usage-mode retrieval gating
- crisis intercept before normal generation
- crisis source cards for intercepted crisis turns
- local corpus cleanup script
- Karthik assigned to build evaluation dataset
- validation script ready for Karthik's eval delivery

## Best-Case Path

The best-case class demo uses V2 as the main story:

1. User asks normal student-support prompt.
2. Emotion classifier labels the turn.
3. Safety triage stays at `pass` or `wellbeing_support`.
4. Curated retrieval pulls UMD/NIMH/NAMI/988/ADS/Ombuds sources depending on need.
5. Demo side panel shows source names, topics, risk levels, and links.
6. Crisis prompt is safely intercepted.
7. Crisis source cards show 988/UMD crisis resources.
8. We present this as a safer evolution from Reddit-research RAG to campus-resource RAG.

Best-case message:

> EmpathRAG V2 is a safety-aware student-support RAG prototype that routes ordinary support questions to curated resources, gates crisis content away from normal generation, and exposes auditable safety/retrieval metadata.

## Worst-Case Path

If V2 has runtime problems during presentation:

1. Use V1/Reddit path as fallback.
2. Explain V2 work as completed architecture/hardening, shown through docs/audit outputs.
3. Show curated index validation and retrieval spot-check outputs instead of live generation.
4. Avoid live crisis prompts if guardrail/model loading is unstable.

Fallback command:

```powershell
$env:EMPATHRAG_RETRIEVAL_CORPUS='reddit_research'
.\venv\Scripts\python.exe demo\app.py
```

V2 command:

```powershell
$env:EMPATHRAG_RETRIEVAL_CORPUS='curated_support'
.\venv\Scripts\python.exe demo\app.py
```

## Demo Readiness Checklist

### Corpus

- [x] Karthik V2 corpus received.
- [x] Raw V2 corpus audited.
- [x] Local cleanup script added.
- [x] Cleaned local corpus generated.
- [x] Broken `umd_counseling_005` removed.
- [x] Too-short popup-cleaned `988_lifeline_003` removed.
- [x] `url: N/A` eliminated from JSONL.
- [x] Duplicates removed.
- [x] Local corpus validates.
- [x] Curated index built.
- [ ] Add corpus audit command that automatically checks boilerplate, duplicate text, source inventory mismatch, and risky labels.
- [ ] Add a short corpus card for demo/research documentation.

### Retrieval

- [x] Curated retrieval path exists.
- [x] `retrieval_corpus` supports `reddit_research`, `curated_support`, and `auto`.
- [x] Normal prompts retrieve `usage_mode=retrieval` only.
- [x] Wellbeing-support prompts can retrieve `retrieval` plus `wellbeing_only`.
- [x] Crisis retrieval, if directly called, uses `crisis_only`.
- [x] Source repetition is limited in curated top results.
- [ ] Run curated retrieval audit after latest pipeline changes.
- [ ] Add evaluator that scores Karthik's eval queries when received.
- [ ] Add source-match metrics: expected source type/name/topic hit rate.

### Safety

- [x] Fail-closed guardrail behavior added.
- [x] Triage levels added: `pass`, `wellbeing_support`, `crisis`, `emergency`.
- [x] Explicit/imminent lexical backup patterns added.
- [x] Crisis turns intercept before normal retrieval/generation.
- [x] Crisis source cards can be shown without normal generation.
- [ ] Re-run adversarial safety eval after latest changes.
- [ ] Review false positives on academic idioms.
- [ ] Decide whether demo uses direct crisis prompt or only describes crisis handling.

### Demo App

- [x] Demo shows retrieval corpus.
- [x] Demo shows safety level and safety reason.
- [x] Demo shows top source metadata.
- [x] Sharing/logging disabled by default.
- [ ] Add demo prompt buttons/examples.
- [ ] Clean source card formatting.
- [ ] Add concise visible disclaimer.
- [ ] Add "V2 curated mode" label so audience knows it is not raw Reddit.
- [ ] Run local demo end-to-end and note startup time.
- [ ] Prepare a 5-prompt demo script.

### Evaluation

- [x] Karthik assigned eval dataset task.
- [x] Eval delivery validator added.
- [ ] Validate Karthik's eval delivery.
- [ ] Convert eval CSV into automated retrieval audit.
- [ ] Add safety-intercept scoring.
- [ ] Add source/topic hit-rate scoring.
- [ ] Save results as JSON/CSV for presentation.

### Git And Reproducibility

- [x] V2 work isolated on branch `codex-v2-safety-hardening`.
- [x] Raw/cleaned corpora and indexes ignored.
- [x] Cleanup script is committed candidate.
- [ ] Commit current V2 checkpoint.
- [ ] Push branch after verification.
- [ ] Keep `Data_Karthik/` untracked unless explicitly approved.

## Things That Can Fall Apart

### 1. Model Loading Fails

Risk:

- DeBERTa guardrail, RoBERTa classifier, sentence-transformer, or Mistral path fails.

Impact:

- Demo cannot start or generation fails.

Mitigation:

- Test demo before presentation.
- Keep v1 fallback path ready.
- Have screenshots or terminal validation outputs ready.
- Do not change model paths close to demo.

### 2. Guardrail Fails Closed

Risk:

- Real guardrail checkpoint fails to load and pipeline refuses to use stub.

Impact:

- Safer behavior but demo startup may fail.

Mitigation:

- Verify `models/safety_guardrail/` is present before demo.
- For internal retrieval-only testing, use explicit development overrides only.
- For class demo, do not silently use stub.

### 3. Mistral Latency Is Too Slow

Risk:

- Local 7B generation may take too long during live demo.

Impact:

- Presentation feels sluggish.

Mitigation:

- Use prepared prompts.
- Keep responses short.
- Pre-warm the app.
- Use one or two live turns, not a long conversation.
- If needed, show retrieval/safety panels first and let generation finish.

### 4. Crisis Prompt Takes Too Long

Risk:

- Integrated Gradients attribution can be slow.

Impact:

- Crisis demo stalls.

Mitigation:

- The demo already does a fast pass and computes IG after.
- For live presentation, describe IG rather than waiting too long.
- Use only one crisis prompt.

### 5. Retrieval Gives Odd Source

Risk:

- Dense retrieval returns a semantically plausible but not ideal source.

Impact:

- Audience sees mismatch.

Mitigation:

- Use tested prompt set.
- Add source-diversity and usage-mode gating already done.
- Run curated retrieval audit before presentation.
- Avoid improvising too many new prompts live.

### 6. Safety False Positive On Academic Idiom

Risk:

- Phrases like "this thesis is killing me" trigger crisis handling.

Impact:

- Demo appears oversensitive.

Mitigation:

- Mention this as a known research challenge.
- Use it as a discussion point only if prepared.
- Continue improving academic idiom patterns.

### 7. Safety False Negative

Risk:

- Crisis language is missed.

Impact:

- Highest-risk failure.

Mitigation:

- Use explicit lexical backups.
- Re-run adversarial eval.
- Avoid claiming clinical safety.
- Present as prototype with safety triage, not deployment-ready tool.

### 8. Corpus Licensing Concern

Risk:

- NAMI/JED content may not be redistributable the same way government content is.

Impact:

- Research/publication dataset release may be constrained.

Mitigation:

- For class demo, cite links.
- For publication, separate official UMD/government from third-party nonprofit content.
- Do not publish full scraped corpus without license review.

### 9. User Data/Privacy Concern

Risk:

- Demo logging captures sensitive text.

Impact:

- Ethics/privacy issue.

Mitigation:

- Logging disabled by default.
- Do not use real student data.
- If logging for study later, get IRB/institutional guidance.

### 10. Overclaiming

Risk:

- Presentation frames system as therapy or counseling replacement.

Impact:

- Scientifically and ethically unsafe.

Mitigation:

- Frame as retrieval/navigation/support prototype.
- Say it is not diagnosis, therapy, or emergency care.
- Emphasize escalation and source-aware support.

## Speed And Latency Optimization

Highest-impact options:

- Pre-warm the Gradio app before presenting.
- Keep Mistral loaded once; do not restart the app during demo.
- Use curated index for demo; it is only 177 vectors and very fast.
- Keep `top_k=5`.
- Avoid long multi-turn histories.
- Keep generation max tokens low.
- Use crisis intercept path to skip Mistral generation.

Possible code optimizations:

- Keep sentence-transformer on CPU for curated index because 177 vectors is tiny and GPU transfer may not be worth it.
- Add optional retrieval-only demo mode for faster safety/retrieval walkthrough.
- Add cached responses for prepared demo prompts if absolutely needed.
- Reduce `max_tokens` from 200 to 120 for demo mode.
- Add env var for demo `top_k`.

## Quality Optimization

Highest-impact options:

- Use prepared prompts.
- Show source cards prominently.
- Add a short disclaimer and scope statement.
- Prefer UMD-specific sources for campus navigation.
- Keep crisis resources separate from normal generation.
- Use Karthik's eval dataset to measure source-hit rate.

Quality checks:

- Normal counseling prompt should retrieve UMD Counseling Center.
- Accessibility prompt should retrieve UMD ADS.
- Advisor conflict prompt should retrieve UMD Graduate School Ombuds.
- Crisis prompt should intercept and show 988/UMD crisis resources.
- Academic idiom should not intercept unless explicit risk appears.

## Karthik Dependency

Karthik is currently working on:

```text
empathrag_eval_delivery_v1/
```

When received, run:

```powershell
.\venv\Scripts\python.exe eval\validate_eval_delivery.py path\to\empathrag_eval_delivery_v1
```

Then build:

- automated retrieval evaluation
- safety intercept scoring
- source/topic hit-rate report

## Immediate Next Actions

Recommended order:

1. Polish Gradio demo UI and source panel.
2. Add prepared example prompt buttons.
3. Re-run curated retrieval audit.
4. Re-run adversarial safety eval.
5. Start demo locally in curated mode.
6. Commit current V2 checkpoint.
7. Prepare 5-prompt MSML demo script.
8. Validate and integrate Karthik eval dataset when it arrives.

## Presentation Positioning

Use this phrasing:

> This is a research prototype for safety-aware student-support retrieval. It is not a therapist and not an emergency service. The contribution is the pipeline design: emotion-aware routing, fail-closed safety triage, curated campus-resource retrieval, and auditable source/safety metadata.

Avoid:

- "mental health counselor"
- "diagnoses"
- "treats"
- "safe for deployment"
- "replaces counseling"

Say:

- "student support navigation"
- "campus resource retrieval"
- "safety-aware triage"
- "research prototype"
- "human review required before deployment"
