# EmpathRAG — Full Project Write-Up

A long-form narrative companion to the slide deck. Written so any one section can be quoted directly into a slide's speaker notes or expanded into talking points if a slide needs more depth than the per-slide note allows.

Mapped to the MSML641 rubric — Problem Definition · Approach · Data · Model and Method · Evaluation and Results · Discussion.

---

## 1. Problem Definition

University students often need help that sits in the gap between a counseling appointment and a Google search. They have a moment of distress, a question about a campus office, a worry about an advisor, an immigration concern that touches their visa status. They need a system that will listen, decide what kind of help is appropriate, and point them to a real resource.

A general-purpose chatbot can sound supportive in this setting. The trouble is that it has two structural weaknesses that matter specifically for student wellbeing.

The first is **fabrication**. A general chatbot can produce a phone number, an office name, an eligibility rule, or a referral pathway that does not exist. In a low-stakes domain that is annoying. In student wellbeing it can keep someone from finding the actual office that would help.

The second is **softening of risk signals**. A general chatbot, optimized to be agreeable, can fail to recognize when a student is escalating, can soften crisis language, can validate harmful framings that a human counselor would gently reframe.

The narrow technical question I set out to answer was: *can a layered architecture wrapped around an off-the-shelf large language model close those two structural gaps without giving up the conversational quality that makes a chatbot feel useful in the first place?*

The narrow practical question was: *for a UMD student in a moment of difficulty, can the resulting system point them at the right campus or national resource, with provenance, in language that does not sound clinical or canned?*

EmpathRAG is the prototype answer to both questions. It is explicitly not a counselor, therapist, diagnostic tool, or emergency service. It is a research prototype built as a class project, evaluated honestly, and bounded honestly.

---

## 2. Approach

The architectural pattern is **plan and rephrase**.

A deterministic planner — auditable, rule-based code — is the source of truth for what the system says. The language model is a controlled paraphrasing layer that cannot invent advice, resources, or claims. After the model rephrases the planner's template, a verifier inspects the rephrased text for scope drift, fabricated resources, sycophantic agreement under explicit pressure, and length blowup. If the verifier rejects the rephrased candidate, the deterministic template is shown instead. Crisis content bypasses the model entirely and is rendered from a vetted template.

This separation gives the system its safety properties. The planner is auditable line-by-line. The resource registry is grounded in verified UMD and national service entries. The verifier is the trust boundary between deterministic intent and generated text.

The system is the result of three deliberate design iterations, each one motivated by what the previous iteration got wrong.

### 2.1 Open Retrieval Baseline

A five-stage pipeline modeled on the standard emotion-aware open RAG approach: a RoBERTa-base emotion classifier fine-tuned on GoEmotions, a DeBERTa-v3 NLI safety guardrail trained on the Suicide Detection dataset, an emotion-conditioned query rewrite, FAISS retrieval over 1.67 million Reddit Mental Health passages, and a quantized Mistral 7B Instruct generator. Single-turn. Strong on standard benchmarks: emotion F1 of 0.7127 weighted, crisis recall of 0.9629 on a held-out NLI set of 23,000 samples, BERTScore F1 of 0.8266 against Empathetic Dialogues, Wilcoxon p-value of 3.62e-08 versus a BM25 baseline.

The standard metrics were good. Adversarial probing surfaced four structural failures.

**Failure one — bait-and-switch.** A message that opens with positive framing and then introduces crisis content, such as "Things have actually been good lately, but I have been planning how to end it," fooled the NLI guardrail forty percent of the time. Real students rarely lead with crisis language. They lead with deflection. A guardrail that fails on the realistic phrasing is the wrong place to put the safety floor.

**Failure two — domain-transfer false positives.** Academic idioms such as "this thesis is killing me" triggered crisis intercept at high confidence. The DeBERTa model trained on r/SuicideWatch had never seen graduate-student hyperbole. Crisis intercept on a stressed grad student is harmful in a different way than a missed real crisis: it teaches students the system is not trustworthy on academic content.

**Failure three — generic empathetic generation.** The Mistral generator authored responses directly from retrieved Reddit chunks. The output was warm but ungrounded. It recommended generic "talk to a professional" without naming the actual UMD office, hotline, or website that would help. For a UMD-deployed system this means students get redirected away from the on-campus options that would actually help.

**Failure four — no multi-turn dynamics.** A student saying "I am fine" on turn one and "I have a plan for tonight" on turn three received independent assessments. The session trajectory was lost. Crisis recognition that requires three turns to develop never developed.

### 2.2 Guarded Architecture

The redesign moved every safety-relevant decision out of the language model. Each baseline failure has an architectural response.

| Baseline failure mode | Architectural response |
|---|---|
| Bait-and-switch (40% NLI recall) | Stage-1 lexical precheck runs before NLI; trajectory tracker locks sessions after three consecutive high-risk turns; four-tier safety ladder forces categorization. |
| Domain-transfer false positives | Stage-1 routes academic idioms to `academic_setback`, not `imminent_safety`. NLI is the second opinion only. |
| Generic empathetic generation | Curated UMD resource registry (34 verified entries) replaces the open Reddit corpus. The planner authors recommendations; the model only paraphrases. |
| No multi-turn dynamics | Session-aware state: tier history, F-1 sub-topic decay, locked-session flag, conversation history threaded into rephraser context. |

### 2.3 Listening Layer

Conversation review showed that the guarded architecture still felt prescriptive on turn one. Students wanted to be heard before being routed. The listening layer introduced a four-stage planner — *listen, ask permission, offer, clarify*. On turn one of a non-crisis listen-eligible route, no resources surface at all. The system invites the student to share more, then offers paths only when the conversation has earned them.

### 2.4 Verified Rephrasing — current architecture

The current pipeline. The planner sends a template, the user message, and recent history to the language model under a strict system prompt. The model returns a paraphrased candidate. A post-rephrase verifier (`verify_rephrased_safety`) inspects the candidate for scope drift, fabricated resources, ungrounded phone numbers, sycophantic capitulation under explicit pressure, length sanity, and a small number of regression patterns from the baseline. If any check fails, the deterministic template is returned. Crisis content never enters this path.

Subsequent polish added: response streaming, a support-plan export in Markdown and PDF, voice input via Whisper turbo, an ISSS document side-panel, an authority-misconduct route, a sycophancy guard, F-1 session decay, prompt-injection auditing, per-layer ablation evaluation, the same-model unguarded baseline, an in-UI safety pipeline visualization, mobile CSS, and HIPAA / privacy gap documentation.

---

## 3. Data

EmpathRAG combines public mental-health corpora used for the baseline iteration with custom UMD-specific datasets built for the guarded architecture.

### 3.1 Public datasets used for the baseline

- **GoEmotions** (Google Research) — 58k Reddit comments labeled across 27 emotion classes, collapsed to a 5-class schema (distress, anxiety, frustration, neutral, hopeful) for the RoBERTa fine-tune. Apache 2.0.
- **Reddit Mental Health corpus** (Zenodo) — 1.67 million passages from mental-health-adjacent subreddits, used as the open retrieval corpus for the baseline iteration. CC BY 4.0.
- **Suicide Detection (r/SuicideWatch on Kaggle)** — approximately 230k labeled examples, used to fine-tune the DeBERTa-v3 NLI safety guardrail.
- **Empathetic Dialogues** (Facebook Research) — 25k empathetic conversation pairs used as the BERTScore reference set for evaluating the baseline generator. CC BY-NC 4.0.

### 3.2 Custom UMD datasets

- **UMD Student Support Conversational Dataset** — 360 single-turn examples with a 216 / 72 / 72 train / dev / test split, plus 50 multi-turn scenarios, plus 22 high-risk cases. Used to train the route classifier and to drive single-turn and multi-turn safety evaluation.
- **UMD Resource Knowledge Base** — 177 passages drawn from UMD Counseling Center, ISSS, ADS, Graduate Ombuds, and national authorities (NIMH, NAMI, SAMHSA, CDC, 988). Used as the curated retrieval corpus when retrieval is enabled.
- **UMD Service Graph** (`data/curated/service_graph.jsonl`) — 34 verified service entries, each with source URL, last-verified date, and source-authority attribution. This is the primary grounding registry — every recommendation the system makes comes from here.
- **Adversarial Probe Dataset** *(in development)* — authority-misconduct scenarios, sycophancy probes, topic-shift cases, and anonymized real turns. Planned re-evaluation set.

Multi-turn evaluation scenarios are tracked at `eval/multiturn_scenarios.jsonl` (50 scenarios) and `eval/multiturn_safety_supplement.jsonl` (24 scenarios).

---

## 4. Model and Method

### 4.1 Components

| Component | Model | Role | Iteration |
|---|---|---|---|
| Emotion classifier | RoBERTa-base + LoRA | Five-class emotion labels on each turn. | Baseline |
| Safety guardrail | DeBERTa-v3 NLI + Integrated Gradients | Crisis classification with token-level attribution. | Baseline (optional in current) |
| Retrieval embeddings | sentence-transformers/all-mpnet-base-v2 | Embedding for FAISS retrieval. | Baseline |
| Generator | Mistral 7B Instruct (Q4_K_M GGUF) | Empathetic generation from retrieved context. | Baseline only |
| Route classifier | TF-IDF + logistic regression | Hybrid rule and ML route prediction (14 routes × 4 tiers). | Current |
| Rephraser, primary | Groq Llama 3.3 70B Versatile | Plan-and-rephrase paraphrasing layer. | Current |
| Rephraser, fallback | Anthropic Claude Haiku 4.5 | Provider chain fallback when Groq is unavailable. | Current |
| Voice input | Groq Whisper Large v3 Turbo | Speech-to-text for the voice-input toggle. | Current |

### 4.2 The pipeline, in execution order

1. **Length cap.** The user message is rejected if it exceeds 2000 characters.
2. **Stage-1 lexical safety precheck.** Deterministic regex over crisis-language patterns. The per-layer ablation reported in §5 shows that this stage alone catches 22 of 28 escalation scenarios.
3. **Hybrid route classifier.** Rule-based keyword matching combined with the TF-IDF logistic model. Maps the message to one of fourteen routes (such as `low_mood`, `anxiety_panic`, `academic_setback`, `f1_immigration`, `authority_misconduct`) at one of four safety tiers.
4. **Resource registry filter.** Retrieval is restricted to the 34 verified service entries with provenance attached.
5. **Stage-aware planner.** Selects one of four conversational stages — listen, ask permission, offer, clarify — based on the route, the turn index, and the explicit-ask phrasing of the message. Authors a deterministic response template.
6. **Plan and rephrase.** The template, the user message, and the recent conversation history are sent to the language model under a strict system-prompt contract.
7. **Post-rephrase verifier (`verify_rephrased_safety`).** Checks the rephrased candidate against a set of regression patterns — scope drift, baseline-style failures, ungrounded phone numbers, fabricated resource names, sycophantic capitulation, length sanity. Failures fall through to the deterministic template.
8. **Output guard.** Runs at the offer stage. Rejects responses that lack an actionable next step, reinforce dependency on the bot, or agree with a harmful framing.

Three crisis variants are rendered without invoking the model: self-harm ideation routes to 988 plus the UMD Counseling Center, interpersonal danger routes to 911 plus UMD CARE, and the peer-helper case is handled with a "you can support them but not as their only safety plan" template.

### 4.3 Why a deterministic planner

The planner is the load-bearing safety component because it is the only place where the system's intent is *explicitly authored* in code that can be reviewed, tested, and version-controlled. The language model can produce many wordings of that intent. The verifier checks that the wording stayed faithful to the intent. The pattern lets the model do what it is good at (natural language) while removing it from the decisions where it is least reliable (what to actually recommend).

---

## 5. Evaluation and Results

All numbers are reproducible from the repository with a Groq API key. Reproduction commands and expected outputs are in `docs/research/REPRODUCIBILITY.md`.

### 5.1 Headline — same-model guarded vs unguarded

On a 28-scenario multi-turn safety benchmark, both systems use Llama 3.3 70B as the underlying model.

| System | Missed escalation | 95% CI | Harm endorsement |
|---|---:|---|---:|
| EmpathRAG (full pipeline) | 0 / 28 (0.0%) | [0.000, 0.000] | 0 |
| Unguarded same-model baseline | 9 / 28 (32.1%) | [0.148, 0.494] | 2 turns |

Same model, different architecture. The 95% confidence intervals do not overlap. Because the underlying model is identical, the entire delta is attributable to the surrounding architecture rather than to model capability.

### 5.2 Per-layer ablation

Each row disables exactly one layer and re-runs the multi-turn benchmark.

| Layer disabled | Missed escalation | Δ vs full pipeline |
|---|---:|---:|
| (none — full pipeline) | 0 / 28 | — |
| Stage-1 lexical precheck | 22 / 28 | +22 |
| Output guard | 0 / 28 | — |
| Post-rephrase verifier | 0 / 28 | — |
| Resource registry filter | 0 / 28 | — |

The Stage-1 lexical precheck is load-bearing for the missed-escalation metric specifically. The output guard, verifier, and registry filter each show zero missed escalations on this metric, but they protect orthogonal failure modes — drift, sycophancy, fabrication — that surface in the targeted sweeps below. Removing any of those three layers does not change the missed-escalation number, but it reopens specific failure surfaces that the targeted sweeps were designed to catch.

### 5.3 Targeted failure-mode sweeps

| Sweep | Cells | Clean |
|---|---:|---:|
| Drift sweep (14 routes × 3 stages) | 29 | 29 |
| F-1 stage × ISSS contract | 12 | 12 |
| Sycophancy probes (single and multi-turn pressure) | 25 | 25 |
| Prompt-injection probes (9 attack categories) | 16 | 16 |
| Fairness spot-check (paired demographic perturbation) | 18 | 18 |
| Diversity probes (10 underexplored message types) | 30 | 30 |
| Resource URL audit | 63 | 60 live |
| Regression tests | 21 | 21 |

The three URL audit failures are TLS handshake quirks against SAMHSA endpoints, not real outages.

### 5.4 Baseline reference numbers (for academic continuity)

| Metric | Value |
|---|---:|
| RoBERTa emotion F1 (weighted) | 0.7127 |
| DeBERTa crisis recall (held-out NLI, 23k) | 0.9629 |
| DeBERTa crisis precision | 0.7951 |
| BERTScore F1 vs Empathetic Dialogues | 0.8266 |
| Wilcoxon p-value (full vs BM25) | 3.62e-08 |
| Euphemistic crisis recall vs keyword filter | 100% vs 20% |

These numbers describe the baseline iteration and are preserved here so the design history is traceable. The full V1 evaluation context is in `docs/research/PAPER_FRAMING.md`.

---

## 6. Discussion

### 6.1 What the evaluation shows and does not show

The evaluation shows that a deterministic safety architecture wrapped around an off-the-shelf model can hold up under a multi-turn safety benchmark where the same model alone does not. The non-overlapping confidence intervals on the headline result are the strongest statement the data supports.

The evaluation does not show that EmpathRAG would behave the same way against real students. All evaluation uses synthetic curated scenarios. Real student phrasing differs from synthetic phrasing in ways the curated dataset does not capture. The numbers in this write-up are prototype evidence, not deployment claims.

### 6.2 Honest bounds

- **Synthetic-data ceiling.** The escalation benchmark is small (n = 28) and synthetic. Confidence intervals are wide. Stronger absolute claims need a larger and more naturalistic sample.
- **Route classifier ceiling.** The hybrid TF-IDF logistic classifier reaches 0.86 accuracy on the held-out test split. The remaining 14 percent fall into the `general_student_support` route, which degrades gracefully and does not fabricate, but it is a real ceiling.
- **Compliance posture.** The architecture is HIPAA- and FERPA-compatible by design. The current deployment is not. Groq does not sign Business Associate Agreements for commercial chat workloads. Any real deployment requires a BAA-signed model provider.
- **Cross-cutting concerns coverage.** F-1 international students are the only first-class cross-cutting concern in the planner today. Queer, undocumented, parenting, Black, and first-generation students each warrant similar layered treatment.
- **No real-world pilot.** The next validation milestone is a UMD Counseling Center clinician walkthrough, not public release.

### 6.3 Challenges encountered

The two non-obvious engineering challenges were:

**Streaming under a verifier.** The post-rephrase verifier conceptually wants the full rephrased text in hand before deciding to accept or reject. Streaming wants to render tokens as they arrive. The compromise is to stream the candidate to the user while the verifier runs in parallel; if the verifier rejects mid-stream, the buffer is replaced with the deterministic template. This is good enough for the demo but introduces a brief visual flicker on rejection that I have not yet smoothed.

**Iframe sizing on Hugging Face Spaces.** The deployed Gradio app, when embedded in the Spaces page chrome via iframe-resizer in `taggedElement` mode, was clipping the top of the app because the height-anchor element was rendered by Svelte after iframe-resizer had already given up scanning. The fix is a `data-iframe-height` sentinel rendered into the DOM plus a JS hook that calls `parentIFrame.size()` on mount and on every DOM mutation. The direct Space subdomain serves the app full-page without iframes and was unaffected.

### 6.4 What I would do differently

Three things, in order of leverage.

First, **build the dataset before the model**. A larger and more naturalistic dataset would have surfaced the four baseline failure modes earlier and would let the route classifier go past 0.86.

Second, **separate the streaming layer cleanly from the verification layer**. The current implementation entangles them more than it should because streaming was added after the verifier was already in place.

Third, **plan the deployment surface earlier**. Gradio is the right choice for evaluation and demo screenshots. It is the wrong choice for a real student-facing pilot, where a custom front-end and server-side persistence would matter. Recognizing that earlier would have saved some CSS-vs-iframe debugging time.

### 6.5 Future work

In progress: a richer Adversarial Probe Dataset to re-run all evaluations against. Near term: a UMD Counseling Center clinician walkthrough, a fine-tuned route classifier on the new dataset, and a scheduled weekly URL audit via GitHub Actions. Longer term: first-class layered treatment for additional cross-cutting concerns, multilingual reflection openers, a custom FastAPI plus HTML/JS front-end for any future deployment context, and server-side persistence and authentication contingent on a BAA-signed model provider.

---

## 7. Reproducibility and Links

- Live demo: `https://mukulray-empathrag.hf.space/`
- Repository: `https://github.com/MukulRay1603/Empath-RAG`
- Reproducibility commands: `docs/research/REPRODUCIBILITY.md`
- Architecture deep-dive: `docs/architecture/EMPATHRAG_CORE_ARCHITECTURE.md`
- Error analysis: `docs/research/ERROR_ANALYSIS.md`
- Privacy and data flow: `docs/research/PRIVACY_AND_DATA_FLOW.md`
- HIPAA / FERPA gap analysis: `docs/research/HIPAA_FERPA_GAP_ANALYSIS.md`

---

## 8. One-Paragraph Summary (for the abstract slot, if needed)

EmpathRAG is a guarded conversational support navigator for UMD students. It separates *what to say* (a deterministic planner authoring responses from a curated UMD resource registry) from *how to say it* (a controlled language-model paraphrasing layer), with a verifier rejecting any rephrased output that drifts outside the planner's intent. On a 28-scenario multi-turn safety benchmark, the system records zero missed escalations using Llama 3.3 70B as the underlying model; the same model without the architecture records nine missed escalations on the same benchmark. The 95% confidence intervals do not overlap. The contribution is the architectural pattern — plan and rephrase, with a trust boundary — not a new model.
