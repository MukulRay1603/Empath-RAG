# EmpathRAG — Slide Deck Plan

A visual-first plan for the MSML641 final presentation. Eighteen slides, target ten minutes (~30–35 seconds per slide on average). Each slide entry below has three sections:

- **Visual** — what to put on the slide canvas (image, diagram, table, or one large headline number).
- **On-slide text** — the few words actually printed on the slide. Treat this as a hard cap. If a slide needs more, split it.
- **Speaker notes** — what to say out loud. This is where the depth lives.

Design rules for the deck itself:

1. One idea per slide.
2. Three to five short bullets max. No paragraphs.
3. Charts and diagrams over walls of text.
4. Use the live Mermaid diagram from the README as a baked image for the architecture slide.
5. Headline numbers (`0 / 28`, `9 / 28`) get their own large-typography slides. Do not bury them.

---

## Slide 1 — Title

**Visual:** EmpathRAG wordmark centered, subtle teal-to-indigo gradient background, small UMD MSML641 mark in the corner. No body text.

**On-slide text:**
- **EmpathRAG**
- *A guarded conversational support navigator for UMD students*
- Mukul Rayana — MSML641, University of Maryland

**Speaker notes:** *(15 seconds)*
> Hi, I'm Mukul Rayana. For my MSML641 final project I built EmpathRAG, a guarded conversational support navigator for University of Maryland students. The next ten minutes are about what I built, why the architecture matters more than the model, and what the evaluation actually proves.

---

## Slide 2 — The Problem

**Visual:** A two-panel illustration. Left panel: a student's chat bubble saying "I haven't been okay lately." Right panel: a chatbot response that fabricates a phone number ("Call UMD Support at 301-555-CARE") with a red strike-through. Underneath both panels, a single line in muted text.

**On-slide text:**
- A general-purpose chatbot can sound supportive — and still fabricate or soften crisis signals.

**Speaker notes:** *(35 seconds)*
> University students often need help that sits between a counseling appointment and a Google search. They have a worry, a question, a tough moment. A general-purpose chatbot can sound supportive in this setting — but it has two structural weaknesses that matter for student wellbeing. It can fabricate resources, phone numbers, eligibility rules. And it can fail to recognize, or actively soften, language that signals risk. EmpathRAG is an attempt to address both without giving up the natural conversational quality.

---

## Slide 3 — The Headline Result

**Visual:** Single full-bleed slide. Two huge numbers side by side, separated by a vs.

```
   0 / 28          9 / 28
   ──────          ──────
   EmpathRAG       Same model,
   (full pipeline) no pipeline
```

Subtitle in small text underneath: *"Same Llama 3.3 70B model. Non-overlapping 95% confidence intervals."*

**On-slide text:** *(only the numbers shown above)*

**Speaker notes:** *(40 seconds)*
> Here's the headline result, before any of the how. On a 28-scenario multi-turn safety benchmark, with both systems using the same underlying Llama 3.3 70B model, EmpathRAG missed escalation in 0 out of 28 scenarios. The same model without our architecture missed 9. The 95% confidence intervals do not overlap. Because the underlying model is identical, the entire delta is attributable to the surrounding architecture, not the model. That's the contribution. The rest of this talk is what's in that surrounding architecture and why each piece is needed.

---

## Slide 4 — The Pattern: Plan and Rephrase

**Visual:** A simple horizontal three-box diagram with arrows.

```
  ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
  │ DETERMINISTIC│ →  │     LLM     │ →  │   VERIFIER   │
  │   PLANNER    │    │  REPHRASER  │    │ (trust line) │
  └─────────────┘    └─────────────┘    └──────────────┘
   what to say        how to say it       reject drift
```

**On-slide text:**
- **Plan and rephrase** — separate *what* the system says from *how* it says it.
- The LLM cannot invent advice or resources.

**Speaker notes:** *(40 seconds)*
> The architectural pattern I'm calling plan and rephrase is the central design decision. A deterministic planner — auditable, rule-based code — is the source of truth for what the system says. The language model is only a controlled paraphrasing layer. It cannot invent advice, resources, or claims. After the model rephrases, a verifier inspects the output for scope drift, fabricated resources, sycophancy, and minimization. If the rephrased text drifts outside the planner's intent, the verifier rejects it and the deterministic template is shown instead. Crisis content bypasses the model entirely.

---

## Slide 5 — Architecture

**Visual:** The Mermaid pipeline diagram from the README, exported as a clean image. Color-coded by layer (yellow: precheck, red: crisis, teal: planner, purple: LLM, orange: trust boundary). Take a screenshot from GitHub's rendered Mermaid for highest fidelity.

**On-slide text:**
- *(no text — the diagram speaks)*

**Speaker notes:** *(50 seconds)*
> Here's the full pipeline. A student message hits a length cap, then a Stage-1 lexical safety check that runs in roughly five milliseconds with no network call. If crisis language is detected, we intercept immediately and never invoke the model. Otherwise the message routes to a hybrid classifier — fourteen routes at four safety tiers — and a resource registry filter restricted to thirty-four verified UMD and national service entries. A stage-aware planner picks one of four conversational stages: listen, ask permission, offer, or clarify. The planner authors a deterministic template. That template plus the user message goes to the LLM rephraser, which paraphrases under a strict system prompt. The post-rephrase verifier and an output guard then bracket the model's output before it reaches the student.

---

## Slide 6 — Iteration 1: The Open Retrieval Baseline

**Visual:** A simpler five-stage diagram showing the V1 pipeline horizontally. Below it, a small box of headline V1 metrics.

```
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ RoBERTa  │ → │ DeBERTa  │ → │ Query    │ → │  FAISS   │ → │ Mistral  │
  │ emotion  │   │   NLI    │   │ rewrite  │   │ 1.67M    │   │   7B     │
  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

Tiny stats panel: `RoBERTa F1 0.71  ·  DeBERTa recall 0.96  ·  BERTScore 0.83`

**On-slide text:**
- Five-stage pipeline, single-turn, open Reddit corpus.
- Strong on standard metrics — until adversarial probing.

**Speaker notes:** *(35 seconds)*
> The first iteration was an emotion-aware open RAG pipeline. RoBERTa for emotion classification, DeBERTa for crisis NLI, FAISS retrieval over 1.67 million Reddit Mental Health passages, Mistral 7B as the generator. Single-turn. The standard metrics were good — emotion F1 of 0.71, crisis recall of 0.96, BERTScore F1 of 0.83. Then I ran adversarial probes against it and four structural failures surfaced.

---

## Slide 7 — Why The Baseline Failed (Four Cases)

**Visual:** A 2×2 grid of stylized chat bubbles. Each cell shows one failure mode with a brief student message and a red icon indicating the failure.

```
┌──────────────────────────────┬──────────────────────────────┐
│ "Things have been good but   │ "This thesis is killing me." │
│  I've been planning to end   │                              │
│  it."                        │  → False-positive crisis     │
│  → Bait-and-switch           │    intercept                 │
│    (40% recall fail)         │                              │
├──────────────────────────────┼──────────────────────────────┤
│ "I'm overwhelmed."           │ Turn 1: "I'm fine."          │
│                              │ Turn 3: "I have a plan."     │
│  → Generic open-corpus       │                              │
│    advice, no UMD resources  │  → No multi-turn awareness   │
└──────────────────────────────┴──────────────────────────────┘
```

**On-slide text:**
- Bait-and-switch · Domain-transfer false positives · Generic generation · No multi-turn state

**Speaker notes:** *(45 seconds)*
> Four failures. One: messages that opened with positive framing and then introduced crisis content fooled the NLI guardrail forty percent of the time. Real students don't lead with crisis language. They lead with deflection. Two: academic idioms like "this thesis is killing me" triggered false-positive crisis intercept, because the DeBERTa model trained on r/SuicideWatch had never seen graduate-student hyperbole. Three: the open Reddit corpus produced warm but ungrounded responses that recommended generic advice instead of naming the specific UMD office that would help. Four: each turn was treated independently, so escalation that developed across three turns was never recognized. These four failures motivated the redesign.

---

## Slide 8 — The Architectural Response

**Visual:** A two-column comparison table styled like a side-by-side card.

| Failure mode | Architectural response |
|---|---|
| Bait-and-switch (40% recall) | Lexical precheck before NLI, trajectory tracker locks sessions |
| Academic-idiom false positives | Idioms route to `academic_setback`, not `imminent_safety` |
| Generic, ungrounded generation | Curated resource registry replaces open retrieval |
| No multi-turn dynamics | Session-aware state, history threaded into model context |

**On-slide text:** *(use the table)*

**Speaker notes:** *(40 seconds)*
> Each baseline failure has a structural response in the new design. For bait-and-switch, a deterministic lexical precheck runs before the NLI guardrail, and a trajectory tracker locks sessions after three consecutive high-risk turns. For academic-idiom false positives, the lexical layer routes idioms to an academic-setback route, not to imminent-safety. For ungrounded generation, the curated UMD resource registry replaces the open Reddit corpus. For multi-turn awareness, the session carries tier history, sub-topic decay, a locked-session flag, and the recent conversation history is threaded into the model's context.

---

## Slide 9 — The Listening Layer

**Visual:** A four-circle loop diagram. The circles read LISTEN → PERMISSION → OFFER → CLARIFY, arranged left-to-right with arrows. Underneath: a small annotation reading "Turn 1: no resources surface."

**On-slide text:**
- Four-stage planner — listen, permission, offer, clarify.
- A student is heard before being routed.

**Speaker notes:** *(30 seconds)*
> The third design iteration came from real-conversation review. The previous design still felt prescriptive on turn one. Students wanted to be heard before being routed. I introduced a four-stage planner — listen, permission, offer, clarify. On turn one of a non-crisis listen-eligible route, no resources surface at all. The system invites the student to share more, then offers paths only when the conversation has earned them.

---

## Slide 10 — The Trust Boundary

**Visual:** A vertical flow showing template → LLM → verifier → either accepted or fallback. Use icons rather than words where possible. The verifier box should have four small chips reading: scope drift, fabrication, sycophancy, length sanity.

**On-slide text:**
- Verifier rejects rephrased output that drifts outside the planner's intent.
- Crisis content never enters this path.

**Speaker notes:** *(40 seconds)*
> The current architecture adds a post-rephrase verifier — the trust boundary. The planner sends a template, the user message, and recent history to the model. The model returns a paraphrased candidate. The verifier inspects that candidate for scope drift, fabricated resources, sycophantic agreement under explicit pressure, and length blowup. If any check fails, we fall back to the deterministic template. Crisis content never enters this path at all — it's intercepted upstream and rendered from a vetted template, with three variants for self-harm, interpersonal danger, and the peer-helper case.

---

## Slide 11 — Datasets

**Visual:** A clean table, eight rows, three columns: Dataset / Size / Role. Use small dataset logos where possible (GoEmotions, Reddit, Kaggle, HF). The four bold rows below are the UMD-specific custom datasets — highlight them.

| Dataset | Size | Role |
|---|---|---|
| GoEmotions | 58k | Emotion classifier training |
| Reddit Mental Health | 1.67M | Open retrieval (baseline) |
| r/SuicideWatch | ~230k | NLI guardrail training |
| Empathetic Dialogues | 25k | Reference set |
| **UMD Student Support Conv. Dataset** | 360 + 50 + 22 | Routing, eval |
| **UMD Resource Knowledge Base** | 177 passages | Curated retrieval |
| **UMD Service Graph** | 34 entries | Primary grounding |
| **Adversarial Probe Dataset** | *in progress* | Re-evaluation |

**On-slide text:** *(table only)*

**Speaker notes:** *(35 seconds)*
> Datasets — public ones used for the baseline iteration, plus four UMD-specific datasets I built for the guarded architecture. The grounding source that matters most is the UMD Service Graph: thirty-four verified service entries, each with a source URL, a last-verified date, and a source authority. Every recommendation the system makes comes from this registry.

---

## Slide 12 — Models

**Visual:** Two-column logo grid. Left column "Baseline iteration" with RoBERTa, DeBERTa, sentence-transformers, Mistral. Right column "Current architecture" with TF-IDF + sklearn, Groq Llama 3.3 70B, Anthropic Claude Haiku 4.5, Whisper.

**On-slide text:**
- Eight components. None of the heavy ML lives in the runtime path.

**Speaker notes:** *(30 seconds)*
> Eight components total. The baseline iteration uses fine-tuned RoBERTa for emotion, DeBERTa for crisis NLI, sentence-transformers for embeddings, Mistral 7B as the generator. The current architecture is intentionally lighter at runtime: TF-IDF plus logistic regression for routing, Groq's Llama 3.3 70B as the primary rephraser, Anthropic Claude Haiku 4.5 as the fallback, and Whisper turbo for voice input. The auditable pieces are deterministic; the heavy ML lives only in the baseline notebooks, not the live system.

---

## Slide 13 — Per-Layer Ablation

**Visual:** Horizontal bar chart. X-axis: missed escalations (0 to 28). Five rows: Full pipeline, Stage-1 disabled, Output guard disabled, Verifier disabled, Registry disabled. The "Stage-1 disabled" bar is dramatically longer than the others (22 vs 0).

**On-slide text:**
- Stage-1 lexical precheck is load-bearing for missed-escalation specifically.
- Other layers protect orthogonal failure modes.

**Speaker notes:** *(40 seconds)*
> To show that each layer earns its place, I ran a per-layer ablation. Disable one layer at a time, re-run the multi-turn benchmark. The Stage-1 lexical precheck is load-bearing for missed-escalation specifically — without it, missed escalations jump from zero to twenty-two out of twenty-eight. The other three layers — output guard, verifier, registry filter — each show zero missed escalations on this metric, but they protect orthogonal failure modes that show up in the targeted sweeps on the next slide. They are not redundant. They guard against drift, sycophancy, and fabrication.

---

## Slide 14 — Targeted Failure-Mode Sweeps

**Visual:** Eight green checkmark badges in a 2×4 grid. Each badge shows the sweep name and "n / n clean".

```
✅ Drift sweep         29/29
✅ F-1 × ISSS contract 12/12
✅ Sycophancy probes   25/25
✅ Prompt-injection    16/16
✅ Fairness spot-check 18/18
✅ Diversity probes    30/30
✅ Resource URL audit  60/63
✅ Regression tests    21/21
```

**On-slide text:** *(badge grid only)*

**Speaker notes:** *(35 seconds)*
> Beyond the headline benchmark, I built six targeted sweeps for orthogonal failure modes that the missed-escalation metric does not capture. Drift sweep across all routes and all stages. F-1 immigration vs counseling separation. Sycophancy probes under explicit pressure. Prompt-injection across nine attack categories. Fairness spot-check with paired demographic perturbation. Ten under-explored message types. All clean. The three URL audit failures are TLS handshake quirks against SAMHSA endpoints, not real outages.

---

## Slide 15 — Honest Bounds On The Claims

**Visual:** A vertical list of six concise limitation chips, each with a muted color. Use a "caution" tone, not panic.

**On-slide text:**
- Synthetic evaluation data only
- n = 28 escalation scenarios, wide CIs
- Route classifier ceiling at 0.86
- Architecture HIPAA-compatible, deployment is not
- F-1 only first-class cross-cutting concern
- No real student pilot yet

**Speaker notes:** *(40 seconds)*
> What I will not claim. All evaluation uses synthetic curated scenarios. Real student phrasing differs in ways the dataset does not capture. The escalation benchmark has twenty-eight scenarios. The confidence intervals are wide. The route classifier reaches 0.86 accuracy on the held-out split. The remaining fourteen percent degrade gracefully. The architecture is HIPAA-compatible by design but the current deployment is not — Groq does not sign business associate agreements. F-1 international students are the only first-class cross-cutting concern in the planner today. Queer, undocumented, parenting, Black, and first-generation students each warrant similar treatment. And no real student pilot has been conducted. The next milestone is a counseling center clinician walkthrough, not public release.

---

## Slide 16 — What's Next

**Visual:** A three-column roadmap card.

```
┌─────────────────┬─────────────────┬─────────────────┐
│   IN PROGRESS   │    NEAR TERM    │    LONGER       │
├─────────────────┼─────────────────┼─────────────────┤
│ Adversarial     │ Counseling      │ Cross-cutting   │
│ probe dataset   │ Center walk-    │ concerns        │
│                 │ through         │ expansion       │
│                 │                 │                 │
│                 │ Fine-tuned      │ Multilingual    │
│                 │ classifier      │ openers         │
│                 │                 │                 │
│                 │ Weekly URL      │ FastAPI front-  │
│                 │ audit           │ end + auth      │
└─────────────────┴─────────────────┴─────────────────┘
```

**On-slide text:** *(roadmap card only)*

**Speaker notes:** *(30 seconds)*
> What's next. In progress: a richer adversarial probe dataset to re-run all evaluations on. Near term: a counseling center clinician walkthrough, a fine-tuned route classifier on the larger dataset, and a scheduled weekly URL audit. Longer term: first-class layered treatment for additional cross-cutting concerns, multilingual reflection openers, and a custom front-end with server-side persistence — contingent on a BAA-signed model provider for any real deployment.

---

## Slide 17 — Live Demo

**Visual:** A QR code that resolves to `https://mukulray-empathrag.hf.space/`. To the right of the QR, the URL printed in large monospace.

**On-slide text:**
- 🚀 Live demo: **mukulray-empathrag.hf.space**
- 📦 Code: github.com/MukulRay1603/Empath-RAG

**Speaker notes:** *(20 seconds)*
> The live demo is hosted on Hugging Face Spaces at mukulray-empathrag.hf.space. The repository, with reproducibility commands and full documentation, is at github dot com slash MukulRay1603 slash Empath-RAG. Both are public.

---

## Slide 18 — Thanks

**Visual:** A clean sign-off slide. Just a centered "Thanks." in large type, with the live demo URL and the MSML641 mark beneath in smaller type.

**On-slide text:**
- Thanks.
- mukulray-empathrag.hf.space
- MSML641 · University of Maryland

**Speaker notes:** *(10 seconds)*
> Thanks. Happy to take questions.

---

## Timing Sheet

| Slide | Topic | Target time |
|---|---|---:|
| 1 | Title | 0:15 |
| 2 | Problem | 0:35 |
| 3 | Headline result | 0:40 |
| 4 | Plan-and-rephrase pattern | 0:40 |
| 5 | Architecture diagram | 0:50 |
| 6 | Baseline iteration | 0:35 |
| 7 | Four failure cases | 0:45 |
| 8 | Architectural response table | 0:40 |
| 9 | Listening layer | 0:30 |
| 10 | Trust boundary | 0:40 |
| 11 | Datasets | 0:35 |
| 12 | Models | 0:30 |
| 13 | Ablation | 0:40 |
| 14 | Targeted sweeps | 0:35 |
| 15 | Limitations | 0:40 |
| 16 | Roadmap | 0:30 |
| 17 | Demo links | 0:20 |
| 18 | Thanks | 0:10 |
| | **Total** | **~9:50** |

---

## Visual Asset Checklist

Pre-build these once and reuse:

| Asset | Source | Notes |
|---|---|---|
| Architecture pipeline diagram | `README.md` Mermaid block | Render on GitHub, screenshot at 2× retina, save as `architecture.png` |
| Plan-and-rephrase three-box | Build in slides | Three rounded rectangles, arrows |
| Baseline pipeline horizontal | Build in slides | Five rounded rectangles, single row |
| Headline numbers slide | Build in slides | Two giant numbers, one slide |
| Ablation bar chart | Build in slides or matplotlib | One bar dramatically longer than others |
| Sweeps badge grid | Build in slides | 8 green checkmark cards |
| Roadmap card | Build in slides | Three-column card |
| QR code for demo URL | qr-code-generator.com | Resolves to `https://mukulray-empathrag.hf.space/` |
