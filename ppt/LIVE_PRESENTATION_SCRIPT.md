# EmpathRAG — Live Presentation Script

**Two-presenter, ~10-minute recorded PPT.** Read line-by-line. Each line is a single breath group.

---

## Reading conventions

- **M:** = Mukul speaks
- **K:** = Karthik speaks
- *Italics* = stage direction, do not say aloud
- `[ time ]` = budget for that slide
- `—` (em-dash on its own line) = handoff pause, ~1 second of silence

---

# 🎬 SLIDE 1 — TITLE  `[ 0:00 – 0:12 ]`

**M:** &nbsp; Hi, I'm Mukul.

**K:** &nbsp; And I'm Karthik.

**K:** &nbsp; This is EmpathRAG — a guarded conversational support navigator for UMD students.

**M:** &nbsp; Ten-minute walkthrough — problem, architecture, evaluation, and a quick demo.

**M:** &nbsp; Let's get into it.

---

# 🎬 SLIDE 2 — THE PROBLEM  `[ 0:12 – 0:55 ]`

*Click.*

**M:** &nbsp; Two failure modes get students in trouble when they turn to chatbots in distress.

**M:** &nbsp; The first is fabricated resources.

**M:** &nbsp; The model invents a phone number, a service, an eligibility rule.

**M:** &nbsp; The example on the right is exactly that — a number that looks real and isn't.

—

**K:** &nbsp; The second is missed risk signals.

**K:** &nbsp; Generic models soften language that signals real distress instead of intercepting it.

**K:** &nbsp; Either failure, at a vulnerable moment, is a serious problem.

—

**M:** &nbsp; EmpathRAG is built to fail in neither direction.

**M:** &nbsp; Without losing the conversational quality students actually need.

---

# 🎬 SLIDE 3 — THE HEADLINE RESULT  `[ 0:55 – 1:30 ]`

*Click.*

**M:** &nbsp; Same Llama 3.3 70B model. Two configurations.

**M:** &nbsp; Unguarded on the left misses escalation 9 times out of 28.

**M:** &nbsp; Our guarded pipeline on the right — zero out of 28.

**M:** &nbsp; The 95% confidence intervals don't overlap.

—

**K:** &nbsp; And this isn't a benchmark we tuned for.

**K:** &nbsp; It's an external safety contract — does the system intercept crisis language.

**K:** &nbsp; That's the bar everyone should be evaluated against.

---

# 🎬 SLIDE 4 — THE PATTERN: PLAN AND REPHRASE  `[ 1:30 – 2:05 ]`

*Click.*

**M:** &nbsp; The pattern that makes this work is simple.

**M:** &nbsp; Separate *what* the system says from *how* it says it.

**M:** &nbsp; A deterministic planner picks the route, the safety tier, and the resources.

**M:** &nbsp; Only then does the LLM rephrase the plan into natural language.

—

**K:** &nbsp; The LLM never decides what to recommend.

**K:** &nbsp; It only paraphrases.

**K:** &nbsp; That means it physically cannot invent advice or resources — the planner already chose.

---

# 🎬 SLIDE 5 — ARCHITECTURE  `[ 2:05 – 2:55 ]`

*Click.*  *(Densest slide — pace yourself ~10% slower.)*

**M:** &nbsp; Five layers.

**M:** &nbsp; Stage-1 lexical pre-check catches crisis language before anything else runs.

**M:** &nbsp; The router classifies into one of sixteen routes.

**M:** &nbsp; Curated retrieval pulls from a registry of verified UMD resources.

**M:** &nbsp; No open web. No Reddit at inference.

—

**K:** &nbsp; The planner builds a deterministic response plan.

**K:** &nbsp; The rephraser turns it into natural language.

**K:** &nbsp; The post-rephrase verifier rejects anything that drifts off the plan.

—

**M:** &nbsp; And about what's actually trained:

**M:** &nbsp; The route classifier is TF-IDF plus logistic regression.

**M:** &nbsp; Trained on our 216-72-72 UMD dataset, about 86% test accuracy.

**M:** &nbsp; The LLM is Llama 3.3 70B via Groq — pretrained, not fine-tuned by us.

**M:** &nbsp; The templates and crisis regex are deterministic.

**M:** &nbsp; That separation is the safety contract.

---

# 🎬 SLIDE 6 — ITERATION 1: OPEN RETRIEVAL BASELINE  `[ 2:55 – 3:25 ]`

*Click.*

**K:** &nbsp; The first version was a five-stage pipeline.

**K:** &nbsp; Open Reddit retrieval. Single-turn responses.

**K:** &nbsp; It scored well on standard metrics.

—

**M:** &nbsp; It only fell over when we ran adversarial probes against it.

**M:** &nbsp; Four specific failure cases told us the architecture had to change.

---

# 🎬 SLIDE 7 — WHY THE BASELINE FAILED (FOUR CASES)  `[ 3:25 – 4:10 ]`

*Click.*

**M:** &nbsp; Case one — generic emotional prompts dropped into a default route with no specificity.

**M:** &nbsp; Case two — F-1 framing on turn one hijacked every later turn in the session.

—

**K:** &nbsp; Case three — a counselor allegedly suggesting harm got routed to academic-setback.

**K:** &nbsp; The system implicitly validated the authority figure.

**K:** &nbsp; Case four — under explicit pressure to agree, the rephraser leaked a "you're right" capitulation before the verifier caught it.

—

**M:** &nbsp; Each of these became an architectural fix.

---

# 🎬 SLIDE 8 — THE ARCHITECTURAL RESPONSE  `[ 4:10 – 4:35 ]`

*Click.*

**K:** &nbsp; So we layered the response.

**K:** &nbsp; Listening for tone.

**K:** &nbsp; A trust boundary for the LLM.

**K:** &nbsp; Session-level state for context that should carry.

**K:** &nbsp; And a new route for authority misconduct.

---

# 🎬 SLIDE 9 — THE LISTENING LAYER  `[ 4:35 – 5:15 ]`

*Click.*

**M:** &nbsp; Four stages — LISTEN, PERMISSION, OFFER, CLARIFY.

**M:** &nbsp; A student is heard before being routed.

—

**M:** &nbsp; Turn one validates without dumping resources.

**M:** &nbsp; Turn two names a few options but asks permission.

**M:** &nbsp; Turn three offers.

**M:** &nbsp; CLARIFY catches single-word or incomplete replies, so the system doesn't barrel forward on insufficient input.

—

**K:** &nbsp; Concretely — the chatbot stops feeling like an FAQ bot.

**K:** &nbsp; And starts feeling like someone paying attention.

---

# 🎬 SLIDE 10 — THE TRUST BOUNDARY  `[ 5:15 – 5:50 ]`

*Click.*

**M:** &nbsp; After the LLM rephrases, the verifier checks for drift.

**M:** &nbsp; Fabricated resources. Scope creep. Capitulation under pressure. AI-tells. Length blow-up.

**M:** &nbsp; If it fails, we fall back to the deterministic template.

—

**K:** &nbsp; Crisis content never enters this path at all.

**K:** &nbsp; The Stage-1 pre-check intercepts before the LLM ever sees it.

---

# 🎬 SLIDE 11 — ARCHITECTURAL RESPONSE (CONTINUED)  `[ 5:50 – 6:15 ]`

*Click.*

**M:** &nbsp; And the session layer carries context where it should.

**M:** &nbsp; F-1 status. Prior offers.

**M:** &nbsp; And decays it after two silent turns.

**M:** &nbsp; So the system can fully shift topic without one early signal dominating.

---

# 🎬 SLIDE 12 — DATASETS  `[ 6:15 – 6:55 ]`

*Click.*

**K:** &nbsp; We worked with three layers of data.

**K:** &nbsp; A curated UMD student-support conversational dataset.

**K:** &nbsp; Split 216 / 72 / 72 for training, dev, and test on single-turn routing.

**K:** &nbsp; A 74-scenario multi-turn evaluation set for the safety contract.

**K:** &nbsp; And a verified registry of sixty-plus UMD resource URLs with last-verified dates.

—

**M:** &nbsp; Karthik led dataset curation, source verification, and the annotation conventions for routing and safety tiers.

---

# 🎬 SLIDE 13 — PER-LAYER ABLATION  `[ 6:55 – 7:35 ]`

*Click.*  *(This is the proof slide — land it.)*

**M:** &nbsp; We disabled each layer one at a time.

**M:** &nbsp; And re-ran the 28-scenario escalation eval.

**M:** &nbsp; The Stage-1 lexical pre-check is load-bearing for missed escalation specifically.

**M:** &nbsp; Disabling it takes us from zero misses to 22 out of 28.

—

**K:** &nbsp; The other layers protect orthogonal failure modes.

**K:** &nbsp; Registry filtering prevents fabrication.

**K:** &nbsp; The verifier catches LLM drift.

**K:** &nbsp; Each one earns its place in the pipeline.

—

**M:** &nbsp; And this is the cleanest answer to *"is the architecture real or scripted?"*

**M:** &nbsp; If a layer weren't doing real work, disabling it wouldn't change the numbers.

**M:** &nbsp; The 0-to-22 swing is the proof.

---

# 🎬 SLIDE 14 — TARGETED FAILURE-MODE SWEEPS  `[ 7:35 – 8:05 ]`

*Click.*

**M:** &nbsp; Beyond the headline eval we ran five targeted sweeps.

**M:** &nbsp; Rephraser drift across 29 cells.

**M:** &nbsp; F-1 stage and ISSS contract across 12 cells.

**M:** &nbsp; 25 sycophancy probes.

**M:** &nbsp; 16 prompt-injection probes.

**M:** &nbsp; 18 fairness paired prompts.

**M:** &nbsp; All clean within the stochastic LLM tolerance.

---

# 🎬 SLIDE 15 — HONEST BOUNDS ON THE CLAIMS  `[ 8:05 – 8:35 ]`

*Click.*

**K:** &nbsp; N is 28 escalation scenarios. That's small.

**K:** &nbsp; We don't claim zero missed escalation in deployment.

**K:** &nbsp; What we claim is zero versus the unguarded baseline's nine.

**K:** &nbsp; With non-overlapping confidence intervals.

—

**M:** &nbsp; The data is synthetic. Real student phrasing is messier.

**M:** &nbsp; This is prototype-stage evidence, not a deployment claim.

**M:** &nbsp; We say that out loud because it matters.

---

# 🎬 SLIDE 16 — WHAT'S NEXT  `[ 8:35 – 9:00 ]`

*Click.*

**M:** &nbsp; A real student dataset.

**M:** &nbsp; A RoBERTa router on top of the rule layer.

**M:** &nbsp; Scheduled URL re-verification.

**M:** &nbsp; A multilingual opener for international students.

**M:** &nbsp; This is a prototype. The gaps we know about are the roadmap.

---

# 🎬 SLIDE 17 — LIVE DEMO  `[ 9:00 – 9:45 ]`

*Click.  Embedded 30-second auto-loop GIF starts playing on slide entry.*

**M:** &nbsp; Quick look at the system in action.

**M:** &nbsp; A student opens vaguely.

**M:** &nbsp; Notice — no resources dumped. Just validation. That's the LISTEN stage.

—

**K:** &nbsp; Then context. The system asks permission before offering anything.

—

**M:** &nbsp; Permission granted. OFFER surfaces real cards, real links from the verified registry.

—

**K:** &nbsp; A single "ok" doesn't re-render the same template.

**K:** &nbsp; The system advances.

**K:** &nbsp; Full five-minute demo with three more scenarios and the Support Plan export is linked in the README.

---

# 🎬 SLIDE 18 — THANK YOU  `[ 9:45 – 10:00 ]`

**M:** &nbsp; Thanks for watching.

**M:** &nbsp; Code, datasets, evaluations, and the full write-up are on GitHub.

**K:** &nbsp; Happy to take questions.

---

## Pre-record checklist

- [ ] Both webcams placed in same corner. Don't move them per slide.
- [ ] One full dry-run with a stopwatch. If you hit 10:30, cut the longest sentence on the slide where you went over.
- [ ] Slide 5 (Architecture) and Slide 13 (Ablation) — pace yourself ~10% slower. Those are the dense slides.
- [ ] Slide 17 — Mukul's first line should hit as the GIF's first response streams. Time the entry.
- [ ] Don't say "uhh" — pause silently. Pauses read as confidence on camera. Filler words don't.

## If you go long

| Slide | Sentence to cut first |
|---|---|
| 5 | *"The templates and crisis regex are deterministic."* |
| 7 | *"Each of these became an architectural fix."* |
| 13 | *"Each one earns its place in the pipeline."* |
| 14 | Drop one sweep from the list (any of them) |
| 15 | *"We say that out loud because it matters."* |

## Q&A defense cheatsheet

- **"Is this trained?"** → Slide 5 already named what's trained. Reference it.
- **"Is this real?"** → Slide 13 ablation. Disabling Stage-1 changes the numbers, ergo the layers are doing real work.
- **"What about substance / privacy / typos?"** → All have dedicated routes (`substance_use_concern`, `privacy_confidentiality`, typo-aware crisis detection). Demo'd in the linked five-minute video.
- **"Why not fine-tune the LLM?"** → Plan-and-rephrase is the architectural commitment. Fine-tuning the LLM doesn't give you the deterministic safety contract.
