# EmpathRAG — Live Presentation Script

**Two-presenter, ~10-minute recorded PPT.** Read line-by-line. Each line is a single breath group.

---

## Reading conventions

- **M:** = Mukul speaks
- **K:** = Karthik speaks
- *Italics* = stage direction, do not say aloud
- `[ time ]` = budget for that slide
- `—` (em-dash on its own line) = handoff pause, ~1 second of silence
- Handoffs are written as a *continuation* — the speaker picking up should sound like they're finishing the previous speaker's thought, not starting a new lecture.

---

# 🎬 SLIDE 1 — TITLE  `[ 0:00 – 0:12 ]`

**M:** &nbsp; Hi everyone, I'm Mukul.

**K:** &nbsp; And I'm Karthik. Together we built EmpathRAG.

**M:** &nbsp; It's a guarded conversational support navigator for UMD students.

**K:** &nbsp; Ten minutes, four sections — the problem, the architecture, the evidence, and a quick demo.

**M:** &nbsp; Let's start with why we built it.

---

# 🎬 SLIDE 2 — THE PROBLEM  `[ 0:12 – 0:55 ]`

*Click.*

**M:** &nbsp; When students reach for a chatbot in distress, two things tend to go wrong.

**M:** &nbsp; The first one is fabrication.

**M:** &nbsp; The model invents a phone number, a service, an eligibility rule that doesn't exist.

**M:** &nbsp; The example on the right is exactly that — a number that *looks* official and isn't.

—

**K:** &nbsp; And the second is the opposite problem — missed signals.

**K:** &nbsp; Generic models tend to soften language that signals real distress.

**K:** &nbsp; They reassure when they should be intercepting.

**K:** &nbsp; And in a vulnerable moment, either failure is dangerous.

—

**M:** &nbsp; So our goal was to fix both of those without losing the conversational quality students actually need.

---

# 🎬 SLIDE 3 — THE HEADLINE RESULT  `[ 0:55 – 1:30 ]`

*Click.*

**M:** &nbsp; This is the headline number — same Llama 3.3 70B model, two configurations.

**M:** &nbsp; On the left, the unguarded model misses escalation 9 times out of 28.

**M:** &nbsp; On the right, our guarded pipeline misses zero out of 28.

**M:** &nbsp; And the 95% confidence intervals don't overlap.

—

**K:** &nbsp; What's important here is that we didn't tune *for* this benchmark.

**K:** &nbsp; This is an external safety contract — does the system intercept crisis language, yes or no.

**K:** &nbsp; That's the bar a system like this should be evaluated against.

—

**M:** &nbsp; Now, how do you actually get from a 32% miss rate to zero, with the same model underneath? That's the architecture.

---

# 🎬 SLIDE 4 — THE PATTERN: PLAN AND REPHRASE  `[ 1:30 – 2:05 ]`

*Click.*

**M:** &nbsp; The core pattern is what we call *plan and rephrase*.

**M:** &nbsp; We separate *what* the system says from *how* it says it.

**M:** &nbsp; A deterministic planner picks the route, the safety tier, and the resources.

**M:** &nbsp; Only then does the LLM rephrase the plan into natural language.

—

**K:** &nbsp; So the LLM never decides what to recommend. It only paraphrases.

**K:** &nbsp; Which means it physically *cannot* invent a resource the planner didn't already authorize.

**K:** &nbsp; That's the safety contract.

**K:** &nbsp; And it's the foundation everything else builds on.

---

# 🎬 SLIDE 5 — ARCHITECTURE  `[ 2:05 – 2:55 ]`

*Click.*  *(Densest slide — pace yourself ~10% slower.)*

**M:** &nbsp; Here's the full pipeline. Five layers.

**M:** &nbsp; Stage-1 is a lexical pre-check that catches crisis language before anything else runs.

**M:** &nbsp; If it passes, the router classifies the message into one of sixteen routes.

**M:** &nbsp; Then curated retrieval pulls from a verified UMD resource registry — no open web, no Reddit at inference.

—

**K:** &nbsp; From there the planner builds a deterministic response plan.

**K:** &nbsp; The rephraser turns it into natural language.

**K:** &nbsp; And the post-rephrase verifier rejects anything that drifted off the plan.

—

**M:** &nbsp; And to be clear about what's actually trained here, because it matters:

**M:** &nbsp; The route classifier is TF-IDF plus logistic regression, trained on our 216-72-72 UMD dataset, about 86% test accuracy.

**M:** &nbsp; The LLM is Llama 3.3 70B via Groq — pretrained, *not* fine-tuned by us.

**M:** &nbsp; The templates and the crisis regex are deterministic.

**M:** &nbsp; That separation between learned and hand-crafted is the entire safety story.

---

# 🎬 SLIDE 6 — ITERATION 1: OPEN RETRIEVAL BASELINE  `[ 2:55 – 3:25 ]`

*Click.*

**K:** &nbsp; The architecture you just saw is version three. Let me walk you through how we got there.

**K:** &nbsp; Our first version was a five-stage pipeline with open Reddit retrieval and single-turn responses.

**K:** &nbsp; And on standard metrics it actually scored well.

—

**M:** &nbsp; The problem only showed up when we ran adversarial probes against it.

**M:** &nbsp; Four specific failure cases told us the architecture had to change.

---

# 🎬 SLIDE 7 — WHY THE BASELINE FAILED (FOUR CASES)  `[ 3:25 – 4:10 ]`

*Click.*

**M:** &nbsp; Case one — generic emotional prompts dropped into a default route, with no specificity.

**M:** &nbsp; Case two — F-1 framing on turn one hijacked every later turn in the session, even after the topic moved on.

—

**K:** &nbsp; Case three was the worst one — a counselor was reported to have suggested harm.

**K:** &nbsp; And the system routed it to *academic-setback*, which implicitly validated the authority figure.

**K:** &nbsp; Case four — under explicit pressure to agree, the rephraser leaked a "you're right" capitulation before the verifier caught it.

—

**M:** &nbsp; Each one of these became an architectural fix in the next iteration.

---

# 🎬 SLIDE 8 — THE ARCHITECTURAL RESPONSE  `[ 4:10 – 4:35 ]`

*Click.*

**K:** &nbsp; So we layered the response across four areas.

**K:** &nbsp; Listening for tone instead of pushing resources immediately.

**K:** &nbsp; A trust boundary on the LLM.

**K:** &nbsp; Session-level state to carry context that should carry — and decay it when it shouldn't.

**K:** &nbsp; And a dedicated route for authority misconduct.

---

# 🎬 SLIDE 9 — THE LISTENING LAYER  `[ 4:35 – 5:15 ]`

*Click.*

**M:** &nbsp; Let's go deeper on the listening layer, because it's the most user-visible change.

**M:** &nbsp; Four stages — LISTEN, PERMISSION, OFFER, CLARIFY.

**M:** &nbsp; A student is *heard* before being routed.

—

**M:** &nbsp; Turn one validates without dumping resources.

**M:** &nbsp; Turn two names a few options but asks permission before pushing further.

**M:** &nbsp; Turn three offers the full plan.

**M:** &nbsp; And CLARIFY catches single-word or incomplete replies, so the system doesn't barrel forward on insufficient input.

—

**K:** &nbsp; The practical effect is that the chatbot stops feeling like an FAQ bot.

**K:** &nbsp; And starts feeling like someone who's actually paying attention.

---

# 🎬 SLIDE 10 — THE TRUST BOUNDARY  `[ 5:15 – 5:50 ]`

*Click.*

**M:** &nbsp; The other major change is the trust boundary on the LLM.

**M:** &nbsp; After the model rephrases, the verifier checks for drift.

**M:** &nbsp; Fabricated resources, scope creep, capitulation under pressure, AI-tells, length blow-up — any of those, and we fall back to the deterministic template.

—

**K:** &nbsp; And just to underline this — crisis content never enters the LLM path *at all.*

**K:** &nbsp; The Stage-1 pre-check intercepts it before the model ever sees the message.

---

# 🎬 SLIDE 11 — ARCHITECTURAL RESPONSE (CONTINUED)  `[ 5:50 – 6:15 ]`

*Click.*

**M:** &nbsp; And the session layer carries context where it actually should — F-1 status, prior offers.

**M:** &nbsp; Then it decays that context after two silent turns.

**M:** &nbsp; So the system can fully shift topic without one early signal dominating the rest of the conversation.

---

# 🎬 SLIDE 12 — DATASETS  `[ 6:15 – 6:55 ]`

*Click.*

**M:** &nbsp; Karthik, walk us through the data.

—

**K:** &nbsp; Sure. We worked with three layers of data.

**K:** &nbsp; First, a curated UMD student-support conversational dataset — split 216, 72, 72 for training, dev, and test on single-turn routing.

**K:** &nbsp; Second, a 74-scenario multi-turn evaluation set for the safety contract.

**K:** &nbsp; And third, a verified registry of sixty-plus UMD resource URLs, each one annotated with a last-verified date.

—

**M:** &nbsp; And Karthik led all of that — dataset curation, source verification, the annotation conventions for routing and safety tiers.

---

# 🎬 SLIDE 13 — PER-LAYER ABLATION  `[ 6:55 – 7:35 ]`

*Click.*  *(This is the proof slide — land it.)*

**M:** &nbsp; So how do we know the layers are actually doing work, and not just sitting in the diagram?

**M:** &nbsp; We disabled each layer one at a time and re-ran the same 28-scenario escalation eval.

**M:** &nbsp; The Stage-1 lexical pre-check is load-bearing for missed escalation specifically.

**M:** &nbsp; Disabling it alone takes us from zero misses to twenty-two out of twenty-eight.

—

**K:** &nbsp; And the other layers protect orthogonal failure modes.

**K:** &nbsp; Registry filtering prevents fabrication.

**K:** &nbsp; The verifier catches LLM drift.

**K:** &nbsp; Each layer earns its place.

—

**M:** &nbsp; This is also our cleanest answer to a fair question — *is the architecture real, or is it a scripted demo?*

**M:** &nbsp; If a layer weren't doing real work, disabling it wouldn't change the numbers.

**M:** &nbsp; The zero-to-twenty-two swing is the proof.

---

# 🎬 SLIDE 14 — TARGETED FAILURE-MODE SWEEPS  `[ 7:35 – 8:05 ]`

*Click.*

**M:** &nbsp; Beyond the headline eval, we ran five targeted sweeps to stress specific failure modes.

**M:** &nbsp; Rephraser drift across 29 cells.

**M:** &nbsp; F-1 stage and ISSS contract across 12 cells.

**M:** &nbsp; 25 sycophancy probes, 16 prompt-injection probes, 18 fairness paired prompts.

**M:** &nbsp; All clean within the stochastic LLM tolerance.

---

# 🎬 SLIDE 15 — HONEST BOUNDS ON THE CLAIMS  `[ 8:05 – 8:35 ]`

*Click.*

**K:** &nbsp; A quick reality check before we move on.

**K:** &nbsp; N is twenty-eight escalation scenarios. That's small.

**K:** &nbsp; We're not claiming zero missed escalation in deployment.

**K:** &nbsp; What we *are* claiming is zero versus the unguarded baseline's nine, with non-overlapping confidence intervals.

—

**M:** &nbsp; And the data is synthetic — real student phrasing is messier than anything we've evaluated on.

**M:** &nbsp; This is prototype-stage evidence, not a deployment claim. We say that out loud because it matters.

---

# 🎬 SLIDE 16 — WHAT'S NEXT  `[ 8:35 – 9:00 ]`

*Click.*

**M:** &nbsp; The roadmap follows the gaps we just admitted.

**M:** &nbsp; A real student dataset.

**M:** &nbsp; A RoBERTa router on top of the rule layer.

**M:** &nbsp; Scheduled URL re-verification.

**M:** &nbsp; And a multilingual opener for international students.

**M:** &nbsp; Now let's see the system in action.

---

# 🎬 SLIDE 17 — LIVE DEMO  `[ 9:00 – 9:45 ]`

*Click.  Embedded 30-second auto-loop GIF starts playing on slide entry.*

**M:** &nbsp; This is the system live. Let's narrate as it streams.

**M:** &nbsp; The student opens vaguely.

**M:** &nbsp; Notice — no resources dumped, just validation. That's the LISTEN stage doing its job.

—

**K:** &nbsp; Then they add context. Watch what the system does — it asks permission before offering anything.

—

**M:** &nbsp; Permission granted, OFFER surfaces real cards, real links from the verified registry.

—

**K:** &nbsp; And when the student says "ok," the system doesn't re-render the same template. It advances.

**K:** &nbsp; The full five-minute demo with three more scenarios — F-1, substance use and confidentiality, crisis with sycophancy resistance — plus the Support Plan export, is linked in the README.

---

# 🎬 SLIDE 18 — THANK YOU  `[ 9:45 – 10:00 ]`

**M:** &nbsp; That's EmpathRAG. Thanks for watching.

**M:** &nbsp; Code, datasets, evaluations, and the full write-up are all on GitHub.

**K:** &nbsp; Happy to take any questions.

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
| 4 | *"And it's the foundation everything else builds on."* |
| 5 | *"The templates and the crisis regex are deterministic."* |
| 7 | *"Each one of these became an architectural fix in the next iteration."* |
| 11 | *"So the system can fully shift topic without one early signal dominating the rest of the conversation."* |
| 13 | *"Each layer earns its place."* |
| 14 | Drop one sweep from the list (any of them) |
| 15 | *"We say that out loud because it matters."* |

## Q&A defense cheatsheet

- **"Is this trained?"** → Slide 5 already named what's trained. Reference it.
- **"Is this real?"** → Slide 13 ablation. Disabling Stage-1 changes the numbers, ergo the layers are doing real work.
- **"What about substance / privacy / typos?"** → All have dedicated routes (`substance_use_concern`, `privacy_confidentiality`, typo-aware crisis detection). Demo'd in the linked five-minute video.
- **"Why not fine-tune the LLM?"** → Plan-and-rephrase is the architectural commitment. Fine-tuning the LLM doesn't give you the deterministic safety contract.

## Continuity audit (quick reference)

Each slide hands off to the next. If you forget a transition line, here's the bridge:

| End of slide | Bridge to next slide |
|---|---|
| 3 → 4 | *"How do you get from 32% to zero with the same model? That's the architecture."* |
| 5 → 6 | *"That's version three. Let me walk you through how we got there."* |
| 7 → 8 | *"Each one became an architectural fix."* (then K opens 8 with "So we layered the response...") |
| 11 → 12 | (slight beat — M hands to K with "Karthik, walk us through the data.") |
| 12 → 13 | *"How do we know the layers are actually doing work?"* |
| 16 → 17 | *"Now let's see the system in action."* |
