# EmpathRAG — Live Presentation Script

**Two-presenter, ~10-minute recorded PPT.** Open this file in **rendered Markdown view** while presenting (GitHub web view, VS Code preview, Obsidian, etc.) — the speaker blocks render as visual cards.

---

## 🎯 How to use this script

| Marker | Meaning |
|---|---|
| 🔷 **Mukul** | Mukul speaks the lines below |
| 🔶 **Karthik** | Karthik speaks the lines below |
| *(italics, plain text)* | Stage direction — do not say aloud |
| Empty line **between blocks** | ~1-second silent pause / handoff |
| Empty line **inside a block** | Brief breath, same speaker continues |

Speakers are visually separated by **alternating card colors** (blue / orange) so you can spot at a glance who's about to talk. Each line within a card is one breath group — read one sentence, look up at camera if you want, then read the next.

---

# 🎬 SLIDE 1 — Title

`[ 0:00 – 0:12 ]`

> 🔷 **MUKUL**
>
> Hi everyone, I'm Mukul.

> 🔶 **KARTHIK**
>
> And I'm Karthik. Together we built EmpathRAG.

> 🔷 **MUKUL**
>
> It's a guarded conversational support navigator for UMD students.

> 🔶 **KARTHIK**
>
> Ten minutes, four sections — the problem, the architecture, the evidence, and a quick demo.

> 🔷 **MUKUL**
>
> Let's start with why we built it.

---

# 🎬 SLIDE 2 — The Problem

`[ 0:12 – 0:55 ]` &nbsp;·&nbsp; *Click.*

> 🔷 **MUKUL**
>
> When students reach for a chatbot in distress, two things tend to go wrong.
>
> The first one is fabrication.
>
> The model invents a phone number, a service, an eligibility rule that doesn't exist.
>
> The example on the right is exactly that — a number that *looks* official and isn't.

> 🔶 **KARTHIK**
>
> And the second is the opposite problem — missed signals.
>
> Generic models tend to soften language that signals real distress.
>
> They reassure when they should be intercepting.
>
> And in a vulnerable moment, either failure is dangerous.

> 🔷 **MUKUL**
>
> So our goal was to fix both of those without losing the conversational quality students actually need.

---

# 🎬 SLIDE 3 — The Headline Result

`[ 0:55 – 1:30 ]` &nbsp;·&nbsp; *Click.*

> 🔷 **MUKUL**
>
> This is the headline number — same Llama 3.3 70B model, two configurations.
>
> On the left, the unguarded model misses escalation 9 times out of 28.
>
> On the right, our guarded pipeline misses zero out of 28.
>
> And the 95% confidence intervals don't overlap.

> 🔶 **KARTHIK**
>
> What's important here is that we didn't tune *for* this benchmark.
>
> This is an external safety contract — does the system intercept crisis language, yes or no.
>
> That's the bar a system like this should be evaluated against.

> 🔷 **MUKUL**
>
> Now — how do you actually get from a 32% miss rate to zero, with the same model underneath?
>
> That's the architecture.

---

# 🎬 SLIDE 4 — The Pattern: Plan and Rephrase

`[ 1:30 – 2:05 ]` &nbsp;·&nbsp; *Click.*

> 🔷 **MUKUL**
>
> The core pattern is what we call *plan and rephrase*.
>
> We separate *what* the system says from *how* it says it.
>
> A deterministic planner picks the route, the safety tier, and the resources.
>
> Only then does the LLM rephrase the plan into natural language.

> 🔶 **KARTHIK**
>
> So the LLM never decides what to recommend. It only paraphrases.
>
> Which means it physically cannot invent a resource the planner didn't already authorize.
>
> That's the safety contract. And it's the foundation everything else builds on.

---

# 🎬 SLIDE 5 — Architecture

`[ 2:05 – 2:55 ]` &nbsp;·&nbsp; *Click. Densest slide — pace ~10% slower.*

> 🔷 **MUKUL**
>
> Here's the full pipeline. Five layers.
>
> Stage-1 is a lexical pre-check that catches crisis language before anything else runs.
>
> If it passes, the router classifies the message into one of sixteen routes.
>
> Then curated retrieval pulls from a verified UMD resource registry — no open web, no Reddit at inference.

> 🔶 **KARTHIK**
>
> From there the planner builds a deterministic response plan.
>
> The rephraser turns it into natural language.
>
> And the post-rephrase verifier rejects anything that drifted off the plan.

> 🔷 **MUKUL**
>
> And to be clear about what's actually trained here, because it matters:
>
> The route classifier is TF-IDF plus logistic regression, trained on our 216-72-72 UMD dataset, about 86% test accuracy.
>
> The LLM is Llama 3.3 70B via Groq — pretrained, *not* fine-tuned by us.
>
> The templates and crisis regex are deterministic.
>
> That separation between learned and hand-crafted is the entire safety story.

---

# 🎬 SLIDE 6 — Iteration 1: Open Retrieval Baseline

`[ 2:55 – 3:25 ]` &nbsp;·&nbsp; *Click.*

> 🔶 **KARTHIK**
>
> The architecture you just saw is version three. Let me walk you through how we got there.
>
> Our first version was a five-stage pipeline with open Reddit retrieval and single-turn responses.
>
> And on standard metrics, it actually scored well.

> 🔷 **MUKUL**
>
> The problem only showed up when we ran adversarial probes against it.
>
> Four specific failure cases told us the architecture had to change.

---

# 🎬 SLIDE 7 — Why the Baseline Failed (Four Cases)

`[ 3:25 – 4:10 ]` &nbsp;·&nbsp; *Click.*

> 🔷 **MUKUL**
>
> Case one — generic emotional prompts dropped into a default route, with no specificity.
>
> Case two — F-1 framing on turn one hijacked every later turn in the session, even after the topic moved on.

> 🔶 **KARTHIK**
>
> Case three was the worst one — a counselor was reported to have suggested harm.
>
> And the system routed it to *academic-setback*, which implicitly validated the authority figure.
>
> Case four — under explicit pressure to agree, the rephraser leaked a "you're right" capitulation before the verifier caught it.

> 🔷 **MUKUL**
>
> Each one of these became an architectural fix in the next iteration.

---

# 🎬 SLIDE 8 — The Architectural Response

`[ 4:10 – 4:35 ]` &nbsp;·&nbsp; *Click.*

> 🔶 **KARTHIK**
>
> So we layered the response across four areas.
>
> Listening for tone, instead of pushing resources immediately.
>
> A trust boundary on the LLM.
>
> Session-level state to carry context that should carry — and decay it when it shouldn't.
>
> And a dedicated route for authority misconduct.

---

# 🎬 SLIDE 9 — The Listening Layer

`[ 4:35 – 5:15 ]` &nbsp;·&nbsp; *Click.*

> 🔷 **MUKUL**
>
> Let's go deeper on the listening layer, because it's the most user-visible change.
>
> Four stages — LISTEN, PERMISSION, OFFER, CLARIFY.
>
> A student is *heard* before being routed.

> 🔷 **MUKUL**
>
> Turn one validates without dumping resources.
>
> Turn two names a few options but asks permission before pushing further.
>
> Turn three offers the full plan.
>
> And CLARIFY catches single-word or incomplete replies, so the system doesn't barrel forward on insufficient input.

> 🔶 **KARTHIK**
>
> The practical effect is that the chatbot stops feeling like an FAQ bot.
>
> And starts feeling like someone who's actually paying attention.

---

# 🎬 SLIDE 10 — The Trust Boundary

`[ 5:15 – 5:50 ]` &nbsp;·&nbsp; *Click.*

> 🔷 **MUKUL**
>
> The other major change is the trust boundary on the LLM.
>
> After the model rephrases, the verifier checks for drift.
>
> Fabricated resources, scope creep, capitulation under pressure, AI-tells, length blow-up — any of those, and we fall back to the deterministic template.

> 🔶 **KARTHIK**
>
> And just to underline this — crisis content never enters the LLM path *at all.*
>
> The Stage-1 pre-check intercepts it before the model ever sees the message.

---

# 🎬 SLIDE 11 — Architectural Response (continued)

`[ 5:50 – 6:15 ]` &nbsp;·&nbsp; *Click.*

> 🔷 **MUKUL**
>
> And the session layer carries context where it actually should — F-1 status, prior offers.
>
> Then it decays that context after two silent turns.
>
> So the system can fully shift topic without one early signal dominating the rest of the conversation.

---

# 🎬 SLIDE 12 — Datasets

`[ 6:15 – 6:55 ]` &nbsp;·&nbsp; *Click.*

> 🔷 **MUKUL**
>
> Karthik, walk us through the data.

> 🔶 **KARTHIK**
>
> Sure. We worked with three layers of data.
>
> First, a curated UMD student-support conversational dataset — split 216, 72, 72 for training, dev, and test on single-turn routing.
>
> Second, a 74-scenario multi-turn evaluation set for the safety contract.
>
> And third, a verified registry of sixty-plus UMD resource URLs, each one annotated with a last-verified date.

> 🔷 **MUKUL**
>
> And Karthik led all of that — dataset curation, source verification, the annotation conventions for routing and safety tiers.

---

# 🎬 SLIDE 13 — Per-Layer Ablation

`[ 6:55 – 7:35 ]` &nbsp;·&nbsp; *Click. This is the proof slide — land it.*

> 🔷 **MUKUL**
>
> So how do we know the layers are actually doing work, and not just sitting in the diagram?
>
> We disabled each layer one at a time and re-ran the same 28-scenario escalation eval.
>
> The Stage-1 lexical pre-check is load-bearing for missed escalation specifically.
>
> Disabling it alone takes us from zero misses to twenty-two out of twenty-eight.

> 🔶 **KARTHIK**
>
> And the other layers protect orthogonal failure modes.
>
> Registry filtering prevents fabrication.
>
> The verifier catches LLM drift.
>
> Each layer earns its place.

> 🔷 **MUKUL**
>
> This is also our cleanest answer to a fair question — *is the architecture real, or is it a scripted demo?*
>
> If a layer weren't doing real work, disabling it wouldn't change the numbers.
>
> The zero-to-twenty-two swing is the proof.

---

# 🎬 SLIDE 14 — Targeted Failure-Mode Sweeps

`[ 7:35 – 8:05 ]` &nbsp;·&nbsp; *Click.*

> 🔷 **MUKUL**
>
> Beyond the headline eval, we ran five targeted sweeps to stress specific failure modes.
>
> Rephraser drift across 29 cells.
>
> F-1 stage and ISSS contract across 12 cells.
>
> 25 sycophancy probes, 16 prompt-injection probes, 18 fairness paired prompts.
>
> All clean within the stochastic LLM tolerance.

---

# 🎬 SLIDE 15 — Honest Bounds on the Claims

`[ 8:05 – 8:35 ]` &nbsp;·&nbsp; *Click.*

> 🔶 **KARTHIK**
>
> A quick reality check before we move on.
>
> N is twenty-eight escalation scenarios. That's small.
>
> We're not claiming zero missed escalation in deployment.
>
> What we *are* claiming is zero versus the unguarded baseline's nine, with non-overlapping confidence intervals.

> 🔷 **MUKUL**
>
> And the data is synthetic — real student phrasing is messier than anything we've evaluated on.
>
> This is prototype-stage evidence, not a deployment claim. We say that out loud because it matters.

---

# 🎬 SLIDE 16 — What's Next

`[ 8:35 – 9:00 ]` &nbsp;·&nbsp; *Click.*

> 🔷 **MUKUL**
>
> The roadmap follows the gaps we just admitted.
>
> A real student dataset.
>
> A RoBERTa router on top of the rule layer.
>
> Scheduled URL re-verification.
>
> And a multilingual opener for international students.
>
> Now let's see the system in action.

---

# 🎬 SLIDE 17 — Live Demo

`[ 9:00 – 9:45 ]` &nbsp;·&nbsp; *Click. Embedded 30-second auto-loop GIF starts playing on slide entry.*

> 🔷 **MUKUL**
>
> This is the system live. Let's narrate as it streams.
>
> The student opens vaguely.
>
> Notice — no resources dumped, just validation. That's the LISTEN stage doing its job.

> 🔶 **KARTHIK**
>
> Then they add context. Watch what the system does — it asks permission before offering anything.

> 🔷 **MUKUL**
>
> Permission granted. OFFER surfaces real cards, real links from the verified registry.

> 🔶 **KARTHIK**
>
> And when the student says "ok," the system doesn't re-render the same template. It advances.
>
> The full five-minute demo with three more scenarios — F-1, substance use and confidentiality, crisis with sycophancy resistance — plus the Support Plan export, is linked in the README.

---

# 🎬 SLIDE 18 — Thank You

`[ 9:45 – 10:00 ]`

> 🔷 **MUKUL**
>
> That's EmpathRAG. Thanks for watching.
>
> Code, datasets, evaluations, and the full write-up are all on GitHub.

> 🔶 **KARTHIK**
>
> Happy to take any questions.

---

## ✅ Pre-record checklist

- [ ] Both webcams placed in same corner. Don't move them per slide.
- [ ] One full dry-run with a stopwatch. If you hit 10:30, cut the longest sentence on the slide where you went over.
- [ ] Slide 5 (Architecture) and Slide 13 (Ablation) — pace yourself ~10% slower. Those are the dense slides.
- [ ] Slide 17 — Mukul's first line should hit as the GIF's first response streams. Time the entry.
- [ ] Don't say "uhh" — pause silently. Pauses read as confidence on camera. Filler words don't.

## ✂️ If you go long — sentences to cut first

| Slide | Cut this sentence |
|---|---|
| 4 | *"And it's the foundation everything else builds on."* |
| 5 | *"The templates and crisis regex are deterministic."* |
| 7 | *"Each one of these became an architectural fix in the next iteration."* |
| 11 | *"So the system can fully shift topic without one early signal dominating the rest of the conversation."* |
| 13 | *"Each layer earns its place."* |
| 14 | Drop one sweep from the list (any of them) |
| 15 | *"We say that out loud because it matters."* |

## 💬 Q&A defense cheatsheet

- **"Is this trained?"** → Slide 5 already named what's trained. Reference it.
- **"Is this real?"** → Slide 13 ablation. Disabling Stage-1 changes the numbers, ergo the layers are doing real work.
- **"What about substance / privacy / typos?"** → All have dedicated routes (`substance_use_concern`, `privacy_confidentiality`, typo-aware crisis detection). Demo'd in the linked five-minute video.
- **"Why not fine-tune the LLM?"** → Plan-and-rephrase is the architectural commitment. Fine-tuning the LLM doesn't give you the deterministic safety contract.

## 🔗 Continuity bridges (if you forget a transition mid-recording)

| Going from → to | Bridge sentence |
|---|---|
| Slide 3 → 4 | *"How do you get from 32% to zero with the same model? That's the architecture."* |
| Slide 5 → 6 | *"That's version three. Let me walk you through how we got there."* (Karthik takes over) |
| Slide 7 → 8 | *"Each became an architectural fix."* (Karthik opens 8 with "So we layered the response...") |
| Slide 11 → 12 | (Beat — Mukul hands to Karthik with "Karthik, walk us through the data.") |
| Slide 12 → 13 | *"How do we know the layers are actually doing work?"* |
| Slide 16 → 17 | *"Now let's see the system in action."* |
