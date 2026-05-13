# EmpathRAG — Live Presentation Script

A two-presenter livestream-style script for the 10-minute recorded PPT. Mukul leads ~65%, Karthik ~35%. Both are on camera the entire time. Read aloud — the lines are written so they sound conversational, not scripted, even when read verbatim.

**Format conventions**
- **M** = Mukul, **K** = Karthik
- Bold opening line per slide is the lead-in. Italics are stage directions for the camera or click.
- ~600 spoken words total. At a measured 140 wpm with pauses, this lands at 9:30–9:55.
- Numbers in brackets are slide time budgets. If you're tracking long, trim the last sentence of the longer Mukul slides — Karthik's slides should not be cut.

---

## Slide 1 — Title [0:00 – 0:12]

**M:** Hi, I'm Mukul.
**K:** And I'm Karthik. This is EmpathRAG — a guarded conversational support navigator for UMD students.
**M:** Ten-minute walkthrough — problem, architecture, the things that actually work, and a quick demo. Let's get into it.

---

## Slide 2 — The Problem [0:12 – 0:55]

**M:** *Click.* Two failure modes get students in trouble when they turn to chatbots in distress. The first is fabricated resources. The model invents a phone number, a service, an eligibility rule. The example on the right is exactly that — a number that looks real and isn't.
**K:** The second is missed risk signals. Generic models soften language that signals real distress instead of intercepting it. Either failure, at a vulnerable moment, is a serious problem.
**M:** EmpathRAG is built to fail in neither direction without losing the conversational quality students actually need.

---

## Slide 3 — The Headline Result [0:55 – 1:30]

**M:** *Click.* Same Llama 3.3 70B model, two configurations. Unguarded on the left misses escalation 9 times out of 28. Our guarded pipeline on the right — zero out of 28. The 95% confidence intervals don't overlap.
**K:** And this isn't a benchmark we tuned for. It's an external safety contract — does the system intercept crisis language. That's the bar everyone should be evaluated against.

---

## Slide 4 — The Pattern: Plan and Rephrase [1:30 – 2:05]

**M:** *Click.* The pattern that makes this work is simple. Separate *what* the system says from *how* it says it. A deterministic planner picks the route, the safety tier, and the resources. Only then does the LLM rephrase the plan into natural language.
**K:** The LLM never decides what to recommend. It only paraphrases. That means it physically cannot invent advice or resources — the planner already chose.

---

## Slide 5 — Architecture [2:05 – 2:55]

**M:** *Click.* Five layers. Stage-1 lexical pre-check catches crisis language before anything else runs. The router classifies into one of sixteen routes. Curated retrieval pulls from a registry of verified UMD resources — no open web, no Reddit at inference.
**K:** The planner builds a deterministic response plan, the rephraser turns it into natural language, and the post-rephrase verifier rejects anything that drifts off the plan.
**M:** And about what's trained: the route classifier is a TF-IDF plus logistic regression model trained on our 216-72-72 UMD dataset, about 86% test accuracy. The LLM is Llama 3.3 70B via Groq — pretrained, not fine-tuned by us. The templates and crisis regex are deterministic. That separation is the safety contract.

---

## Slide 6 — Iteration 1: Open Retrieval Baseline [2:55 – 3:25]

**K:** *Click.* The first version was a five-stage pipeline with open Reddit retrieval and single-turn responses. It scored well on standard metrics.
**M:** It only fell over when we ran adversarial probes against it. Four specific failure cases told us the architecture had to change.

---

## Slide 7 — Why the Baseline Failed (Four Cases) [3:25 – 4:10]

**M:** *Click.* Case one — generic emotional prompts dropped into a default route with no specificity. Case two — F-1 framing on turn one hijacked every later turn in the session.
**K:** Case three — a counselor allegedly suggesting harm got routed to academic-setback. The system implicitly validated the authority figure. Case four — under explicit pressure to agree, the rephraser leaked a "you're right" capitulation before the verifier caught it.
**M:** Each of these became an architectural fix.

---

## Slide 8 — The Architectural Response [4:10 – 4:35]

**K:** *Click.* So we layered the response. Listening for tone, a trust boundary for the LLM, session-level state for context that should carry, and a new route for authority misconduct.

---

## Slide 9 — The Listening Layer [4:35 – 5:15]

**M:** *Click.* Four stages — LISTEN, PERMISSION, OFFER, CLARIFY. A student is heard before being routed. Turn one validates without dumping resources. Turn two names a few options but asks permission. Turn three offers. CLARIFY catches single-word or incomplete replies so the system doesn't barrel forward on insufficient input.
**K:** Concretely, this means the chatbot stops feeling like an FAQ bot and starts feeling like someone paying attention.

---

## Slide 10 — The Trust Boundary [5:15 – 5:50]

**M:** *Click.* After the LLM rephrases, the verifier checks for drift — fabricated resources, scope creep, capitulation under pressure, AI-tells, length blow-up. If it fails, we fall back to the deterministic template.
**K:** Crisis content never enters this path at all. The Stage-1 pre-check intercepts before the LLM ever sees it.

---

## Slide 11 — Architectural Response (continued) [5:50 – 6:15]

**M:** *Click.* And the session layer carries context where it should — F-1 status, prior offers — and decays it after two silent turns. So the system can fully shift topic without one early signal dominating.

---

## Slide 12 — Datasets [6:15 – 6:55]

**K:** *Click.* We worked with three layers of data. A curated UMD student-support conversational dataset, split 216 / 72 / 72 for training, dev, and test on single-turn routing. A 74-scenario multi-turn evaluation set for the safety contract. And a verified registry of sixty-plus UMD resource URLs with last-verified dates.
**M:** Karthik led dataset curation, source verification, and annotation conventions for routing and safety tiers.

---

## Slide 13 — Per-Layer Ablation [6:55 – 7:35]

**M:** *Click.* We disabled each layer one at a time and re-ran the 28-scenario escalation eval. The Stage-1 lexical pre-check is load-bearing for missed escalation specifically — disabling it takes us from zero misses to 22 out of 28.
**K:** The other layers protect orthogonal failure modes — registry filtering prevents fabrication, the verifier catches LLM drift. Each one earns its place in the pipeline.
**M:** And this is the cleanest answer to "is the architecture real or scripted?" — if a layer weren't doing real work, disabling it wouldn't change the numbers. The 0-to-22 swing is the proof.

---

## Slide 14 — Targeted Failure-Mode Sweeps [7:35 – 8:05]

**M:** *Click.* Beyond the headline eval we ran five targeted sweeps — rephraser drift across 29 cells, F-1 stage and ISSS contract across 12 cells, 25 sycophancy probes, 16 prompt-injection probes, and 18 fairness paired prompts. All clean within the stochastic LLM tolerance.

---

## Slide 15 — Honest Bounds on the Claims [8:05 – 8:35]

**K:** *Click.* N is 28 escalation scenarios. That's small. We don't claim zero missed escalation in deployment — what we claim is zero versus the unguarded baseline's nine, with non-overlapping confidence intervals.
**M:** The data is synthetic. Real student phrasing is messier. This is prototype-stage evidence, not a deployment claim. We say that out loud because it matters.

---

## Slide 16 — What's Next [8:35 – 9:00]

**M:** *Click.* A real student dataset, a RoBERTa router on top of the rule layer, scheduled URL re-verification, and a multilingual opener for international students. This is a prototype. The gaps we know about are the roadmap.

---

## Slide 17 — Live Demo [9:00 – 9:45]

*The embedded 30-second auto-loop GIF starts playing on slide entry.*

**M:** *Click.* Quick look at the system in action. A student opens vaguely — and notice, no resources dumped, just validation. That's the LISTEN stage.
**K:** Then context, the system asks permission before offering anything.
**M:** Permission granted, OFFER surfaces real cards, real links from the verified registry.
**K:** A single "ok" doesn't re-render the same template. The system advances. Full five-minute demo with three more scenarios and the Support Plan export is linked in the README.

---

## Slide 18 — Thank You [9:45 – 10:00]

**M:** Thanks for watching. Code, datasets, evaluations, and the full write-up are on GitHub.
**K:** Happy to take questions.

---

## Total runtime check

- Mukul speaking blocks: ~410 words
- Karthik speaking blocks: ~210 words
- Total ~620 words at 140 wpm = **9:30** (with normal slide-transition pauses lands at **9:45–9:55**)
- Hard cap: **10:00**

## Recording technique

- **Same room, single mic** is the easiest path — audio levels match, no post-production audio sync work.
- If remote: each record audio per slide on your own mic with the deck on screen-share. Edit audio together in Premiere, lay it under the slide capture, add picture-in-picture webcams over the slides.
- **Webcam positions stay fixed** in the same corner the entire video — do not move them per slide.
- **One full dry-run** at speaking pace with a stopwatch. If you hit 10:30, identify the slide that ran long and cut a sentence — do not "speak faster," it shows on camera.
- **Slide 17 (demo)**: the GIF starts auto-playing on slide entry. Time the narration to land on the right beat — Mukul's first line should hit as the first response streams in the GIF.
- **Pace yourself on Slide 5 (Architecture)** — that's the densest slide, with the trained-vs-handcrafted disclosure. Slow it ~10%.
- **Pace yourself on Slide 13 (Ablation)** — the 0-to-22 swing is the proof point. Land it.

## What to skip in Q&A (if asked)

- "Is this trained?" → Slide 5 already named what's trained. Reference it.
- "Is this real?" → Slide 13 ablation is the answer. Disabling Stage-1 changes the numbers, ergo the layers are doing real work.
- "What about substance use / FERPA?" → Both have dedicated routes (`substance_use_concern`, `privacy_confidentiality`). Demo'd in the linked five-minute video.
- "What about typos?" → The safety layer has a typo-aware second pass; the demo shows it firing on a deliberate typo.
- "Why not fine-tune the LLM?" → Plan-and-rephrase is the architectural commitment. Fine-tuning the LLM doesn't give you the deterministic safety contract that the verifier-bounded paraphrase pattern does.
