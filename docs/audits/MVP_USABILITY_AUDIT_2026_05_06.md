# EmpathRAG Core MVP Usability Audit

Date: 2026-05-06  
Branch: `codex/v2.5-support-navigator`

## Bottom Line

The MVP is technically credible but the conversational layer is still the
weakest part. The safety/retrieval architecture is visible and defensible, but
ordinary low-risk emotional prompts can still feel over-routed into formal
support navigation. The app should feel like a warm support navigator first and
a RAG/eval system second.

## What Is Working

- The app runs locally at `http://127.0.0.1:7860/`.
- It uses one Gradio app, not multiple services.
- Local trained TF-IDF/logistic router artifacts are present.
- Curated UMD resource index/database are present.
- Stage-1 safety precheck is visible.
- Peer-helper mode exists.
- Crisis/imminent prompts bypass normal generation.
- Out-of-scope prompts avoid support-source retrieval.
- Eval A and Eval B run.
- Regression tests are passing.

## Current Product Shape

EmpathRAG Core is best described as:

> A guarded conversational RAG support navigator that helps a student name the
> kind of support they need, retrieve grounded resources, take one practical
> next step, and escalate safely when risk appears.

It is not a therapist, counselor, diagnostic tool, emergency service, or clinical
intervention.

## Major Usability Issue Found Today

Prompt:

```text
I'm nervous to meet a girl I asked out tomorrow
```

Bad behavior:

- Routed to `exam_stress` because `tomorrow` was over-broadly treated as an exam/study signal.
- Generated test-prep language about high-yield topics and sleep.
- Felt like a conditional routing bot, not emotional support.

Fix implemented:

- Removed `tomorrow` as a standalone exam-stress trigger.
- Added social/date nerves detection:
  - `asked out`
  - `first date`
  - `meet a girl`
  - `meet a guy`
  - `meet someone`
  - `going on a date`
  - `date tomorrow`
  - `nervous to meet`
  - `romantic`
- Routes those prompts to `anxiety_panic`, not `exam_stress`.
- Added a conversational response for ordinary date/social nerves.

## What The Response Should Feel Like

For low-risk emotional prompts, the app should:

- reassure without over-validating extreme conclusions
- normalize ordinary anxiety
- help brainstorm options
- avoid forcing campus resources when not necessary
- keep source cards as a backup/support option
- ask one natural follow-up question

Example target style:

```text
That kind of nervousness before meeting someone you like is very normal. It
does not mean something is wrong; it usually means the moment matters to you.

For right now: keep it simple. Decide one easy opener, one genuine question you
can ask her, and one way to exit gracefully if either of you feels awkward.

Want to brainstorm a relaxed opening line, a few questions to ask, or how to
calm down beforehand?
```

## What Still Feels Too Mechanical

- The right panel is still useful for presentation, but too debug-heavy for a
  user-facing product.
- The phrase "Grounded Resources" is better than "Retrieval Sources", but source
  cards still dominate even for ordinary lightweight emotional support.
- Formal resource suggestions can feel excessive for normal life stress.
- Route labels are visible and useful for evaluation, but they should be
  visually secondary to the support plan.
- The app needs a clearer primary "Support Plan" card.

## Recommended Next MVP Improvements

### 1. Add A Support Plan Panel

Make the main user-facing side panel:

- What I heard
- Support path
- For right now
- Backup if this gets heavier
- Optional resource

Keep route/tier/debug details collapsible or lower on the page.

### 2. Add Conversational Micro-Templates

Add message-sensitive templates for:

- ordinary date/social nerves
- roommate/friend conflict
- homesickness
- presentation anxiety
- job/internship rejection
- procrastination shame
- asking for help from a professor/TA

These should not always push counseling resources.

### 3. Add Brainstorm Mode

For low-risk prompts, offer:

- brainstorm what to say
- make a tiny plan
- calm down first

This makes the app feel interactive and meaningful.

### 4. Add Copyable Scripts

For practical routes:

- professor/TA email
- ADS message
- advisor/Ombuds timeline note
- asking a friend for support
- peer-helper safety wording

### 5. De-Emphasize Debug Language

Rename or hide:

- classifier confidence
- registry-filtered
- output guard
- lexical precheck

Keep these available for presentation/evaluation but not as the emotional center
of the UI.

## What Opus Should Review

Ask Opus to focus on:

1. Conversational UX: Does this feel supportive without pretending to be therapy?
2. Route taxonomy: Do we need a `social_anxiety` or `everyday_support` route, or
   is `anxiety_panic` with message-sensitive templates enough?
3. Product framing: Is "guarded support navigator" still too formal for a user
   app?
4. UI hierarchy: What should be first-class for users versus hidden for demo/eval?
5. Evaluation: How do we score conversational usefulness without drifting into
   clinical claims?

## Current Honest Assessment

EmpathRAG Core is a strong technical MVP and a good class project foundation.
It is not yet a delightful support app. The next product leap is not more
retrieval; it is a better conversational planning layer that adapts to everyday
student concerns without making everything feel like a campus-resource triage.
