# Direction: counselor-assist deployment

Captured 2026-05-10 during V4 final review. Not pivoting current scope; this
documents where the project could go after the class demo.

## The shift

EmpathRAG today is framed as a **support-navigation prototype for students**:
the student talks to the system, the system reflects + points to UMD resources.

A stronger product framing emerged from review: position EmpathRAG as a
**counselor-assist tool for the UMD Counseling Center to host on their website**.

In this framing:

- Students arrive at the Counseling Center's site, optionally authenticated.
- They talk to EmpathRAG before scheduling a session, or while waiting for
  an opening.
- At the end of the conversation, the system produces a **PDF intake summary**
  the student can share with their assigned counselor (or that the system
  hands directly to the counselor via the booking flow).
- The counselor reviews the summary alongside the student's profile **before**
  the in-person session, so the session starts further along than "what's
  going on?"
- The PDF includes: what the student is working on (their own words), what
  they've tried, what the navigator surfaced (route, F-1 sub-topic, suggested
  next steps), resources mentioned.

Reference: the Anthropic claude.ai hackathon entries for the Cerebral AI
event — they used the model as a structured pre-intake layer that produced
a downloadable artifact for the clinician.

## What V4 already supports

- **Support Plan export** (Markdown + PDF) — already produces the artifact.
  The PDF format is the counselor-handoff format.
- **Stage-aware planner** (LISTEN / PERMISSION / OFFER / CLARIFY) — already
  shapes the conversation in a pre-intake-friendly arc.
- **Resource registry filter** — already keeps the system within UMD's
  authoritative resource set.
- **Crisis intercept** — already routes imminent-safety messages to 988 /
  human support without going through the LLM.

## What this direction would require beyond V4

1. **Student authentication** — terpconnect SSO or similar so the system can
   key conversations to a student profile.
2. **Counselor side** — a small console showing pending intake summaries,
   linked to upcoming session bookings.
3. **Student opt-in to share** — the artifact is the student's record;
   sharing with a specific counselor must be an explicit consent action.
4. **Booking integration** — handoff into whatever scheduling system the
   Counseling Center uses (Penn State uses Titanium; UMD CAPS' system may
   be similar).
5. **HIPAA / FERPA review** — pre-intake notes about mental health are
   sensitive. Storage, retention, deletion, and access controls would need
   to meet whatever the Counseling Center already operates under.
6. **Clinician walkthrough + sign-off** — the existing post-MVP item #1.
   Now extra-load-bearing because clinicians would be downstream consumers,
   not just adjacent reviewers.
7. **Service-level redesign of the framing copy** — current copy speaks to
   the student directly. Counselor-facing copy would emphasize "before your
   session" / "what your student already named."

## Why this framing is stronger than student-facing standalone

- The student doesn't have to decide whether to trust a chatbot to "really
  help" — the system explicitly hands off to their real counselor.
- The Counseling Center's existing safety net catches anything the system
  misses.
- The PDF artifact is genuinely useful for a 50-minute session that would
  otherwise spend the first 15 minutes on intake questions.
- It avoids the "this is too risky to deploy student-facing" objection that
  any stand-alone mental-health-adjacent chatbot will face.

## When to revisit

After the class demo. The CAPS clinician walkthrough (post-MVP #1) is the
gate — if a CAPS clinician says "yes, this would help my pre-session
preparation," the direction is worth committing engineering time to.

If they don't see clinical value in the pre-intake artifact, the
student-facing navigator framing remains valid for the class / paper.
