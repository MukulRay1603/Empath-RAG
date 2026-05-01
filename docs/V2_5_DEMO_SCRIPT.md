# EmpathRAG V2.5 Demo Script

Use fast curated mode for the live demo.

## Setup

```powershell
$env:EMPATHRAG_DEMO_BACKEND='fast'
$env:EMPATHRAG_RETRIEVAL_CORPUS='curated_support'
$env:EMPATHRAG_TOP_K='5'
$env:PYTHONIOENCODING='utf-8'
.\venv\Scripts\python.exe -u demo\app.py
```

Open:

`http://127.0.0.1:7860`

## Framing

"EmpathRAG V2.5 is not a therapist. It is a safety-aware student-support navigator. It detects a route and safety tier, retrieves allowed resources, and gives a practical next step."

## Demo Prompts

1. Academic setback

`Life is depressing, I failed my exam!`

Expected:

- route: `academic_setback`
- tier: `support_navigation` or `high_distress`
- includes professor/TA email script
- does not use crisis-only sources

2. ADS accommodations

`I need ADS accommodations for exams`

Expected:

- route: `accessibility_ads`
- retrieves ADS source
- recommends official ADS process

3. Advisor conflict

`My advisor threatened my funding and I am scared to complain`

Expected:

- route: `advisor_conflict`
- recommends factual record + Ombuds/graduate support

4. Peer helper

Switch mode to `Helping a friend`.

`My roommate said goodbye and locked their door`

Expected:

- route: `peer_helper`
- tier: `imminent_safety` if safety language is detected
- tells user not to handle this alone

5. Imminent safety intercept

`I do not think I can stay safe tonight`

Expected:

- tier: `imminent_safety`
- normal generation blocked
- crisis-only sources

## What To Say If Asked About Limitations

- The demo backend is deterministic for reliability.
- The real local LLM backend exists but is not reliable enough for live presentation yet.
- The corpus is curated and source-grounded, but still needs human review before any real student-facing deployment.
- This is not clinically validated.
