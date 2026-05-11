# HIPAA / FERPA gap analysis

Honest accounting of the regulatory gaps between the current EmpathRAG
prototype and what a UMD Counseling Center pilot would require. This doc
exists so the gaps are named explicitly rather than hand-waved.

**Bottom line:** EmpathRAG is a research / class prototype and is **not**
HIPAA-compliant, **not** FERPA-compliant for student records, and would
need substantial work to be deployable on UMD infrastructure.

---

## Why we even discuss this

EmpathRAG conversations contain content that is plausibly:

- **Protected health information (PHI)** under HIPAA, when students disclose
  mental-health-adjacent symptoms, treatments, or care relationships.
- **Educational records** under FERPA, when students discuss grades,
  academic standing, disciplinary history, or advisor relationships.
- **Sensitive personal information** that, even if not technically PHI
  or educational records, students reasonably expect to be private.

If UMD CAPS or the CC hosts EmpathRAG and treats it as a pre-intake or
adjunct support tool, the conversations almost certainly become subject
to HIPAA's privacy and security rules.

## Where we fail HIPAA

### Privacy Rule (45 CFR 164.500 et seq.)

| Requirement | Current state | Gap |
|---|---|---|
| Business Associate Agreements (BAAs) with subprocessors | Groq does not sign BAAs for commercial chat completions API; Anthropic API does for some plans but not by default. | **Hard blocker.** No BAA = sending PHI to Groq is itself a violation. |
| Minimum necessary disclosure | Full user messages sent to Groq for paraphrasing; full conversation history sent for context | **Gap.** Could be mitigated by truncation / redaction before send, but only partial. |
| Notice of Privacy Practices | None today; demo has a 1-line scope disclaimer | **Gap.** Required document if deployed; not present. |
| Patient access to records (right to obtain) | Support Plan export is the closest thing; not formally a "designated record set" | **Partial.** Student-facing export exists; structured record-set retention does not. |
| Patient amendment of records | Not supported | **Gap.** |
| Accounting of disclosures | Not logged | **Gap.** |
| Authorization for use beyond TPO (treatment / payment / operations) | No mechanism today | **Gap.** Required for any research / pilot use. |

### Security Rule (45 CFR 164.302 et seq.)

| Requirement | Current state | Gap |
|---|---|---|
| Access control | None — local demo runs on http:// 127.0.0.1 with no auth | **Hard blocker** for deployment. |
| Audit controls | Optional `EMPATHRAG_LOG_TURNS=1` writes raw user text to JSONL | **Gap.** Logs PHI in plain text if enabled. |
| Encryption in transit | Local: http; deployed: depends on reverse proxy | **Gap.** Requires TLS termination + HTTPS-only. |
| Encryption at rest | session_message_history is in-memory; support plan PDF is written unencrypted to OS temp | **Gap.** OS temp dirs are often world-readable on multi-user systems. |
| Workforce training | N/A — no workforce | **N/A.** |
| Incident response | No documented IRP | **Gap.** |
| Backup / contingency | No designed plan | **Gap.** |

### Breach Notification Rule

Not applicable until breach risk exists; once deployed, would need to be addressed.

## Where we fail FERPA

| Requirement | Current state | Gap |
|---|---|---|
| Annual notification of FERPA rights | N/A — not a school yet | **Gap if deployed by UMD.** |
| Records access by parents (under 18) / students | Support plan export exists; no formal records system | **Partial.** |
| Consent before disclosure of education records | No consent flow today | **Gap.** Any disclosure (e.g., to a counselor) needs documented consent. |
| Directory information opt-out | N/A | **N/A.** |

## What it would actually take to close the gaps

### Tier 1: bare minimum for an IRB-light pilot (probably 3-6 months of work)

1. **Custom LLM provider with BAA.** Replace Groq with Azure OpenAI (signs BAA for $$$$), AWS Bedrock (BAA available), or self-hosted Llama 3.3 on UMD-controlled infra. Cost goes from ~$0.0005/turn to ~$0.05-0.10/turn — 100x increase.
2. **TLS-terminated reverse proxy.** Cloudflare in front, https-only.
3. **Auth via terpconnect SSO** so student identity is known and access-controlled.
4. **Server-side session storage with encryption at rest.** Postgres or similar; encrypt user-message columns; access logged.
5. **Explicit consent flow** at the start of every session: scope, data handling, retention, sharing.
6. **Retention policy.** Default delete after N days; student can request earlier deletion.
7. **Audit log** of every access to PHI columns.
8. **IRB-light or full IRB review** depending on whether the pilot generates publishable findings about students.
9. **Notice of Privacy Practices.**
10. **Updated scope disclaimer with HIPAA-specific language.**

### Tier 2: full HIPAA-covered deployment (probably 12+ months and $$$$)

11. All Tier-1, plus:
12. Workforce training records.
13. Documented incident response plan.
14. Disaster recovery / backup plan.
15. Regular security audit / penetration test.
16. Documented data flow / risk assessment.
17. Patient access / amendment / accounting-of-disclosures mechanisms.
18. Sub-BAA chain documented for every dependency that touches PHI.

## What we CAN say today

- The architecture is HIPAA-compatible in design (deterministic planner, no PHI required for routing, all retrieval is from a fixed verified registry, no learned-from-users behavior).
- Crisis intercept never touches the LLM, so PHI in crisis messages never leaves UMD if the LLM provider is the only off-network dependency.
- The Support Plan export is student-controlled — no automatic sharing.
- Server-side persistence is currently zero by design (browser-state only).

These are starting positions, not compliance claims. They make a future
HIPAA-deployed version easier to build; they don't make the current
prototype HIPAA-compliant.

## What we will NOT say

- "EmpathRAG is HIPAA-compliant."
- "Student conversations are private."
- "Your messages stay on your device." (They go to Groq.)
- "This is safe to use for real student support."
