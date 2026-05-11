# Privacy and data flow

Student-facing summary of what data EmpathRAG sees, where it goes, who can
read it, and how long it lasts. Written to be readable by a non-engineer
student or a UMD CAPS clinician deciding whether to recommend the tool.

For the underlying compliance gap analysis, see
`HIPAA_FERPA_GAP_ANALYSIS.md`.

---

## Quick reference

| Data | Where it goes | Who can read it | How long it lasts |
|---|---|---|---|
| Your message text | Browser → local server → Groq Llama 3.3 70B (US-based, commercial) | Engineers running the server; Groq's commercial-tier logs (~30 days per their policy) | Until the server is restarted (browser state); Groq retention separate |
| Voice recording (if used) | Browser microphone → local server → Groq Whisper turbo | Same as above | Same as above; audio file deleted from server after transcription |
| Conversation history | Browser session state only | Only you in your browser tab; no server-side database | Cleared when you click "New conversation" or close the browser tab |
| Diagnostic metadata (route, tier, latency) | Local server in-memory only | Engineer running the server | Until server restart |
| Support Plan export | Your local downloads folder | Only you, unless you share it | Forever, on your device, until you delete it |

**No content is shared with UMD, ISSS, the Counseling Center, or any other
party unless you explicitly download a Support Plan and choose to share it.**

## What actually happens when you send a message

1. You type or speak a message in your browser.
2. The browser sends it over your network connection to the EmpathRAG server.
3. The server runs **Stage-1 safety check** locally (no network call) to look for crisis language.
4. If the message is a crisis disclosure, the server returns a deterministic crisis-response template that includes 988 and UMD Counseling Center contacts. **The crisis message never leaves the server.** The language model is not called.
5. If not crisis, the server runs a **route classifier** (locally; no network call) to decide which support topic this is.
6. The server picks a **planner template** for that route and stage of the conversation.
7. If LLM rephrasing is enabled, the server sends to Groq Llama 3.3 70B (over the internet, to api.groq.com):
   - A fixed system prompt (the "trust boundary" — same every turn).
   - Your current message.
   - The planner's authored response template.
   - The last 2 user-assistant exchanges, if any.
8. Groq returns a paraphrased response over the same connection.
9. The server runs a **post-rephrase safety check** locally on Groq's output. If the check fails, the server falls back to the deterministic template.
10. The server returns the response to your browser. Your browser displays it.
11. Conversation state stays in your browser; nothing is written to a database on the server.

## What is NOT sent to any external service

- Your name, email, or any other identifier — unless you type one into a message.
- Your IP address — only as a routing detail of the HTTPS connection itself; not stored beyond that.
- Any clinical record, terpconnect identity, or UMD-internal data.
- Voice audio after transcription completes — the temp file is deleted.

## What IS sent to Groq

- The text of your message.
- The planner's response template (which is a fixed UMD-resource-aware template; no user-specific data).
- The last 2 user-assistant exchanges for context.
- Voice recordings only if you use the voice feature.

Groq's retention policy for commercial API: per their published policy, completions are retained for ~30 days for abuse-monitoring purposes. They do not train on your data unless you opt in (you do not; we don't enable opt-in).

## What is NOT logged or stored on our side

By default, with `EMPATHRAG_LOG_TURNS` unset:

- No user messages written to disk.
- No response text written to disk.
- No session IDs persisted.
- No timestamps written beyond what shows in console logs.

The "Support Plan" download is the **only** way conversation content leaves
your browser tab durably, and that file goes to your local downloads folder —
not to any server we control.

## Things you should still know

- **This is a research / class prototype.** Privacy guarantees here are
  best-effort architectural choices, not legal compliance with HIPAA or FERPA.
  See `HIPAA_FERPA_GAP_ANALYSIS.md` for the explicit gaps.
- **Your network operator can see that you're talking to EmpathRAG.**
  Standard TLS hides message content from network sniffers but not the
  fact of the connection.
- **Public deployment risks.** If we ever expose this on a public URL,
  multi-user concurrency, session isolation, and rate-limiting are
  things we have not yet hardened. We won't do public deployment without
  closing those.
- **If you are in immediate danger:** the system intercepts crisis
  language and redirects to 988 / UMD Counseling Center. That intercept
  happens on the local server BEFORE any network call to the LLM. **988
  is always the right answer for immediate danger, with or without this
  tool.**

## How to clear your data right now

- Click `↺ New conversation` in the top bar. Conversation state is cleared
  from your browser session immediately.
- Close the browser tab. Session state is destroyed.
- If you downloaded a Support Plan, delete the file from your local
  Downloads folder.

## What we cannot clear

- Whatever Groq retains under their commercial-tier retention policy
  (~30 days, abuse-monitoring purposes, per their published terms).
  We have no API or means to request earlier deletion of that data.

## Contact

Questions: open an issue at https://github.com/MukulRay1603/Empath-RAG.

Concerns about a specific incident or message: include the timestamp from
your conversation and the route + tier shown in the right-hand Live Thread
panel. There is no log on our side to look up; the timestamp and metadata
are what we'd correlate against.
