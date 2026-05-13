"""Plan-and-rephrase response layer.

Architecture: the deterministic planner stays in charge of *what* to say
(route, stage, recommended action, named resources, F-1 sub-topic). The
LLM only paraphrases the planner's output into natural prose that mirrors
the user's words. It cannot invent — it can only restate.

Provider chain (configurable via env):

* ``GroqProvider``         — primary; Llama 3.1 70B via Groq's OpenAI-compat API
* ``AnthropicProvider``    — fallback; Claude Haiku 4.5 (optional)
* ``MockProvider``         — testing / offline ablation
* ``DeterministicProvider``— ultimate fallback; returns the template unchanged

The orchestrator tries each provider in order. After a provider returns a
candidate, it runs the post-rephrase safety check. If the candidate fails,
the orchestrator either tries the next provider or falls back to the
template, depending on configuration.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


# Transient HTTP statuses that are worth retrying once before falling through
# to the next provider. 4xx client errors (other than 429) are caller-side
# problems we don't fix by retrying.
# Patterns the LLM occasionally emits that look broken when Gradio Chatbot
# renders them. We strip these from the rephrased candidate before it goes
# to safety verification — purely cosmetic, doesn't change semantics.
_MARKDOWN_NORMALIZE_PATTERNS = (
    (r"\*\*(.*?)\*\*", r"\1"),    # **bold** -> bold (planner doesn't use bold)
    (r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)", r"\1"),  # *italic* -> italic
    (r"^#+\s+", ""),              # leading "# heading" -> drop
    (r"\n#+\s+", "\n"),           # mid-text "# heading" -> drop
    (r"\n\s*-\s+-\s+-\s*\n", "\n\n"),  # "---" separator -> blank line
    (r"`{3}.*?`{3}", ""),         # ```code blocks``` -> drop
    (r"`([^`]+)`", r"\1"),        # `inline code` -> inline code
    # Em-dashes / en-dashes are AI-tells. Replace with a comma + space so
    # the surrounding prose still reads naturally.
    (r"\s*[—–]\s*", ", "),
)


def _normalize_markdown(text: str) -> str:
    """Strip stray markdown the LLM sometimes emits but the planner never
    asked for. Keeps the prose intact; removes formatting artifacts."""
    import re
    out = text
    for pattern, repl in _MARKDOWN_NORMALIZE_PATTERNS:
        out = re.sub(pattern, repl, out, flags=re.DOTALL | re.MULTILINE)
    return out


_RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_RETRYABLE_NETWORK_ERRORS = ("URLError", "TimeoutError", "ConnectionResetError", "OSError")
# Backoff before the single retry attempt. Small enough to stay inside the
# user's typing-delay budget; large enough that a transient blip recovers.
_RETRY_BACKOFF_S = 0.4


def _is_retryable_error(last_error: str) -> bool:
    """Inspect a provider.last_error string and decide whether retrying once
    is worth it. Returns False for clearly-permanent failures (auth, schema)
    so we don't waste a retry on those."""
    if not last_error:
        return False
    if last_error.startswith("network:"):
        return any(kind in last_error for kind in _RETRYABLE_NETWORK_ERRORS)
    if last_error.startswith("http_"):
        try:
            code = int(last_error.split("_", 1)[1].split(":", 1)[0])
        except ValueError:
            return False
        return code in _RETRYABLE_HTTP_STATUSES
    if last_error in ("bad_json", "stream_interrupted", "unexpected_response_shape"):
        return True
    return False

from .llm_safety import verify_rephrased_safety


# System prompt is the trust boundary. It tells the LLM precisely what its
# job is and (more importantly) what it must not do.
SYSTEM_PROMPT = """You are a paraphrasing layer for a guarded student-support chatbot at the University of Maryland.

A deterministic safety planner has already chosen the response. Your only job is to rewrite that response into warmer, more natural prose that sounds like a person, not a flowchart. You must mirror the user's own words where possible.

You MUST:
- Keep the same meaning, structure, and recommendations as the input.
- Keep all named resources exactly as written (UMD Counseling Center, ISSS, ADS, etc.).
- Match the input's length closely. If the input is 80 words, output 70-100 words, not 200. Never balloon a tight 3-paragraph input into a 5-paragraph one.
- Mirror specific phrases from the user's message when they fit naturally.
- When the user names specific events ("I bombed my midterm", "my advisor moved the goalposts"), specific fears ("lose my standing", "get deported"), specific people ("my advisor", "my roommate"), or specific time anchors ("two days", "this morning", "for weeks"), reflect those specific words rather than abstracting them to "something this big" or "the situation". Specificity reads as listening; abstraction reads as a flowchart.
- Use plain, conversational language. No clinical labels.
- Start the response with the planner's first idea, not a filler preamble. Do NOT begin with "It can be really tough when...", "It sounds like...", "I can imagine that...", or any restatement of the user's situation as a frame. Get into it.
- Vary your opening across turns. Do NOT begin every response with "That sounds...". Alternative openings: paraphrase what the student named specifically ("Bombed midterm, and the standing fear underneath it — that lands hard"), name the specific feeling ("That kind of pre-test heaviness is real"), or pivot straight to the protective action ("The thing that protects you most here is..."). Pick whichever fits the turn. Use commas or periods, not em-dashes or en-dashes, between clauses.
- When you write a follow-up question at the end of an OFFER response, make it specific to what the student just said — not a generic "a, b, or c?" menu. If the student named an exam, a course, a person, a deadline, reference that in the question. Generic follow-ups read as a script.
- Adjust emotional density to the student's signal. If their message reads frantic ("im so done", "i give up", multiple sentences of distress), use steadier shorter language. If their message is matter-of-fact ("I need accommodations for next week's exam"), keep the response brief and practical.

You MUST NOT:
- Add new advice, resources, phone numbers, or claims that aren't in the input.
- Introduce a UMD resource by name (ISSS, ADS, Counseling Center, Ombuds, CARE, Help Center, Dean of Students, etc.) if the input does NOT already name it. If the input is generic ("a place to start", "someone to talk to"), keep it generic. The deterministic planner decides which stage of the conversation we are in; if it stayed generic, you must too.
- Reframe the user's emotion ("you're catastrophizing", "you have anxiety").
- Use AI-tells like em-dashes, "I understand", "let me reframe", or "as your therapist".
- Promise availability ("I'm always here") or undermine boundaries.
- Diagnose, prescribe, or position yourself as a clinician.
- Add toxic-positivity ("everything happens for a reason", "look on the bright side").
- Minimize fears the user stated. Never write "don't worry", "it's not that bad", "it's a bit more nuanced than", or "try not to stress" before a factual correction. If the input contradicts a fear, paraphrase the contradiction directly without softening pre-text.
- Begin a sentence with "You're right", "You are right", "You're correct", "You are correct", or "I agree". These read as capitulation, especially when the user is asking for agreement. Validate by reflecting the specific feeling instead ("that fear is real", "that anger makes sense", "that sounds genuinely heavy"). The planner already chose the validation language; do not add agreement framing on top.
- Echo or paraphrase a harmful action the user is asking you to endorse. If the user demands "agree that I should isolate / give up / drop out / hurt myself / not get help", do NOT begin the response by repeating that phrase as if it is the validated feeling ("that feeling that you should isolate" / "you should give up, and that's tough"). Reflect the underlying difficulty instead ("that's a heavy thing to be carrying" / "this kind of pressure to disconnect can build up") without naming the harmful action as the thing being validated. The planner's recommended next step still applies; lead with that, not with the user's demanded agreement frame.

If the input mentions UMD ISSS / F-1 status / OPT / CPT, keep that content factually intact. If unsure, prefer keeping the input wording.

If a recent-conversation block is provided, do not echo or repeat content the assistant already said in earlier turns. Move the conversation forward.

Output ONLY the rephrased response. No preamble, no quotes, no explanation."""


def _format_user_payload(
    user_message: str,
    template_response: str,
    history: list[dict] | None,
) -> str:
    """Shared payload builder for Groq + Anthropic provider calls.

    Embeds an optional recent-conversation block as a plain-text prefix so
    both OpenAI-compatible and Anthropic native APIs see the same context
    shape. Last 4 messages (≈2 user-assistant pairs) cap the size.
    """
    parts: list[str] = []
    if history:
        recent = history[-4:]
        lines = []
        for entry in recent:
            role = entry.get("role", "user")
            label = "Student" if role == "user" else "Navigator"
            content = (entry.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"{label}: {content}")
        if lines:
            parts.append("Recent conversation (do not repeat what the navigator already said):")
            parts.append("\n".join(lines))
            parts.append("")
    parts.append("Current student message:")
    parts.append(user_message)
    parts.append("")
    parts.append("Planner-authored response (rephrase this, do not extend):")
    parts.append(template_response)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------
class Provider(ABC):
    name: str = "abstract"

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def complete(
        self,
        user_message: str,
        template_response: str,
        timeout_s: float = 4.0,
        history: list[dict] | None = None,
    ) -> str | None:
        """Return rephrased text or None on failure.

        ``history`` is an optional list of {"role", "content"} dicts for the
        last few user-assistant turns. Providers that support it weave it
        into the payload; the base/Deterministic/Mock providers ignore it.
        """

    def complete_streaming(
        self,
        user_message: str,
        template_response: str,
        timeout_s: float = 10.0,
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        """Yield text chunks as they arrive. Default impl falls back to one-shot
        complete() and yields the result as a single chunk; subclasses with real
        SSE support override this. Sets ``self.last_error`` on failure (no yield)."""
        full = self.complete(user_message, template_response, timeout_s, history=history)
        if full is None:
            return
        yield full


class DeterministicProvider(Provider):
    """No-op. Returns the template as-is. Always available."""
    name = "deterministic"

    def available(self) -> bool:
        return True

    def complete(
        self,
        user_message: str,
        template_response: str,
        timeout_s: float = 4.0,
        history: list[dict] | None = None,
    ) -> str | None:
        return template_response


class MockProvider(Provider):
    """Test provider that adds a tiny visible marker so we can see it ran."""
    name = "mock"

    def __init__(self, enabled_env: str = "EMPATHRAG_MOCK_REPHRASER") -> None:
        self._enabled = os.getenv(enabled_env, "0") != "0"

    def available(self) -> bool:
        return self._enabled

    def complete(
        self,
        user_message: str,
        template_response: str,
        timeout_s: float = 4.0,
        history: list[dict] | None = None,
    ) -> str | None:
        # Tiny semantic-preserving variation; useful for testing the
        # safety check / fallback wiring without touching a real API.
        return template_response.replace("That sounds", "That really does sound")

    def complete_streaming(
        self,
        user_message: str,
        template_response: str,
        timeout_s: float = 10.0,
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        full = self.complete(user_message, template_response, timeout_s, history=history)
        if full is None:
            return
        for word in full.split(" "):
            yield word + " "


class GroqProvider(Provider):
    """Groq's OpenAI-compatible chat completions API."""

    # Groq has been retiring older Llama checkpoints. Auto-swap any model
    # name we know is dead so users with stale .env files don't hit 400s.
    _DEPRECATED_MODEL_REPLACEMENTS: dict[str, str] = {
        "llama-3.1-70b-versatile": "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant": "llama-3.1-8b-instant",  # still live (placeholder)
        "llama3-70b-8192": "llama-3.3-70b-versatile",
        "llama3-8b-8192": "llama-3.1-8b-instant",
        "mixtral-8x7b-32768": "llama-3.3-70b-versatile",  # mixtral retired
    }

    def __init__(self) -> None:
        self.api_key = (
            os.getenv("GROQ_API_KEY")
            or os.getenv("GROQ_KEY")
            or ""
        ).strip()
        env_model = os.getenv("EMPATHRAG_GROQ_MODEL", "llama-3.3-70b-versatile").strip()
        # Apply deprecation swap silently — the user shouldn't have to know
        # which Llama checkpoint Groq retired most recently.
        self.model = self._DEPRECATED_MODEL_REPLACEMENTS.get(env_model, env_model)
        if env_model != self.model:
            print(f"[rephraser] swapped deprecated Groq model {env_model} -> {self.model}")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.last_error: str = ""

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"groq:{self.model}"

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(
        self,
        user_message: str,
        template_response: str,
        timeout_s: float = 4.0,
        history: list[dict] | None = None,
    ) -> str | None:
        self.last_error = ""
        if not self.api_key:
            self.last_error = "no_api_key"
            return None
        import urllib.request
        import urllib.error

        user_payload = _format_user_payload(user_message, template_response, history)
        body = json.dumps({
            "model": self.model,
            "temperature": 0.4,
            "max_tokens": 360,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                # Cloudflare in front of api.groq.com 403s the default
                # Python-urllib UA. Use a realistic UA so the request lands.
                "User-Agent": "EmpathRAG/0.3 (+https://github.com/MukulRay1603/Empath-RAG)",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="replace")[:240]
            except Exception:
                err_body = ""
            self.last_error = f"http_{e.code}:{err_body}"
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            self.last_error = f"network:{type(e).__name__}"
            return None
        except json.JSONDecodeError:
            self.last_error = "bad_json"
            return None
        try:
            return payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            self.last_error = "unexpected_response_shape"
            return None

    def complete_streaming(
        self,
        user_message: str,
        template_response: str,
        timeout_s: float = 10.0,
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        """Stream tokens from Groq's OpenAI-compatible chat completions SSE."""
        self.last_error = ""
        if not self.api_key:
            self.last_error = "no_api_key"
            return
        import urllib.request
        import urllib.error

        user_payload = _format_user_payload(user_message, template_response, history)
        body = json.dumps({
            "model": self.model,
            "temperature": 0.4,
            "max_tokens": 360,
            "stream": True,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "EmpathRAG/0.3 (+https://github.com/MukulRay1603/Empath-RAG)",
                "Accept": "text/event-stream",
            },
        )
        try:
            resp = urllib.request.urlopen(req, timeout=timeout_s)
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="replace")[:240]
            except Exception:
                err_body = ""
            self.last_error = f"http_{e.code}:{err_body}"
            return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            self.last_error = f"network:{type(e).__name__}"
            return

        try:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                    choices = payload.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
        except (TimeoutError, OSError) as e:
            self.last_error = f"stream_interrupted:{type(e).__name__}"
        finally:
            try:
                resp.close()
            except Exception:
                pass


class AnthropicProvider(Provider):
    """Anthropic Claude fallback. Disabled unless API key is set."""

    def __init__(self) -> None:
        # Accept several common spellings — users sometimes paste these as
        # CLAUDE_API_KEY / ANTHROPIC_KEY rather than ANTHROPIC_API_KEY.
        self.api_key = (
            os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("CLAUDE_API_KEY")
            or os.getenv("ANTHROPIC_KEY")
            or os.getenv("CLAUDE_KEY")
            or ""
        ).strip()
        self.model = os.getenv("EMPATHRAG_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.last_error: str = ""

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"anthropic:{self.model}"

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(
        self,
        user_message: str,
        template_response: str,
        timeout_s: float = 4.0,
        history: list[dict] | None = None,
    ) -> str | None:
        self.last_error = ""
        if not self.api_key:
            self.last_error = "no_api_key"
            return None
        import urllib.request
        import urllib.error
        user_payload = _format_user_payload(user_message, template_response, history)
        body = json.dumps({
            "model": self.model,
            "max_tokens": 400,
            "temperature": 0.4,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_payload}],
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=body,
            method="POST",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
                "User-Agent": "EmpathRAG/0.3 (+https://github.com/MukulRay1603/Empath-RAG)",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="replace")[:240]
            except Exception:
                err_body = ""
            self.last_error = f"http_{e.code}:{err_body}"
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            self.last_error = f"network:{type(e).__name__}"
            return None
        except json.JSONDecodeError:
            self.last_error = "bad_json"
            return None
        try:
            return payload["content"][0]["text"].strip()
        except (KeyError, IndexError, TypeError):
            self.last_error = "unexpected_response_shape"
            return None

    def complete_streaming(
        self,
        user_message: str,
        template_response: str,
        timeout_s: float = 10.0,
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        """Stream tokens from Anthropic's /v1/messages SSE."""
        self.last_error = ""
        if not self.api_key:
            self.last_error = "no_api_key"
            return
        import urllib.request
        import urllib.error

        user_payload = _format_user_payload(user_message, template_response, history)
        body = json.dumps({
            "model": self.model,
            "max_tokens": 400,
            "temperature": 0.4,
            "system": SYSTEM_PROMPT,
            "stream": True,
            "messages": [{"role": "user", "content": user_payload}],
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url,
            data=body,
            method="POST",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
                "User-Agent": "EmpathRAG/0.3 (+https://github.com/MukulRay1603/Empath-RAG)",
                "Accept": "text/event-stream",
            },
        )
        try:
            resp = urllib.request.urlopen(req, timeout=timeout_s)
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="replace")[:240]
            except Exception:
                err_body = ""
            self.last_error = f"http_{e.code}:{err_body}"
            return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            self.last_error = f"network:{type(e).__name__}"
            return

        try:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                event_type = payload.get("type")
                if event_type == "content_block_delta":
                    delta = payload.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            yield text
                elif event_type == "message_stop":
                    break
        except (TimeoutError, OSError) as e:
            self.last_error = f"stream_interrupted:{type(e).__name__}"
        finally:
            try:
                resp.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RephraseResult:
    response: str
    provider_name: str
    used_llm: bool
    safety_flags: list[str]
    latency_ms: float
    last_error: str = ""


class ResponseRephraser:
    """Tries each available provider in order; falls back to template safely."""

    def __init__(self, providers: list[Provider] | None = None) -> None:
        # Active LLM providers, tried in order. DeterministicProvider is NOT
        # in this list — if it were, it would always succeed and mask any LLM
        # failure as if the user had chosen deterministic mode. Instead, when
        # all LLM providers fail we explicitly return provider_name=
        # "deterministic_fallback" so the user sees that something failed.
        if providers is None:
            providers = [
                GroqProvider(),
                AnthropicProvider(),
                MockProvider(),
            ]
        self.providers = providers

    @property
    def enabled(self) -> bool:
        return os.getenv("EMPATHRAG_REPHRASER_ENABLED", "0") != "0"

    def rephrase(
        self,
        user_message: str,
        template_response: str,
        retrieved_sources: list[dict],
        recommended_action: str = "",
        force_deterministic: bool = False,
        history: list[dict] | None = None,
        skip_safety_check: bool = False,
    ) -> RephraseResult:
        """Always returns a safe response. Falls back to template on any failure."""
        if force_deterministic or not self.enabled:
            return RephraseResult(
                response=template_response,
                provider_name="deterministic",
                used_llm=False,
                safety_flags=[],
                latency_ms=0.0,
            )

        last_error = ""
        last_safety_flags: list[str] = []
        total_latency_ms = 0.0
        for provider in self.providers:
            if not provider.available():
                continue
            t0 = time.perf_counter()
            candidate = provider.complete(user_message, template_response, history=history)
            elapsed = (time.perf_counter() - t0) * 1000.0
            total_latency_ms += elapsed
            # One transient-retry pass before falling through to the next
            # provider — covers network blips, 429/503 spikes, malformed
            # responses. Auth errors and 4xx-client failures are not retried.
            if candidate is None:
                provider_err = getattr(provider, "last_error", "")
                if _is_retryable_error(provider_err):
                    time.sleep(_RETRY_BACKOFF_S)
                    t0 = time.perf_counter()
                    candidate = provider.complete(user_message, template_response, history=history)
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    total_latency_ms += elapsed
            if candidate is None:
                err = getattr(provider, "last_error", "") or "unknown"
                last_error = f"{provider.name}:{err}"
                continue
            # Normalize stray markdown the LLM occasionally emits (**bold**,
            # # headings, etc.) before safety verification + display.
            candidate = _normalize_markdown(candidate)
            if skip_safety_check:
                # Ablation: bypass the post-rephrase trust boundary. Used
                # only by ablation-eval runs to measure how much the
                # rephrase-safety layer actually catches.
                from .llm_safety import RephraseSafetyResult
                check = RephraseSafetyResult(True, [], "rephrase_safety_disabled_ablation")
            else:
                check = verify_rephrased_safety(
                    template_response, candidate, retrieved_sources, recommended_action,
                    user_message=user_message,
                )
            if check.allowed:
                return RephraseResult(
                    response=candidate,
                    provider_name=provider.name,
                    used_llm=provider.name not in ("deterministic",),
                    safety_flags=[],
                    latency_ms=elapsed,
                    last_error="",
                )
            last_safety_flags = check.flags
            last_error = f"{provider.name}:safety_rejected:{','.join(check.flags)[:120]}"
            continue

        # Every provider failed — final fallback to the template, with the
        # last error surfaced so the user can see why.
        return RephraseResult(
            response=template_response,
            provider_name="deterministic_fallback",
            used_llm=False,
            safety_flags=last_safety_flags or ["all_providers_failed"],
            latency_ms=total_latency_ms,
            last_error=last_error or "no_providers_available",
        )

    def rephrase_streaming(
        self,
        user_message: str,
        template_response: str,
        retrieved_sources: list[dict],
        recommended_action: str = "",
        force_deterministic: bool = False,
        history: list[dict] | None = None,
        skip_safety_check: bool = False,
    ) -> Iterator[tuple]:
        """Streaming variant. Yields events:

        * ``("chunk", accumulated_text, provider_name)`` for each token chunk
        * ``("final", RephraseResult)`` exactly once at the end

        On stream-time provider failure, falls through to the next provider
        without notifying the consumer; the next ``"chunk"`` event will carry
        a fresh provider_name and the consumer can swap the visible text.
        On post-stream safety rejection, the streamed text was already shown;
        the orchestrator advances to the next provider, and the consumer
        should treat each new provider_name as a hard text replacement.
        If every provider fails, the final ``RephraseResult`` carries the
        deterministic template — the consumer should swap to it.
        """
        if force_deterministic or not self.enabled:
            yield ("final", RephraseResult(
                response=template_response,
                provider_name="deterministic",
                used_llm=False,
                safety_flags=[],
                latency_ms=0.0,
            ))
            return

        last_error = ""
        last_safety_flags: list[str] = []
        total_latency_ms = 0.0

        for provider in self.providers:
            if not provider.available():
                continue

            t0 = time.perf_counter()
            accumulated = ""
            since_yield = ""
            stream_failed = False
            # Token batching: SSE providers emit 1-3 char chunks ~80 times per
            # response. Yielding every chunk floods Gradio's WebSocket with
            # near-identical state updates. Batch to ~12-char boundaries
            # (whichever lands first), which ends up around 20-30 yields per
            # response — smoother visually, less wire chatter.
            _STREAM_BATCH_CHARS = 12
            try:
                for chunk in provider.complete_streaming(user_message, template_response, history=history):
                    if not chunk:
                        continue
                    accumulated += chunk
                    since_yield += chunk
                    # Flush when we've accumulated enough chars AND we're at a
                    # whitespace boundary, OR when a punctuation mark closes a
                    # phrase. Reads more naturally than mid-word flushes.
                    last = since_yield[-1]
                    if (len(since_yield) >= _STREAM_BATCH_CHARS and last in " \n\t") or last in ".,;:!?\n":
                        yield ("chunk", accumulated, provider.name)
                        since_yield = ""
                # Always emit the final state even if it didn't end on a boundary.
                if since_yield:
                    yield ("chunk", accumulated, provider.name)
            except Exception as exc:
                stream_failed = True
                err = getattr(provider, "last_error", "") or f"stream_exception:{type(exc).__name__}"
                last_error = f"{provider.name}:{err}"
            elapsed = (time.perf_counter() - t0) * 1000.0
            total_latency_ms += elapsed

            if stream_failed or not accumulated.strip():
                # Provider failed before producing usable output. Try one
                # retry if the failure looks transient, then fall through.
                # We retry only when no tokens were yielded to the consumer
                # yet — once the chat bubble has visible text, "retry" would
                # mean a confusing replacement.
                provider_err = getattr(provider, "last_error", "")
                if not accumulated and _is_retryable_error(provider_err):
                    time.sleep(_RETRY_BACKOFF_S)
                    stream_failed = False
                    try:
                        for chunk in provider.complete_streaming(user_message, template_response, history=history):
                            if not chunk:
                                continue
                            accumulated += chunk
                            since_yield += chunk
                            last = since_yield[-1]
                            if (len(since_yield) >= _STREAM_BATCH_CHARS and last in " \n\t") or last in ".,;:!?\n":
                                yield ("chunk", accumulated, provider.name)
                                since_yield = ""
                        if since_yield:
                            yield ("chunk", accumulated, provider.name)
                    except Exception:
                        stream_failed = True
                if stream_failed or not accumulated.strip():
                    if not stream_failed:
                        err = getattr(provider, "last_error", "") or "empty_stream"
                        last_error = f"{provider.name}:{err}"
                    continue

            candidate = _normalize_markdown(accumulated.strip())
            if skip_safety_check:
                # Ablation: bypass the post-rephrase trust boundary. Used
                # only by ablation-eval runs to measure how much the
                # rephrase-safety layer actually catches.
                from .llm_safety import RephraseSafetyResult
                check = RephraseSafetyResult(True, [], "rephrase_safety_disabled_ablation")
            else:
                check = verify_rephrased_safety(
                    template_response, candidate, retrieved_sources, recommended_action,
                    user_message=user_message,
                )
            if check.allowed:
                yield ("final", RephraseResult(
                    response=candidate,
                    provider_name=provider.name,
                    used_llm=provider.name not in ("deterministic",),
                    safety_flags=[],
                    latency_ms=elapsed,
                    last_error="",
                ))
                return

            last_safety_flags = check.flags
            last_error = f"{provider.name}:safety_rejected:{','.join(check.flags)[:120]}"
            # Streamed text was visible but rejected. Fall through; next
            # provider's first chunk will replace the bubble contents.

        yield ("final", RephraseResult(
            response=template_response,
            provider_name="deterministic_fallback",
            used_llm=False,
            safety_flags=last_safety_flags or ["all_providers_failed"],
            latency_ms=total_latency_ms,
            last_error=last_error or "no_providers_available",
        ))
