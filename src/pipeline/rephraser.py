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
from dataclasses import dataclass

from .llm_safety import verify_rephrased_safety


# System prompt is the trust boundary. It tells the LLM precisely what its
# job is and (more importantly) what it must not do.
SYSTEM_PROMPT = """You are a paraphrasing layer for a guarded student-support chatbot at the University of Maryland.

A deterministic safety planner has already chosen the response. Your only job is to rewrite that response into warmer, more natural prose that sounds like a person, not a flowchart. You must mirror the user's own words where possible.

You MUST:
- Keep the same meaning, structure, and recommendations as the input.
- Keep all named resources exactly as written (UMD Counseling Center, ISSS, ADS, etc.).
- Keep the response short (2-4 short paragraphs maximum).
- Mirror specific phrases from the user's message when they fit naturally.
- Use plain, conversational language. No clinical labels.

You MUST NOT:
- Add new advice, resources, phone numbers, or claims that aren't in the input.
- Reframe the user's emotion ("you're catastrophizing", "you have anxiety").
- Use AI-tells like em-dashes, "I understand", "let me reframe", or "as your therapist".
- Promise availability ("I'm always here") or undermine boundaries.
- Diagnose, prescribe, or position yourself as a clinician.
- Add toxic-positivity ("everything happens for a reason", "look on the bright side").

If the input mentions UMD ISSS / F-1 status / OPT / CPT, keep that content factually intact. If unsure, prefer keeping the input wording.

Output ONLY the rephrased response. No preamble, no quotes, no explanation."""


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------
class Provider(ABC):
    name: str = "abstract"

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def complete(self, user_message: str, template_response: str, timeout_s: float = 4.0) -> str | None:
        """Return rephrased text or None on failure."""


class DeterministicProvider(Provider):
    """No-op. Returns the template as-is. Always available."""
    name = "deterministic"

    def available(self) -> bool:
        return True

    def complete(self, user_message: str, template_response: str, timeout_s: float = 4.0) -> str | None:
        return template_response


class MockProvider(Provider):
    """Test provider that adds a tiny visible marker so we can see it ran."""
    name = "mock"

    def __init__(self, enabled_env: str = "EMPATHRAG_MOCK_REPHRASER") -> None:
        self._enabled = os.getenv(enabled_env, "0") != "0"

    def available(self) -> bool:
        return self._enabled

    def complete(self, user_message: str, template_response: str, timeout_s: float = 4.0) -> str | None:
        # Tiny semantic-preserving variation; useful for testing the
        # safety check / fallback wiring without touching a real API.
        return template_response.replace("That sounds", "That really does sound")


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

    def complete(self, user_message: str, template_response: str, timeout_s: float = 4.0) -> str | None:
        self.last_error = ""
        if not self.api_key:
            self.last_error = "no_api_key"
            return None
        import urllib.request
        import urllib.error

        user_payload = (
            f"User message:\n{user_message}\n\n"
            f"Planner-authored response (rephrase this, do not extend):\n{template_response}"
        )
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

    def complete(self, user_message: str, template_response: str, timeout_s: float = 4.0) -> str | None:
        self.last_error = ""
        if not self.api_key:
            self.last_error = "no_api_key"
            return None
        import urllib.request
        import urllib.error
        user_payload = (
            f"User message:\n{user_message}\n\n"
            f"Planner-authored response (rephrase this, do not extend):\n{template_response}"
        )
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
            candidate = provider.complete(user_message, template_response)
            elapsed = (time.perf_counter() - t0) * 1000.0
            total_latency_ms += elapsed
            if candidate is None:
                # Capture the error for surfacing in diagnostics.
                err = getattr(provider, "last_error", "") or "unknown"
                last_error = f"{provider.name}:{err}"
                continue
            check = verify_rephrased_safety(
                template_response, candidate, retrieved_sources, recommended_action
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
