"""Voice transcription via Groq Whisper turbo.

Same Groq key as the text rephraser, same Cloudflare-aware UA. Multipart
upload of the recorded audio file; returns plain text.

Usage:

    transcriber = GroqWhisperTranscriber()
    if transcriber.available():
        text = transcriber.transcribe(audio_path)

Privacy: the audio file is sent to Groq for transcription. The local copy is
the user's own browser-side recording surfaced by Gradio; we never persist it
beyond the request.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranscriptionResult:
    text: str
    provider: str
    latency_ms: float
    error: str = ""

    def ok(self) -> bool:
        return bool(self.text) and not self.error


class GroqWhisperTranscriber:
    """Groq Whisper transcription endpoint (OpenAI-compatible)."""

    # whisper-large-v3-turbo is ~200x realtime, ~$0.0007/min, multilingual.
    DEFAULT_MODEL = "whisper-large-v3-turbo"
    ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"

    # Reject obviously-bogus transcripts: Whisper sometimes hallucinates a
    # default phrase ("Thank you for watching", "Thanks for watching") on
    # silence or noise. Caught here so they don't reach the chat composer.
    _HALLUCINATED_FALLBACKS = (
        "thank you for watching",
        "thanks for watching",
        "thank you.",
        ".",
    )

    def __init__(self) -> None:
        self.api_key = (
            os.getenv("GROQ_API_KEY")
            or os.getenv("GROQ_KEY")
            or ""
        ).strip()
        self.model = os.getenv("EMPATHRAG_WHISPER_MODEL", self.DEFAULT_MODEL).strip()

    def available(self) -> bool:
        return bool(self.api_key)

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        timeout_s: float = 30.0,
    ) -> TranscriptionResult:
        """Transcribe a local audio file. Returns text or an error result."""
        import time

        if not self.api_key:
            return TranscriptionResult(text="", provider="groq_whisper",
                                       latency_ms=0.0, error="no_api_key")
        path = Path(audio_path)
        if not path.exists() or path.stat().st_size == 0:
            return TranscriptionResult(text="", provider="groq_whisper",
                                       latency_ms=0.0, error="empty_or_missing_audio")

        # `requests` handles multipart cleanly; urllib's stdlib multipart is
        # painful and the rephraser already pulls in this dependency tree.
        import requests  # type: ignore

        t0 = time.perf_counter()
        try:
            with path.open("rb") as fh:
                data: dict = {"model": self.model, "response_format": "json"}
                if language:
                    data["language"] = language
                resp = requests.post(
                    self.ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "User-Agent": "EmpathRAG/0.3 (+https://github.com/MukulRay1603/Empath-RAG)",
                    },
                    files={"file": (path.name, fh, "application/octet-stream")},
                    data=data,
                    timeout=timeout_s,
                )
        except requests.RequestException as e:  # type: ignore
            return TranscriptionResult(text="", provider="groq_whisper",
                                       latency_ms=(time.perf_counter() - t0) * 1000.0,
                                       error=f"network:{type(e).__name__}")

        elapsed = (time.perf_counter() - t0) * 1000.0
        if resp.status_code >= 400:
            return TranscriptionResult(text="", provider=f"groq_whisper:{self.model}",
                                       latency_ms=elapsed,
                                       error=f"http_{resp.status_code}:{resp.text[:240]}")
        try:
            payload = resp.json()
            text = (payload.get("text") or "").strip()
        except ValueError:
            return TranscriptionResult(text="", provider=f"groq_whisper:{self.model}",
                                       latency_ms=elapsed, error="bad_json")

        # Length sanity + hallucination filter.
        if len(text) < 2:
            return TranscriptionResult(text="", provider=f"groq_whisper:{self.model}",
                                       latency_ms=elapsed, error="empty_transcript")
        lowered = text.lower().strip()
        if lowered in self._HALLUCINATED_FALLBACKS:
            return TranscriptionResult(text="", provider=f"groq_whisper:{self.model}",
                                       latency_ms=elapsed,
                                       error=f"likely_hallucination:{text[:40]}")

        # Cap at a reasonable length so a runaway transcript doesn't explode
        # downstream prompts.
        if len(text) > 4000:
            text = text[:4000].rstrip() + "..."

        return TranscriptionResult(text=text, provider=f"groq_whisper:{self.model}",
                                   latency_ms=elapsed)
