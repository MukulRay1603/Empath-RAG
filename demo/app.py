"""
demo/app.py
Gradio interface for EmpathRAG V2.
"""

from __future__ import annotations

import datetime
import json
import os
import sqlite3
import sys
import threading
import uuid
from html import escape
from pathlib import Path

import gradio as gr

sys.path.insert(0, "src")

from pipeline.safety_policy import SafetyLevel, SafetyTriagePolicy
from pipeline.core import EmpathRAGCore
from pipeline.output_guard import validate_output
from pipeline.service_graph import match_services
from pipeline.v2_schema import (
    SafetyTier,
    SupportRoute,
    classify_route,
    map_safety_level,
)


LABEL_COLORS = {
    "distress": "#fb7185",
    "anxiety": "#f59e0b",
    "frustration": "#a78bfa",
    "neutral": "#94a3b8",
    "hopeful": "#34d399",
}

LOG_PATH = "eval/human_eval_log.jsonl"
LOG_TURNS = os.getenv("EMPATHRAG_LOG_TURNS") == "1"
SHARE_DEMO = os.getenv("EMPATHRAG_SHARE") == "1"
RETRIEVAL_CORPUS = os.getenv("EMPATHRAG_RETRIEVAL_CORPUS", "auto")
DEMO_TOP_K = int(os.getenv("EMPATHRAG_TOP_K", "5"))
DEMO_MAX_TOKENS = int(os.getenv("EMPATHRAG_MAX_TOKENS", "140"))
DEMO_BACKEND = os.getenv("EMPATHRAG_DEMO_BACKEND", "fast").strip().lower()
CURATED_DB_PATH = Path(os.getenv("EMPATHRAG_CURATED_DB", "data/curated/indexes/metadata_curated.db"))

APP_CSS = """
:root {
  --er-void: #030712;
  --er-space: #07111f;
  --er-deep: #0b1728;
  --er-panel: rgba(8, 20, 34, 0.76);
  --er-panel-solid: #0d1b2d;
  --er-panel-lift: rgba(14, 33, 52, 0.88);
  --er-panel-2: rgba(3, 12, 24, 0.56);
  --er-ink: #f3fbff;
  --er-muted: #9eb4c7;
  --er-soft: #c8d9e8;
  --er-dim: #71869a;
  --er-line: rgba(148, 219, 233, 0.18);
  --er-line-strong: rgba(45, 212, 191, 0.46);
  --er-turquoise: #2dd4bf;
  --er-cyan: #22d3ee;
  --er-blue: #38bdf8;
  --er-amber: #f59e0b;
  --er-rose: #fb7185;
  --er-violet: #a78bfa;
}

html, body {
  min-height: 100% !important;
  background:
    linear-gradient(115deg, rgba(45,212,191,0.10), transparent 34%),
    linear-gradient(245deg, rgba(56,189,248,0.12), transparent 30%),
    linear-gradient(180deg, #030712 0%, #07111f 46%, #0b1728 100%) !important;
  color: var(--er-ink) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

body::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  background-image:
    radial-gradient(circle at 14% 18%, rgba(125, 249, 233, 0.78) 0 1px, transparent 1.5px),
    radial-gradient(circle at 78% 12%, rgba(186, 230, 253, 0.70) 0 1px, transparent 1.5px),
    radial-gradient(circle at 48% 32%, rgba(45, 212, 191, 0.58) 0 1px, transparent 1.4px),
    radial-gradient(circle at 88% 64%, rgba(167, 139, 250, 0.55) 0 1px, transparent 1.4px),
    radial-gradient(circle at 21% 78%, rgba(56, 189, 248, 0.58) 0 1px, transparent 1.4px),
    linear-gradient(rgba(45,212,191,0.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(45,212,191,0.045) 1px, transparent 1px);
  background-size: auto, auto, auto, auto, auto, 72px 72px, 72px 72px;
  mask-image: linear-gradient(to bottom, rgba(0,0,0,0.92), rgba(0,0,0,0.38));
}

body::after {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  background:
    linear-gradient(100deg, transparent 0 38%, rgba(45,212,191,0.10) 38.2%, transparent 39% 100%),
    linear-gradient(144deg, transparent 0 64%, rgba(56,189,248,0.08) 64.2%, transparent 65% 100%);
  opacity: 0.85;
}

.gradio-container {
  position: relative;
  z-index: 1;
  min-height: 100% !important;
  background: transparent !important;
  color: var(--er-ink) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.gradio-container {
  max-width: 1360px !important;
  margin: 0 auto !important;
  padding: 0 22px 28px !important;
}
.gradio-container * {
  border-color: var(--er-line);
}
.gradio-container label,
.gradio-container p,
.gradio-container span,
.gradio-container div,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container textarea,
.gradio-container input {
  color: var(--er-ink);
}
.gradio-container .wrap,
.gradio-container .contain,
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .tabs,
.gradio-container .tabitem {
  background: transparent !important;
  border-color: var(--er-line) !important;
}

.gradio-container label {
  color: var(--er-muted) !important;
}

.er-shell {
  padding: 26px 0 16px;
}
.er-title {
  position: relative;
  overflow: hidden;
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(320px, 0.85fr);
  gap: 26px;
  border: 1px solid rgba(125,249,233,0.24);
  border-radius: 18px;
  padding: 30px;
  background:
    linear-gradient(105deg, rgba(45,212,191,0.20), rgba(34,211,238,0.07) 44%, rgba(167,139,250,0.12)),
    linear-gradient(180deg, rgba(10,25,42,0.94), rgba(6,16,29,0.88));
  box-shadow:
    0 24px 90px rgba(0,0,0,0.46),
    inset 0 1px 0 rgba(255,255,255,0.08);
  backdrop-filter: blur(18px) saturate(140%);
}
.er-title::before {
  content: "";
  position: absolute;
  inset: 18px 20px auto auto;
  width: 420px;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(125,249,233,0.70), transparent);
  transform: rotate(-8deg);
}
.er-title::after {
  content: "";
  position: absolute;
  right: 36px;
  bottom: 24px;
  width: 220px;
  height: 220px;
  border: 1px solid rgba(45,212,191,0.16);
  border-radius: 50%;
  opacity: 0.55;
}
.er-title h1 {
  position: relative;
  font-size: clamp(52px, 8vw, 104px);
  line-height: 0.86;
  margin: 0;
  letter-spacing: 0;
  font-weight: 820;
  color: var(--er-ink);
  text-shadow: 0 0 42px rgba(45,212,191,0.22);
}
.er-kicker {
  position: relative;
  align-self: end;
  color: var(--er-soft);
  font-size: 14px;
  line-height: 1.55;
  max-width: 520px;
  border-left: 1px solid rgba(45,212,191,0.38);
  padding-left: 18px;
}
.er-badges {
  position: relative;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 18px;
}
.er-badge {
  border: 1px solid rgba(148,219,233,0.20);
  border-radius: 999px;
  padding: 6px 10px;
  background: rgba(3,7,18,0.54);
  color: var(--er-soft);
  font-size: 12px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
}
.er-badge:first-child {
  border-color: var(--er-line-strong);
  color: #b8fff2;
  background: rgba(13,148,136,0.22);
}
.er-mission {
  margin-top: 14px;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
.er-metric {
  border: 1px solid rgba(148,219,233,0.16);
  border-radius: 14px;
  padding: 12px;
  background: rgba(4,13,25,0.54);
}
.er-metric strong {
  display: block;
  color: #b8fff2;
  font-size: 15px;
}
.er-metric span {
  display: block;
  color: var(--er-muted);
  font-size: 11px;
  margin-top: 3px;
}
.er-workspace {
  border: 1px solid rgba(148,219,233,0.18);
  border-radius: 18px;
  padding: 18px;
  background:
    radial-gradient(circle at 12% 0%, rgba(45,212,191,0.12), transparent 30%),
    radial-gradient(circle at 88% 24%, rgba(56,189,248,0.10), transparent 34%),
    linear-gradient(180deg, rgba(15,23,42,0.62), rgba(3,7,18,0.42));
  box-shadow: 0 30px 95px rgba(0,0,0,0.38);
  backdrop-filter: blur(12px);
}
.er-workspace::before {
  content: "LIVE SUPPORT ROUTER";
  display: block;
  color: #99f6e4;
  letter-spacing: 0.13em;
  font-size: 11px;
  margin-bottom: 12px;
}
.er-side {
  position: sticky;
  top: 10px;
}
.er-card {
  border: 1px solid rgba(148,219,233,0.18);
  border-radius: 16px;
  background: var(--er-panel);
  padding: 14px;
  box-shadow:
    0 18px 55px rgba(0,0,0,0.26),
    inset 0 1px 0 rgba(255,255,255,0.04);
  backdrop-filter: blur(16px) saturate(135%);
  color: var(--er-ink);
  margin-bottom: 12px;
}
.er-mini-title {
  font-size: 11px;
  color: #a7fff1;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 8px;
}
.er-empty {
  color: var(--er-muted);
  font-size: 13px;
  padding: 10px 2px;
}
.er-status-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.er-status {
  border: 1px solid rgba(148,219,233,0.16);
  border-radius: 12px;
  padding: 10px;
  background: var(--er-panel-2);
}
.er-status span {
  display: block;
  color: var(--er-muted);
  font-size: 11px;
  margin-bottom: 3px;
}
.er-status strong {
  font-size: 13px;
  color: var(--er-ink);
}
.er-source {
  border: 1px solid rgba(148,219,233,0.14);
  border-radius: 14px;
  padding: 11px;
  margin-top: 10px;
  background:
    linear-gradient(135deg, rgba(45,212,191,0.08), rgba(14,33,52,0.56));
}
.er-source-title {
  font-weight: 680;
  font-size: 13px;
  margin-bottom: 3px;
  color: var(--er-ink);
}
.er-source-meta {
  color: var(--er-muted);
  font-size: 12px;
  line-height: 1.35;
}
.er-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-top: 7px;
}
.er-chip {
  border: 1px solid rgba(148,219,233,0.18);
  border-radius: 999px;
  padding: 4px 8px;
  font-size: 11px;
  color: var(--er-soft);
  background: rgba(3,7,18,0.42);
}
.er-chip-risk {
  color: #fcd34d;
  border-color: rgba(245,158,11,0.34);
  background: rgba(245,158,11,0.14);
}
.er-chip-crisis {
  color: #fecdd3;
  border-color: rgba(251,113,133,0.38);
  background: rgba(251,113,133,0.14);
}
.er-link {
  color: #67e8f9;
  font-weight: 620;
  text-decoration: none;
}
.er-link:hover {
  text-decoration: underline;
}
.er-prompt-row button {
  min-height: 44px !important;
  border-radius: 14px !important;
  font-size: 12px !important;
  background:
    linear-gradient(180deg, rgba(30,64,92,0.72), rgba(8,20,34,0.84)) !important;
  color: var(--er-ink) !important;
  border: 1px solid rgba(148,219,233,0.20) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
}
.er-prompt-row button:hover {
  border-color: var(--er-line-strong) !important;
  background:
    linear-gradient(180deg, rgba(20,184,166,0.22), rgba(8,20,34,0.88)) !important;
}
.er-send button {
  min-height: 46px !important;
  border-radius: 14px !important;
}
textarea, input {
  border-radius: 14px !important;
  background: rgba(3,7,18,0.68) !important;
  color: var(--er-ink) !important;
  border: 1px solid rgba(148,219,233,0.20) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}
textarea::placeholder, input::placeholder {
  color: #74869c !important;
}
button.primary, .primary {
  background: linear-gradient(135deg, #0d9488, #0891b2 54%, #2563eb) !important;
  color: #ecfeff !important;
  border: 1px solid rgba(103,232,249,0.42) !important;
  box-shadow: 0 18px 44px rgba(14,165,233,0.26);
}
button.secondary {
  background: rgba(30,41,59,0.88) !important;
  color: var(--er-ink) !important;
}
.gradio-container .chatbot {
  background:
    linear-gradient(180deg, rgba(3,7,18,0.52), rgba(8,20,34,0.70)) !important;
  border: 1px solid rgba(148,219,233,0.18) !important;
  border-radius: 18px !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 24px 70px rgba(0,0,0,0.20);
  min-height: 430px !important;
}
.gradio-container .message,
.gradio-container .bubble-wrap .message,
.gradio-container .user,
.gradio-container .bot {
  color: var(--er-ink) !important;
}
.gradio-container .message.user {
  background: linear-gradient(135deg, rgba(13,148,136,0.30), rgba(14,116,144,0.22)) !important;
  border: 1px solid rgba(45,212,191,0.22) !important;
}
.gradio-container .message.bot {
  background: rgba(15,23,42,0.92) !important;
  border: 1px solid rgba(148,219,233,0.16) !important;
}
.bubble-wrap .message {
  border-radius: 16px !important;
}
.er-terminal-note {
  color: #a7fff1;
  border: 1px solid rgba(45,212,191,0.18);
  border-radius: 14px;
  padding: 10px 12px;
  background: rgba(3,7,18,0.44);
  font-size: 12px;
  margin-top: 10px;
  margin-bottom: 12px;
}
.er-state-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}
.er-state-pill {
  border: 1px solid rgba(148,219,233,0.16);
  border-radius: 14px;
  padding: 10px 12px;
  background: rgba(3,7,18,0.46);
}
.er-state-pill span {
  display: block;
  color: var(--er-dim);
  font-size: 10px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 4px;
}
.er-state-pill strong {
  color: var(--er-ink);
  font-size: 13px;
}
.er-crisis-banner {
  border: 1px solid rgba(251,113,133,0.40);
  border-radius: 14px;
  padding: 12px;
  margin-bottom: 10px;
  background:
    linear-gradient(135deg, rgba(251,113,133,0.16), rgba(15,23,42,0.82));
}
.er-crisis-banner strong {
  display: block;
  color: #fecdd3;
  font-size: 14px;
  margin-bottom: 4px;
}
.er-crisis-banner span {
  color: var(--er-soft);
  font-size: 12px;
}
.er-route {
  border: 1px solid rgba(45,212,191,0.28);
  border-radius: 14px;
  padding: 12px;
  margin-top: 10px;
  background:
    linear-gradient(135deg, rgba(20,184,166,0.16), rgba(14,33,52,0.72));
}
.er-route strong {
  display: block;
  color: #a7fff1;
  font-size: 13px;
  margin-bottom: 4px;
}
.er-route span {
  display: block;
  color: var(--er-soft);
  font-size: 12px;
  line-height: 1.45;
}
.er-why {
  margin-top: 8px;
  color: #a7fff1;
  font-size: 11px;
  line-height: 1.35;
}
.footer, .built-with, .api-docs, footer {
  display: none !important;
}
@media (max-width: 900px) {
  .er-title {
    grid-template-columns: 1fr;
  }
  .er-side {
    position: static;
  }
  .er-mission {
    grid-template-columns: 1fr;
  }
  .er-state-strip {
    grid-template-columns: 1fr 1fr;
  }
}
"""


class FastDemoPipeline:
    """Presentation backend backed by EmpathRAG Core without heavyweight LLM loading."""

    def __init__(self, db_path: Path, retrieval_corpus: str, top_k: int):
        self.db_path = db_path
        self.retrieval_corpus = "curated_support" if db_path.exists() else retrieval_corpus
        self.top_k = top_k
        self.safety_policy = SafetyTriagePolicy()
        self.core = EmpathRAGCore(
            curated_db_path=db_path,
            retrieval_corpus=self.retrieval_corpus,
            top_k=top_k,
        )
        self._turn = 0
        self._tier_history: list[str] = []
        self._crisis_locked = False
        self._last_escalation_reason = ""

    def run(self, user_message: str, audience_mode: str = "student") -> dict:
        core_result = self.core.run_turn(
            message=user_message,
            session_id="demo",
            audience_mode=audience_mode,
            resource_profile="umd",
            backend_mode="hybrid_ml",
        ).to_dict()
        emotion_name = core_result.get("emotion_name", "neutral")
        emotion_label = ["distress", "anxiety", "frustration", "neutral", "hopeful"].index(
            emotion_name if emotion_name in {"distress", "anxiety", "frustration", "neutral", "hopeful"} else "neutral"
        )
        core_result.update(
            {
                "emotion": emotion_label,
                "trajectory": core_result.get("trajectory_state", "active"),
                "crisis_confidence": 1.0 if core_result.get("crisis") else 0.0,
                "safety_level": core_result.get("safety_tier", ""),
            }
        )
        return core_result

    def _legacy_run(self, user_message: str, audience_mode: str = "student") -> dict:
        self._turn += 1
        emotion_name = self._emotion_name(user_message)
        emotion_label = ["distress", "anxiety", "frustration", "neutral", "hopeful"].index(emotion_name)
        safety_decision = self.safety_policy.classify(
            user_message,
            confidence=0.0,
            model_flag=False,
        )
        if safety_decision.level == SafetyLevel.PASS and self._wellbeing_request(user_message):
            safety_level = SafetyLevel.WELLBEING_SUPPORT
            safety_reason = "wellbeing_or_grounding_request"
        else:
            safety_level = safety_decision.level
            safety_reason = safety_decision.reason

        safety_tier = map_safety_level(safety_level, wellbeing_request=self._wellbeing_request(user_message))
        normalized_message = user_message.lower()
        dependency_or_secrecy = any(
            phrase in normalized_message
            for phrase in (
                "you are the only one",
                "only one i can talk to",
                "don't tell anyone",
                "do not tell anyone",
                "keep this secret",
                "no one can help",
            )
        )
        peer_context = audience_mode == "helping_friend" or any(
            phrase in normalized_message
            for phrase in ("my friend", "my roommate", "my labmate", "my teammate", "someone i know")
        )
        peer_imminent = peer_context and (
            "goodbye" in normalized_message
            and any(phrase in normalized_message for phrase in ("locked", "will not answer", "won't answer", "not answering"))
        )
        if peer_imminent:
            safety_tier = SafetyTier.IMMINENT_SAFETY
            safety_reason = "peer_goodbye_unreachable"
        elif dependency_or_secrecy and safety_tier == SafetyTier.SUPPORT_NAVIGATION:
            safety_tier = SafetyTier.HIGH_DISTRESS
            safety_reason = "dependency_or_secrecy_redirect"
        route_decision = classify_route(user_message, safety_tier, audience_mode=audience_mode)
        escalation_reason = self._update_trajectory_lock(user_message, safety_tier, route_decision.route)

        if safety_decision.should_intercept or self._crisis_locked or safety_tier == SafetyTier.IMMINENT_SAFETY:
            retrieved = self._retrieve(
                user_message,
                SafetyLevel.CRISIS,
                route=route_decision.route.value,
                safety_tier=SafetyTier.IMMINENT_SAFETY.value,
                audience_mode=audience_mode,
            )
            if route_decision.route == SupportRoute.PEER_HELPER:
                response = (
                    "I am concerned this could be an immediate safety situation for your friend. "
                    "Do not handle this alone. Contact emergency/crisis support now, and involve a trusted nearby person, RA, supervisor, or campus support while you try to reach them."
                )
            else:
                response = safety_decision.response or (
                    "I am really concerned about your immediate safety. Please call or text 988 now, "
                    "or call emergency services if you may be in immediate danger."
                )
            return self._result(
                response=response,
                emotion_label=emotion_label,
                emotion_name=emotion_name,
                safety_level=safety_decision.level,
                safety_reason=safety_decision.reason,
                crisis=True,
                retrieved=retrieved,
                latency={"demo_backend_ms": 8},
                route_label=route_decision.route.value,
                safety_tier=SafetyTier.IMMINENT_SAFETY.value,
                recommended_action=self._recommended_action(route_decision.route.value),
                escalation_reason=escalation_reason,
                output_guard={"allowed": True, "reason": "crisis_template", "flags": []},
            )

        retrieved = self._retrieve(
            user_message,
            safety_level,
            route=route_decision.route.value,
            safety_tier=safety_tier.value,
            audience_mode=audience_mode,
        )
        route_label = route_decision.route.value
        response = self._response_for(user_message, retrieved, safety_level, route_label, audience_mode)
        guard = validate_output(
            response=response,
            retrieved_sources=self._source_summaries(retrieved),
            safety_tier=safety_tier.value,
            route=route_label,
            conversation_history=[],
        )
        if guard.fallback_required and guard.corrected_response:
            response = guard.corrected_response
        return self._result(
            response=response,
            emotion_label=emotion_label,
            emotion_name=emotion_name,
            safety_level=safety_level,
            safety_reason=safety_reason,
            crisis=False,
            retrieved=retrieved,
            latency={"demo_backend_ms": 8},
            route_label=route_label,
            recommended_action=self._recommended_action(route_label),
            safety_tier=safety_tier.value,
            escalation_reason=escalation_reason,
            output_guard={"allowed": guard.allowed, "reason": guard.reason, "flags": guard.flags},
        )

    def tracker_trajectory(self) -> str:
        return "stable"

    def reset_session(self) -> None:
        self._turn = 0
        self._tier_history = []
        self._crisis_locked = False
        self._last_escalation_reason = ""
        self.core.reset_session("demo")

    def _result(
        self,
        response: str,
        emotion_label: int,
        emotion_name: str,
        safety_level: SafetyLevel,
        safety_reason: str,
        crisis: bool,
        retrieved: list[dict],
        latency: dict,
        route_label: str,
        recommended_action: str,
        safety_tier: str,
        escalation_reason: str,
        output_guard: dict,
    ) -> dict:
        return {
            "response": response,
            "emotion": emotion_label,
            "emotion_name": emotion_name,
            "trajectory": "stable",
            "crisis": crisis,
            "crisis_confidence": 1.0 if crisis else 0.0,
            "safety_level": safety_level.value,
            "safety_tier": safety_tier,
            "safety_reason": safety_reason,
            "escalation_reason": escalation_reason,
            "ig_highlights": [],
            "retrieved_chunks": [row["text"] for row in retrieved],
            "retrieved_sources": self._source_summaries(retrieved),
            "retrieval_corpus": self.retrieval_corpus,
            "latency_ms": latency,
            "route_label": route_label,
            "recommended_action": recommended_action,
            "output_guard": output_guard,
        }

    def _retrieve(
        self,
        message: str,
        safety_level: SafetyLevel,
        route: str | None = None,
        safety_tier: str | None = None,
        audience_mode: str = "student",
    ) -> list[dict]:
        if not self.db_path.exists():
            return [node.as_source("service graph fallback") for node in match_services(route or "", safety_tier or "", audience_mode, limit=self.top_k)]
        topics, source_names = self._targets(message, safety_level, route=route)
        usage_modes = self._usage_modes(safety_level)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, resource_id, text, source_id, source_name, source_type,
                   title, url, topic, audience, risk_level, usage_mode, summary,
                   last_checked, notes
            FROM chunks
            WHERE usage_mode IN ({})
            """.format(",".join("?" * len(usage_modes))),
            tuple(usage_modes),
        ).fetchall()
        conn.close()

        scored = []
        query = message.lower()
        for row in rows:
            score = 0
            reasons = []
            title = row["title"].lower()
            if row["topic"] in topics:
                score += 8
                reasons.append(f"topic match: {row['topic']}")
            if row["source_name"] in source_names:
                score += 7
                reasons.append(f"preferred source: {row['source_name']}")
            if "workshop" in title and any(token in query for token in ("stress", "anxious", "panic", "grades", "exam")):
                score += 6
                reasons.append("student workshop fit")
            if "ptsd" in title and not any(token in query for token in ("ptsd", "trauma", "traumatic", "flashback")):
                score -= 12
            if "eating disorder" in title and not any(token in query for token in ("eating", "food", "body", "weight", "diet")):
                score -= 12
            if "funding" in title and not any(token in query for token in ("funding", "financial", "money", "tuition", "assistantship")):
                score -= 8
            if "admission" in title and not any(token in query for token in ("admission", "admissions", "apply", "application", "admitted")):
                score -= 12
            if "traumatic" in title and not any(token in query for token in ("trauma", "traumatic", "ptsd", "assault", "violence")):
                score -= 8
            haystack = f"{row['title']} {row['summary']} {row['text']}".lower()
            keyword_hits = []
            for token in self._keywords(query):
                if token in haystack:
                    score += 1
                    keyword_hits.append(token)
            if keyword_hits:
                reasons.append("keyword overlap: " + ", ".join(keyword_hits[:3]))
            row_dict = dict(row)
            row_dict["why_retrieved"] = "; ".join(reasons[:2]) if reasons else "semantic support match"
            scored.append((score, row_dict))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = []
        source_counts: dict[str, int] = {}
        seen_cards: set[tuple[str, str]] = set()
        for score, row in scored:
            if score <= 0 and selected:
                continue
            card_key = (row["source_name"], row["title"])
            if card_key in seen_cards:
                continue
            source = row["source_name"]
            if source_counts.get(source, 0) >= 2:
                continue
            selected.append(row)
            seen_cards.add(card_key)
            source_counts[source] = source_counts.get(source, 0) + 1
            if len(selected) == self.top_k:
                break
        if route and safety_tier:
            seen_source_titles = {(row.get("source_name", ""), row.get("title", "")) for row in selected}
            graph_rows = []
            for node in match_services(route, safety_tier, audience_mode, limit=self.top_k):
                source_row = node.as_source("service graph route match")
                key = (source_row.get("source_name", ""), source_row.get("title", ""))
                if key in seen_source_titles:
                    continue
                if source_row.get("usage_mode") not in usage_modes:
                    continue
                graph_rows.append(source_row)
                seen_source_titles.add(key)
            selected = (graph_rows + selected)[: self.top_k]
        return selected

    def _targets(self, message: str, safety_level: SafetyLevel, route: str | None = None) -> tuple[set[str], set[str]]:
        text = message.lower()
        if safety_level in {SafetyLevel.CRISIS, SafetyLevel.EMERGENCY}:
            return (
                {"crisis_immediate_help", "emergency_services"},
                {"988 Suicide & Crisis Lifeline", "UMD Counseling Center"},
            )
        if route == SupportRoute.PEER_HELPER.value:
            return (
                {"crisis_immediate_help", "help_seeking_script", "counseling_services"},
                {"988 Suicide & Crisis Lifeline", "UMD Counseling Center", "JED Foundation"},
            )
        if route == SupportRoute.BASIC_NEEDS.value:
            return (
                {"help_seeking_script", "campus_navigation", "graduate_student_support"},
                {"UMD Dean of Students", "UMD Graduate School", "UMD Counseling Center"},
            )
        if route == SupportRoute.ACCESSIBILITY_ADS.value:
            return (
                {"accessibility_disability", "campus_navigation"},
                {"UMD Accessibility & Disability Service"},
            )
        if route == SupportRoute.ADVISOR_CONFLICT.value:
            return (
                {"advisor_conflict", "graduate_student_support"},
                {"UMD Graduate School Ombuds", "UMD Graduate School"},
            )
        if "accommodation" in text or "disability" in text or "ads" in text:
            return (
                {"accessibility_disability"},
                {"UMD Accessibility & Disability Service"},
            )
        if "advisor" in text or "ombuds" in text or "neutral" in text:
            return (
                {"advisor_conflict", "graduate_student_support"},
                {"UMD Graduate School Ombuds", "UMD Counseling Center"},
            )
        if "ground" in text or "panic" in text or "panicking" in text:
            return (
                {"grounding_exercise", "anxiety_stress", "counseling_services"},
                {"UMD Counseling Center", "NAMI", "NIMH"},
            )
        if any(word in text for word in ("stress", "stressful", "stressed", "overwhelmed", "too much", "spiral")):
            return (
                {"anxiety_stress", "academic_burnout", "counseling_services", "grounding_exercise"},
                {"UMD Counseling Center", "NIMH"},
            )
        if any(word in text for word in ("failed", "fail", "exam", "grades", "grade", "doomed", "class", "course", "semester")):
            return (
                {"academic_burnout", "anxiety_stress", "counseling_services", "graduate_student_support"},
                {"UMD Counseling Center", "UMD Graduate School", "NIMH"},
            )
        if any(word in text for word in ("depressing", "depressed", "depression", "low mood")):
            return (
                {"depression_support", "counseling_services", "anxiety_stress"},
                {"UMD Counseling Center", "NIMH", "NAMI"},
            )
        if any(word in text for word in ("grade", "grades", "doomed", "failing", "failed", "class", "course", "semester")):
            return (
                {"academic_burnout", "anxiety_stress", "counseling_services", "graduate_student_support"},
                {"UMD Counseling Center", "UMD Graduate School", "NIMH"},
            )
        if "counsel" in text or "therapy" in text or "start" in text:
            return (
                {"counseling_services", "campus_navigation", "therapy_expectations"},
                {"UMD Counseling Center"},
            )
        if "isolated" in text or "lonely" in text:
            return (
                {"isolation_loneliness", "counseling_services"},
                {"UMD Counseling Center", "NAMI"},
            )
        return (
            {"anxiety_stress", "counseling_services", "academic_burnout"},
            {"UMD Counseling Center", "NIMH"},
        )

    def _usage_modes(self, safety_level: SafetyLevel) -> tuple[str, ...]:
        if safety_level in {SafetyLevel.CRISIS, SafetyLevel.EMERGENCY}:
            return ("crisis_only",)
        if safety_level == SafetyLevel.WELLBEING_SUPPORT:
            return ("retrieval", "wellbeing_only")
        return ("retrieval",)

    def _keywords(self, query: str) -> list[str]:
        return [token for token in query.replace("?", " ").replace(".", " ").split() if len(token) > 4]

    def _source_summaries(self, rows: list[dict]) -> list[dict]:
        return [
            {
                "title": row.get("title", ""),
                "source_name": row.get("source_name", ""),
                "url": row.get("url", ""),
                "topic": row.get("topic", ""),
                "risk_level": row.get("risk_level", ""),
                "usage_mode": row.get("usage_mode", ""),
                "source_type": row.get("source_type", ""),
                "why_retrieved": row.get("why_retrieved", ""),
            }
            for row in rows
        ]

    def _emotion_name(self, message: str) -> str:
        text = message.lower()
        if any(word in text for word in ("safe tonight", "hurt myself", "hopeless", "die", "suicide")):
            return "distress"
        if any(word in text for word in ("depressing", "depressed", "depression", "failed my exam")):
            return "distress"
        if any(word in text for word in ("anxious", "panic", "panicking", "overwhelmed", "exam", "grades", "grade", "doomed", "failing", "stress", "stressful", "stressed")):
            return "anxiety"
        if any(word in text for word in ("advisor", "dismiss", "angry", "rejected")):
            return "frustration"
        if any(word in text for word in ("finished", "better", "proud", "hopeful")):
            return "hopeful"
        return "neutral"

    def _wellbeing_request(self, message: str) -> bool:
        text = message.lower()
        return any(word in text for word in ("grounding", "ground", "panic", "breathing", "cope"))

    def _response_for(
        self,
        message: str,
        rows: list[dict],
        safety_level: SafetyLevel,
        route_label: str,
        audience_mode: str,
    ) -> str:
        source = rows[0]["source_name"] if rows else "a student-support resource"
        topic = rows[0]["topic"].replace("_", " ") if rows else "student support"
        source_line = self._source_line(rows)
        if route_label == SupportRoute.PEER_HELPER.value:
            return (
                "Route detected: peer-helper support. This is not something your friend should have to handle alone, and it is not something you should handle alone either.\n\n"
                "Recommended next action: if there may be immediate danger, contact emergency/crisis support now and involve a trusted nearby person, RA, supervisor, or campus support. Do not promise secrecy when safety may be at risk.\n\n"
                f"Sources matched: {source_line}\n\n"
                "A safer thing to say: I care about you, and I am worried enough that we need to get another person involved right now."
            )
        if route_label == SupportRoute.BASIC_NEEDS.value:
            return (
                "Route detected: basic needs / student support. Food, housing, and money stress are not motivation problems; they are support-navigation problems.\n\n"
                "Recommended next action: contact a campus student-support office or Dean of Students-style support path and say plainly what you need help with today. I will not invent Pantry or Thrive details unless they are in the verified corpus.\n\n"
                f"Sources matched: {source_line}"
            )
        if route_label == SupportRoute.ACADEMIC_SETBACK.value:
            return (
                "Route detected: academic setback with distress. Failing an exam can feel catastrophic, but this is exactly the kind of moment where the next step matters more than the spiral.\n\n"
                "Recommended next action: send a short office-hours note instead of trying to solve the whole semester tonight.\n\n"
                "Email script: Hi Professor/TA [Name], I am trying to understand what went wrong on [exam/assignment] and what I can do differently before the next assessment. Could I come to office hours or schedule a short meeting to review my mistakes?\n\n"
                f"Sources matched: {source_line}"
            )
        if route_label == SupportRoute.LOW_MOOD.value:
            return (
                "Route detected: low mood / depression support. I am not reading this as an emergency from the wording alone, but it is serious enough to deserve support instead of being minimized.\n\n"
                f"Recommended next action: tell one trusted person what is going on, then use a campus counseling starting point. If this shifts into not feeling safe, use crisis support immediately.\n\n"
                f"Sources matched: {source_line}"
            )
        if route_label == SupportRoute.EXAM_STRESS.value:
            return (
                "That sounds like the kind of grade panic that can make everything feel bigger and more permanent than it actually is.\n\n"
                f"Recommended next action: choose one academic action for the next 24 hours: office hours, TA email, syllabus policy check, or advisor check-in. I found {topic} resources anchored around {source}.\n\n"
                f"Sources matched: {source_line}"
            )
        if route_label == SupportRoute.ANXIETY_PANIC.value:
            return (
                "That sounds like stress has moved from background noise into something that is taking over the whole room.\n\n"
                f"Recommended next action: first do one short grounding step, then choose whether you need a campus support path or a simple next-step plan. I found {topic} resources anchored around {source}.\n\n"
                f"Sources matched: {source_line}"
            )
        if route_label == SupportRoute.ACCESSIBILITY_ADS.value:
            return (
                "Route detected: accessibility / accommodations support. This is a practical support path, not something you have to improvise alone.\n\n"
                f"Recommended next action: identify the class or exam barrier, then use the official ADS student process so the request is traceable.\n\n"
                f"Sources matched: {source_line}"
            )
        if route_label == SupportRoute.ADVISOR_CONFLICT.value:
            return (
                "Route detected: advisor conflict / graduate support. The safest next step is to keep the record factual and use a neutral campus channel before the situation escalates.\n\n"
                f"Recommended next action: write down the specific concern, separate urgent academic deadlines from relationship issues, and consider Ombuds or graduate support resources.\n\n"
                f"Sources matched: {source_line}"
            )
        if safety_level == SafetyLevel.WELLBEING_SUPPORT:
            return (
                f"That sounds like a sharp spike of student stress, and it makes sense to want something steadying rather than another wall of advice.\n\n"
                f"Recommended next action: take one short grounding step, then decide whether you need who to contact or what to expect next. I found {topic} resources anchored around {source}."
            )
        return (
            f"That sounds like a real student-support concern, and you should not have to untangle it from scratch.\n\n"
            f"Recommended next action: pick one concrete support path before trying to solve the whole situation. I found {topic} resources anchored around {source}. What would help most to focus on first: next steps, who to contact, or what to expect?\n\n"
            f"Sources matched: {source_line}"
        )

    def _need_label(self, message: str, safety_level: SafetyLevel) -> str:
        text = message.lower()
        if safety_level in {SafetyLevel.CRISIS, SafetyLevel.EMERGENCY}:
            return "immediate safety"
        if "accommodation" in text or "disability" in text or "ads" in text:
            return "accessibility"
        if "advisor" in text or "neutral" in text or "ombuds" in text:
            return "advisor conflict"
        if any(word in text for word in ("failed", "failed my exam", "fail", "exam")):
            return "academic setback"
        if any(word in text for word in ("depressing", "depressed", "depression", "low mood")):
            return "low mood"
        if "counsel" in text or "therapy" in text:
            return "counseling navigation"
        if "panic" in text or "ground" in text:
            return "anxiety"
        if any(word in text for word in ("stress", "stressful", "stressed", "overwhelmed", "too much", "spiral")):
            return "stress overload"
        if any(word in text for word in ("grade", "grades", "doomed", "failing", "class", "course", "semester")):
            return "academic stress"
        return "student-support"

    def _source_line(self, rows: list[dict]) -> str:
        if not rows:
            return "no source cards available"
        labels = []
        seen = set()
        for row in rows[:3]:
            label = f"{row['source_name']} - {row['title']}"
            if label in seen:
                continue
            seen.add(label)
            labels.append(label)
        return "; ".join(labels)

    def _recommended_action(self, route_label: str) -> str:
        actions = {
            SupportRoute.CRISIS_IMMEDIATE.value: "Stop normal advice. Show 988, emergency, and campus crisis options first.",
            SupportRoute.PEER_HELPER.value: "Do not ask the peer to handle risk alone. Escalate to a trusted person, campus support, or crisis help when safety may be at risk.",
            SupportRoute.ACADEMIC_SETBACK.value: "Send a short office-hours note and identify the next academic policy/support step.",
            SupportRoute.LOW_MOOD.value: "Tell one trusted person and use a campus counseling starting point; escalate if safety changes.",
            SupportRoute.EXAM_STRESS.value: "Choose one academic action for the next 24 hours: office hours, TA email, syllabus policy check, or advisor check-in.",
            SupportRoute.ANXIETY_PANIC.value: "Start with one grounding step, then choose a support path if symptoms keep interfering.",
            SupportRoute.ACCESSIBILITY_ADS.value: "Route to the official ADS process and keep the accommodations request traceable.",
            SupportRoute.ADVISOR_CONFLICT.value: "Keep the record factual and consider Ombuds or graduate support before escalating the conflict.",
            SupportRoute.COUNSELING_NAVIGATION.value: "Explain how to start with UMD Counseling and what to expect from first contact.",
            SupportRoute.BASIC_NEEDS.value: "Route to verified campus student-support resources without inventing Pantry/Thrive details.",
        }
        return actions.get(route_label, "Keep the answer practical, source-grounded, and student-support oriented.")

    def _update_trajectory_lock(self, message: str, safety_tier: SafetyTier, route: SupportRoute) -> str:
        self._tier_history.append(safety_tier.value)
        self._tier_history = self._tier_history[-3:]
        text = message.lower()
        reason = ""
        if len(self._tier_history) == 3 and all(tier in {"imminent_safety", "high_distress"} for tier in self._tier_history):
            self._crisis_locked = True
            reason = "three_consecutive_high_risk_turns"
        dependency_or_secrecy = any(
            phrase in text
            for phrase in (
                "you are the only one",
                "only one i can talk to",
                "don't tell anyone",
                "do not tell anyone",
                "keep this secret",
                "no one can help",
            )
        )
        if dependency_or_secrecy:
            reason = reason or "dependency_or_secrecy_redirect"
            if safety_tier == SafetyTier.IMMINENT_SAFETY:
                self._crisis_locked = True
        if self._crisis_locked and not reason:
            reason = "crisis_locked"
        self._last_escalation_reason = reason
        return reason


pipeline_lock = threading.Lock()
pipeline = None


def get_pipeline() -> EmpathRAGPipeline:
    global pipeline
    if pipeline is None:
        if DEMO_BACKEND == "real":
            print("[Demo] Initialising full EmpathRAG pipeline...", flush=True)
            from pipeline.pipeline import EmpathRAGPipeline

            pipeline = EmpathRAGPipeline(
                use_real_guardrail=True,
                guardrail_threshold=0.5,
                retrieval_corpus=RETRIEVAL_CORPUS,
                top_k=DEMO_TOP_K,
                generation_max_tokens=DEMO_MAX_TOKENS,
            )
            print("[Demo] Full pipeline ready.", flush=True)
        else:
            print("[Demo] Initialising fast presentation backend.", flush=True)
            pipeline = FastDemoPipeline(
                db_path=CURATED_DB_PATH,
                retrieval_corpus=RETRIEVAL_CORPUS,
                top_k=DEMO_TOP_K,
            )
    return pipeline


def new_session_id() -> str:
    return uuid.uuid4().hex[:6].upper()


def new_session_state() -> dict:
    return {
        "session_id": new_session_id(),
        "emotion_history": [],
        "tracker_history": [],
        "conv_history": [],
    }


def log_turn(session_id, turn, user_message, result):
    if not LOG_TURNS:
        return
    try:
        log_entry = {
            "session_id": session_id,
            "turn": turn,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "user_message": user_message,
            "response": result["response"],
            "emotion_label": result["emotion"],
            "emotion_name": result["emotion_name"],
            "trajectory": result["trajectory"],
            "crisis_fired": result["crisis"],
            "crisis_confidence": result["crisis_confidence"],
            "retrieval_corpus": result.get("retrieval_corpus", ""),
            "safety_level": result.get("safety_level", ""),
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"[Warning] Failed to log turn: {e}")


def format_emotion_timeline(history, trajectory) -> str:
    if not history:
        return (
            "<div class='er-card'><div class='er-mini-title'>Emotion Timeline</div>"
            "<div class='er-empty'>Waiting for the first turn.</div></div>"
        )

    trajectory_badge_colors = {
        "stable": "#64748b",
        "stable_positive": "#047857",
        "stable_negative": "#b42318",
        "escalating": "#b42318",
        "de_escalating": "#0f766e",
        "volatile": "#b45309",
    }

    traj_color = trajectory_badge_colors.get(trajectory, "#64748b")
    html = "<div class='er-card'><div class='er-mini-title'>Emotion Timeline</div>"
    html += (
        f"<div style='margin-bottom:10px;padding:7px 10px;background:{traj_color};"
        "color:white;border-radius:8px;font-size:12px;font-weight:650;'>"
        f"Session: {escape(str(trajectory))}</div>"
    )
    html += "<div style='display:flex;flex-wrap:wrap;gap:6px;'>"
    for item in history:
        label = escape(str(item["label_name"]))
        turn = escape(str(item["turn"]))
        color = escape(str(item["color"]))
        html += (
            f"<span style='padding:5px 8px;background:{color};color:white;"
            f"border-radius:999px;font-size:11px;'>T{turn}: {label}</span>"
        )
    html += "</div></div>"
    return html


def format_ig_panel(is_crisis, confidence, ig_tokens, loading, explanation_reason="") -> str:
    if not is_crisis:
        return (
            "<div class='er-card'><div class='er-mini-title'>Safety Guardrail</div>"
            "<div class='er-empty'>No crisis intercept on this turn.</div></div>"
        )

    if loading:
        return (
            "<div class='er-card' style='border-color:rgba(180,35,24,0.26);'>"
            "<div class='er-mini-title'>Safety Guardrail</div>"
            f"<div style='font-weight:700;color:var(--er-danger);margin-bottom:4px;'>"
            f"Crisis signal detected - {confidence:.1%}</div>"
            "<div class='er-empty'>Computing token attributions...</div></div>"
        )

    conf_pct = int(confidence * 100)
    html = "<div class='er-card' style='border-color:rgba(180,35,24,0.26);'>"
    html += "<div class='er-mini-title'>Safety Guardrail</div>"
    html += (
        f"<div style='font-weight:700;color:var(--er-danger);margin-bottom:8px;'>"
        f"Crisis Confidence: {confidence:.1%}</div>"
    )
    html += (
        "<div style='background:rgba(3,7,18,0.72);height:8px;border-radius:999px;overflow:hidden;margin-bottom:10px;'>"
        f"<div style='background:var(--er-danger);height:100%;width:{conf_pct}%;'></div></div>"
    )

    if ig_tokens:
        valid_tokens = [(tok, score) for tok, score in ig_tokens if tok.strip()]
        if valid_tokens:
            max_score = max(score for _, score in valid_tokens)
            html += (
                "<div style='font-size:11px;color:#fecdd3;margin-bottom:4px;font-weight:650;'>"
                "Top Crisis Signals</div>"
            )
            html += "<div style='display:flex;flex-wrap:wrap;gap:4px;'>"
            for tok, score in valid_tokens[:10]:
                opacity = score / max_score if max_score > 0 else 0.5
                bg_color = f"rgba(180,35,24,{opacity:.2f})"
                html += (
                    f"<span style='padding:3px 7px;background:{bg_color};"
                    f"border:1px solid #b42318;border-radius:999px;font-size:10px;'>"
                    f"{escape(tok)}</span>"
                )
            html += "</div>"
    elif explanation_reason:
        html += (
            "<div class='er-source-meta' style='margin-top:8px;'>"
            f"Explanation source: {escape(str(explanation_reason))}</div>"
        )

    html += "</div>"
    return html


def format_retrieval_panel(result=None) -> str:
    if not result:
        return (
            "<div class='er-card'><div class='er-mini-title'>Retrieval Sources</div>"
            "<div class='er-empty'>No sources retrieved yet.</div></div>"
        )

    safety_level = escape(str(result.get("safety_level", "unknown")))
    safety_reason = escape(str(result.get("safety_reason", "")))
    safety_tier = escape(str(result.get("safety_tier", "unknown")))
    escalation_reason = escape(str(result.get("escalation_reason", "")))
    corpus = escape(str(result.get("retrieval_corpus", "unknown")))
    route_label = escape(str(result.get("route_label", "student-support")))
    recommended_action = escape(str(result.get("recommended_action", "")))
    output_guard = result.get("output_guard", {}) or {}
    output_guard_reason = escape(str(output_guard.get("reason", "not_checked")))
    safety_precheck = result.get("safety_precheck", {}) or {}
    precheck_reason = escape(str(safety_precheck.get("reason", "not_recorded")))
    precheck_level = escape(str(safety_precheck.get("level", "unknown")))
    safety_explanation = result.get("safety_explanation", {}) or {}
    explanation_reason = escape(str(safety_explanation.get("reason", "not_checked")))
    classifier_confidence = result.get("classifier_confidence", {}) or {}
    route_conf = float(classifier_confidence.get("route", 0.0) or 0.0)
    tier_conf = float(classifier_confidence.get("tier", 0.0) or 0.0)
    classifier_label = "ml" if classifier_confidence.get("used_ml") else "fallback"
    retrieval_mode = escape(str(result.get("retrieval_mode", "graph_filtered_faiss_plus_router")))
    html = (
        "<div class='er-card'>"
        "<div class='er-mini-title'>Retrieval Sources</div>"
        "<div class='er-status-grid'>"
        f"<div class='er-status'><span>Corpus</span><strong>{corpus}</strong></div>"
        f"<div class='er-status'><span>Tier</span><strong>{safety_tier}</strong></div>"
        f"<div class='er-status'><span>Safety</span><strong>{safety_level}</strong></div>"
        f"<div class='er-status'><span>Output guard</span><strong>{output_guard_reason}</strong></div>"
        f"<div class='er-status'><span>Classifier</span><strong>{classifier_label} {route_conf:.2f}/{tier_conf:.2f}</strong></div>"
        f"<div class='er-status'><span>Retrieval</span><strong>{retrieval_mode}</strong></div>"
        f"<div class='er-status'><span>Stage-1 check</span><strong>{precheck_level}</strong></div>"
        f"<div class='er-status'><span>Safety model</span><strong>{explanation_reason}</strong></div>"
        "</div>"
        f"<div class='er-source-meta' style='margin-top:8px;'>Reason: {safety_reason}; precheck: {precheck_reason}</div>"
        "<div class='er-route'>"
        f"<strong>Support route: {route_label}</strong>"
        f"<span>{recommended_action}</span>"
        "</div>"
    )
    if escalation_reason:
        html += f"<div class='er-source-meta' style='margin-top:8px;'>Escalation: {escalation_reason}</div>"

    if safety_level in {"crisis", "emergency"}:
        html += (
            "<div class='er-crisis-banner'>"
            "<strong>Normal generation intercepted</strong>"
            "<span>Crisis resources are shown as source cards; the chat response uses the safety template.</span>"
            "</div>"
        )

    sources = result.get("retrieved_sources", [])
    if not sources:
        html += "<div class='er-empty'>No sources retrieved for this turn.</div></div>"
        return html

    for source in sources[:5]:
        title = escape(str(source.get("title", "") or "Untitled source"))
        source_name = escape(str(source.get("source_name", "") or "Unknown source"))
        topic = escape(str(source.get("topic", "") or ""))
        risk = escape(str(source.get("risk_level", "") or ""))
        usage = escape(str(source.get("usage_mode", "") or ""))
        source_type = escape(str(source.get("source_type", "") or ""))
        why = escape(str(source.get("why_retrieved", "") or "matched prompt intent"))
        url = escape(str(source.get("url", "") or ""))
        risk_class = "er-chip-crisis" if "crisis" in risk else "er-chip-risk" if risk else ""
        html += (
            "<div class='er-source'>"
            f"<div class='er-source-title'>{title}</div>"
            f"<div class='er-source-meta'>{source_name}</div>"
            "<div class='er-chip-row'>"
            f"<span class='er-chip'>{topic}</span>"
            f"<span class='er-chip {risk_class}'>{risk}</span>"
            f"<span class='er-chip'>{usage}</span>"
            f"<span class='er-chip'>{source_type}</span>"
            "</div>"
            f"<div class='er-why'>{why}</div>"
        )
        if url:
            html += f"<div style='margin-top:7px;'><a class='er-link' href='{url}' target='_blank'>Open source</a></div>"
        html += "</div>"
    html += "</div>"
    return html


def respond(message, chat_history, session_state, audience_mode):
    if not session_state:
        session_state = new_session_state()

    emotion_history = session_state["emotion_history"]
    session_id = session_state["session_id"]

    if not message.strip():
        yield (
            chat_history,
            format_emotion_timeline(emotion_history, "stable"),
            "stable",
            format_ig_panel(False, 0.0, [], False),
            format_retrieval_panel(),
            session_id,
            session_state,
        )
        return

    with pipeline_lock:
        active_pipeline = get_pipeline()
        if hasattr(active_pipeline, "tracker"):
            active_pipeline.tracker.reset()
            for label in session_state.get("tracker_history", []):
                active_pipeline.tracker.update(label, token_count=5)
            active_pipeline.conv_history = list(session_state.get("conv_history", []))

            original_check = active_pipeline.guardrail.check

            def fast_check(text, threshold=0.5, skip_ig=False):
                return original_check(text, threshold=threshold, skip_ig=True)

            active_pipeline.guardrail.check = fast_check
            result = active_pipeline.run(message)
            active_pipeline.guardrail.check = original_check
            session_state["tracker_history"] = active_pipeline.tracker.history()
            session_state["conv_history"] = list(active_pipeline.conv_history)
        else:
            result = active_pipeline.run(message, audience_mode=audience_mode or "student")
            session_state["tracker_history"] = session_state.get("tracker_history", []) + [result["emotion"]]
            session_state["conv_history"] = session_state.get("conv_history", [])

    chat_history.append((message, result["response"]))
    emotion_history.append(
        {
            "turn": len(emotion_history) + 1,
            "label_name": result["emotion_name"],
            "color": LABEL_COLORS[result["emotion_name"]],
        }
    )

    log_turn(session_id, len(emotion_history), message, result)
    timeline_html = format_emotion_timeline(emotion_history, result["trajectory"])

    if result["crisis"]:
        safety_explanation = result.get("safety_explanation", {}) or {}
        explanation_available = bool(safety_explanation.get("available"))
        ig_tokens = safety_explanation.get("ig_tokens") or []
        explanation_reason = safety_explanation.get("reason", "")
        if not hasattr(get_pipeline(), "guardrail") and not explanation_available:
            yield (
                chat_history,
                timeline_html,
                result["trajectory"],
                format_ig_panel(True, result["crisis_confidence"], ig_tokens, loading=False, explanation_reason=explanation_reason),
                format_retrieval_panel(result),
                session_id,
                session_state,
            )
            return
        yield (
            chat_history,
            timeline_html,
            result["trajectory"],
            format_ig_panel(True, result["crisis_confidence"], ig_tokens, loading=True),
            format_retrieval_panel(result),
            session_id,
            session_state,
        )

        with pipeline_lock:
            active_pipeline = get_pipeline()
            if hasattr(active_pipeline, "guardrail"):
                _, confidence, ig_tokens = active_pipeline.guardrail.check(message, threshold=0.5, skip_ig=False)
            else:
                confidence, ig_tokens = result["crisis_confidence"], []

        yield (
            chat_history,
            timeline_html,
            result["trajectory"],
            format_ig_panel(True, confidence, ig_tokens, loading=False),
            format_retrieval_panel(result),
            session_id,
            session_state,
        )
    else:
        yield (
            chat_history,
            timeline_html,
            result["trajectory"],
            format_ig_panel(False, 0.0, [], False),
            format_retrieval_panel(result),
            session_id,
            session_state,
        )


def reset_session_handler():
    session_state = new_session_state()
    return (
        [],
        format_emotion_timeline([], "stable"),
        "stable",
        format_ig_panel(False, 0.0, [], False),
        format_retrieval_panel(),
        session_state["session_id"],
        session_state,
    )


def set_prompt(prompt: str) -> str:
    return prompt


theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="amber",
    neutral_hue="stone",
    radius_size="sm",
)

with gr.Blocks(theme=theme, title="EmpathRAG Core", css=APP_CSS) as demo:
    initial_state = new_session_state()
    session_state = gr.State(value=initial_state)

    gr.HTML(
        f"""
        <div class="er-shell">
          <div class="er-title">
            <div>
              <h1>EmpathRAG Core</h1>
              <div class="er-badges">
                <span class="er-badge">Guarded conversational RAG</span>
                <span class="er-badge">{escape(RETRIEVAL_CORPUS)}</span>
                <span class="er-badge">logging off by default</span>
              </div>
              <div class="er-mission">
                <div class="er-metric"><strong>177</strong><span>curated support chunks</span></div>
                <div class="er-metric"><strong>gated</strong><span>retrieval by usage mode</span></div>
                <div class="er-metric"><strong>fail-closed</strong><span>safety-first pipeline</span></div>
              </div>
            </div>
            <div class="er-kicker">
              Guarded conversational RAG for emotional and student-support navigation.
              This prototype is not therapy, diagnosis, or emergency care.
            </div>
          </div>
        </div>
        """
    )

    session_id_box = gr.Textbox(
        label="Session ID",
        interactive=False,
        value=initial_state["session_id"],
    )
    audience_mode_box = gr.Radio(
        choices=[("Student", "student"), ("Helping a friend", "helping_friend")],
        value="student",
        label="Mode",
        interactive=True,
    )

    gr.HTML(
        f"""
        <div class="er-state-strip">
          <div class="er-state-pill"><span>Backend</span><strong>hybrid_ml</strong></div>
          <div class="er-state-pill"><span>Corpus</span><strong>{escape(RETRIEVAL_CORPUS)}</strong></div>
          <div class="er-state-pill"><span>Retrieval</span><strong>graph-filtered</strong></div>
          <div class="er-state-pill"><span>Logging</span><strong>{"on" if LOG_TURNS else "off"}</strong></div>
        </div>
        """
    )

    with gr.Row(elem_classes=["er-workspace"]):
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", height=500, bubble_full_width=False)
            note = (
                "Fast curated router is active for the class demo."
                if DEMO_BACKEND != "real"
                else "Full local model stack is active; first response may prewarm models."
            )
            gr.HTML(f"<div class='er-terminal-note'>{escape(note)}</div>")
            with gr.Row(elem_classes=["er-prompt-row"]):
                prompt_counseling = gr.Button("Start counseling")
                prompt_ads = gr.Button("ADS accommodations")
                prompt_ombuds = gr.Button("Advisor conflict")
                prompt_grounding = gr.Button("Grounding help")
                prompt_crisis = gr.Button("Crisis redirect")
            msg_box = gr.Textbox(
                placeholder="Type a student-support prompt...",
                label="",
                autofocus=True,
            )
            with gr.Row(elem_classes=["er-send"]):
                send_btn = gr.Button("Send", variant="primary")
                reset_btn = gr.Button("Reset Session")

        with gr.Column(scale=1, elem_classes=["er-side"]):
            timeline_out = gr.HTML(value=format_emotion_timeline([], "stable"))
            trajectory_out = gr.Textbox(label="Trajectory", value="stable", interactive=False)
            crisis_out = gr.HTML(value=format_ig_panel(False, 0.0, [], False))
            retrieval_out = gr.HTML(value=format_retrieval_panel())

    submit_outputs = [
        chatbot,
        timeline_out,
        trajectory_out,
        crisis_out,
        retrieval_out,
        session_id_box,
        session_state,
    ]

    msg_box.submit(
        respond,
        inputs=[msg_box, chatbot, session_state, audience_mode_box],
        outputs=submit_outputs,
    ).then(lambda: "", outputs=msg_box)

    send_btn.click(
        respond,
        inputs=[msg_box, chatbot, session_state, audience_mode_box],
        outputs=submit_outputs,
    ).then(lambda: "", outputs=msg_box)

    reset_btn.click(reset_session_handler, outputs=submit_outputs)

    prompt_counseling.click(
        lambda: set_prompt("I think I need counseling at UMD, but I do not know how to start."),
        outputs=msg_box,
    )
    prompt_ads.click(
        lambda: set_prompt("I need disability accommodations for my graduate assistantship work at UMD."),
        outputs=msg_box,
    )
    prompt_ombuds.click(
        lambda: set_prompt("My advisor keeps dismissing my concerns and I need someone neutral to talk to."),
        outputs=msg_box,
    )
    prompt_grounding.click(
        lambda: set_prompt("I am panicking before my exam. Can you help me with a grounding exercise?"),
        outputs=msg_box,
    )
    prompt_crisis.click(
        lambda: set_prompt("I do not think I can stay safe tonight."),
        outputs=msg_box,
    )


if __name__ == "__main__":
    os.makedirs("eval", exist_ok=True)
    demo.launch(share=SHARE_DEMO)
