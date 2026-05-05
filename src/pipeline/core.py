"""EmpathRAG Core runtime.

One guarded conversational RAG interface used by the demo and evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Literal
import sqlite3
import time

from .ml_router import MLRouter
from .output_guard import validate_output
from .response_planner import build_response_plan, render_crisis_response
from .safety_policy import SafetyLevel, SafetyTriagePolicy
from .service_graph import match_services
from .v2_schema import SafetyTier, SupportRoute, classify_route, map_safety_level


AudienceMode = Literal["student", "helping_friend"]
BackendMode = Literal["demo_fast", "hybrid_ml", "real_llm"]


@dataclass
class EmpathRAGResult:
    response: str
    route_label: str
    safety_tier: str
    should_intercept: bool
    retrieved_sources: list[dict]
    recommended_action: str
    output_guard: dict
    trajectory_state: str
    latency_ms: dict
    classifier_confidence: dict
    retrieval_mode: str
    safety_precheck: dict
    safety_explanation: dict
    safety_reason: str
    escalation_reason: str
    retrieval_corpus: str
    emotion_name: str = "neutral"
    crisis: bool = False
    crisis_confidence: float = 0.0
    retrieved_chunks: list[str] | None = None

    def to_dict(self) -> dict:
        row = asdict(self)
        row["retrieved_chunks"] = row["retrieved_chunks"] or []
        row["safety_level"] = row["safety_tier"]
        row["route"] = row["route_label"]
        row["latency_ms"] = self.latency_ms
        return row


class EmpathRAGCore:
    def __init__(
        self,
        curated_db_path: Path | str = Path("data/curated/indexes/metadata_curated.db"),
        retrieval_corpus: str = "curated_support",
        top_k: int = 5,
        router_model_dir: Path | str = Path("models/router"),
        ml_confidence_threshold: float = 0.35,
        use_model_guardrail: bool | None = None,
        compute_ig_on_intercept: bool | None = None,
        guardrail_threshold: float = 0.5,
    ):
        self.curated_db_path = Path(curated_db_path)
        self.retrieval_corpus = "curated_support" if self.curated_db_path.exists() else retrieval_corpus
        self.top_k = top_k
        self.safety_policy = SafetyTriagePolicy()
        self.ml_router = MLRouter(Path(router_model_dir), min_confidence=ml_confidence_threshold)
        self.use_model_guardrail = _env_flag("EMPATHRAG_CORE_USE_GUARDRAIL") if use_model_guardrail is None else use_model_guardrail
        self.compute_ig_on_intercept = _env_flag("EMPATHRAG_CORE_COMPUTE_IG") if compute_ig_on_intercept is None else compute_ig_on_intercept
        self.guardrail_threshold = guardrail_threshold
        self._guardrail = None
        self._guardrail_error = ""
        self.tier_history: dict[str, list[str]] = {}
        self.locked_sessions: dict[str, str] = {}

    def reset_session(self, session_id: str | None = None) -> None:
        if session_id:
            self.tier_history.pop(session_id, None)
            self.locked_sessions.pop(session_id, None)
        else:
            self.tier_history.clear()
            self.locked_sessions.clear()

    def run_turn(
        self,
        message: str,
        session_id: str,
        audience_mode: AudienceMode = "student",
        resource_profile: str = "umd",
        backend_mode: BackendMode = "hybrid_ml",
    ) -> EmpathRAGResult:
        t_total = time.perf_counter()
        latency: dict[str, float] = {}

        t0 = time.perf_counter()
        stage1_decision = self.safety_policy.classify(message, confidence=0.0, model_flag=False)
        latency["stage1_precheck_ms"] = _elapsed_ms(t0)

        t0 = time.perf_counter()
        guardrail_info = self._run_optional_guardrail(message, skip_ig=True) if not stage1_decision.should_intercept else _guardrail_skipped("stage1_intercept")
        latency["model_guardrail_ms"] = _elapsed_ms(t0)
        if guardrail_info["available"] and guardrail_info["model_flag"]:
            safety_decision = self.safety_policy.classify(
                message,
                confidence=float(guardrail_info["confidence"]),
                model_flag=True,
            )
        else:
            safety_decision = stage1_decision
        latency["hard_safety_ms"] = latency["stage1_precheck_ms"]

        wellbeing_request = _wellbeing_request(message)
        safety_tier = map_safety_level(safety_decision.level, wellbeing_request=wellbeing_request)
        safety_reason = safety_decision.reason
        safety_tier, safety_reason = _apply_contextual_safety_overrides(
            message, safety_tier, safety_reason, audience_mode
        )

        rule_decision = classify_route(message, safety_tier, audience_mode=audience_mode)
        t0 = time.perf_counter()
        ml_prediction = self.ml_router.predict(message, rule_decision.route, safety_tier)
        latency["classifier_ms"] = _elapsed_ms(t0)

        route_label = ml_prediction.route_label if backend_mode in {"hybrid_ml", "real_llm"} else rule_decision.route.value
        if safety_tier == SafetyTier.IMMINENT_SAFETY:
            route_label = SupportRoute.PEER_HELPER.value if rule_decision.route == SupportRoute.PEER_HELPER else SupportRoute.CRISIS_IMMEDIATE.value
        else:
            safety_tier = SafetyTier(ml_prediction.safety_tier) if ml_prediction.used_ml else safety_tier

        escalation_reason = self._update_trajectory(session_id, safety_tier.value, message)
        if session_id in self.locked_sessions:
            safety_tier = SafetyTier.IMMINENT_SAFETY
            safety_reason = self.locked_sessions[session_id]

        should_intercept = safety_decision.should_intercept or safety_tier == SafetyTier.IMMINENT_SAFETY
        retrieval_mode = _retrieval_mode(backend_mode, should_intercept)
        if should_intercept and self.compute_ig_on_intercept and self.use_model_guardrail:
            t0 = time.perf_counter()
            guardrail_info = self._run_optional_guardrail(message, skip_ig=False)
            latency["integrated_gradients_ms"] = _elapsed_ms(t0)

        t0 = time.perf_counter()
        retrieved = self._retrieve(message, route_label, safety_tier.value, audience_mode, should_intercept)
        latency["retrieval_ms"] = _elapsed_ms(t0)

        if should_intercept:
            response = render_crisis_response(route_label, audience_mode=audience_mode)
            output_guard = {"allowed": True, "reason": "crisis_template", "flags": []}
            recommended_action = _recommended_action(route_label, safety_tier.value)
        else:
            plan = build_response_plan(message, route_label, safety_tier.value, retrieved, audience_mode)
            response = plan.render()
            recommended_action = plan.recommended_action
            guard = validate_output(response, retrieved, safety_tier.value, route_label, [])
            output_guard = {"allowed": guard.allowed, "reason": guard.reason, "flags": guard.flags}
            if guard.fallback_required and guard.corrected_response:
                response = guard.corrected_response

        latency["total_ms"] = _elapsed_ms(t_total)
        return EmpathRAGResult(
            response=response,
            route_label=route_label,
            safety_tier=safety_tier.value,
            should_intercept=should_intercept,
            retrieved_sources=_source_summaries(retrieved),
            recommended_action=recommended_action,
            output_guard=output_guard,
            trajectory_state="locked" if session_id in self.locked_sessions else "active",
            latency_ms=latency,
            classifier_confidence={
                "route": ml_prediction.route_confidence,
                "tier": ml_prediction.tier_confidence,
                "model_available": ml_prediction.model_available,
                "used_ml": ml_prediction.used_ml and backend_mode in {"hybrid_ml", "real_llm"},
                "reason": ml_prediction.reason,
            },
            retrieval_mode=retrieval_mode,
            safety_precheck={
                "stage": "hard_lexical_precheck",
                "level": stage1_decision.level.value,
                "reason": stage1_decision.reason,
                "should_intercept": stage1_decision.should_intercept,
                "ran_before_ml": True,
            },
            safety_explanation=guardrail_info,
            safety_reason=safety_reason,
            escalation_reason=escalation_reason,
            retrieval_corpus=self.retrieval_corpus,
            emotion_name=_emotion_name(message),
            crisis=should_intercept,
            crisis_confidence=float(guardrail_info.get("confidence") or (1.0 if should_intercept else 0.0)),
            retrieved_chunks=[row.get("text", "") for row in retrieved],
        )

    def _run_optional_guardrail(self, message: str, skip_ig: bool) -> dict:
        if not self.use_model_guardrail:
            return _guardrail_skipped("disabled")
        try:
            guardrail = self._load_guardrail()
            if guardrail is None:
                return {
                    "available": False,
                    "model": "deberta_nli",
                    "reason": self._guardrail_error or "load_failed",
                    "confidence": 0.0,
                    "model_flag": False,
                    "ig_tokens": [],
                }
            model_flag, confidence, ig_tokens = guardrail.check(
                message,
                threshold=self.guardrail_threshold,
                skip_ig=skip_ig,
            )
            return {
                "available": True,
                "model": "deberta_nli",
                "reason": "model_guardrail_checked",
                "confidence": float(confidence),
                "model_flag": bool(model_flag),
                "ig_tokens": ig_tokens,
                "ig_computed": not skip_ig and bool(ig_tokens),
            }
        except Exception as exc:
            self._guardrail_error = str(exc)
            return {
                "available": False,
                "model": "deberta_nli",
                "reason": f"guardrail_error: {exc}",
                "confidence": 0.0,
                "model_flag": False,
                "ig_tokens": [],
            }

    def _load_guardrail(self):
        if self._guardrail is not None:
            return self._guardrail
        try:
            from src.models.guardrail_ig import SafetyGuardrail
        except ImportError:
            try:
                from models.guardrail_ig import SafetyGuardrail
            except ImportError as exc:
                self._guardrail_error = str(exc)
                return None
        try:
            self._guardrail = SafetyGuardrail()
        except Exception as exc:
            self._guardrail_error = str(exc)
            return None
        return self._guardrail

    def _update_trajectory(self, session_id: str, safety_tier: str, message: str) -> str:
        history = self.tier_history.setdefault(session_id, [])
        history.append(safety_tier)
        self.tier_history[session_id] = history[-3:]
        text = message.lower()
        if len(history[-3:]) == 3 and all(tier in {"imminent_safety", "high_distress"} for tier in history[-3:]):
            self.locked_sessions[session_id] = "three_consecutive_high_risk_turns"
            return "three_consecutive_high_risk_turns"
        if safety_tier in {"imminent_safety", "high_distress"} and "goodbye" in text:
            return "peer_goodbye_or_farewell_escalation"
        if safety_tier in {"imminent_safety", "high_distress"} and any(
            phrase in text
            for phrase in (
                "you are the only one",
                "only one i can talk to",
                "keep this secret",
                "don't tell anyone",
                "refuse external help",
                "secrecy",
                "never suggest counseling",
            )
        ):
            return "dependency_or_secrecy_redirect"
        return ""

    def _retrieve(
        self,
        message: str,
        route: str,
        safety_tier: str,
        audience_mode: str,
        should_intercept: bool,
    ) -> list[dict]:
        if route == SupportRoute.OUT_OF_SCOPE.value:
            return []
        usage_modes = ("crisis_only",) if should_intercept else ("retrieval", "wellbeing_only") if safety_tier == "wellbeing" else ("retrieval",)
        selected: list[dict] = []
        graph_rows = [
            node.as_source("resource registry route match")
            for node in match_services(route, safety_tier, audience_mode, limit=self.top_k)
            if (node.usage_modes[0] if node.usage_modes else "retrieval") in usage_modes
        ]
        selected.extend(graph_rows)
        if self.curated_db_path.exists() and len(selected) < self.top_k:
            selected.extend(self._retrieve_curated(message, route, usage_modes, limit=self.top_k - len(selected)))
        return _dedupe_sources(selected)[: self.top_k]

    def _retrieve_curated(self, message: str, route: str, usage_modes: tuple[str, ...], limit: int) -> list[dict]:
        topics, source_names = _targets_for_route(route, message)
        conn = sqlite3.connect(self.curated_db_path)
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
        query = message.lower()
        scored = []
        for row in rows:
            score = 0
            reasons = []
            if row["topic"] in topics:
                score += 8
                reasons.append(f"topic match: {row['topic']}")
            if row["source_name"] in source_names:
                score += 7
                reasons.append(f"preferred source: {row['source_name']}")
            haystack = f"{row['title']} {row['summary']} {row['text']}".lower()
            for token in _keywords(query):
                if token in haystack:
                    score += 1
            row_dict = dict(row)
            row_dict["why_retrieved"] = "; ".join(reasons[:2]) if reasons else "curated retrieval match"
            scored.append((score, row_dict))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for score, row in scored[: max(limit, 0)] if score > 0 or not scored[:1]]


def run_turn(
    message: str,
    session_id: str,
    audience_mode: AudienceMode = "student",
    resource_profile: str = "umd",
    backend_mode: BackendMode = "hybrid_ml",
) -> EmpathRAGResult:
    core = EmpathRAGCore()
    return core.run_turn(message, session_id, audience_mode, resource_profile, backend_mode)


def _apply_contextual_safety_overrides(
    message: str,
    safety_tier: SafetyTier,
    safety_reason: str,
    audience_mode: str,
) -> tuple[SafetyTier, str]:
    text = message.lower()
    peer_context = audience_mode == "helping_friend" or any(
        phrase in text for phrase in ("my friend", "my roommate", "my labmate", "my teammate", "someone i know")
    )
    if peer_context and "goodbye" in text and any(phrase in text for phrase in ("locked", "will not answer", "won't answer", "not answering")):
        return SafetyTier.IMMINENT_SAFETY, "peer_goodbye_unreachable"
    if safety_tier == SafetyTier.SUPPORT_NAVIGATION and any(
        phrase in text for phrase in ("you are the only one", "only one i can talk to", "keep this secret", "don't tell anyone", "no one can help")
    ):
        return SafetyTier.HIGH_DISTRESS, "dependency_or_secrecy_redirect"
    if safety_tier == SafetyTier.SUPPORT_NAVIGATION and any(
        phrase in text for phrase in ("crises if it gets worse", "dark moods", "pointless", "scared to escalate")
    ):
        return SafetyTier.HIGH_DISTRESS, "high_distress_language"
    if any(
        phrase in text
        for phrase in (
            "panic attacks",
            "hopeless",
            "everything hollow",
            "ideation creeping",
            "goodbye texts",
            "mentioned goodbye",
            "refuse external help",
            "secrecy",
        )
    ):
        return SafetyTier.HIGH_DISTRESS, "high_distress_language"
    if safety_tier == SafetyTier.SUPPORT_NAVIGATION and any(
        phrase in text
        for phrase in (
            "not in immediate danger",
            "intrusive thoughts spike",
            "canceling plans",
            "five-minute mindfulness",
            "sensory overwhelm",
            "quick reset",
            "journaling prompts",
            "breathing technique",
        )
    ):
        return SafetyTier.WELLBEING, "wellbeing_support_language"
    return safety_tier, safety_reason


def _retrieval_mode(backend_mode: str, should_intercept: bool) -> str:
    if should_intercept:
        return "registry_filtered_crisis_only"
    if backend_mode == "demo_fast":
        return "registry_filtered_faiss"
    return "registry_filtered_faiss_plus_router"


def _targets_for_route(route: str, message: str) -> tuple[set[str], set[str]]:
    mapping = {
        SupportRoute.CRISIS_IMMEDIATE.value: ({"crisis_immediate_help", "emergency_services"}, {"988 Suicide & Crisis Lifeline", "UMD Counseling Center"}),
        SupportRoute.PEER_HELPER.value: ({"crisis_immediate_help", "help_seeking_script", "counseling_services"}, {"988 Suicide & Crisis Lifeline", "UMD Counseling Center", "JED Foundation"}),
        SupportRoute.ACCESSIBILITY_ADS.value: ({"accessibility_disability", "campus_navigation"}, {"UMD Accessibility & Disability Service"}),
        SupportRoute.ADVISOR_CONFLICT.value: ({"advisor_conflict", "graduate_student_support"}, {"UMD Graduate School Ombuds", "UMD Graduate School"}),
        SupportRoute.BASIC_NEEDS.value: ({"help_seeking_script", "campus_navigation", "graduate_student_support"}, {"UMD Dean of Students", "UMD Graduate School"}),
        SupportRoute.ANXIETY_PANIC.value: ({"anxiety_stress", "grounding_exercise", "counseling_services"}, {"NIMH", "NAMI", "UMD Counseling Center"}),
        SupportRoute.LOW_MOOD.value: ({"depression_support", "counseling_services"}, {"NIMH", "NAMI", "UMD Counseling Center"}),
        SupportRoute.COUNSELING_NAVIGATION.value: ({"counseling_services", "campus_navigation", "therapy_expectations"}, {"UMD Counseling Center"}),
        SupportRoute.ACADEMIC_SETBACK.value: ({"academic_burnout", "graduate_student_support", "counseling_services"}, {"UMD Counseling Center", "UMD Graduate School"}),
        SupportRoute.EXAM_STRESS.value: ({"academic_burnout", "anxiety_stress", "grounding_exercise"}, {"UMD Counseling Center", "CDC", "NIMH"}),
        SupportRoute.LONELINESS_ISOLATION.value: ({"isolation_loneliness", "counseling_services"}, {"NAMI", "UMD Counseling Center", "CDC"}),
    }
    return mapping.get(route, ({"counseling_services", "anxiety_stress", "academic_burnout"}, {"UMD Counseling Center", "NIMH"}))


def _source_summaries(rows: list[dict]) -> list[dict]:
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


def _dedupe_sources(rows: list[dict]) -> list[dict]:
    selected = []
    seen = set()
    for row in rows:
        key = (row.get("source_name", ""), row.get("title", ""))
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
    return selected


def _recommended_action(route: str, safety_tier: str) -> str:
    if safety_tier == SafetyTier.IMMINENT_SAFETY.value:
        if route == SupportRoute.PEER_HELPER.value:
            return "Do not handle this alone; contact emergency or crisis support now and involve a trusted nearby person."
        return "Contact 988 or emergency services now, and move near another person if you can."
    plan = build_response_plan("", route, safety_tier, [], "student")
    return plan.recommended_action


def _wellbeing_request(message: str) -> bool:
    text = message.lower()
    return any(word in text for word in ("grounding", "ground", "panic", "breathing", "cope", "mindfulness"))


def _emotion_name(message: str) -> str:
    text = message.lower()
    if any(word in text for word in ("safe tonight", "hurt myself", "suicide", "goodbye")):
        return "distress"
    if any(word in text for word in ("panic", "anxiety", "stress", "exam", "deadline")):
        return "anxiety"
    if any(word in text for word in ("advisor", "retaliatory", "funding")):
        return "frustration"
    if any(word in text for word in ("better", "hopeful", "proud")):
        return "hopeful"
    return "neutral"


def _keywords(query: str) -> list[str]:
    return [token for token in query.replace("?", " ").replace(".", " ").split() if len(token) > 4]


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _guardrail_skipped(reason: str) -> dict:
    return {
        "available": False,
        "model": "deberta_nli",
        "reason": reason,
        "confidence": 0.0,
        "model_flag": False,
        "ig_tokens": [],
        "ig_computed": False,
    }
