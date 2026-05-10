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
from .response_planner import (
    INTERNATIONAL_SOURCE_HINT,
    build_response_plan,
    classify_intl_topic,
    decide_stage,
    has_international_concern,
    render_crisis_response,
    render_intl_factual_offer,
)
from .rephraser import RephraseResult, ResponseRephraser
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
    international_concern: bool = False
    conversation_stage: str = "offer"
    turn_index: int = 1
    intl_topic: str = ""
    rephraser_provider: str = "deterministic"
    rephraser_used_llm: bool = False
    rephraser_latency_ms: float = 0.0
    rephraser_last_error: str = ""

    def to_dict(self) -> dict:
        row = asdict(self)
        row["retrieved_chunks"] = row["retrieved_chunks"] or []
        row["safety_level"] = row["safety_tier"]
        row["route"] = row["route_label"]
        row["latency_ms"] = self.latency_ms
        return row


@dataclass
class _TurnPlan:
    """Everything ``_plan_turn`` decides before the rephrase step.

    Shared between sync ``run_turn`` and ``run_turn_streaming`` so the
    pre-rephrase planning logic isn't duplicated across the two paths.
    """
    message: str
    session_id: str
    audience_mode: str
    backend_mode: str
    turn_index: int
    template_response: str
    retrieved: list[dict]
    recommended_action: str
    route_label: str
    safety_tier: SafetyTier
    safety_reason: str
    stage: str
    intl_concern: bool
    intl_topic: str
    should_intercept: bool
    retrieval_mode: str
    latency: dict
    t_total: float
    stage1_level: str
    stage1_reason: str
    stage1_should_intercept: bool
    ml_prediction: object
    guardrail_info: dict
    escalation_reason: str


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
        self.rephraser = ResponseRephraser()
        self._guardrail = None
        self._guardrail_error = ""
        self.tier_history: dict[str, list[str]] = {}
        self.locked_sessions: dict[str, str] = {}
        # Session-scoped sticky flags so context detected on one turn carries
        # forward (international concern, last specific route) instead of being
        # re-derived from each message in isolation.
        self.session_intl_flag: dict[str, bool] = {}
        self.session_last_specific_route: dict[str, str] = {}

    def reset_session(self, session_id: str | None = None) -> None:
        if session_id:
            self.tier_history.pop(session_id, None)
            self.locked_sessions.pop(session_id, None)
            self.session_intl_flag.pop(session_id, None)
            self.session_last_specific_route.pop(session_id, None)
        else:
            self.tier_history.clear()
            self.locked_sessions.clear()
            self.session_intl_flag.clear()
            self.session_last_specific_route.clear()

    def run_turn(
        self,
        message: str,
        session_id: str,
        audience_mode: AudienceMode = "student",
        resource_profile: str = "umd",
        backend_mode: BackendMode = "hybrid_ml",
        turn_index: int | None = None,
    ) -> EmpathRAGResult:
        plan = self._plan_turn(message, session_id, audience_mode, backend_mode, turn_index)
        if plan.should_intercept:
            return self._finalize_turn(plan, plan.template_response, None)
        rephrase_result = self.rephraser.rephrase(
            user_message=message,
            template_response=plan.template_response,
            retrieved_sources=plan.retrieved,
            recommended_action=plan.recommended_action,
        )
        return self._finalize_turn(plan, rephrase_result.response, rephrase_result)

    def run_turn_streaming(
        self,
        message: str,
        session_id: str,
        audience_mode: AudienceMode = "student",
        resource_profile: str = "umd",
        backend_mode: BackendMode = "hybrid_ml",
        turn_index: int | None = None,
    ):
        """Generator variant of :meth:`run_turn`.

        Yields:

        * ``("token", accumulated_text)`` for each streamed chunk
        * ``("done", EmpathRAGResult)`` exactly once at the end

        Crisis interception, deterministic mode, and safety-rejected rephrases
        all collapse to a single deterministic ``("token", final_text)`` plus
        a ``("done", ...)``. Real Groq/Anthropic streaming only kicks in when
        ``EMPATHRAG_REPHRASER_ENABLED=1`` and the route is non-crisis.
        """
        plan = self._plan_turn(message, session_id, audience_mode, backend_mode, turn_index)

        if plan.should_intercept:
            result = self._finalize_turn(plan, plan.template_response, None)
            yield ("token", result.response)
            yield ("done", result)
            return

        if not self.rephraser.enabled:
            # Deterministic mode: no LLM, no real streaming. Hand the template
            # through and let the demo's word-chunk reveal layer fake-stream it
            # the way it always has.
            rephrase_result = self.rephraser.rephrase(
                user_message=message,
                template_response=plan.template_response,
                retrieved_sources=plan.retrieved,
                recommended_action=plan.recommended_action,
            )
            result = self._finalize_turn(plan, rephrase_result.response, rephrase_result)
            yield ("token", result.response)
            yield ("done", result)
            return

        accumulated = ""
        rephrase_result: RephraseResult | None = None
        for event in self.rephraser.rephrase_streaming(
            user_message=message,
            template_response=plan.template_response,
            retrieved_sources=plan.retrieved,
            recommended_action=plan.recommended_action,
        ):
            if event[0] == "chunk":
                _, accumulated, _provider_name = event
                yield ("token", accumulated)
            elif event[0] == "final":
                _, rephrase_result = event

        # rephrase_streaming guarantees exactly one ("final", ...).
        assert rephrase_result is not None
        result = self._finalize_turn(plan, rephrase_result.response, rephrase_result)
        # If the output guard corrected the response or the streamed text was
        # safety-rejected and replaced with the deterministic fallback, the
        # bubble currently shows stale text. Yield one more correction.
        if result.response != accumulated:
            yield ("token", result.response)
        yield ("done", result)

    # ------------------------------------------------------------------
    # Internal: plan + finalize
    # ------------------------------------------------------------------
    def _plan_turn(
        self,
        message: str,
        session_id: str,
        audience_mode: AudienceMode,
        backend_mode: BackendMode,
        turn_index: int | None,
    ) -> _TurnPlan:
        if turn_index is None:
            turn_index = len(self.tier_history.get(session_id, [])) + 1
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

        # Cross-cutting: when an F-1 / visa / international-status worry shows up
        # at any point in the session, keep the ISSS surface and the
        # international-aware framing for subsequent turns.
        intl_now = has_international_concern(message)
        if intl_now:
            self.session_intl_flag[session_id] = True
        intl_session = self.session_intl_flag.get(session_id, False)

        if not should_intercept and intl_session:
            already_has_isss = any(
                (s.get("source_name") or "").lower().startswith("umd international")
                or (s.get("service_id") or "").startswith("umd_isss")
                for s in retrieved
            )
            if not already_has_isss:
                retrieved = [INTERNATIONAL_SOURCE_HINT] + retrieved

        intl_topic = ""
        if should_intercept:
            template_response = render_crisis_response(route_label, audience_mode=audience_mode)
            recommended_action = _recommended_action(route_label, safety_tier.value)
            stage = "offer"
        else:
            response_plan = build_response_plan(
                message,
                route_label,
                safety_tier.value,
                retrieved,
                audience_mode,
                international_concern_override=intl_session,
            )
            stage = decide_stage(message, route_label, safety_tier.value, turn_index)
            intl_topic = classify_intl_topic(message) if intl_session else ""
            # Minimal-affirmation handler. "yes" / "no" / "ok" with no other
            # content has no new intent — re-rendering the OFFER template just
            # repeats the previous turn. Instead, route to a clarifying-question
            # response that asks the student which thread they want to pull.
            minimal_kind = _minimal_response_kind(message)
            if minimal_kind:
                template_response = _render_minimal_followup(minimal_kind, intl_session)
                stage = "clarify"
                recommended_action = response_plan.recommended_action
            elif intl_topic and stage == "offer":
                template_response = render_intl_factual_offer(intl_topic, message)
                recommended_action = response_plan.recommended_action
            else:
                template_response = response_plan.render(stage)
                recommended_action = response_plan.recommended_action

        return _TurnPlan(
            message=message,
            session_id=session_id,
            audience_mode=audience_mode,
            backend_mode=backend_mode,
            turn_index=turn_index,
            template_response=template_response,
            retrieved=retrieved,
            recommended_action=recommended_action,
            route_label=route_label,
            safety_tier=safety_tier,
            safety_reason=safety_reason,
            stage=stage,
            intl_concern=intl_session,
            intl_topic=intl_topic,
            should_intercept=should_intercept,
            retrieval_mode=retrieval_mode,
            latency=latency,
            t_total=t_total,
            stage1_level=stage1_decision.level.value,
            stage1_reason=stage1_decision.reason,
            stage1_should_intercept=stage1_decision.should_intercept,
            ml_prediction=ml_prediction,
            guardrail_info=guardrail_info,
            escalation_reason=escalation_reason,
        )

    def _finalize_turn(
        self,
        plan: _TurnPlan,
        response: str,
        rephrase_result: RephraseResult | None,
    ) -> EmpathRAGResult:
        if plan.should_intercept:
            output_guard = {"allowed": True, "reason": "crisis_template", "flags": []}
        elif plan.stage == "clarify":
            # Clarifying replies after minimal user input (yes/no/maybe) are
            # intentionally short open-ended invitations. The OFFER-stage
            # missing-action / pure-validation checks would punish them.
            output_guard = {"allowed": True, "reason": "minimal_response_clarify", "flags": []}
        elif plan.stage == "offer":
            # Output guard runs after rephrase (or fall-through to template)
            # only at OFFER stage. LISTEN/PERMISSION are intentionally
            # reflective and would fail missing-action / pure-validation checks.
            guard = validate_output(
                response, plan.retrieved, plan.safety_tier.value, plan.route_label, []
            )
            output_guard = {"allowed": guard.allowed, "reason": guard.reason, "flags": guard.flags}
            if guard.fallback_required and guard.corrected_response:
                response = guard.corrected_response
        else:
            output_guard = {"allowed": True, "reason": f"listening_stage_{plan.stage}", "flags": []}

        latency = dict(plan.latency)
        latency["total_ms"] = _elapsed_ms(plan.t_total)

        return EmpathRAGResult(
            response=response,
            route_label=plan.route_label,
            safety_tier=plan.safety_tier.value,
            should_intercept=plan.should_intercept,
            retrieved_sources=_source_summaries(plan.retrieved),
            recommended_action=plan.recommended_action,
            output_guard=output_guard,
            trajectory_state="locked" if plan.session_id in self.locked_sessions else "active",
            latency_ms=latency,
            classifier_confidence={
                "route": plan.ml_prediction.route_confidence,
                "tier": plan.ml_prediction.tier_confidence,
                "model_available": plan.ml_prediction.model_available,
                "used_ml": plan.ml_prediction.used_ml and plan.backend_mode in {"hybrid_ml", "real_llm"},
                "reason": plan.ml_prediction.reason,
            },
            retrieval_mode=plan.retrieval_mode,
            safety_precheck={
                "stage": "hard_lexical_precheck",
                "level": plan.stage1_level,
                "reason": plan.stage1_reason,
                "should_intercept": plan.stage1_should_intercept,
                "ran_before_ml": True,
            },
            safety_explanation=plan.guardrail_info,
            safety_reason=plan.safety_reason,
            escalation_reason=plan.escalation_reason,
            retrieval_corpus=self.retrieval_corpus,
            emotion_name=_emotion_name(plan.message),
            crisis=plan.should_intercept,
            crisis_confidence=float(plan.guardrail_info.get("confidence") or (1.0 if plan.should_intercept else 0.0)),
            retrieved_chunks=[row.get("text", "") for row in plan.retrieved],
            international_concern=plan.intl_concern,
            conversation_stage=plan.stage,
            turn_index=plan.turn_index,
            intl_topic=plan.intl_topic,
            rephraser_provider=(rephrase_result.provider_name if rephrase_result else "deterministic"),
            rephraser_used_llm=(rephrase_result.used_llm if rephrase_result else False),
            rephraser_latency_ms=(rephrase_result.latency_ms if rephrase_result else 0.0),
            rephraser_last_error=(rephrase_result.last_error if rephrase_result else ""),
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
            "devastated",
            "terrified",
            "scared about",
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
            "documents": row.get("documents", []) or [],
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


_MINIMAL_AFFIRM = frozenset({
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "kk", "k",
    "alright", "fine", "definitely", "absolutely",
})
_MINIMAL_NEGATE = frozenset({"no", "nope", "nah", "not really", "nuh"})
_MINIMAL_UNSURE = frozenset({"maybe", "idk", "dunno", "unsure", "not sure", "perhaps"})


def _minimal_response_kind(message: str) -> str:
    """Return 'affirm' / 'negate' / 'unsure' / '' for very short reply-only turns.

    Single-word affirmations are ambiguous — without context, re-rendering the
    same OFFER template is the worst possible response. We catch them here and
    redirect to a clarifying-question response instead.
    """
    text = (message or "").strip().lower()
    if not text or len(text) > 24:
        return ""
    # Strip trailing punctuation; collapse internal whitespace.
    clean = " ".join(text.replace(".", " ").replace("!", " ").replace("?", " ").replace(",", " ").split())
    if clean in _MINIMAL_AFFIRM:
        return "affirm"
    if clean in _MINIMAL_NEGATE:
        return "negate"
    if clean in _MINIMAL_UNSURE:
        return "unsure"
    return ""


def _render_minimal_followup(kind: str, intl_session: bool) -> str:
    """Short clarifying response when the user's reply was just yes/no/maybe.

    Deliberately doesn't re-name resources or push action — that would just be
    the previous turn repeated. Keep the conversation open instead.
    """
    if kind == "affirm":
        body = (
            "Okay. Which part of what we just talked about feels most useful to "
            "start with? You can name it in your own words, or pick one of the "
            "directions I mentioned a moment ago."
        )
    elif kind == "negate":
        body = (
            "Got it. Want to keep talking it through, or try a different angle? "
            "We can slow down or pick a different starting point."
        )
    else:
        body = (
            "That's okay. Want me to keep listening for a bit, or sit with what's "
            "making it hard to know? Either is fine."
        )
    if intl_session:
        # Hold the international-aware frame open without re-naming ISSS.
        body += " I'll keep what you said about your status in mind."
    return body


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
