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
    CLARIFY,
    INTERNATIONAL_SOURCE_HINT,
    OFFER,
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
    # Safety layers that can be selectively disabled for ablation evaluation.
    # See eval/run_ablation_eval.py for the harness that uses this.
    VALID_DISABLED_LAYERS = frozenset({
        "stage1_precheck",     # Stage-1 lexical safety policy precheck
        "output_guard",        # validate_output at OFFER stage
        "rephrase_safety",     # verify_rephrased_safety on LLM output
        "registry_filter",     # resource registry + retrieval filter
    })

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
        disable_layers: set[str] | None = None,
    ):
        self.disable_layers = set(disable_layers or set())
        invalid = self.disable_layers - self.VALID_DISABLED_LAYERS
        if invalid:
            raise ValueError(
                f"Unknown disable_layers: {invalid}. "
                f"Valid: {sorted(self.VALID_DISABLED_LAYERS)}"
            )
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
        # Turns since the user last mentioned F-1 / visa / international
        # status keywords. ISSS auto-surfacing decays after 2 silent turns
        # so the conversation can fully shift topic without ISSS hijacking
        # every subsequent response.
        self.session_turns_since_intl: dict[str, int] = {}
        self.session_last_specific_route: dict[str, str] = {}
        # Per-session "open offer" state. Set whenever the planner renders an
        # OFFER turn that ends in a follow-up question. Used to handle the
        # next-turn minimal-affirm response intelligently — a bare "yes" /
        # "ok" after an offer should *advance* the conversation (acknowledge
        # the consent, then deliver something concrete), not re-render the
        # same OFFER template back at the user. State machine per session:
        #   {
        #     "route": str,       # the OFFER route that set this state
        #     "question": str,    # the follow-up question that was asked
        #     "offer_turn": int,  # turn index when the OFFER was rendered
        #     "stage": "initial" | "acknowledged" | "delivered",
        #   }
        self.session_open_offer: dict[str, dict] = {}
        # Rolling per-session message log so the rephraser has continuity.
        # Only the last few user-assistant pairs are passed to the LLM; full
        # history is kept here for diagnostics. Each entry: {"role", "content"}.
        self.session_message_history: dict[str, list[dict]] = {}

    def reset_session(self, session_id: str | None = None) -> None:
        if session_id:
            self.tier_history.pop(session_id, None)
            self.locked_sessions.pop(session_id, None)
            self.session_intl_flag.pop(session_id, None)
            self.session_turns_since_intl.pop(session_id, None)
            self.session_last_specific_route.pop(session_id, None)
            self.session_open_offer.pop(session_id, None)
            self.session_message_history.pop(session_id, None)
        else:
            self.tier_history.clear()
            self.locked_sessions.clear()
            self.session_intl_flag.clear()
            self.session_turns_since_intl.clear()
            self.session_last_specific_route.clear()
            self.session_open_offer.clear()
            self.session_message_history.clear()

    # Input length cap: defends against (1) accidental wall-of-text pastes
    # ballooning LLM cost and planner regex matching, (2) deliberate DoS
    # via runaway-length payloads. Configurable via env so larger contexts
    # can be unlocked for evaluation runs that legitimately need them.
    MAX_USER_MESSAGE_CHARS = int(os.getenv("EMPATHRAG_MAX_USER_MESSAGE_CHARS", "2000"))

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
            history=self._recent_history(session_id),
            skip_safety_check="rephrase_safety" in self.disable_layers,
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

        history = self._recent_history(session_id)
        if not self.rephraser.enabled:
            # Deterministic mode: no LLM, no real streaming. Hand the template
            # through and let the demo's word-chunk reveal layer fake-stream it
            # the way it always has.
            rephrase_result = self.rephraser.rephrase(
                user_message=message,
                template_response=plan.template_response,
                retrieved_sources=plan.retrieved,
                recommended_action=plan.recommended_action,
                history=history,
                skip_safety_check="rephrase_safety" in self.disable_layers,
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
            history=history,
            skip_safety_check="rephrase_safety" in self.disable_layers,
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

    def _recent_history(self, session_id: str, max_messages: int = 4) -> list[dict]:
        """Return the last ``max_messages`` entries from this session's
        message history. Used to give the rephraser continuity across turns
        without leaking older context the planner has already moved past."""
        hist = self.session_message_history.get(session_id, [])
        return hist[-max_messages:] if hist else []

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

        # Length cap: a very long input is either an accidental wall-of-text
        # paste or a DoS attempt. We truncate before the planner sees it (so
        # regex matching + downstream LLM cost stay bounded) and surface a
        # short clarify-style response if the truncation actually mattered.
        message_was_truncated = False
        if message and len(message) > self.MAX_USER_MESSAGE_CHARS:
            message = message[: self.MAX_USER_MESSAGE_CHARS]
            message_was_truncated = True

        t0 = time.perf_counter()
        if "stage1_precheck" in self.disable_layers:
            # Ablation: pretend the message passes safety. Used to measure
            # what fraction of escalation catches depend on the Stage-1
            # lexical check vs. downstream layers (ML router, contextual
            # overrides, output guard). Never set in production.
            from .safety_policy import SafetyDecision, SafetyLevel as _SL
            stage1_decision = SafetyDecision(
                level=_SL.PASS, confidence=0.0,
                reason="stage1_disabled_ablation", should_intercept=False,
            )
        else:
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
        if "registry_filter" in self.disable_layers:
            # Ablation: no resource-registry filtering, no curated retrieval.
            # Planner falls back to generic templates with no named UMD
            # resources. Used to measure how much of the system's
            # resource-grounding depends on the registry layer.
            retrieved = []
        else:
            retrieved = self._retrieve(message, route_label, safety_tier.value, audience_mode, should_intercept)
        latency["retrieval_ms"] = _elapsed_ms(t0)

        # Cross-cutting: when an F-1 / visa / international-status worry shows up
        # at any point in the session, keep the ISSS surface and the
        # international-aware framing for subsequent turns — but decay it so
        # ISSS doesn't hijack every later turn after the topic has shifted.
        intl_now = has_international_concern(message)
        if intl_now:
            self.session_intl_flag[session_id] = True
            self.session_turns_since_intl[session_id] = 0
        else:
            self.session_turns_since_intl[session_id] = (
                self.session_turns_since_intl.get(session_id, 0) + 1
            )
        intl_session = self.session_intl_flag.get(session_id, False)
        # "Active" means F-1 framing should drive retrieval + sub-topic logic
        # this turn. After 2 turns with no F-1 keyword the flag remains True
        # for diagnostics but stops dominating responses.
        intl_active = intl_session and self.session_turns_since_intl.get(session_id, 0) <= 2

        if not should_intercept and intl_active:
            already_has_isss = any(
                (s.get("source_name") or "").lower().startswith("umd international")
                or (s.get("service_id") or "").startswith("umd_isss")
                for s in retrieved
            )
            if not already_has_isss:
                retrieved = [INTERNATIONAL_SOURCE_HINT] + retrieved

        intl_topic = ""
        if should_intercept:
            template_response = render_crisis_response(
                route_label, audience_mode=audience_mode, user_message=message,
            )
            recommended_action = _recommended_action(route_label, safety_tier.value)
            stage = "offer"
        else:
            response_plan = build_response_plan(
                message,
                route_label,
                safety_tier.value,
                retrieved,
                audience_mode,
                international_concern_override=intl_active,
            )
            stage = decide_stage(message, route_label, safety_tier.value, turn_index)
            # Only classify F-1 sub-topic when the framing is actively in play.
            # After 2 silent turns, decayed; don't pin the planner to OPT/RCL/etc.
            intl_topic = classify_intl_topic(message) if intl_active else ""
            # Conversational utterances (greeting / goodbye / meta) route to
            # short purposeful templates BEFORE the regular minimal /
            # incomplete checks. These aren't support content — they're
            # conversation openers and closers. Treating them as OFFER
            # responses produces jarring "let me tell you about UMD CC"
            # replies to a simple "hi".
            conv_intent = (
                "" if message_was_truncated else _conversational_intent(message)
            )
            # Length-cap / minimal / incomplete short-circuits, in priority
            # order. Each picks a clarify-style template instead of trying
            # to OFFER a full route response on insufficient or excessive
            # input. Output guard skipped for the clarify stage.
            minimal_kind = "" if (message_was_truncated or conv_intent) else _minimal_response_kind(message)
            if message_was_truncated:
                template_response = (
                    "That's a lot, and I want to make sure I focus on what matters most to you. "
                    "Your message was long enough that I only have the first part. Could you say "
                    "which piece feels most pressing right now, in a sentence or two?"
                )
                stage = CLARIFY
                recommended_action = response_plan.recommended_action
            elif conv_intent == "greeting":
                template_response = _render_greeting()
                stage = CLARIFY
                recommended_action = response_plan.recommended_action
            elif conv_intent == "goodbye":
                template_response = _render_goodbye()
                stage = CLARIFY
                recommended_action = response_plan.recommended_action
            elif conv_intent == "meta":
                template_response = _render_meta()
                stage = CLARIFY
                recommended_action = response_plan.recommended_action
            elif minimal_kind == "affirm" and _open_offer_recent(
                self.session_open_offer.get(session_id), turn_index
            ):
                # The student just said yes/ok to an OFFER we made on a
                # recent turn. Re-rendering the same template at this point
                # was the bug: the user already consented, the planner
                # should advance. Two-stage advance:
                #   1st affirm  -> acknowledge consent + naturalize the
                #                  binary disambiguation that the OFFER asked
                #   2nd affirm  -> stop asking, deliver a concrete starter
                #                  appropriate to the route
                offer_state = dict(self.session_open_offer[session_id])
                if offer_state.get("stage") == "initial":
                    template_response = _render_consent_acknowledged(
                        offer_state.get("route", ""),
                        offer_state.get("question", ""),
                    )
                    offer_state["stage"] = "acknowledged"
                    offer_state["acknowledged_turn"] = turn_index
                    self.session_open_offer[session_id] = offer_state
                    stage = OFFER
                else:
                    template_response = _render_consent_delivered(
                        offer_state.get("route", "")
                    )
                    offer_state["stage"] = "delivered"
                    offer_state["delivered_turn"] = turn_index
                    self.session_open_offer[session_id] = offer_state
                    stage = OFFER
                recommended_action = response_plan.recommended_action
            elif minimal_kind:
                template_response = _render_minimal_followup(minimal_kind, intl_session)
                stage = CLARIFY
                recommended_action = response_plan.recommended_action
            elif _is_incomplete_message(message):
                # The student's message trails off ("what should i do to",
                # "honestly", "kind of"). Guessing the intent and producing
                # a full OFFER response is worse than asking them to finish.
                template_response = _render_incomplete_followup()
                stage = CLARIFY
                recommended_action = response_plan.recommended_action
            elif intl_topic and stage == "offer":
                template_response = render_intl_factual_offer(intl_topic, message)
                recommended_action = response_plan.recommended_action
            else:
                template_response = response_plan.render(stage)
                recommended_action = response_plan.recommended_action

            # Capture an open-offer slot whenever this turn rendered an OFFER
            # with a follow-up question, so the next turn can recognize a
            # bare "yes" / "ok" as consent and advance the conversation.
            # Skip when the consent flow itself just rendered this turn —
            # the slot has already been updated above and overwriting it
            # would reset the consent progression.
            existing_offer = self.session_open_offer.get(session_id) or {}
            advanced_this_turn = (
                existing_offer.get("acknowledged_turn") == turn_index
                or existing_offer.get("delivered_turn") == turn_index
            )
            if (
                stage == OFFER
                and not should_intercept
                and response_plan.follow_up_question
                and not advanced_this_turn
            ):
                self.session_open_offer[session_id] = {
                    "route": route_label,
                    "question": response_plan.follow_up_question.strip(),
                    "offer_turn": turn_index,
                    "stage": "initial",
                }
            elif (
                stage not in (OFFER, CLARIFY)
                and not should_intercept
                and not advanced_this_turn
            ):
                # LISTEN / PERMISSION turns reset any stale open-offer so we
                # don't bind a much-later "yes" to an offer the user already
                # moved past.
                self.session_open_offer.pop(session_id, None)

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
        if "output_guard" in self.disable_layers:
            output_guard = {"allowed": True, "reason": "output_guard_disabled_ablation", "flags": []}
        elif plan.should_intercept:
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

        # Append this turn to the per-session history so the next turn has
        # context. Trimmed to last ~6 messages to bound memory + LLM payload.
        hist = self.session_message_history.setdefault(plan.session_id, [])
        hist.append({"role": "user", "content": plan.message})
        hist.append({"role": "assistant", "content": response})
        if len(hist) > 12:  # 6 user-assistant pairs
            del hist[: len(hist) - 12]

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
        SupportRoute.AUTHORITY_MISCONDUCT.value: (
            {"authority_misconduct", "care_violence_confidential", "campus_navigation"},
            {"UMD Office of Civil Rights & Sexual Misconduct (OCRSM)", "UMD Office of Student Conduct", "UMD Dean of Students", "UMD CARE to Stop Violence"},
        ),
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
            # Trust signal: surface the verification date in the UI so users
            # (especially clinicians) can see how fresh the link is.
            "last_verified": row.get("last_verified") or row.get("last_checked", ""),
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


# Words that, when they're the last token of a short message, strongly
# suggest the user trailed off mid-sentence. Conservative list: only
# prepositions/conjunctions/articles/auxiliaries that almost never close
# a real sentence. We deliberately omit "is", "are", "was" — those can
# legitimately close a sentence ("It is what it is").
_INCOMPLETE_TERMINAL_WORDS = frozenset({
    # Prepositions
    "to", "for", "with", "about", "of", "on", "at", "by", "from", "into",
    "over", "under", "between", "without", "through", "during",
    # Conjunctions / subordinators
    "and", "but", "or", "because", "if", "when", "while", "though",
    "although", "since", "until", "whereas",
    # Articles
    "a", "an", "the",
    # Auxiliary fragments
    "should", "would", "could", "might", "may", "will", "can", "do",
    "does", "did",
    # Fragment determiners (only ones that never legitimately close a sentence)
    "my", "your", "their", "our",
    # Transitive verbs that almost always need an object
    "want", "need", "wish",
})
# Deliberately NOT in the terminal set: "this", "that", "these", "those",
# "her", "his", "him". These DO legitimately close sentences ("I don't know
# what to do with that.", "I told her."), and putting them in the terminal
# set produces false positives on real emotional disclosure.

# Words that close a sentence legitimately in longer messages ("It is what
# it is") but signal incompleteness in very short ones ("the thing is").
_SHORT_ONLY_TERMINAL_WORDS = frozenset({
    "is", "are", "was", "were", "i",
})

# Whole-message hedges that carry no content. Distinct from minimal
# affirmations (yes/no/ok) — those have a routing meaning; these don't.
_HEDGE_ONLY_MESSAGES = frozenset({
    "honestly", "kind of", "kinda", "sort of", "sorta", "i mean",
    "well", "uh", "um", "you know", "like", "i guess", "hmm",
    "idk maybe", "idk i guess", "maybe idk",
})


def _is_incomplete_message(message: str) -> bool:
    """True if the message looks like it trailed off or is hedge-only."""
    if not message:
        return True
    raw = message.strip()
    if not raw:
        return True
    # Pure ellipsis / punctuation-only
    if all(c in ".,;:!? \t-" for c in raw):
        return True
    # Strip trailing punctuation/whitespace before testing the terminal word
    text = raw.rstrip(".,!?;:- \t").strip()
    if not text:
        return True
    text_lower = " ".join(text.lower().split())
    if text_lower in _HEDGE_ONLY_MESSAGES:
        return True
    # Last word ending the (short) message is a known incomplete-terminal
    words = text_lower.split()
    if not words:
        return True
    if words[-1] in _INCOMPLETE_TERMINAL_WORDS and len(text) < 80:
        return True
    # Short-only terminals: "is" / "are" close real sentences in longer
    # messages but signal incompleteness in fragments like "the thing is".
    if words[-1] in _SHORT_ONLY_TERMINAL_WORDS and len(words) <= 3:
        return True
    return False


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


# Conversation openers / closers / meta — these are NOT student-support
# content. They route to short purposeful templates instead of a heavy
# OFFER response that would feel jarring for an opener or premature for a
# close.
_GREETING_PATTERNS = frozenset({
    "hi", "hello", "hey", "yo", "hiya", "hey there", "hi there", "hello there",
    "good morning", "good afternoon", "good evening",
    "sup", "what's up", "whats up", "wassup",
    "hey empathrag", "hi empathrag",
})
_GOODBYE_PATTERNS = frozenset({
    "thanks", "thank you", "thx", "ty", "thank u",
    "bye", "goodbye", "see you", "see ya", "later",
    "i'll think about it", "ill think about it", "let me think",
    "got it", "cool", "appreciate it", "that helps",
    "alright then", "ok thanks", "thanks bye",
})
_META_PATTERNS = (
    "are you a bot", "are you an ai", "are you a robot", "are you human",
    "what are you", "who are you", "are you real", "are you a person",
    "is this an ai", "is this a bot", "is this real",
    "what can you do", "how do you work", "what is this",
)


def _conversational_intent(message: str) -> str:
    """Returns 'greeting' / 'goodbye' / 'meta' / '' for non-support
    conversational utterances. These are routed to short purposeful
    templates rather than full OFFER responses."""
    text = (message or "").strip().lower()
    if not text or len(text) > 60:
        return ""
    clean = " ".join(text.rstrip(".,!?;:").split())
    if clean in _GREETING_PATTERNS:
        return "greeting"
    if clean in _GOODBYE_PATTERNS:
        return "goodbye"
    if any(pattern in clean for pattern in _META_PATTERNS):
        return "meta"
    return ""


def _render_greeting() -> str:
    return (
        "Hi. I'm here if there's something on your mind — academic stress, "
        "visa or status worry, feeling low, helping a friend through "
        "something, or anything else weighing on you. Say what feels "
        "easiest to share first."
    )


def _render_goodbye() -> str:
    return (
        "Take care of yourself. The resources we talked through will still "
        "be here when you're ready to follow up. If anything shifts in the "
        "meantime — especially anything that feels urgent — UMD Counseling "
        "Center is at counseling.umd.edu and 988 is always there."
    )


def _render_meta() -> str:
    return (
        "I'm EmpathRAG — a research prototype built at UMD to help students "
        "navigate campus support. I'm not a counselor, therapist, or "
        "emergency service. I listen, then point to verified UMD resources "
        "when it feels like the right moment. Anything you'd like to start with?"
    )


def _render_incomplete_followup() -> str:
    """Response for messages that trail off mid-sentence or are content-free
    hedges. Invite completion without guessing at intent."""
    return (
        "Looks like the thought might have cut off. Take your time and say "
        "the rest when you're ready, in whatever words feel right."
    )


def _open_offer_recent(state: dict | None, current_turn: int) -> bool:
    """Is there an OFFER on the table that the user could be saying yes to?

    True if an OFFER was rendered on the previous or current turn (current
    can match in unusual harness flows where turn_index is not strictly
    incrementing) and we haven't already delivered the concrete starter.
    """
    if not state:
        return False
    offer_turn = state.get("offer_turn", 0)
    if not isinstance(offer_turn, int):
        return False
    # Allow consent on the immediately following turn or two — leaves a
    # small margin for the second affirm in a row, which is the bug we're
    # fixing. After "delivered" we stop binding new affirms here.
    if state.get("stage") == "delivered":
        return False
    return current_turn - offer_turn in (1, 2)


def _render_consent_acknowledged(route: str, prior_question: str) -> str:
    """First minimal-affirm after an OFFER. Acknowledge the consent and
    naturalize the binary disambiguation that the OFFER asked for. The
    point is to *advance* the turn: the user said yes, the system should
    acknowledge that yes and ask a more direct, conversational pick-one
    question instead of repeating the OFFER template verbatim."""
    question = (prior_question or "").strip().rstrip("?")
    if question:
        return (
            "Glad you're up for it. To land in the right place — "
            f"{question}? You can answer in a sentence, or just name the "
            "part you want to start with."
        )
    return (
        "Glad you're up for it. Could you say in your own words which "
        "piece you'd like to start with? Even one sentence is enough."
    )


def _render_consent_delivered(route: str) -> str:
    """Second minimal-affirm in a row after an OFFER. Stop asking the
    student to choose and deliver a concrete starter scoped to the route.
    Keeps the conversation moving and gives the student something usable
    to take with them."""
    if route == "academic_setback":
        return (
            "Okay, let's just start. Here's a short email you can adapt "
            "for your professor or TA — change the bracketed bits:\n\n"
            "Subject: Quick check-in during office hours\n\n"
            "Hi Professor [Last Name], I'm in your [Course]. I've been "
            "having a hard time keeping up the last few weeks and I'd like "
            "to use a few minutes of office hours to figure out the best "
            "way forward. Would [day/time] work, or is there another time "
            "you'd suggest?\n\n"
            "Thanks for your time, [Your Name]\n\n"
            "Send that, then come back and we can talk through anything "
            "else that's been weighing on you."
        )
    if route == "exam_stress":
        return (
            "Okay, small concrete start. Tonight: pick two topics from the "
            "exam that you actually want to be solid on, set a 25-minute "
            "timer for the first one, then take 5. That's it. The rest of "
            "the prep can wait until after a real reset.\n\n"
            "If the stress spikes during the reset, the UMD Counseling "
            "Center has same-day phone consultations at counseling.umd.edu."
        )
    if route == "anxiety_panic":
        return (
            "Okay, let's start with a short grounding reset. Try this:\n\n"
            "1. Name five things you can see right now.\n"
            "2. Name four things you can hear.\n"
            "3. Three slow breaths — in for four counts, out for six.\n\n"
            "Take it slow. When you're done, tell me how it landed."
        )
    if route == "low_mood":
        return (
            "Okay, small start. Pick one thing today that feels almost "
            "too easy — drink a glass of water, step outside for two "
            "minutes, message one person to say hi. Just pick the one. "
            "Then come back and tell me how it went.\n\n"
            "If anything tips into not feeling safe, UMD Counseling "
            "Center is at counseling.umd.edu and 988 is always there."
        )
    if route == "loneliness_isolation":
        return (
            "Okay, one small reach. Think of one person you used to talk "
            "to who wouldn't feel hard to message. The text doesn't have "
            "to be deep — \"hey, been thinking about you, how are you?\" "
            "is enough. You don't have to send it now; just pick the "
            "person.\n\n"
            "If you want, name them and we can plan what to say."
        )
    if route in ("advisor_conflict", "authority_misconduct"):
        return (
            "Okay, let's start by writing it down. A short factual "
            "timeline — date, who was there, what was said, in your own "
            "words. Even three or four lines is enough for now. That "
            "gives you something concrete to bring to whichever office "
            "fits, without committing to anything yet.\n\n"
            "Want to draft those few lines together?"
        )
    if route == "accessibility_ads":
        return (
            "Okay, one small step: open accessibility.umd.edu and start "
            "the ADS intake form — you don't have to finish it tonight, "
            "just open it and save your name. The form asks about the "
            "specific course and barrier, which is easier to fill out "
            "with the syllabus in front of you.\n\n"
            "When you're ready, we can talk through what to put in the "
            "barrier section."
        )
    if route == "basic_needs":
        return (
            "Okay, one concrete first step: today, contact UMD's Dean of "
            "Students Office (dos.umd.edu) and tell them plainly what "
            "you need help with — food, housing, or money. They route "
            "students to Campus Pantry and emergency funds and they're "
            "used to getting these calls.\n\n"
            "Want help drafting a short message to send them?"
        )
    if route == "counseling_navigation":
        return (
            "Okay, one concrete step: open counseling.umd.edu and book "
            "the brief 20-minute initial consultation. That's the "
            "intake — it's not therapy yet, just a quick conversation "
            "where they listen and tell you what fits (group, "
            "individual, off-campus referral). You can do it by phone.\n\n"
            "If a deadline or finals window is making this urgent, say "
            "that on the call — they have same-day options."
        )
    # Default — generic small concrete starter
    return (
        "Okay, let's just start small. In one sentence, what feels most "
        "pressing right now — not the whole picture, just whatever's "
        "loudest today? We'll work from there, one step at a time."
    )


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
