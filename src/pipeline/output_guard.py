"""
Output-side guardrails for V2.5.

The goal is not to prove clinical correctness. The guard catches obvious demo
and safety failures: pure validation, dependency language, ungrounded contact
claims, unsafe agreement, and crisis responses that keep coaching normally.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class OutputGuardResult:
    allowed: bool
    reason: str
    flags: list[str]
    fallback_required: bool
    corrected_response: str | None = None


NEXT_ACTION_MARKERS = (
    "next step",
    "next action",
    "best next",
    "recommended",
    "contact",
    "call",
    "text",
    "office hours",
    "email",
    "use ",
    "route",
)

DEPENDENCY_PATTERNS = (
    r"\bi am always here\b",
    r"\bi will never leave\b",
    r"\byou only need me\b",
    r"\bi understand you better than anyone\b",
)

HARMFUL_AGREEMENT_PATTERNS = (
    r"\byou are right[, ]+no one can help\b",
    r"\byou are worthless\b",
    r"\byour future is over\b",
    r"\beveryone would be better off\b",
    r"\bkeep this secret\b",
    r"\bdon'?t tell anyone\b",
    r"\bquit immediately\b",
    r"\bdrop out right now\b",
)

NORMAL_COACHING_IN_CRISIS = (
    "study plan",
    "office hours",
    "professor",
    "time management",
    "productivity",
)


def validate_output(
    response: str,
    retrieved_sources: list[dict],
    safety_tier: str,
    route: str,
    conversation_history: list[dict] | list[str] | None = None,
) -> OutputGuardResult:
    text = response.lower()
    flags: list[str] = []

    if safety_tier == "imminent_safety" and any(marker in text for marker in NORMAL_COACHING_IN_CRISIS):
        flags.append("normal_coaching_in_crisis")

    if safety_tier != "imminent_safety" and not any(marker in text for marker in NEXT_ACTION_MARKERS):
        flags.append("missing_recommended_next_action")

    if _looks_like_pure_validation(text):
        flags.append("pure_validation_no_redirect")

    if any(re.search(pattern, text) for pattern in DEPENDENCY_PATTERNS):
        flags.append("dependency_reinforcement")

    if any(re.search(pattern, text) for pattern in HARMFUL_AGREEMENT_PATTERNS):
        flags.append("harmful_or_sycophantic_agreement")

    if _self_degrading_compliance(text):
        flags.append("self_degrading_compliance")

    if _has_ungrounded_contact_claim(response, retrieved_sources):
        flags.append("ungrounded_contact_claim")

    if _has_unsupported_resource_recommendation(response, retrieved_sources):
        flags.append("unsupported_resource_recommendation")

    if flags:
        return OutputGuardResult(
            allowed=False,
            reason=";".join(flags),
            flags=flags,
            fallback_required=True,
            corrected_response=_fallback_response(safety_tier, route),
        )

    return OutputGuardResult(
        allowed=True,
        reason="passed_output_guard",
        flags=[],
        fallback_required=False,
    )


def _looks_like_pure_validation(text: str) -> bool:
    validation_words = ("that sounds", "makes sense", "i hear", "valid", "understandable", "painful")
    has_validation = any(word in text for word in validation_words)
    has_action = any(marker in text for marker in NEXT_ACTION_MARKERS)
    return has_validation and not has_action


def _self_degrading_compliance(text: str) -> bool:
    return bool(re.search(r"\byou are (worthless|a failure|pathetic|stupid|unlovable)\b", text))


def _has_ungrounded_contact_claim(response: str, retrieved_sources: list[dict]) -> bool:
    phone_like = re.findall(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b|\b\d{3}\b", response)
    if not phone_like:
        return False
    grounded_blob = " ".join(
        str(value)
        for source in retrieved_sources
        for value in source.values()
        if value is not None
    )
    for claim in phone_like:
        if claim == "988":
            continue
        if claim not in grounded_blob:
            return True
    return False


def _has_unsupported_resource_recommendation(response: str, retrieved_sources: list[dict]) -> bool:
    text = response.lower()
    if "will not invent" in text or "not invent" in text:
        return False
    if "source-grounded option:" not in text and "grounded support:" not in text:
        return False
    known_names = {
        str(source.get("source_name", "")).lower()
        for source in retrieved_sources
        if source.get("source_name")
    }
    known_titles = {
        str(source.get("title", "")).lower()
        for source in retrieved_sources
        if source.get("title")
    }
    known_blob = " ".join(known_names | known_titles)
    flagged_resources = ("campus pantry", "thrive", "mheart", "help center", "care to stop violence")
    return any(resource in text and resource not in known_blob for resource in flagged_resources)


def _fallback_response(safety_tier: str, route: str) -> str:
    if safety_tier == "imminent_safety":
        return (
            "I am concerned this may be an immediate safety moment. Please call or text 988 now, "
            "or call emergency services if there is immediate danger. If someone nearby can stay with you, "
            "move near them while you get help."
        )
    return (
        "That sounds difficult, and I do not want to turn it into empty reassurance. "
        "Recommended next action: make this smaller and use one relevant campus or human support option now. "
        "If safety becomes urgent, use 988 or emergency services instead of continuing here."
    )
