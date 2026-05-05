"""
Canonical V2.5 route and safety-tier schema.

This module is intentionally rule-based and dependency-free so it can be used by
the fast demo backend, tests, and the heavier pipeline without slowing startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from .safety_policy import SafetyLevel


class SafetyTier(str, Enum):
    IMMINENT_SAFETY = "imminent_safety"
    HIGH_DISTRESS = "high_distress"
    SUPPORT_NAVIGATION = "support_navigation"
    WELLBEING = "wellbeing"


class SupportRoute(str, Enum):
    ACADEMIC_SETBACK = "academic_setback"
    EXAM_STRESS = "exam_stress"
    ACCESSIBILITY_ADS = "accessibility_ads"
    ADVISOR_CONFLICT = "advisor_conflict"
    COUNSELING_NAVIGATION = "counseling_navigation"
    BASIC_NEEDS = "basic_needs"
    CARE_VIOLENCE_CONFIDENTIAL = "care_violence_confidential"
    PEER_HELPER = "peer_helper"
    LONELINESS_ISOLATION = "loneliness_isolation"
    ANXIETY_PANIC = "anxiety_panic"
    LOW_MOOD = "low_mood"
    CRISIS_IMMEDIATE = "crisis_immediate"
    GENERAL_STUDENT_SUPPORT = "general_student_support"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True)
class RouteDecision:
    route: SupportRoute
    tier: SafetyTier
    reason: str
    audience_mode: str = "student"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def map_safety_level(level: SafetyLevel | str, wellbeing_request: bool = False) -> SafetyTier:
    value = level.value if isinstance(level, SafetyLevel) else str(level)
    if value in {SafetyLevel.EMERGENCY.value, SafetyLevel.CRISIS.value, "emergency", "crisis"}:
        return SafetyTier.IMMINENT_SAFETY
    if value == SafetyLevel.WELLBEING_SUPPORT.value:
        return SafetyTier.HIGH_DISTRESS if not wellbeing_request else SafetyTier.WELLBEING
    return SafetyTier.WELLBEING if wellbeing_request else SafetyTier.SUPPORT_NAVIGATION


def classify_route(
    message: str,
    safety_tier: SafetyTier,
    audience_mode: str = "student",
) -> RouteDecision:
    text = normalize_text(message)

    if safety_tier == SafetyTier.IMMINENT_SAFETY:
        if _is_peer_helper(text, audience_mode):
            return RouteDecision(SupportRoute.PEER_HELPER, safety_tier, "peer_imminent_safety", audience_mode)
        return RouteDecision(SupportRoute.CRISIS_IMMEDIATE, safety_tier, "imminent_safety_tier", audience_mode)

    if _is_peer_helper(text, audience_mode):
        return RouteDecision(SupportRoute.PEER_HELPER, safety_tier, "peer_helper_language", audience_mode)

    if _has_any(text, ("prescribe", "diagnose", "medication dosage", "hipaa complaint", "legal complaint", "lawsuit", "litigation", "dining hall rumor", "rumor about mold")):
        return RouteDecision(SupportRoute.OUT_OF_SCOPE, safety_tier, "out_of_scope_language", audience_mode)

    if _has_any(text, ("counseling center", "referral", "off-campus care", "after hours", "after-hours", "brief assessment", "group therapy", "individual counseling", "mental health crises", "number to call")):
        return RouteDecision(SupportRoute.COUNSELING_NAVIGATION, safety_tier, "counseling_navigation_language", audience_mode)

    if _has_any(text, ("accommodation", "disability", "504", "extended time", "assistive tech", "paratransit")) or _has_word(text, "ads"):
        return RouteDecision(SupportRoute.ACCESSIBILITY_ADS, safety_tier, "accessibility_language", audience_mode)

    if _has_any(text, ("advisor", "ombuds", "funding threatened", "threatened my funding", "funding might disappear", "pi is", "my pi", "committee feedback", "retaliatory", "neutral process", "power dynamics")):
        return RouteDecision(SupportRoute.ADVISOR_CONFLICT, safety_tier, "advisor_or_ombuds_language", audience_mode)

    if _has_any(text, ("no food", "out of money", "hungry because", "food rent", "food or rent", "can't afford food", "cannot afford food", "nowhere to sleep")):
        return RouteDecision(SupportRoute.BASIC_NEEDS, safety_tier, "basic_needs_language", audience_mode)

    if _has_any(text, ("violence", "assault", "stalking", "abuse", "harassment", "dating violence")):
        return RouteDecision(SupportRoute.CARE_VIOLENCE_CONFIDENTIAL, safety_tier, "care_or_violence_language", audience_mode)

    if _has_any(text, ("failed", "fail", "failed my exam", "future is over", "grade", "grades")):
        return RouteDecision(SupportRoute.ACADEMIC_SETBACK, safety_tier, "academic_setback_language", audience_mode)

    if _has_any(text, ("exam", "midterm", "final", "qualifying exam", "study", "deadline", "comps prep", "labs plus ta", "syllabus")):
        return RouteDecision(SupportRoute.EXAM_STRESS, safety_tier, "exam_stress_language", audience_mode)

    if _has_any(text, ("counseling", "counselling", "therapy", "therapist", "appointment", "get started")):
        return RouteDecision(SupportRoute.COUNSELING_NAVIGATION, safety_tier, "counseling_navigation_language", audience_mode)

    if _has_any(text, ("lonely", "isolated", "no one cares", "no friends", "alone", "roommate moved out", "dorm feels hollow", "nobody texts", "burden people", "disappear socially")):
        return RouteDecision(SupportRoute.LONELINESS_ISOLATION, safety_tier, "loneliness_language", audience_mode)

    if _has_any(text, ("panic", "panicking", "anxiety", "anxious", "grounding", "breathing", "stomach is wrecked", "freeze in social", "intrusive thoughts", "heart rate", "drinking more", "mindfulness", "sensory overwhelm", "quick reset", "journaling", "worry loops")):
        return RouteDecision(SupportRoute.ANXIETY_PANIC, safety_tier, "anxiety_or_panic_language", audience_mode)

    if _has_any(text, ("depressed", "depressing", "depression", "low mood", "hopeless", "feel numb", "motivation disappeared", "canceling plans", "guilty", "dark moods", "pointless")):
        return RouteDecision(SupportRoute.LOW_MOOD, safety_tier, "low_mood_language", audience_mode)

    return RouteDecision(SupportRoute.GENERAL_STUDENT_SUPPORT, safety_tier, "default_support_navigation", audience_mode)


def action_ladder(tier: SafetyTier | str) -> dict[str, str]:
    value = tier.value if isinstance(tier, SafetyTier) else str(tier)
    return {
        SafetyTier.IMMINENT_SAFETY.value: {
            "mode": "imminent_safety",
            "generation": "blocked",
            "retrieval": "crisis_only",
            "goal": "human handoff now",
        },
        SafetyTier.HIGH_DISTRESS.value: {
            "mode": "high_distress",
            "generation": "template_or_guarded",
            "retrieval": "retrieval + wellbeing_only",
            "goal": "stabilize and connect to support",
        },
        SafetyTier.SUPPORT_NAVIGATION.value: {
            "mode": "support_navigation",
            "generation": "route_template",
            "retrieval": "retrieval",
            "goal": "practical next step",
        },
        SafetyTier.WELLBEING.value: {
            "mode": "wellbeing",
            "generation": "brief coping support",
            "retrieval": "retrieval + wellbeing_only",
            "goal": "low-risk support and campus option",
        },
    }.get(value, {"mode": value, "generation": "guarded", "retrieval": "retrieval", "goal": "support navigation"})


def _is_peer_helper(text: str, audience_mode: str) -> bool:
    if audience_mode == "helping_friend":
        return True
    return _has_any(text, ("my friend", "my roommate", "my labmate", "my teammate", "someone i know", "they said goodbye", "not tell anyone"))


def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _has_word(text: str, word: str) -> bool:
    return bool(re.search(rf"\b{re.escape(word)}\b", text))
