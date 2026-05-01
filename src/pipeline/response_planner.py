"""Constrained response planning for EmpathRAG Core."""

from __future__ import annotations

from dataclasses import dataclass

from .v2_schema import SupportRoute


@dataclass(frozen=True)
class ResponsePlan:
    route: str
    safety_tier: str
    validation: str
    reframe: str
    recommended_action: str
    support_option: str
    backup_option: str
    follow_up_question: str

    def render(self) -> str:
        return (
            f"{self.validation} {self.reframe}\n\n"
            f"Recommended next action: {self.recommended_action}\n\n"
            f"Source-grounded option: {self.support_option}\n\n"
            f"Backup option: {self.backup_option}\n\n"
            f"{self.follow_up_question}"
        )


def build_response_plan(
    message: str,
    route: str,
    safety_tier: str,
    retrieved_sources: list[dict],
    audience_mode: str = "student",
) -> ResponsePlan:
    source_label = _source_label(retrieved_sources)
    source_names = _source_names(retrieved_sources)

    if route == SupportRoute.ACADEMIC_SETBACK.value:
        return ResponsePlan(
            route,
            safety_tier,
            "That sounds like a painful academic setback.",
            "One exam or assignment can feel huge, but the safer move is to make the next step concrete instead of treating it as your whole future.",
            "Send a short professor/TA office-hours note asking what went wrong and what to change before the next assessment.",
            f"Use {source_label} if the stress is affecting sleep, panic, or your ability to function.",
            "If the situation starts feeling unsafe, switch from academic planning to crisis or human support immediately.",
            "Do you want the short email script or the next-step checklist first?",
        )

    if route == SupportRoute.ACCESSIBILITY_ADS.value:
        return ResponsePlan(
            route,
            safety_tier,
            "That is a practical accommodations question, not something you need to improvise alone.",
            "The strongest path is to use the official ADS process so the request is documented and traceable.",
            "Identify the course/exam barrier and start from the ADS source shown here.",
            f"Use {source_label} for the official accommodations workflow.",
            "If a deadline is urgent, also contact the instructor or program staff with a brief factual note.",
            "Is this for an exam, assignment deadline, temporary condition, or ongoing accommodation?",
        )

    if route == SupportRoute.ADVISOR_CONFLICT.value:
        return ResponsePlan(
            route,
            safety_tier,
            "That sounds stressful, especially when power and funding are involved.",
            "The safer path is to separate facts, deadlines, and relationship concerns before escalating.",
            "Write a factual timeline and use a neutral graduate support or Ombuds route before making irreversible decisions.",
            f"Use {source_label} as the campus support starting point.",
            "If the stress becomes unsafe or overwhelming, use counseling or crisis support before continuing the conflict process.",
            "Do you need help turning this into a neutral message or a timeline?",
        )

    if route == SupportRoute.BASIC_NEEDS.value:
        return ResponsePlan(
            route,
            safety_tier,
            "Food, housing, or money stress can make everything else harder very quickly.",
            "This is a support-navigation problem, not a personal failure.",
            "Contact a campus student-support office and state the concrete need for today.",
            f"Use {source_label}; I will not invent Pantry, Thrive, hours, or eligibility details unless they are in the verified source metadata.",
            "If your safety or shelter is immediately at risk, use emergency or crisis support instead of waiting.",
            "Is the most urgent need food, housing, money, or a campus contact?",
        )

    if route == SupportRoute.PEER_HELPER.value:
        return ResponsePlan(
            route,
            safety_tier,
            "It makes sense that you are worried about your friend.",
            "You can support them, but you should not be the only safety plan.",
            "Ask directly whether they are safe right now, stay with them or keep them connected if possible, and involve emergency/crisis support if there may be immediate danger.",
            f"Use {source_label} for helping-someone-else or crisis guidance.",
            "What to say: I care about you, I am worried, and I want to get another person involved so you are not alone. What not to say: do not promise secrecy or agree to handle safety risk by yourself.",
            "Are they reachable right now, and is someone physically nearby who can check on them?",
        )

    if route == SupportRoute.ANXIETY_PANIC.value:
        return ResponsePlan(
            route,
            safety_tier,
            "That sounds like anxiety is taking up a lot of space right now.",
            "The goal is to lower the intensity first, then decide whether you need a campus support path.",
            "Do one short grounding step, then pick one follow-up action if the anxiety keeps interfering.",
            f"Use {source_label} for anxiety, grounding, or counseling support.",
            "If the anxiety shifts into not feeling safe, use crisis support instead of continuing here.",
            "Would you rather start with grounding or with finding who to contact?",
        )

    if route == SupportRoute.LOW_MOOD.value:
        return ResponsePlan(
            route,
            safety_tier,
            "That sounds heavy, and it deserves support instead of being minimized.",
            "Low mood can make isolation feel logical, but that does not mean handling it alone is the safest path.",
            "Tell one trusted person what is going on and use a counseling or support starting point.",
            f"Use {source_label} as a grounded starting point.",
            "If this turns into not feeling safe, use 988 or emergency support immediately.",
            "Is the hardest part right now motivation, isolation, sleep, or asking for help?",
        )

    if route == SupportRoute.COUNSELING_NAVIGATION.value:
        return ResponsePlan(
            route,
            safety_tier,
            "It is completely reasonable to want a clear first step.",
            "Counseling navigation should be practical and source-grounded, not vague encouragement.",
            "Start with the official counseling source and look for intake, scheduling, or referral instructions.",
            f"Use {source_label} for the current official path.",
            "If the need becomes urgent or safety-related, use crisis resources instead of ordinary scheduling.",
            "Are you trying to start care, understand confidentiality, or find referral options?",
        )

    if route == SupportRoute.OUT_OF_SCOPE.value:
        return ResponsePlan(
            route,
            safety_tier,
            "That does not look like a student-support navigation request.",
            "I should avoid pretending this system is a general authority outside its scope.",
            "Use an appropriate official or professional source for that topic.",
            "No support source is retrieved for this route because the request is outside the system scope.",
            "For medical, legal, or emergency issues, use qualified human support.",
            "Do you want to rephrase this as a campus-support question?",
        )

    return ResponsePlan(
        route,
        safety_tier,
        "That sounds like a real support concern.",
        "The safest way to help is to turn it into one grounded next step instead of a broad reassurance loop.",
        "Pick one concrete support path from the retrieved sources.",
        f"Use {source_label} as the starting point.",
        "If this becomes urgent or safety-related, switch to crisis or emergency support.",
        "What would help most first: next steps, who to contact, or what to expect?",
    )


def render_crisis_response(route: str, audience_mode: str = "student") -> str:
    if route == SupportRoute.PEER_HELPER.value or audience_mode == "helping_friend":
        return (
            "I am concerned this could be an immediate safety situation for your friend. "
            "Do not handle this alone. Contact emergency/crisis support now, and involve a trusted nearby person, RA, supervisor, or campus support while you try to reach them."
        )
    return (
        "I am concerned this may be an immediate safety moment. Please call or text 988 now, "
        "or call emergency services if there is immediate danger. If someone nearby can stay with you, move near them while you get help."
    )


def _source_label(retrieved_sources: list[dict]) -> str:
    names = _source_names(retrieved_sources)
    return names or "the retrieved source cards"


def _source_names(retrieved_sources: list[dict]) -> str:
    seen = []
    for source in retrieved_sources:
        name = str(source.get("source_name") or source.get("title") or "").strip()
        if name and name not in seen:
            seen.append(name)
        if len(seen) == 2:
            break
    return " and ".join(seen)
