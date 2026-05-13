"""Constrained response planning for EmpathRAG Core.

Three layers compose the chat response:

1. Per-route content (validation / reframe / recommended action / follow-up).
2. Conversation stage (LISTEN / PERMISSION / OFFER) — controls *how much* of the
   plan surfaces and how directive the wording is.
3. Cross-cutting concern flags (international_concern) — append context-aware
   paragraphs without changing the route.

Stages:

* ``LISTEN``     - turn 1, listen-eligible route, no explicit ask.
                   Acknowledge with reflection in the user's frame, plus a soft
                   options hint so they know what's available without being
                   pushed.
* ``PERMISSION`` - turn 2, still no explicit ask. Different opener than
                   listen (no repetition), names 1-2 specific resources
                   gently, asks permission to share more.
* ``OFFER``      - turn 3+, explicit ask, always-direct route, or imminent
                   safety. Full plan with named resources and follow-up.

Always-direct routes skip LISTEN entirely so users with explicit asks get help
immediately. Crisis bypasses everything via render_crisis_response().
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Tuple

from .v2_schema import SupportRoute


LISTEN = "listen"
PERMISSION = "permission"
OFFER = "offer"
# CLARIFY is set by the planner for minimal-affirm / incomplete / truncated
# messages — short open-ended responses that intentionally skip the
# OFFER-stage output guard. Exported here as a string constant so callers
# don't reach for stringly-typed values.
CLARIFY = "clarify"


# Routes that always go straight to OFFER. The prompt itself is an explicit
# practical ask (ADS, counseling navigation, basic needs) or the safety/scope
# contract requires immediate, direct guidance.
ALWAYS_DIRECT_ROUTES = frozenset({
    SupportRoute.ACCESSIBILITY_ADS.value,
    SupportRoute.COUNSELING_NAVIGATION.value,
    SupportRoute.BASIC_NEEDS.value,
    SupportRoute.ADVISOR_CONFLICT.value,
    SupportRoute.PEER_HELPER.value,
    SupportRoute.CRISIS_IMMEDIATE.value,
    SupportRoute.OUT_OF_SCOPE.value,
    SupportRoute.CARE_VIOLENCE_CONFIDENTIAL.value,
    SupportRoute.AUTHORITY_MISCONDUCT.value,
})

EXPLICIT_ASK_PATTERNS = (
    # Direct asks
    "what should i", "what do i do", "how do i", "where do i",
    "what can i do", "how can i", "who do i", "where can i",
    "any advice", "any tips", "can you help", "help me",
    "options", "resources", "what are the", "tell me how",
    "should i", "i need to", "i want to", "i'm ready",
    "next step", "what now",
    # Yes/no factual questions ("is it legal", "will I get deported", "can I work")
    "is it ", "is this ", "is that ", "is there ",
    "will i ", "will it ", "will they ", "will that ",
    "can i ", "could i ", "could it ", "may i ",
    "do i need", "do they ", "does it ", "does this ",
    "am i allowed", "am i able",
    # Factual question stems often used by international students
    "is it legal", "is it allowed", "is it ok", "is it fine", "is it safe",
    "is it true", "is it possible", "is it fair",
    "what are the rules", "what's the rule", "what's the policy",
)


# Sub-topic detection for international/F-1 students.  When we hit any of these
# the OFFER stage uses a topic-specific factual template instead of the generic
# academic-stress one - so a question like "can I work on campus after
# graduation?" gets a real factual answer routed through ISSS instead of a
# generic "that sounds heavy" reflection.
INTL_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "work_authorization": (
        "work on campus", "work off campus", "on-campus work", "off-campus work",
        "after graduation", "post-graduation", "post graduation",
        "opt", "cpt", "stem extension", "stem opt",
        "employment authorization", "work authorization",
        "ead card", "i-765", "h-1b", "h1b", "h1-b",
        "internship", "job offer", "paid work", "legally work",
    ),
    "academic_standing": (
        "academic standing", "academic probation",
        "fail my class", "fail my course", "fail a class", "fail a course",
        "fail one of", "failing my class", "failing my course",
        "failing a class", "failing one of",
        "withdraw", "withdrawal", "drop a class", "drop the class",
        "drop a course", "course load", "credit load",
        "full-time enrollment", "minimum credits", "full time",
        "f-1 status", "f1 status", "affect my status", "affect my f-1",
        "affect my visa", "affect my i-20",
    ),
    "visa_status": (
        "visa renewal", "visa expir", "visa stamp", "visa appointment",
        "consulate", "embassy", "visa interview",
        "i-20 expir", "extension of stay", "change of status",
    ),
    "deportation_fear": (
        "deport", "sent home", "lose my status", "lose status",
        "out of status", "fall out of status",
        "back to my country", "go back home", "have to leave",
    ),
}


def classify_intl_topic(message: str) -> str:
    """Return the most specific international sub-topic, or '' if none."""
    text = (message or "").lower()
    # Order matters: more specific topics win. Work authorization first because
    # it's the most common and most factual.
    for topic in ("work_authorization", "visa_status", "academic_standing", "deportation_fear"):
        for kw in INTL_TOPIC_KEYWORDS[topic]:
            if kw in text:
                return topic
    return ""

# Cues that the student is talking about visa / international-status pressure.
# F-1 students carry a different weight on academic outcomes (status tied to
# enrollment + grades + SEVIS reporting). Catching these signals lets us
# acknowledge the layered fear and surface ISSS, which handles things regular
# counseling can't.
INTERNATIONAL_PATTERNS = (
    "deport", "deportation", "f-1", "f1 ", " f1", " f1.",
    "visa", "i-20", "i20", "sevis", "sevp",
    "international student", "international students",
    "opt ", " opt.", "cpt ", " cpt.",
    "academic standing",
    "back to my country", "sent home", "lose my status",
)

# Reusable ISSS source dict that we inject into retrieved sources whenever we
# detect an international concern, so the Support card actually shows ISSS
# alongside whatever the route normally surfaces.
INTERNATIONAL_SOURCE_HINT = {
    "service_id": "umd_isss",
    "source_id": "umd_isss",
    "source_name": "UMD International Student & Scholar Services (ISSS)",
    "title": "UMD International Student & Scholar Services (ISSS)",
    "topic": "international_student_support",
    "risk_level": "safe",
    "usage_mode": "retrieval",
    "source_type": "university_resource",
    "url": "https://isss.umd.edu/",
    "why_retrieved": "international_concern_detected",
}


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
    listen_reflection: str = ""
    listen_invite: str = ""
    permission_opener: str = ""
    international_concern: bool = False
    retrieved_sources: Tuple[dict, ...] = field(default_factory=tuple)

    def render(self, stage: str = OFFER) -> str:
        if self.safety_tier == "imminent_safety":
            stage = OFFER
        if stage == LISTEN:
            return self._render_listen()
        if stage == PERMISSION:
            return self._render_permission()
        return self._render_offer()

    # ------------------------------------------------------------------
    def _resource_names(self, n: int = 2) -> list[str]:
        names: list[str] = []
        for src in self.retrieved_sources[:8]:
            nm = (src.get("source_name") or src.get("title") or "").strip()
            if nm and nm not in names:
                names.append(nm)
            if len(names) >= n:
                break
        return names

    def _phrase_resources(self, names: list[str]) -> str:
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return f"{names[0]}, {names[1]}, and a couple of others"

    # ------------------------------------------------------------------
    # Stage renderers
    # ------------------------------------------------------------------
    def _render_listen(self) -> str:
        opener = self.listen_reflection or f"{self.validation} {self.reframe}".strip()

        intl_ack = ""
        if self.international_concern:
            intl_ack = (
                "And I noticed visa or status worry is in this too. That's a different "
                "layer of fear, not just \"academic\" stress. It changes who's actually "
                "the right person to talk to."
            )

        invite = self.listen_invite or "I'm here. Want to tell me more about what's coming up for you?"
        options = (
            "If it would help, I can keep listening, sit with what's underneath, "
            "or point you toward a place to start when you're ready. "
            "Whichever you want."
        )

        parts = [opener]
        if intl_ack:
            parts.append(intl_ack)
        parts.append(invite)
        parts.append(options)
        return "\n\n".join(parts)

    def _render_permission(self) -> str:
        # Use a different opener than listen so back-to-back turns don't repeat.
        opener = self.permission_opener or "Thanks for staying with this."

        names = self._resource_names(2)
        phrase = self._phrase_resources(names)
        if phrase:
            hint = (
                f"When you want to widen this out: {phrase} are usually the most useful "
                "starting points for what you've described — or we can keep talking, "
                "no pressure either way."
            )
        else:
            hint = (
                "When you want to widen this out, I can point you toward a couple of "
                "specific places that fit — or we can keep talking, no pressure either way."
            )

        intl_line = ""
        if self.international_concern:
            intl_line = (
                "If the visa or academic-standing piece is part of why this feels so big, "
                "UMD's International Student & Scholar Services (ISSS) has advisors who "
                "specifically handle that. That's a different conversation from regular "
                "counseling, and it's a good one to have early."
            )

        parts = [opener]
        if intl_line:
            parts.append(intl_line)
        parts.append(hint)
        return "\n\n".join(parts)

    def _render_offer(self) -> str:
        parts = [f"{self.validation} {self.reframe}".strip()]
        if self.recommended_action:
            parts.append(self.recommended_action)

        # Soft, named resource mention. Only for routes where it adds value.
        # The Support card UI on the right surfaces verified links + last-
        # verified dates already, so the chat bubble doesn't need to point
        # at the panel — that text used to leak into the LLM rephrase and
        # read as a UI instruction inside what should be conversational
        # prose.
        if self.route not in ("crisis_immediate", "out_of_scope", "peer_helper"):
            names = self._resource_names(2)
            phrase = self._phrase_resources(names)
            if phrase:
                parts.append(f"Specific places worth knowing: {phrase}.")

        if self.international_concern and self.route not in ("crisis_immediate", "out_of_scope"):
            parts.append(
                "Because visa or international-status worry is part of this: UMD's "
                "International Student & Scholar Services (ISSS) has academic-difficulty "
                "advising specifically for F-1 students. They can clarify what your "
                "academic standing actually means for SEVIS and your status. That's a "
                "different conversation from regular counseling, and worth having early "
                "rather than late."
            )

        # Essential safety / scope content for specific routes
        if self.route in ("peer_helper", "out_of_scope") and self.support_option:
            parts.append(self.support_option)
        if self.route == "peer_helper" and self.backup_option:
            parts.append(self.backup_option)
        if self.follow_up_question:
            parts.append(self.follow_up_question)
        return "\n\n".join(p for p in parts if p)


def has_international_concern(message: str) -> bool:
    text = (message or "").lower()
    return any(p in text for p in INTERNATIONAL_PATTERNS)


def decide_stage(message: str, route: str, safety_tier: str, turn_index: int) -> str:
    """Decide which conversation stage applies for this turn."""
    if safety_tier == "imminent_safety":
        return OFFER
    if route in ALWAYS_DIRECT_ROUTES:
        return OFFER
    if _has_explicit_ask(message):
        return OFFER
    # If the user asked a factual question (question mark) about a specific
    # F-1 / international sub-topic, that deserves a real answer. Not a
    # generic LISTEN reflection. Catches phrasings like "Will get deported?"
    # or "Is this OK?" that don't match the regular ask patterns.
    if "?" in (message or "") and classify_intl_topic(message):
        return OFFER
    if turn_index >= 3:
        return OFFER
    if turn_index == 2:
        return PERMISSION
    return LISTEN


def _has_explicit_ask(message: str) -> bool:
    text = (message or "").lower()
    return any(pattern in text for pattern in EXPLICIT_ASK_PATTERNS)


def build_response_plan(
    message: str,
    route: str,
    safety_tier: str,
    retrieved_sources: list[dict],
    audience_mode: str = "student",
    international_concern_override: bool | None = None,
) -> ResponsePlan:
    source_label = _source_label(retrieved_sources)
    if international_concern_override is None:
        international_concern = has_international_concern(message)
    else:
        international_concern = bool(international_concern_override)

    # When international concern is detected, ensure ISSS is in the retrieved
    # source list so it shows up on the Support card alongside whatever the
    # router otherwise surfaces.
    enriched_sources: list[dict] = list(retrieved_sources or [])
    if international_concern:
        already_has_isss = any(
            (s.get("source_name") or "").lower().startswith("umd international")
            or (s.get("service_id") or "").startswith("umd_isss")
            for s in enriched_sources
        )
        if not already_has_isss:
            enriched_sources = [INTERNATIONAL_SOURCE_HINT] + enriched_sources

    sources_tuple = tuple(enriched_sources)
    common = {
        "international_concern": international_concern,
        "retrieved_sources": sources_tuple,
    }

    if route == SupportRoute.ACADEMIC_SETBACK.value:
        return ResponsePlan(
            route,
            safety_tier,
            "That sounds like a really painful moment.",
            "Failing something you cared about can land hard, especially when the brain immediately starts treating it as proof of something bigger.",
            "When you're ready, a small grounded step is a short office-hours note to your professor or TA. Just to understand what went wrong and what to do differently next time.",
            f"Use {source_label} if the stress is affecting sleep, panic, or your ability to function.",
            "If the situation starts feeling unsafe, switch from academic planning to crisis or human support immediately.",
            "Do you want me to share an email script you could adapt, or would you rather just talk it through more first?",
            listen_reflection=(
                "That sounds really heavy. Failing something you cared about can land "
                "hard. And the part of your mind that's catastrophizing right now isn't "
                "lying, it's just turned up too loud."
            ),
            listen_invite="Want to tell me more about what it's been bringing up for you?",
            permission_opener=(
                "I hear you. The piece about everything stacking at once is real. When "
                "one thing tips, the rest start feeling fragile too."
            ),
            **common,
        )

    if route == SupportRoute.EXAM_STRESS.value:
        return ResponsePlan(
            route,
            safety_tier,
            "That kind of pre-test heaviness makes a lot of sense.",
            "When the test is this close, the mind tends to grab every worst-case scenario at once. That doesn't mean any of them are true, it just means the stakes feel real.",
            "If a small grounded step would help: a 10-minute reset, then picking two high-yield topics and a tiny study plan for tonight is usually more useful than trying to fix everything.",
            f"Use {source_label} if the stress is affecting sleep, panic, or your ability to function.",
            "If this shifts into not feeling safe or being unable to stay with yourself, use crisis or emergency support instead of continuing study planning.",
            "Would a quick study plan, a grounding reset, or drafting a message to your instructor help most right now?",
            listen_reflection=(
                "That kind of pre-test heaviness is real. When something feels this "
                "close and this big, the body tends to react like it's already happening."
            ),
            listen_invite="Want to tell me a bit more about what's been swirling?",
            permission_opener=(
                "Thanks for naming a bit more. The way the worry is spreading past the "
                "test itself makes sense, especially when the consequences feel like they "
                "go further than one grade."
            ),
            **common,
        )

    if route == SupportRoute.ACCESSIBILITY_ADS.value:
        return ResponsePlan(
            route,
            safety_tier,
            "That's a practical accommodations question, and you don't have to figure it out alone.",
            "Going through the official ADS process gets the request documented and traceable, which protects you later.",
            "A good first move: identify the specific course or exam barrier, then start with the ADS intake page.",
            f"Use {source_label} for the official accommodations workflow.",
            "If a deadline is urgent, also send a brief factual note to the instructor or program staff in parallel.",
            "Is this for an exam, an assignment deadline, a temporary condition, or an ongoing accommodation?",
            **common,
        )

    if route == SupportRoute.ADVISOR_CONFLICT.value:
        return ResponsePlan(
            route,
            safety_tier,
            "That sounds genuinely stressful, especially when funding or power is involved.",
            "Keeping the record factual and using a neutral channel before escalating tends to protect you the most.",
            "A grounded next move: write down a short factual timeline of what was said and when, then look at a neutral graduate-support or Ombuds path before doing anything irreversible.",
            f"Use {source_label} as the campus support starting point.",
            "If the stress becomes unsafe or overwhelming, use counseling or crisis support before continuing the conflict process.",
            "Would it help to draft the timeline together, or would you rather talk through what's actually been happening first?",
            **common,
        )

    if route == SupportRoute.AUTHORITY_MISCONDUCT.value:
        # The user is reporting an authority figure (counselor, professor,
        # advisor, RA, coach, etc.) did or said something problematic. The
        # planner deliberately does NOT validate the authority's standing or
        # treat the alleged conduct as ordinary advice. It reflects the
        # report, names the institutional reporting channel, and avoids
        # legal advice or judgment of the person.
        return ResponsePlan(
            route,
            safety_tier,
            "What you are describing isn't ordinary feedback. When someone in a position of authority over you says or does something that crosses a line, you deserve a separate channel to look at it.",
            "You don't have to decide right now whether to file anything. Naming what happened to the right office is itself a step, and offices that handle this are required to walk you through options before any action is taken.",
            "A grounded next move: contact UMD's Office of Civil Rights & Sexual Misconduct (OCRSM) or the Office of Student Conduct, depending on what occurred. Both will explain what reporting does and does not commit you to. The UMD Dean of Students Office is also a safe first contact if you're not sure which channel fits.",
            f"Use {source_label} to start with a confidential conversation before you decide anything.",
            "If you are in immediate danger or facing retaliation now, contact emergency services or use crisis support first; reporting can come after.",
            "Would it help to talk through what happened first, or would you rather skip straight to which office probably fits best?",
            **common,
        )

    if route == SupportRoute.BASIC_NEEDS.value:
        return ResponsePlan(
            route,
            safety_tier,
            "Food, housing, or money stress can make everything else harder very fast.",
            "This is a support-navigation problem, not a personal failure, and it's exactly the kind of thing campus offices are built for.",
            "A grounded next move: contact a campus student-support office and say plainly what you need help with today.",
            f"Use {source_label}; rely on the source card for verified contact, hours, and eligibility details.",
            "If your safety or shelter is immediately at risk, use emergency or crisis support instead of waiting.",
            "What's the most urgent piece right now. Food, housing, money, or finding the right campus contact?",
            **common,
        )

    if route == SupportRoute.PEER_HELPER.value:
        return ResponsePlan(
            route,
            safety_tier,
            "It makes sense that you're worried about your friend.",
            "You can support them, and you should not be the only safety plan.",
            "If you're with them now: ask directly whether they're safe, stay with them or keep them connected if possible, and involve emergency or crisis support if there may be immediate danger.",
            f"Use {source_label} for helping-someone-else or crisis guidance.",
            "What to say: I care about you, I'm worried, and I want to get another person involved so you're not alone. What not to say: do not promise secrecy or agree to handle safety risk by yourself.",
            "Are they reachable right now, and is someone physically nearby who can check on them?",
            **common,
        )

    if route == SupportRoute.ANXIETY_PANIC.value:
        if _is_social_or_date_nerves(message):
            return ResponsePlan(
                route,
                safety_tier,
                "That kind of nervousness before meeting someone you like is very normal.",
                "It does not mean something is wrong; it usually means the moment matters to you and your brain is trying to predict every possible outcome.",
                "If it would help: keep it simple. One easy opener, one genuine question you could ask, and one graceful exit if either of you feels awkward.",
                "You probably do not need a formal campus resource for ordinary date nerves; if anxiety starts disrupting sleep, eating, or daily functioning, the source cards can point you toward support.",
                "If this shifts into panic, feeling unsafe, or being unable to function, use a human support option rather than trying to push through alone.",
                "Want to brainstorm a relaxed opening line, a few questions to ask, or how to calm down beforehand?",
                listen_reflection=(
                    "That kind of nervousness before meeting someone you like is very "
                    "normal. It usually means the moment matters to you and your brain "
                    "is trying to predict every possible outcome."
                ),
                listen_invite="Want to tell me a bit more about what you're hoping for or worried about?",
                permission_opener=(
                    "I hear you. Pre-date jitters can balloon when you've been thinking "
                    "about it for a while. They don't always shrink with more thinking."
                ),
                **common,
            )
        return ResponsePlan(
            route,
            safety_tier,
            "That sounds like anxiety is taking up a lot of space right now.",
            "Anxiety like this can feel huge in the body even when nothing visible has changed yet. That's exhausting to ride out alone.",
            "If a grounded step would help: one short grounding reset, and then we can decide whether finding a campus support path is the next move.",
            f"Use {source_label} for anxiety, grounding, or counseling support.",
            "If the anxiety shifts into not feeling safe, use crisis support instead of continuing here.",
            "Would you rather try a grounding reset together, or look at who you could talk to?",
            listen_reflection=(
                "That sounds like anxiety is taking up a lot of room right now. "
                "Even when nothing visible has changed, the body can feel like it's "
                "in the middle of something. And that's exhausting."
            ),
            listen_invite="Want to tell me a bit more about what's been going on?",
            permission_opener=(
                "Thanks for sharing more. Anxiety this loud is a real thing to be sitting "
                "with. It's not a small ask of yourself."
            ),
            **common,
        )

    if route == SupportRoute.LOW_MOOD.value:
        return ResponsePlan(
            route,
            safety_tier,
            "That sounds heavy.",
            "Low mood has a way of making everything quieter and farther away. Including the parts of life that usually help.",
            "If a small step would feel doable: telling one trusted person what's been going on, and looking at a counseling starting point. We don't have to do both today.",
            f"Use {source_label} as a grounded starting point.",
            "If this turns into not feeling safe, use 988 or emergency support immediately.",
            "What's been the hardest part lately. Motivation, isolation, sleep, or asking for help?",
            listen_reflection=(
                "That sounds heavy. Low mood has a way of pulling everything that "
                "usually helps a little farther out of reach."
            ),
            listen_invite="Take your time. What's been on your mind most?",
            permission_opener=(
                "I hear you. The way it's been pulling everything farther away. "
                "that's a real thing, not weakness."
            ),
            **common,
        )

    if route == SupportRoute.LONELINESS_ISOLATION.value:
        return ResponsePlan(
            route,
            safety_tier,
            "That sounds really lonely.",
            "Isolation can quietly become its own thing. It's not weakness, and it doesn't mean you haven't tried.",
            "If a small step would help: reaching out to one person you used to feel close to, even with a tiny message, is often more useful than trying to rebuild a whole social life at once.",
            f"Use {source_label} for connection and support.",
            "If isolation starts feeling unsafe or hopeless, use a counseling or crisis support path.",
            "Has there been one person you keep almost reaching out to but haven't?",
            listen_reflection=(
                "That sounds really lonely. Isolation has a way of growing in the "
                "background until everything starts feeling far away."
            ),
            listen_invite="Want to tell me a bit about what it's been like?",
            permission_opener=(
                "Thanks for staying with this. Loneliness is one of those feelings that "
                "gets quieter the longer it sits, and that makes it harder to name."
            ),
            **common,
        )

    if route == SupportRoute.COUNSELING_NAVIGATION.value:
        return ResponsePlan(
            route,
            safety_tier,
            "It is completely reasonable to want a clear first step.",
            "Counseling navigation works best when it's practical and source-grounded, not vague encouragement.",
            "A solid first move: start with the official counseling source and look for intake, scheduling, or referral instructions.",
            f"Use {source_label} for the current official path.",
            "If the need becomes urgent or safety-related, use crisis resources instead of ordinary scheduling.",
            "Are you trying to start care, understand confidentiality, or find referral options?",
            **common,
        )

    if route == SupportRoute.OUT_OF_SCOPE.value:
        return ResponsePlan(
            route,
            safety_tier,
            "That doesn't look like a student-support navigation request.",
            "I should avoid pretending this system is a general authority outside its scope.",
            "Use an appropriate official or professional source for that topic.",
            "No support source is retrieved for this route because the request is outside the system scope.",
            "For medical, legal, or emergency issues, use qualified human support.",
            "Do you want to rephrase this as a campus-support question?",
            **common,
        )

    # Generic / general_student_support fallback - listen-eligible.
    # OFFER `validation` deliberately differs from `listen_reflection` so a
    # session that walks LISTEN -> PERMISSION -> OFFER doesn't open all
    # three turns with "That sounds like a lot to carry."
    return ResponsePlan(
        route,
        safety_tier,
        "Here's where I'd actually start.",
        "You shouldn't have to sort through this from scratch on your own.",
        "When you're ready, a useful first move is naming the kind of support that fits. Whether that's someone to talk to, a campus office, or a small concrete next step.",
        f"Use {source_label} as the starting point.",
        "If this becomes urgent or safety-related, switch to crisis or emergency support.",
        "What would help most first — a next-step checklist, who to contact, or just talking it through more?",
        listen_reflection=(
            "That sounds like a lot to carry. Whatever's underneath this, you don't "
            "have to have it figured out right now."
        ),
        listen_invite="What's been weighing on you most lately?",
        permission_opener=(
            "I hear you. The way it's all collapsing at once is exhausting. That "
            "kind of overload usually has more than one thing pulling on it."
        ),
        **common,
    )


def render_intl_factual_offer(topic: str, message: str = "") -> str:
    """Topic-specific factual orientation for F-1 / international students.

    These responses *engage with what was actually asked* (work rules, visa
    timing, academic standing, deportation fear) instead of giving the generic
    academic-stress reflection. The format is:

    1. Acknowledge the actual question (not the emotion).
    2. General factual orientation (always with the ISSS-is-authoritative
       disclaimer).
    3. The specific next move with ISSS by name.
    4. One clarifying question to understand their case.
    """
    if topic == "work_authorization":
        return (
            "That fear isn't unfounded. But the rule is more specific than "
            "\"you'll get deported.\" On-campus work doesn't continue after "
            "graduation; you'd need OPT (Optional Practical Training) authorized "
            "first, or you'd fall out of status. Good news: it's avoidable with "
            "the right timing.\n\n"
            "Talk to UMD ISSS now, before graduation rather than after. They'll tell "
            "you what's authorized for your case. What's your graduation timing?"
        )
    if topic == "visa_status":
        return (
            "Visa questions are case-specific, so I'll keep this general and "
            "point you to ISSS for the real answer. F-1 needs three things "
            "aligned: an unexpired I-20, valid status (full-time enrollment + "
            "SEVIS), and (only for re-entering the US) an unexpired visa "
            "stamp. The stamp can expire while you're still in valid status; "
            "that's fine as long as you don't leave.\n\n"
            "Schedule time with UMD ISSS to review your I-20 and any travel "
            "plans. Is this about the visa stamp, the I-20, a travel plan, or "
            "something else?"
        )
    if topic == "academic_standing":
        return (
            "Academic standing as F-1 isn't only academic. Your enrollment is "
            "what keeps your status valid. But: failing one course doesn't "
            "auto-end status. What can end it is dropping below full-time "
            "enrollment without ISSS-authorized RCL (reduced course load). The "
            "key phrase is *prior authorization*.\n\n"
            "Contact ISSS *before* any drop or withdrawal posts. They can "
            "usually authorize an RCL for medical, academic, or final-term "
            "reasons. Is this about a failed grade, a class you're thinking of "
            "dropping, or your overall load?"
        )
    if topic == "deportation_fear":
        return (
            "That fear is real, but the actual mechanism is more specific than "
            "\"one bad grade and you're sent home.\" Deportation as F-1 usually "
            "follows from *falling out of status*. Unauthorized work, dropping "
            "below full-time without RCL, or an expired I-20. A single failed "
            "course typically doesn't end status, and there's a grace period "
            "plus reinstatement path.\n\n"
            "Talk to ISSS now about what specifically worries you. They can "
            "almost always tell you whether the worst case is real for your "
            "situation. What's the specific worry. A grade, course load, work, "
            "or an I-20 date?"
        )
    return ""


# Regex patterns for active interpersonal danger (intimate-partner violence,
# family abuse, stalking, assault threat). Distinct from self-harm ideation;
# right redirect is 911 + safe location + UMD CARE, not 988. Use word
# boundaries so "hits me" and "hit me" both match.
_INTERPERSONAL_DANGER_REGEX = re.compile(
    r"\b(hitting|hits|hit) me\b|"
    r"\b(beats|beating|beat) me\b|"
    r"\bscared (?:to go home|of him|of her|of them)\b|"
    r"\bafraid (?:of him|of her|of them)\b|"
    r"\babusive (?:partner|relationship|boyfriend|girlfriend|husband|wife|parent|dad|mom)\b|"
    r"\b(?:my )?(?:dad|mom|father|mother|brother|sister) (?:hits|beats|hurts|abuses) me\b|"
    r"\b(?:domestic violence|abuse|abusing me)\b|"
    r"\b(stalker|stalking me)\b|"
    r"\bwon'?t leave me alone\b|"
    r"\b(?:threatening|threatened) (?:to kill|me)\b|"
    r"\bnot safe (?:at home|in my)\b|"
    r"\b(?:raped|assaulted|forced) me\b|"
    r"\b(?:he|she)(?:'s)? coming (?:back|after me)\b",
    re.IGNORECASE,
)


def _is_interpersonal_danger(message: str) -> bool:
    return bool(_INTERPERSONAL_DANGER_REGEX.search(message or ""))


def render_crisis_response(
    route: str,
    audience_mode: str = "student",
    user_message: str = "",
) -> str:
    # Peer-helper crisis: user is worried about someone else.
    if route == SupportRoute.PEER_HELPER.value or audience_mode == "helping_friend":
        return (
            "I'm really glad you told me about this. This sounds like an immediate "
            "safety situation for your friend, and you should not handle it alone. "
            "Please contact emergency or crisis support now, and involve a trusted "
            "nearby person, an RA, a supervisor, or campus support while you try "
            "to reach them. What you can say if you do reach them: I care about you, "
            "I'm worried, and I want to get another person involved so you're not alone."
        )

    # Active interpersonal danger (DV / stalking / assault threat).
    # 911 + safe location + UMD CARE, not 988.
    if _is_interpersonal_danger(user_message):
        return (
            "I'm really glad you told me, and I'm taking what you said seriously. "
            "If you're not safe right now or someone might be coming to you, call 911 "
            "or get to a public location with other people around. When you're "
            "somewhere safe, UMD CARE to Stop Violence is a confidential resource "
            "that can walk through next steps with you, including safety planning and "
            "what your options look like. No pressure to file or report anything right now."
        )

    # Default: self-harm ideation / imminent safety to self.
    return (
        "I'm really glad you told me. What you're describing sounds like a moment "
        "where having a real person with you matters. Please call or text 988 now, "
        "or call emergency services if there's immediate danger. If someone nearby "
        "can stay with you, move near them while you get help. UMD Counseling "
        "Center after-hours is at 301-314-HELP (4357) and is staffed for moments "
        "exactly like this. I'm here while you reach out."
    )


def _source_label(retrieved_sources: list[dict]) -> str:
    names = _source_names(retrieved_sources)
    return names or "the retrieved source cards"


def _is_social_or_date_nerves(message: str) -> bool:
    text = message.lower()
    return any(
        phrase in text
        for phrase in (
            "asked out",
            "first date",
            "meet a girl",
            "meet a guy",
            "meet someone",
            "going on a date",
            "date tomorrow",
            "nervous to meet",
            "romantic",
        )
    )


def _source_names(retrieved_sources: list[dict]) -> str:
    seen: list[str] = []
    for source in retrieved_sources:
        name = str(source.get("source_name") or source.get("title") or "").strip()
        if name and name not in seen:
            seen.append(name)
        if len(seen) == 2:
            break
    return " and ".join(seen)
