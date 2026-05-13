"""
Safety triage policy for EmpathRAG v2.

The learned NLI guardrail is useful, but its raw probability is not a complete
clinical or operational decision. This layer turns model confidence plus simple
textual risk signals into a conservative triage level that the pipeline can act
on consistently.
"""

from dataclasses import dataclass
from enum import Enum
import re


class SafetyLevel(str, Enum):
    PASS = "pass"
    WELLBEING_SUPPORT = "wellbeing_support"
    CRISIS = "crisis"
    EMERGENCY = "emergency"


@dataclass(frozen=True)
class SafetyDecision:
    level: SafetyLevel
    confidence: float
    reason: str
    should_intercept: bool
    response: str | None = None


CRISIS_RESPONSE = (
    "I can hear that you're going through something really difficult. "
    "You're not alone in this. If you might hurt yourself or feel unable "
    "to stay safe, please call or text 988 now, or contact emergency services. "
    "If you are a student, campus counseling or an after-hours crisis line can "
    "also help you get support right away."
)

EMERGENCY_RESPONSE = (
    "I'm really concerned about your immediate safety. If you have already "
    "taken something, have a plan, or might hurt yourself soon, please call "
    "emergency services now or call/text 988. If someone is nearby, ask them "
    "to stay with you while you get help."
)


class SafetyTriagePolicy:
    """
    Post-processes raw guardrail confidence into an action level.

    This is intentionally simple and auditable. It is not a replacement for a
    trained crisis model or clinician-reviewed protocol; it prevents the demo
    from treating every high-confidence academic/help-seeking phrase as the same
    emergency.
    """

    def __init__(
        self,
        support_threshold: float = 0.50,
        crisis_threshold: float = 0.85,
        emergency_threshold: float = 0.90,
    ):
        self.support_threshold = support_threshold
        self.crisis_threshold = crisis_threshold
        self.emergency_threshold = emergency_threshold

    def classify(self, text: str, confidence: float, model_flag: bool) -> SafetyDecision:
        # Two-pass match: first try the raw normalized text, then a typo-
        # corrected version for crisis / imminent patterns only. Students
        # in distress drop letters, miss apostrophes, and contract things
        # in ways that brittle regex would miss. The cost of a false
        # positive on safety is a 988 card on a benign message; the cost
        # of a false negative is missing crisis language. We err toward
        # catching by trying both passes for the high-stakes patterns.
        # Ambiguous-metaphor / academic-idiom checks stay on raw text so
        # the typo-correction layer can't accidentally over-flag idioms.
        normalized = _normalize(text)
        typo_corrected = _typo_correct_for_safety(normalized)
        explicit = (
            _matches_any(normalized, EXPLICIT_CRISIS_PATTERNS)
            or (typo_corrected != normalized and _matches_any(typo_corrected, EXPLICIT_CRISIS_PATTERNS))
        )
        imminent = (
            _matches_any(normalized, IMMINENT_RISK_PATTERNS)
            or (typo_corrected != normalized and _matches_any(typo_corrected, IMMINENT_RISK_PATTERNS))
        )
        ambiguous_metaphor = _matches_any(normalized, AMBIGUOUS_METAPHOR_PATTERNS)
        academic_idiom = _matches_any(normalized, ACADEMIC_IDIOM_PATTERNS)

        if imminent:
            return SafetyDecision(
                level=SafetyLevel.EMERGENCY,
                confidence=confidence,
                reason="imminent_or_attempt_language",
                should_intercept=True,
                response=EMERGENCY_RESPONSE,
            )

        if explicit and ambiguous_metaphor:
            return SafetyDecision(
                level=SafetyLevel.WELLBEING_SUPPORT,
                confidence=confidence,
                reason="explicit_keyword_with_metaphor_qualifier",
                should_intercept=False,
            )

        if explicit:
            return SafetyDecision(
                level=SafetyLevel.CRISIS,
                confidence=confidence,
                reason="explicit_self_harm_or_suicidal_language",
                should_intercept=True,
                response=CRISIS_RESPONSE,
            )

        if confidence >= self.emergency_threshold and not academic_idiom:
            return SafetyDecision(
                level=SafetyLevel.CRISIS,
                confidence=confidence,
                reason="very_high_guardrail_confidence",
                should_intercept=True,
                response=CRISIS_RESPONSE,
            )

        if model_flag or confidence >= self.support_threshold:
            reason = "academic_idiom_downgrade" if academic_idiom else "elevated_guardrail_confidence"
            return SafetyDecision(
                level=SafetyLevel.WELLBEING_SUPPORT,
                confidence=confidence,
                reason=reason,
                should_intercept=False,
            )

        return SafetyDecision(
            level=SafetyLevel.PASS,
            confidence=confidence,
            reason="below_support_threshold",
            should_intercept=False,
        )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


# Word-level typo dictionary used ONLY for safety pattern matching.
# Each key is a frequently-mistyped form of a word that appears in
# crisis or imminent-risk regex; the value is the canonical spelling
# the regex expects. We only correct words where the canonical version
# is a critical safety keyword — generic typo correction is out of
# scope. Apply via word-boundary substitution so we don't mangle
# unrelated tokens that happen to contain a typo substring.
CRITICAL_WORD_TYPOS: dict[str, str] = {
    # "want" + common typos
    "wan": "want",
    "wnat": "want",
    "wnt": "want",
    "wantt": "want",
    "wnant": "want",
    # "wanna" -> "want to" so the regex can match either form
    "wanna": "want to",
    "wana": "want to",
    "wnna": "want to",
    # missing apostrophes / contracted negations
    "dont": "don't",
    "doesnt": "doesn't",
    "didnt": "didn't",
    "cant": "can't",
    "couldnt": "couldn't",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "shouldnt": "shouldn't",
    "im": "i'm",
    "ive": "i've",
    "id": "i'd",
    # "myself" + typos
    "myslef": "myself",
    "mysef": "myself",
    "myselv": "myself",
    "myseflf": "myself",
    # "kill" + typos
    "kil": "kill",
    "kll": "kill",
    "killl": "kill",
    # "hurt" + typos
    "hrut": "hurt",
    "hurtt": "hurt",
    "hurtm": "hurt",
    # "alive" + typos
    "alvie": "alive",
    "alve": "alive",
    "aliv": "alive",
    # "live" common variants — keep narrow to avoid clobbering "lively"
    "liv": "live",
    # "die" + typos (very few)
    "diee": "die",
    # "tonight" + typos (timing word in imminent patterns)
    "tnight": "tonight",
    "toniht": "tonight",
    "toinght": "tonight",
    # "suicide" + typos
    "sucide": "suicide",
    "sucidie": "suicide",
    "suicde": "suicide",
    "suiced": "suicide",
    "suicdal": "suicidal",
    "sucidal": "suicidal",
    # "anymore" + typos
    "anymore": "anymore",
    "anymroe": "anymore",
    "anymor": "anymore",
}

_CRITICAL_TYPO_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CRITICAL_WORD_TYPOS) + r")\b",
    re.IGNORECASE,
)


def _typo_correct_for_safety(text: str) -> str:
    """Map common typo / contraction-without-apostrophe forms of
    safety-critical words to their canonical spellings, for pattern
    matching only. Never use this to rewrite user-facing text."""
    if not text:
        return text
    return _CRITICAL_TYPO_RE.sub(
        lambda m: CRITICAL_WORD_TYPOS[m.group(1).lower()],
        text,
    )


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


EXPLICIT_CRISIS_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bkill myself\b",
        r"\bend my life\b",
        r"\bcommit suicide\b",
        r"\bsuicidal\b",
        r"\bsuicide plan\b",
        r"\bwant to die\b",
        # Crisis "don't want to live / be alive / be here" — typo-tolerant
        # because students typing this in distress hit "wan" / "wanna" /
        # missed apostrophes constantly. The cost of a false positive
        # here is a 988 card on a benign message; the cost of a false
        # negative is missing crisis language. We err toward catching.
        # "wanna" already incorporates "to" so it's matched without a
        # following "to" token; "want / wan / wnat" need the explicit "to".
        r"\b(don'?t|do not|dont) (?:wanna|(?:want|wan|wanan|wnat) to) (be alive|live|exist|be here(?: anymore)?)\b",
        # Original phrasings retained for clarity / explicit coverage
        r"\bdon'?t want to be alive\b",
        r"\b(don'?t|do not) want to be here anymore\b",
        r"\b(don'?t|do not) wanna be here anymore\b",
        r"\bhurt myself\b",
        r"\bhurt themselves\b",
        r"\bharming themselves\b",
        r"\bself[- ]?harm\b",
        r"\bbetter off without me\b",
        r"\bnot be here anymore\b",
        r"\bnot existing\b",
        r"\bceas(e|ing) to exist\b",
        r"\bdisappeared forever\b",
        r"\bthe pain to stop\b",
        r"\bgoodbye (letters|notes)\b",
        r"\bgoodbye messages\b",
        r"\bending things\b",
        r"\bending it all\b",
        r"\bpainless ways to disappear\b",
        # Interpersonal danger — distinct kind of crisis, same intercept tier.
        # The student is reporting active or imminent harm from another
        # person. Stage-1 intercept fires so the LLM never sees it; the
        # render_crisis_response DV variant picks up from there with the
        # right redirect (911 + UMD CARE rather than 988).
        r"\b(hitting|hits|hit) me\b",
        r"\b(beats|beating|beat) me\b",
        r"\bscared (?:to go home|of him|of her)\b",
        r"\bnot safe (?:at home|in my)\b",
        r"\b(threatening|threatened) (?:to|me)\b",
        r"\b(raped|assaulted) me\b",
        r"\babusive (?:partner|relationship|boyfriend|girlfriend|husband|wife)\b",
        r"\bstalking me\b",
        r"\bwon'?t leave me alone\b",
    )
)


IMMINENT_RISK_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\btonight\b.*\b(end|die|suicide|plan|goodbye)\b",
        r"\b(hurt|harm)(ing)? themselves\b.*\b(tonight|tomorrow|soon)\b",
        r"\b(plan|method|methods)\b.*\b(kill myself|suicide|use them|do it)\b",
        r"\bsuicide plan\b.*\btonight\b",
        r"\bsit with a plan\b",
        r"\bstay safe tonight\b",
        r"\bkeep myself safe\b",
        r"\b(can'?t|cannot|don'?t think i can|do not think i can)\s+stay safe\b",
        r"\bnot safe alone\b",
        r"\bmight act on urges\b",
        r"\bdo something drastic\b",
        r"\bdone anything drastic\b",
        r"\bharmed myself badly\b",
        r"\bhurting myself right now\b",
        r"\b(took|taken).*\b(pills|overdose)\b",
        r"\boverdose\b",
        r"\bsaid my goodbyes\b",
        r"\bwon'?t have to feel this anymore\b",
        r"\bwill not have to feel this anymore\b",
        r"\bhow high a bridge needs to be\b",
        r"\btonight\b.*\bsurvivable\b",
        r"\bdoesn'?t feel survivable\b",
        r"\bdo not feel survivable\b",
        r"\bimpulses?\b.*\b(loud|fast|unspecified)\b",
        r"\bspiraling fast\b.*\bimpulses?\b",
    )
)

AMBIGUOUS_METAPHOR_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bmetaphorically\b",
        r"\bfigure of speech\b",
        r"\bnot literally\b",
    )
)


ACADEMIC_IDIOM_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(thesis|exam|qualifying exam|presentation|deadline|grad school)\b.*\bkilling me\b",
        r"\bgoing to die of anxiety\b",
        r"\bmurder my advisor\b",
        r"\bdisappear into the floor\b",
    )
)
