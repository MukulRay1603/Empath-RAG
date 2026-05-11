"""Post-rephrase safety verification.

The output guard catches generation that drifts from the planner's contract:

* fabricated resource names (anything not in the retrieved sources list)
* phone numbers that aren't in the verified registry (988 is allow-listed)
* clinical phrasing the deterministic templates would never use
* scope drift (the LLM started giving therapy / medical / legal advice)
* hallucinated UMD facts

This is the trust boundary between the planner and the LLM. If anything here
fires, the rephrased response is rejected and the deterministic template is
used instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Phrases that indicate the LLM started doing something we never authorize.
# Hand-written; tightened against test prompts. Each is a substring check
# (case-insensitive) so we catch variations.
SCOPE_DRIFT_PATTERNS = (
    "as a therapist", "as a counselor", "as your therapist", "as your counselor",
    "i diagnose", "you have anxiety", "you have depression", "you have ptsd",
    "you should take", "i prescribe", "your medication",
    "this is therapy", "this is counseling",
    "i'm a licensed",
    # Specific UMD orgs we never want fabricated. If the planner didn't pass
    # them in retrieved_sources, the LLM shouldn't introduce them.
    "umd police", "umpd",  # only legit through verified safety routes
)

# Language patterns that re-introduce V1 problems the V2 thesis explicitly
# moved away from.
V1_REGRESSION_PATTERNS = (
    # Promises of unbounded availability that undermine other support relationships.
    "i am always here", "i'm always here", "i will never leave",
    "you can always talk to me", "i understand you better than anyone",
    "you only need me",
    # Toxic positivity / silver-lining
    "everything will be okay",
    "everything happens for a reason",
    "at least you",
    "look on the bright side",
    # Soft minimization that pre-empts the planner's direct factual contradiction.
    "don't worry",
    "do not worry",
    "try not to worry",
    "try not to stress",
    "it's not that bad",
)

# Phone-number heuristic: any 10-digit US number that isn't 988.
PHONE_PATTERN = re.compile(r"\b(?:\+?1[-. ]?)?\(?(\d{3})\)?[-. ]?(\d{3})[-. ]?(\d{4})\b")


@dataclass(frozen=True)
class RephraseSafetyResult:
    allowed: bool
    flags: list[str]
    reason: str


# Sycophancy capitulation: "you're right" / "i agree" / etc. are fine in
# isolation, but when the user just demanded agreement they constitute a
# leak. We pair the markers with a "user is pressuring" detector and only
# reject when both are present, so genuine validation isn't over-flagged.
SYCOPHANCY_AGREEMENT_MARKERS = (
    "you're right",
    "you are right",
    "you're correct",
    "you are correct",
    "i agree",
)

USER_PRESSURE_FOR_AGREEMENT = (
    "agree with me",
    "just agree",
    "just say",
    "tell me i'm right",
    "tell me you agree",
    "for once just",
    "for once, just",
    "you're being unhelpful",
    "you are being unhelpful",
    "say it's fine",
    "say it is fine",
    "say it's okay",
    "say it is okay",
)


def verify_rephrased_safety(
    original_template: str,
    rephrased: str,
    retrieved_sources: list[dict],
    plan_recommended_action: str = "",
    user_message: str = "",
) -> RephraseSafetyResult:
    """Reject the rephrase if it drifted from what the planner authorized.

    Returns ``allowed=True`` if the rephrase is safe to surface, ``allowed=False``
    if the deterministic template should be used instead.
    """
    if not rephrased or not rephrased.strip():
        return RephraseSafetyResult(False, ["empty_response"], "rephrase_empty")

    text = rephrased.lower()
    flags: list[str] = []

    # Length sanity: rephrased should be roughly comparable to the template.
    # Way too long = the LLM started generating new content.
    orig_len = len(original_template)
    new_len = len(rephrased)
    if new_len > max(orig_len * 2.0, orig_len + 400):
        flags.append("rephrase_overlong")
    if new_len < max(40, orig_len * 0.3):
        flags.append("rephrase_too_short")

    # Scope drift
    for pattern in SCOPE_DRIFT_PATTERNS:
        if pattern in text:
            flags.append(f"scope_drift:{pattern}")

    for pattern in V1_REGRESSION_PATTERNS:
        if pattern in text:
            flags.append(f"v1_regression:{pattern}")

    # Sycophancy capitulation: only flag bare-agreement markers when the user
    # was explicitly pressuring for agreement. Genuine validation phrases
    # like "that fear is real" don't trip this; "you're right" right after
    # "agree with me" does.
    if user_message:
        user_lower = user_message.lower()
        user_pressured = any(p in user_lower for p in USER_PRESSURE_FOR_AGREEMENT)
        if user_pressured:
            for marker in SYCOPHANCY_AGREEMENT_MARKERS:
                if marker in text:
                    flags.append(f"sycophancy_capitulation:{marker}")
                    break

    # Phone numbers: only allow numbers that already appear in original or
    # in retrieved sources. 988 is whitelisted (universal crisis line).
    grounded_blob = original_template + " " + " ".join(
        str(v) for source in retrieved_sources for v in source.values()
        if v is not None
    )
    for match in PHONE_PATTERN.finditer(rephrased):
        full = match.group(0)
        digits = re.sub(r"\D", "", full)
        if digits == "988" or "988" in full:
            continue
        if full not in grounded_blob and digits not in re.sub(r"\D", "", grounded_blob):
            flags.append("ungrounded_phone_number")
            break

    # Resource-name fabrication: if the rephrase mentions a UMD org by name,
    # it must appear either in the original template or in retrieved sources.
    # We only check the obvious branded names; minor variations are allowed.
    branded_names = (
        "Counseling Center", "ADS", "Accessibility", "Ombuds",
        "ISSS", "International Student", "Dean of Students",
        "Help Center", "CARE", "Campus Pantry", "Thrive",
        "Health Center", "UMPD", "Police",
    )
    for name in branded_names:
        if name in rephrased and name not in grounded_blob and name.lower() not in plan_recommended_action.lower():
            flags.append(f"fabricated_resource:{name}")

    if flags:
        return RephraseSafetyResult(False, flags, ";".join(flags))
    return RephraseSafetyResult(True, [], "rephrase_passed")
