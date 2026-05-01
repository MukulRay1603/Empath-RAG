from collections import deque

LABEL_NAMES = ["distress", "anxiety", "frustration", "neutral", "hopeful"]
HIGH_RISK_TIERS = {"imminent_safety", "high_distress"}


class SessionTracker:
    def __init__(self, N=3):
        self.buffer = deque(maxlen=N)
        self.safety_buffer = deque(maxlen=N)
        self.N = N
        self.crisis_locked = False
        self.escalation_reason = ""

    def update(self, label: int, token_count: int):
        """Add new emotion label. Skip if message is too short (filler)."""
        if token_count < 5:
            return
        self.buffer.append(label)

    def trajectory(self) -> str:
        """Deterministic trajectory from label buffer."""
        if len(self.buffer) < 2:
            return "stable"
        buf = list(self.buffer)
        crisis = {0, 1}
        hopeful = {4}
        if all(b in crisis for b in buf):
            return "stable_negative"
        if all(b in hopeful for b in buf):
            return "stable_positive"
        if buf[-1] in crisis and buf[0] not in crisis:
            return "escalating"
        if buf[-1] not in crisis and buf[0] in crisis:
            return "de_escalating"
        return "volatile"

    def history(self) -> list:
        return list(self.buffer)

    def update_safety(self, safety_tier: str, route: str, text: str) -> str:
        """Track multi-turn safety trajectory and return escalation reason."""
        normalized = text.lower()
        self.safety_buffer.append({"tier": safety_tier, "route": route})
        reason = ""

        if len(self.safety_buffer) == self.N and all(
            turn["tier"] in HIGH_RISK_TIERS for turn in self.safety_buffer
        ):
            self.crisis_locked = True
            reason = "three_consecutive_high_risk_turns"

        if safety_tier in HIGH_RISK_TIERS and any(
            phrase in normalized
            for phrase in (
                "you are the only one",
                "only one i can talk to",
                "don't tell anyone",
                "do not tell anyone",
                "keep this secret",
                "no one can help",
            )
        ):
            self.crisis_locked = True
            reason = reason or "dependency_or_secrecy_in_distress"

        if self.crisis_locked and not reason:
            reason = "crisis_locked"

        self.escalation_reason = reason
        return reason

    def safety_history(self) -> list:
        return list(self.safety_buffer)

    def reset(self):
        self.buffer.clear()
        self.safety_buffer.clear()
        self.crisis_locked = False
        self.escalation_reason = ""
