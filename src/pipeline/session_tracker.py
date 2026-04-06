from collections import deque

LABEL_NAMES = ["distress", "anxiety", "frustration", "neutral", "hopeful"]


class SessionTracker:
    def __init__(self, N=3):
        self.buffer = deque(maxlen=N)
        self.N = N

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

    def reset(self):
        self.buffer.clear()
