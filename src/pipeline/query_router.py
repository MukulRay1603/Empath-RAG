TEMPLATES = {
    "distress":    "peer support coping strategies feeling overwhelmed distressed {query}",
    "anxiety":     "managing anxiety stress relief student mental health {query}",
    "frustration": "dealing with frustration academic pressure stress {query}",
    "neutral":     "{query}",
    "hopeful":     "positive coping wellbeing success {query}",
}

TRAJECTORY_PREFIX = {
    "escalating":      "urgent emotional support crisis prevention ",
    "stable_negative": "ongoing support persistent distress ",
    "de_escalating":   "positive reinforcement progress ",
    "stable_positive": "",
    "volatile":        "emotional regulation grounding techniques ",
    "stable":          "",
}

LABEL_NAMES = ["distress", "anxiety", "frustration", "neutral", "hopeful"]


def route_query(raw_query: str, emotion_label: int, trajectory: str) -> str:
    label_name = LABEL_NAMES[emotion_label]
    template = TEMPLATES[label_name]
    prefix = TRAJECTORY_PREFIX.get(trajectory, "")
    routed = prefix + template.format(query=raw_query)
    return routed.strip()
