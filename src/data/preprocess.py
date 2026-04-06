import re
import pandas as pd
from datasets import Dataset

# Emotion label mapping: 27 GoEmotions labels collapsed to 5 coarse classes
LABEL_MAP = {
    # Distress
    "grief": 0, "remorse": 0, "fear": 0, "sadness": 0,
    # Anxiety
    "nervousness": 1, "confusion": 1, "embarrassment": 1,
    # Frustration
    "anger": 2, "annoyance": 2, "disappointment": 2, "disgust": 2,
    # Neutral
    "neutral": 3,
    # Hopeful
    "optimism": 4, "relief": 4, "gratitude": 4, "joy": 4,
    "love": 4, "admiration": 4, "amusement": 4, "approval": 4,
    "caring": 4, "curiosity": 4, "desire": 4, "excitement": 4,
    "pride": 4, "realization": 4, "surprise": 4,
}
LABEL_NAMES = ["distress", "anxiety", "frustration", "neutral", "hopeful"]


def clean_text(text: str) -> str:
    """Remove Reddit artefacts and normalise whitespace."""
    text = re.sub(r"u/\w+", "", text)
    text = re.sub(r"r/\w+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\[deleted\]|\[removed\]", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_length(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def filter_by_length(texts, tokenizer, min_tok=20, max_tok=512):
    return [t for t in texts if min_tok <= token_length(t, tokenizer) <= max_tok]


def map_goemotions_label(label_ids: list, id2label: dict) -> int:
    """Return first matched coarse label, else neutral (3)."""
    for lid in label_ids:
        name = id2label[lid]
        if name in LABEL_MAP:
            return LABEL_MAP[name]
    return 3
