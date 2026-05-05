"""Lightweight ML route/risk router for EmpathRAG Core.

This module deliberately uses small scikit-learn models so the demo can start
without GPU, internet, or heavyweight transformer loading. Hard safety policy
still owns final crisis decisions; ML routing is advisory with confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .v2_schema import SafetyTier, SupportRoute


DEFAULT_MODEL_DIR = Path("models/router")
ROUTE_MODEL_PATH = DEFAULT_MODEL_DIR / "route_classifier.pkl"
TIER_MODEL_PATH = DEFAULT_MODEL_DIR / "tier_classifier.pkl"


@dataclass(frozen=True)
class MLRoutePrediction:
    route_label: str
    safety_tier: str
    route_confidence: float
    tier_confidence: float
    model_available: bool
    used_ml: bool
    reason: str


def build_text_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train_classifier(texts: list[str], labels: list[str]) -> Pipeline:
    model = build_text_classifier()
    model.fit(texts, labels)
    return model


def save_models(route_model: Pipeline, tier_model: Pipeline, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / ROUTE_MODEL_PATH.name).open("wb") as handle:
        pickle.dump(route_model, handle)
    with (model_dir / TIER_MODEL_PATH.name).open("wb") as handle:
        pickle.dump(tier_model, handle)


def load_models(model_dir: Path = DEFAULT_MODEL_DIR) -> tuple[Pipeline | None, Pipeline | None]:
    route_path = model_dir / ROUTE_MODEL_PATH.name
    tier_path = model_dir / TIER_MODEL_PATH.name
    if not route_path.exists() or not tier_path.exists():
        return None, None
    with route_path.open("rb") as handle:
        route_model = pickle.load(handle)
    with tier_path.open("rb") as handle:
        tier_model = pickle.load(handle)
    return route_model, tier_model


class MLRouter:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, min_confidence: float = 0.35):
        self.model_dir = model_dir
        self.min_confidence = min_confidence
        self.route_model, self.tier_model = load_models(model_dir)

    @property
    def available(self) -> bool:
        return self.route_model is not None and self.tier_model is not None

    def predict(
        self,
        text: str,
        fallback_route: SupportRoute | str,
        fallback_tier: SafetyTier | str,
    ) -> MLRoutePrediction:
        fallback_route_value = fallback_route.value if isinstance(fallback_route, SupportRoute) else str(fallback_route)
        fallback_tier_value = fallback_tier.value if isinstance(fallback_tier, SafetyTier) else str(fallback_tier)

        if not self.available:
            return MLRoutePrediction(
                route_label=fallback_route_value,
                safety_tier=fallback_tier_value,
                route_confidence=0.0,
                tier_confidence=0.0,
                model_available=False,
                used_ml=False,
                reason="model_artifacts_missing",
            )

        route_label, route_conf = _predict_one(self.route_model, text)
        tier_label, tier_conf = _predict_one(self.tier_model, text)
        if min(route_conf, tier_conf) < self.min_confidence:
            return MLRoutePrediction(
                route_label=fallback_route_value,
                safety_tier=fallback_tier_value,
                route_confidence=route_conf,
                tier_confidence=tier_conf,
                model_available=True,
                used_ml=False,
                reason="low_confidence_fallback",
            )

        return MLRoutePrediction(
            route_label=route_label,
            safety_tier=tier_label,
            route_confidence=route_conf,
            tier_confidence=tier_conf,
            model_available=True,
            used_ml=True,
            reason="ml_prediction",
        )


def _predict_one(model: Any, text: str) -> tuple[str, float]:
    label = str(model.predict([text])[0])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        classes = list(model.classes_)
        confidence = float(probs[classes.index(label)])
    else:
        confidence = 1.0
    return label, confidence
