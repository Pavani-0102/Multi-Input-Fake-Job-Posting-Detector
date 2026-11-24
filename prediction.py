"""Load the trained pipeline and produce predictions for any input mode."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import joblib

from config import (
    DECISION_THRESHOLD,
    FRAUD_LABEL,
    MAX_CHARS,
    METADATA_PATH,
    MIN_LETTER_RATIO,
    MIN_WORDS,
    PIPELINE_PATH,
)
from preprocessing import build_text
from rules import Indicator, find_indicators

LABEL_FRAUD_TEXT = "Potentially fraudulent"
LABEL_LEGIT_TEXT = "Potentially legitimate"


class InputValidationError(ValueError):
    """Raised for empty, too-short or unreadable input (message is user-safe)."""


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained pipeline has not been generated yet."""


@dataclass
class LoadedModel:
    pipeline: Any
    metadata: dict = field(default_factory=dict)


@dataclass
class PredictionResult:
    label: str
    is_fraud_predicted: bool
    fraud_probability: Optional[float]
    threshold: float
    model_name: str
    word_count: int
    indicators: list[Indicator]


def load_model(pipeline_path: Path = PIPELINE_PATH, metadata_path: Path = METADATA_PATH) -> LoadedModel:
    """Load the joblib pipeline (and optional metadata JSON)."""
    pipeline_path = Path(pipeline_path)
    if not pipeline_path.exists():
        raise ModelNotFoundError(
            f"No trained model found at '{pipeline_path}'. Run: python train_model.py"
        )
    pipeline = joblib.load(pipeline_path)
    metadata: dict = {}
    metadata_path = Path(metadata_path)
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    return LoadedModel(pipeline=pipeline, metadata=metadata)


def validate_text(text: Optional[str]) -> str:
    """Return stripped text or raise InputValidationError."""
    if text is None or not str(text).strip():
        raise InputValidationError("Please enter or extract some job-posting text first.")
    text = str(text).strip()
    if len(text) > MAX_CHARS:
        raise InputValidationError(f"The text is too long (limit {MAX_CHARS:,} characters).")
    words = text.split()
    if len(words) < MIN_WORDS:
        raise InputValidationError(
            f"The text is too short ({len(words)} words). Provide at least {MIN_WORDS} words "
            "so the model has enough content to analyse."
        )
    letters = sum(c.isalpha() for c in text)
    if letters / max(len(text), 1) < MIN_LETTER_RATIO:
        raise InputValidationError(
            "The text contains too few readable letters. If it came from OCR, please correct it or upload a clearer image."
        )
    return text


def _fraud_probability(pipeline: Any, text: str) -> Optional[float]:
    """Return P(fraud) when the final estimator supports predict_proba, else None."""
    if not hasattr(pipeline, "predict_proba"):
        return None
    classes = list(getattr(pipeline, "classes_", []))
    if FRAUD_LABEL not in classes:
        return None
    proba = pipeline.predict_proba([text])[0]
    return float(proba[classes.index(FRAUD_LABEL)])


def predict_text(
    model: LoadedModel,
    text: str,
    fields: Optional[Mapping[str, str]] = None,
    rule_context: str = "",
) -> PredictionResult:
    """Classify free text.

    `rule_context` (e.g. an e-mail sender address) is examined by the rule-based
    checks only; it is never fed to the ML model, keeping inference features
    identical to training.
    """
    text = validate_text(text)
    pipeline = model.pipeline
    threshold = float(model.metadata.get("decision_threshold", DECISION_THRESHOLD))

    probability = _fraud_probability(pipeline, text)
    if probability is not None:
        is_fraud = probability >= threshold
    else:
        is_fraud = int(pipeline.predict([text])[0]) == FRAUD_LABEL

    indicators = find_indicators(f"{text}\n{rule_context}".strip(), fields=fields)
    return PredictionResult(
        label=LABEL_FRAUD_TEXT if is_fraud else LABEL_LEGIT_TEXT,
        is_fraud_predicted=bool(is_fraud),
        fraud_probability=probability,
        threshold=threshold,
        model_name=str(model.metadata.get("model_name", type(pipeline.steps[-1][1]).__name__)),
        word_count=len(text.split()),
        indicators=indicators,
    )


def predict_fields(model: LoadedModel, fields: Mapping[str, str]) -> PredictionResult:
    """Classify the structured job form (Mode A)."""
    text = build_text(fields)
    return predict_text(model, text, fields=fields)
