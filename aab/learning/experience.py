"""Structured prediction and outcome records for experience-based improvement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aab.memory.database import ExperienceStore


@dataclass(frozen=True)
class Prediction:
    id: int
    task: str
    strategy: str
    predicted: str
    confidence: float


@dataclass(frozen=True)
class Outcome:
    prediction_id: int
    actual: str
    correct: bool
    error_type: str | None


class ExperienceTracker:
    """Record predictions and outcomes without treating generated data as truth."""

    def __init__(self, store: ExperienceStore) -> None:
        self.store = store

    def record_prediction(
        self, task: str, strategy: str, predicted: str, confidence: float
    ) -> Prediction:
        if not all(isinstance(value, str) and value.strip() for value in (task, strategy, predicted)):
            raise ValueError("task, strategy, and predicted must be non-empty strings")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        prediction_id = self.store.record_prediction(
            task.strip(), strategy.strip(), predicted.strip(), confidence
        )
        return Prediction(prediction_id, task.strip(), strategy.strip(), predicted.strip(), confidence)

    def record_outcome(
        self,
        prediction: Prediction | int,
        actual: str,
        *,
        correct: bool | None = None,
        error_type: str | None = None,
    ) -> Outcome:
        if not isinstance(actual, str) or not actual.strip():
            raise ValueError("actual must be a non-empty string")
        prediction_id = prediction.id if isinstance(prediction, Prediction) else prediction
        if not isinstance(prediction_id, int) or prediction_id < 1:
            raise ValueError("prediction must be a positive prediction ID")
        predicted = self.store.get_prediction(prediction_id)
        if predicted is None:
            raise ValueError(f"prediction {prediction_id} does not exist")
        is_correct = (
            predicted["predicted"].strip().casefold() == actual.strip().casefold()
            if correct is None
            else correct
        )
        if not is_correct and error_type is None:
            error_type = "incorrect prediction"
        self.store.record_outcome(prediction_id, actual.strip(), is_correct, error_type)
        return Outcome(prediction_id, actual.strip(), is_correct, error_type)

    def recurring_errors(self, minimum_count: int = 2) -> list[dict[str, Any]]:
        if minimum_count < 1:
            raise ValueError("minimum_count must be positive")
        return self.store.recurring_errors(minimum_count)
