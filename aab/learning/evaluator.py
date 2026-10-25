"""Evaluation and benchmark aggregation for measured experience."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from aab.learning.experience import ExperienceTracker, Prediction
from aab.memory.database import ExperienceStore


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    strategy: str
    total: int
    correct: int
    accuracy: float


class LearningEvaluator:
    def __init__(self, store: ExperienceStore) -> None:
        self.store = store
        self.experiences = ExperienceTracker(store)

    def evaluate(
        self,
        prediction: Prediction | int,
        actual: str,
        *,
        error_type: str | None = None,
    ):
        """Compare a prediction with an independently supplied actual outcome."""
        return self.experiences.record_outcome(
            prediction, actual, error_type=error_type
        )

    def benchmark(
        self,
        name: str,
        strategy: str,
        cases: Iterable[tuple[str, str]],
    ) -> BenchmarkResult:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("benchmark name must be a non-empty string")
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError("benchmark strategy must be a non-empty string")
        cases = list(cases)
        if not cases:
            raise ValueError("benchmark requires at least one case")
        for predicted, actual in cases:
            if not isinstance(predicted, str) or not isinstance(actual, str):
                raise ValueError("benchmark cases must contain strings")
        correct = sum(
            predicted.strip().casefold() == actual.strip().casefold()
            for predicted, actual in cases
        )
        result = BenchmarkResult(
            name.strip(), strategy.strip(), len(cases), correct, correct / len(cases)
        )
        self.store.record_benchmark(
            result.name, result.strategy, result.total, result.correct, result.accuracy
        )
        return result

    def strategy_performance(self, strategy: str) -> float | None:
        return self.store.strategy_accuracy(strategy)
