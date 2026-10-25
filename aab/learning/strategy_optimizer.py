"""Evidence-gated strategy alternatives and benchmark comparison."""

from __future__ import annotations

from dataclasses import dataclass

from aab.learning.evaluator import BenchmarkResult, LearningEvaluator


@dataclass(frozen=True)
class StrategyProposal:
    name: str
    rationale: str
    based_on_errors: tuple[str, ...]


@dataclass(frozen=True)
class StrategyComparison:
    baseline: BenchmarkResult
    candidate: BenchmarkResult
    adopted: bool
    rationale: str


class StrategyOptimizer:
    """Propose alternatives and retain only measured improvements."""

    def __init__(self, evaluator: LearningEvaluator) -> None:
        self.evaluator = evaluator

    def alternatives(self, current_strategy: str) -> list[StrategyProposal]:
        if not isinstance(current_strategy, str) or not current_strategy.strip():
            raise ValueError("current_strategy must be a non-empty string")
        errors = tuple(item["error_type"] for item in self.evaluator.experiences.recurring_errors())
        proposals = [
            StrategyProposal(
                f"{current_strategy.strip()} with explicit verification",
                "Add an independent verification step before accepting a conclusion.",
                errors,
            ),
            StrategyProposal(
                f"{current_strategy.strip()} with alternative comparison",
                "Compare at least two approaches before selecting a result.",
                errors,
            ),
        ]
        return proposals

    def compare(
        self,
        baseline: BenchmarkResult,
        candidate: BenchmarkResult,
    ) -> StrategyComparison:
        if baseline.name != candidate.name:
            raise ValueError("baseline and candidate must belong to the same benchmark")
        adopted = candidate.accuracy > baseline.accuracy
        rationale = (
            "Candidate exceeds baseline accuracy and is eligible for controlled adoption."
            if adopted
            else "Candidate did not exceed baseline accuracy; retain the baseline."
        )
        return StrategyComparison(baseline, candidate, adopted, rationale)
