import pytest

from aab.learning.evaluator import LearningEvaluator
from aab.learning.experience import ExperienceTracker
from aab.learning.strategy_optimizer import StrategyOptimizer
from aab.memory.database import ExperienceStore


def test_predictions_and_outcomes_are_compared_and_persisted(tmp_path):
    path = tmp_path / "memory.db"
    with ExperienceStore(path) as store:
        tracker = ExperienceTracker(store)
        prediction = tracker.record_prediction("task", "baseline", "yes", 0.8)
        outcome = tracker.record_outcome(prediction, "no", error_type="wrong direction")
        assert not outcome.correct
    with ExperienceStore(path) as reopened:
        row = reopened._connection.execute(
            "SELECT actual, correct, error_type FROM learning_outcomes"
        ).fetchone()
        assert dict(row) == {"actual": "no", "correct": 0, "error_type": "wrong direction"}


def test_recurring_errors_generate_strategy_alternatives(tmp_path):
    with ExperienceStore(tmp_path / "memory.db") as store:
        tracker = ExperienceTracker(store)
        for _ in range(2):
            prediction = tracker.record_prediction("task", "baseline", "yes", 0.6)
            tracker.record_outcome(prediction, "no", error_type="wrong direction")
        assert tracker.recurring_errors() == [
            {"error_type": "wrong direction", "occurrences": 2}
        ]
        proposals = StrategyOptimizer(LearningEvaluator(store)).alternatives("baseline")
        assert len(proposals) == 2
        assert proposals[0].based_on_errors == ("wrong direction",)


def test_benchmarks_measure_accuracy_and_strategy_comparison_requires_improvement(tmp_path):
    with ExperienceStore(tmp_path / "memory.db") as store:
        evaluator = LearningEvaluator(store)
        baseline = evaluator.benchmark("arithmetic", "baseline", [("2", "2"), ("3", "4")])
        candidate = evaluator.benchmark("arithmetic", "candidate", [("2", "2"), ("3", "3")])
        comparison = StrategyOptimizer(evaluator).compare(baseline, candidate)
        assert baseline.accuracy == 0.5
        assert candidate.accuracy == 1.0
        assert comparison.adopted
        assert evaluator.strategy_performance("candidate") == 1.0


def test_invalid_learning_inputs_are_rejected(tmp_path):
    with ExperienceStore(tmp_path / "memory.db") as store:
        tracker = ExperienceTracker(store)
        with pytest.raises(ValueError):
            tracker.record_prediction("task", "strategy", "prediction", 1.1)
        evaluator = LearningEvaluator(store)
        with pytest.raises(ValueError):
            evaluator.benchmark("empty", "strategy", [])
