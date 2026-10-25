import torch
import pytest

from aab.learning.dataset import DatasetPreparer, TrainingExample
from aab.learning.fine_tuning import FineTuner
from aab.learning.model_registry import ModelRegistry
from aab.learning.training_pipeline import XOR_INPUTS, XOR_TARGETS, train_xor_model


def verified_xor_examples():
    return [
        TrainingExample((0.0, 0.0), 0.0, "independent-xor-spec", True),
        TrainingExample((0.0, 1.0), 1.0, "independent-xor-spec", True),
        TrainingExample((1.0, 0.0), 1.0, "independent-xor-spec", True),
        TrainingExample((1.0, 1.0), 0.0, "independent-xor-spec", True),
    ]


def test_dataset_preparation_rejects_unverified_data():
    with pytest.raises(ValueError, match="unverified"):
        DatasetPreparer().prepare(
            [TrainingExample((0.0, 0.0), 0.0, "generated", False)]
        )


def test_fine_tuning_trains_copy_without_mutating_baseline():
    baseline, _ = train_xor_model(epochs=100)
    state_before = {key: value.clone() for key, value in baseline.state_dict().items()}
    dataset = DatasetPreparer().prepare(verified_xor_examples())
    run = FineTuner().train(baseline, dataset, epochs=5)
    assert run.model is not baseline
    assert run.final_loss <= run.initial_loss
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, state_before[key])


def test_versioned_checkpoint_and_independent_evaluation(tmp_path):
    baseline, _ = train_xor_model(epochs=100)
    dataset = DatasetPreparer().prepare(verified_xor_examples())
    candidate = FineTuner().train(baseline, dataset, epochs=10).model
    registry = ModelRegistry(tmp_path / "checkpoints")
    registry.save("baseline-v1", baseline, {"role": "baseline"})
    registry.save("candidate-v1", candidate, {"role": "candidate", "sources": list(dataset.sources)})
    baseline_eval = registry.evaluate("baseline-v1", XOR_INPUTS, XOR_TARGETS)
    candidate_eval = registry.evaluate("candidate-v1", XOR_INPUTS, XOR_TARGETS)
    result = registry.compare(baseline_eval, candidate_eval)
    assert baseline_eval.examples == 4
    assert candidate_eval.accuracy >= baseline_eval.accuracy
    assert result.rationale


def test_registry_rejects_overwrite_and_invalid_evaluation(tmp_path):
    model, _ = train_xor_model(epochs=10)
    registry = ModelRegistry(tmp_path / "checkpoints")
    registry.save("v1", model, {})
    with pytest.raises(FileExistsError):
        registry.save("v1", model, {})
    with pytest.raises(ValueError):
        registry.evaluate_model("v1", model, torch.zeros((1, 3)), torch.zeros((1, 1)))
