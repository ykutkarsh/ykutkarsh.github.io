import pytest

from aab.brain.meta_reasoning import ExperimentError, MetaReasoningEngine


def test_proposal_is_benchmark_gated_and_adoption_is_reversible():
    engine = MetaReasoningEngine({"threshold": 0.5})
    proposal = engine.propose(
        "raise threshold",
        {"threshold": 0.8},
        rationale="Test a stricter threshold.",
        success_criteria="Score must increase.",
    )

    def benchmark(config, sandbox):
        assert sandbox.exists()
        return config["threshold"]

    results = engine.run([proposal], benchmark, adopt=True)
    assert results[0].improved
    assert engine.active_config == {"threshold": 0.8}
    engine.rollback()
    assert engine.active_config == {"threshold": 0.5}


def test_failed_candidate_is_not_adopted():
    engine = MetaReasoningEngine({"quality": 0.8})
    proposal = engine.propose(
        "lower quality",
        {"quality": 0.2},
        rationale="Negative control.",
        success_criteria="Must exceed baseline.",
    )
    result = engine.run([proposal], lambda config, sandbox: config["quality"], adopt=True)[0]
    assert not result.improved
    assert engine.active_config == {"quality": 0.8}


def test_safety_controls_reject_unknown_changes_and_excess_candidates():
    engine = MetaReasoningEngine({"safe": True}, max_candidates=1)
    with pytest.raises(ExperimentError, match="not allowed"):
        engine.propose(
            "unsafe",
            {"execute_code": True},
            rationale="bad",
            success_criteria="bad",
        )
    proposal = engine.propose(
        "candidate",
        {"safe": False},
        rationale="test",
        success_criteria="test",
    )
    with pytest.raises(ExperimentError, match="candidate limit"):
        engine.run([proposal, proposal], lambda config, sandbox: 0.5)


def test_benchmark_failures_and_invalid_scores_are_explicit():
    engine = MetaReasoningEngine({"value": 1})
    proposal = engine.propose(
        "test",
        {"value": 2},
        rationale="test",
        success_criteria="test",
    )
    with pytest.raises(ExperimentError, match="benchmark failed"):
        engine.run([proposal], lambda config, sandbox: 1 / 0)
    with pytest.raises(ExperimentError, match="between 0 and 1"):
        engine.run([proposal], lambda config, sandbox: 2)


def test_invalid_limits_and_proposals_are_rejected():
    with pytest.raises(ValueError):
        MetaReasoningEngine({}, max_seconds=0)
    engine = MetaReasoningEngine({})
    with pytest.raises(ValueError):
        engine.propose("", {}, rationale="x", success_criteria="x")
