import pytest

from aab.brain.curiosity import CuriosityEngine
from aab.brain.decision_engine import ActionOption, DecisionEngine
from aab.brain.interfaces import ConfiguredProvider, LocalMockProvider, ModelResponse
from aab.brain.reasoning import ReasoningEngine
from aab.memory.database import ExperienceStore


def test_reasoning_returns_structured_uncertainty():
    result = ReasoningEngine().analyze("design a safer experiment")
    assert result.subtasks and result.approaches
    assert result.hypotheses
    assert result.evidence == ()
    assert 0 <= result.confidence <= 1


def test_provider_interface_is_interchangeable():
    class StubProvider:
        name = "stub"

        def generate(self, prompt: str) -> ModelResponse:
            return ModelResponse(
                '{"approaches":["stub approach"],"missing_information":[],'
                '"hypotheses":["stub hypothesis"]}',
                self.name,
            )

    assert ReasoningEngine(StubProvider()).analyze("objective").provider == "stub"
    assert LocalMockProvider().generate("hello").provider == "local-mock"


def test_provider_and_input_validation():
    with pytest.raises(ValueError):
        ReasoningEngine().analyze("")
    with pytest.raises(RuntimeError, match="AAB_MODEL_API_KEY"):
        ConfiguredProvider(api_key=None).generate("hello")


def test_curiosity_ranks_and_avoids_duplicate_questions(tmp_path):
    with ExperienceStore(tmp_path / "memory.db") as store:
        engine = CuriosityEngine(store)
        first = engine.select("learn", ["unknown mechanism"])
        assert first is not None and first.status == "proposal"
        assert engine.select("learn", ["unknown mechanism"]) is None


def test_decision_scores_and_persists_selection(tmp_path):
    with ExperienceStore(tmp_path / "memory.db") as store:
        decision = DecisionEngine(store).choose(
            "select experiment",
            [
                ActionOption("risky", 1.0, 0.1, 0.9, 0.8),
                ActionOption("safe", 0.8, 0.2, 0.1, 0.1),
            ],
        )
        assert decision.selected.name == "safe"
    with ExperienceStore(tmp_path / "memory.db") as reopened:
        row = reopened._connection.execute("SELECT selected_action FROM decisions").fetchone()
        assert row["selected_action"] == "safe"


def test_existing_database_schema_remains_compatible(tmp_path):
    path = tmp_path / "memory.db"
    with ExperienceStore(path) as store:
        store.record(
            task="old", strategy="baseline", prediction="x", outcome="passed",
            confidence=1.0, source=None, verification_status="measured", metrics_json="{}",
        )
    with ExperienceStore(path) as reopened:
        assert reopened.count() == 1
