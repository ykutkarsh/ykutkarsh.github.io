import json

import pytest

from aab.memory.database import ExperienceStore


def test_experience_round_trip(tmp_path):
    path = tmp_path / "experiences.db"
    with ExperienceStore(path) as store:
        experience_id = store.record(
            task="test task",
            strategy="test strategy",
            prediction="yes",
            outcome="passed",
            confidence=0.9,
            source="test",
            verification_status="measured",
            metrics_json=json.dumps({"accuracy": 1.0}),
        )
        assert store.count() == 1
        assert store.get(experience_id)["verification_status"] == "measured"


def test_confidence_is_bounded(tmp_path):
    with ExperienceStore(tmp_path / "experiences.db") as store:
        with pytest.raises(ValueError, match="between 0 and 1"):
            store.record(
                task="task",
                strategy="strategy",
                prediction="prediction",
                outcome="outcome",
                confidence=1.1,
                source=None,
                verification_status="hypothesis",
                metrics_json="{}",
            )
