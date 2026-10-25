import json

from aab.learning.training_pipeline import run_xor_experiment
from aab.memory.database import ExperienceStore


def test_pipeline_persists_evaluation(tmp_path):
    result = run_xor_experiment(tmp_path / "experiences.db")

    assert result.accuracy == 1.0
    with ExperienceStore(tmp_path / "experiences.db") as store:
        saved = store.get(result.experience_id)
    assert saved["outcome"] == "passed"
    assert json.loads(saved["metrics_json"])["accuracy"] == 1.0
