"""Run the Phase 1 neural-network experiment."""

from pathlib import Path

from aab.learning.training_pipeline import run_xor_experiment


if __name__ == "__main__":
    result = run_xor_experiment(Path("data/experiences.db"))
    print(f"accuracy={result.accuracy:.3f}")
    print(f"loss={result.loss:.6f}")
    print(f"experience_id={result.experience_id}")
