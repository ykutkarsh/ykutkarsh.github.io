"""Versioned checkpoints and independent candidate acceptance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from aab.learning.fine_tuning import FineTuner
from aab.models.neural_network import XORNetwork


@dataclass(frozen=True)
class ModelEvaluation:
    version: str
    accuracy: float
    loss: float
    examples: int


@dataclass(frozen=True)
class AcceptanceResult:
    accepted: bool
    baseline: ModelEvaluation
    candidate: ModelEvaluation
    rationale: str


class ModelRegistry:
    """Keep immutable checkpoint versions and require independent comparison."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, version: str, model: XORNetwork, metadata: dict[str, object]) -> Path:
        self._validate_version(version)
        checkpoint = self.directory / f"{version}.pt"
        manifest = self.directory / f"{version}.json"
        if checkpoint.exists() or manifest.exists():
            raise FileExistsError(f"model version already exists: {version}")
        FineTuner.save_model(model, checkpoint)
        manifest.write_text(
            json.dumps({"version": version, **metadata}, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return checkpoint

    def load(self, version: str) -> XORNetwork:
        self._validate_version(version)
        checkpoint = self.directory / f"{version}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"model version does not exist: {version}")
        return FineTuner.load_model(checkpoint)

    def evaluate(
        self,
        version: str,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> ModelEvaluation:
        model = self.load(version)
        return self.evaluate_model(version, model, inputs, targets)

    @staticmethod
    def evaluate_model(
        version: str,
        model: XORNetwork,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> ModelEvaluation:
        if inputs.shape[0] == 0 or inputs.shape != (targets.shape[0], 2):
            raise ValueError("inputs and targets must contain matching non-empty XOR examples")
        criterion = torch.nn.BCEWithLogitsLoss()
        with torch.no_grad():
            logits = model(inputs)
            loss = float(criterion(logits, targets).item())
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            accuracy = float((predictions == targets).float().mean().item())
        return ModelEvaluation(version, accuracy, loss, int(inputs.shape[0]))

    @staticmethod
    def compare(
        baseline: ModelEvaluation,
        candidate: ModelEvaluation,
    ) -> AcceptanceResult:
        if baseline.examples != candidate.examples:
            raise ValueError("baseline and candidate must use the same evaluation set")
        accepted = (
            candidate.accuracy > baseline.accuracy
            or (
                candidate.accuracy == baseline.accuracy
                and candidate.loss < baseline.loss
            )
        )
        rationale = (
            "Candidate outperformed the baseline on the independent evaluation set."
            if accepted
            else "Candidate did not demonstrate an improvement; baseline remains primary."
        )
        return AcceptanceResult(accepted, baseline, candidate, rationale)

    @staticmethod
    def _validate_version(version: str) -> None:
        if not isinstance(version, str) or not version.strip():
            raise ValueError("version must be a non-empty string")
        if Path(version).name != version or version in {".", ".."}:
            raise ValueError("version must be a simple checkpoint name")
