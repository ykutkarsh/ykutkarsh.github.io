"""Controlled fine-tuning for the Phase 1 XOR model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from aab.learning.dataset import PreparedDataset
from aab.models.neural_network import XORNetwork


@dataclass(frozen=True)
class TrainingRun:
    model: XORNetwork
    initial_loss: float
    final_loss: float
    epochs: int


class FineTuner:
    """Train a candidate copy while leaving the supplied baseline untouched."""

    def train(
        self,
        baseline: XORNetwork,
        dataset: PreparedDataset,
        *,
        epochs: int = 100,
        learning_rate: float = 0.05,
    ) -> TrainingRun:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        candidate = XORNetwork()
        candidate.load_state_dict(baseline.state_dict())
        optimizer = torch.optim.Adam(candidate.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        with torch.no_grad():
            initial_loss = float(
                criterion(candidate(dataset.inputs), dataset.targets).item()
            )
        final_loss = initial_loss
        for _ in range(epochs):
            optimizer.zero_grad()
            final_loss_tensor = criterion(candidate(dataset.inputs), dataset.targets)
            final_loss_tensor.backward()
            optimizer.step()
            final_loss = float(final_loss_tensor.detach().item())
        return TrainingRun(candidate, initial_loss, final_loss, epochs)

    @staticmethod
    def save_model(model: XORNetwork, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), destination)

    @staticmethod
    def load_model(path: str | Path) -> XORNetwork:
        model = XORNetwork()
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model
