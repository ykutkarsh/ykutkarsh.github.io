"""Controlled Phase 1 training and evaluation pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from aab.memory.database import ExperienceStore
from aab.models.neural_network import XORNetwork

XOR_INPUTS = torch.tensor(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32
)
XOR_TARGETS = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    loss: float
    experience_id: int


def train_xor_model(epochs: int = 2_000, seed: int = 7) -> tuple[XORNetwork, float]:
    """Train a fresh model and return it with its final binary-cross-entropy loss."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(seed)
    model = XORNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(XOR_INPUTS), XOR_TARGETS)
        loss.backward()
        optimizer.step()
    return model, float(loss.detach().item())


def evaluate_xor_model(model: XORNetwork) -> tuple[float, float]:
    """Measure loss and exact classification accuracy on all XOR examples."""
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        logits = model(XOR_INPUTS)
        loss = float(criterion(logits, XOR_TARGETS).item())
        predictions = (torch.sigmoid(logits) >= 0.5).float()
        accuracy = float((predictions == XOR_TARGETS).float().mean().item())
    return accuracy, loss


def run_xor_experiment(database_path: str | Path = "data/experiences.db") -> EvaluationResult:
    """Train, evaluate, and persist one reproducible experiment."""
    model, _ = train_xor_model()
    accuracy, loss = evaluate_xor_model(model)
    metrics = json.dumps({"accuracy": accuracy, "loss": loss, "examples": 4})
    with ExperienceStore(database_path) as store:
        experience_id = store.record(
            task="Classify the XOR truth table",
            strategy="Two-layer multilayer perceptron trained with Adam",
            prediction=f"Expected accuracy >= 0.75; observed {accuracy:.3f}",
            outcome="passed" if accuracy >= 0.75 else "failed",
            confidence=1.0 if accuracy >= 0.75 else 0.0,
            source="aab.learning.training_pipeline",
            verification_status="measured",
            metrics_json=metrics,
        )
    return EvaluationResult(accuracy, loss, experience_id)
