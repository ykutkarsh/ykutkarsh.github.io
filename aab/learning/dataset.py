"""Validated, explicit datasets for controlled model experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor


@dataclass(frozen=True)
class TrainingExample:
    """One independently identified input/target pair."""

    inputs: tuple[float, ...]
    target: float
    source: str
    verified: bool = False


@dataclass(frozen=True)
class PreparedDataset:
    inputs: Tensor
    targets: Tensor
    sources: tuple[str, ...]


class DatasetPreparer:
    """Prepare only explicitly verified examples for model training."""

    def prepare(
        self,
        examples: Iterable[TrainingExample],
        *,
        input_size: int = 2,
    ) -> PreparedDataset:
        if not isinstance(input_size, int) or input_size < 1:
            raise ValueError("input_size must be a positive integer")
        materialized = list(examples)
        if not materialized:
            raise ValueError("at least one training example is required")
        for example in materialized:
            if len(example.inputs) != input_size:
                raise ValueError(f"each example must contain exactly {input_size} inputs")
            if not example.source.strip():
                raise ValueError("each example must include a source")
            if not example.verified:
                raise ValueError("unverified examples cannot be used for training")
            if not 0 <= example.target <= 1:
                raise ValueError("targets must be between 0 and 1")
        return PreparedDataset(
            inputs=torch.tensor([example.inputs for example in materialized], dtype=torch.float32),
            targets=torch.tensor(
                [[example.target] for example in materialized], dtype=torch.float32
            ),
            sources=tuple(example.source for example in materialized),
        )
