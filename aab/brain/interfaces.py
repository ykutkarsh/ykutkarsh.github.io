"""Provider-neutral language-model interfaces."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelResponse:
    """A model response with provider metadata."""

    text: str
    provider: str


class ModelProvider(Protocol):
    """Minimal contract implemented by interchangeable model providers."""

    name: str

    def generate(self, prompt: str) -> ModelResponse:
        ...


class LocalMockProvider:
    """Deterministic, offline provider used by tests and the local prototype."""

    name = "local-mock"

    def generate(self, prompt: str) -> ModelResponse:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        return ModelResponse(
            text=(
                '{"approaches":["decompose the objective","check available evidence"],'
                '"missing_information":["independent verification"],'
                '"hypotheses":["the stated objective is the current priority"]}'
            ),
            provider=self.name,
        )


class ConfiguredProvider:
    """Placeholder provider configuration that never exposes credentials."""

    def __init__(self, provider_name: str | None = None, api_key: str | None = None) -> None:
        self.name = provider_name or os.getenv("AAB_MODEL_PROVIDER", "unconfigured")
        self.api_key = api_key if api_key is not None else os.getenv("AAB_MODEL_API_KEY")

    def generate(self, prompt: str) -> ModelResponse:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not self.api_key:
            raise RuntimeError(
                "AAB_MODEL_API_KEY is not configured; use LocalMockProvider for offline execution"
            )
        raise NotImplementedError(
            f"provider '{self.name}' is configured but has no adapter implementation"
        )
