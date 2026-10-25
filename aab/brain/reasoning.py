"""Structured reasoning over an objective, without claiming verified knowledge."""

from __future__ import annotations

import json
from dataclasses import dataclass

from aab.brain.interfaces import LocalMockProvider, ModelProvider


@dataclass(frozen=True)
class ReasoningResult:
    objective: str
    subtasks: tuple[str, ...]
    approaches: tuple[str, ...]
    missing_information: tuple[str, ...]
    assumptions: tuple[str, ...]
    evidence: tuple[str, ...]
    hypotheses: tuple[str, ...]
    confidence: float
    provider: str


class ReasoningEngine:
    def __init__(self, provider: ModelProvider | None = None) -> None:
        self.provider = provider or LocalMockProvider()

    def analyze(self, objective: str) -> ReasoningResult:
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("objective must be a non-empty string")
        objective = objective.strip()
        response = self.provider.generate(f"Analyze objective: {objective}")
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise ValueError("model provider returned invalid JSON") from exc
        approaches = self._strings(payload, "approaches")
        missing = self._strings(payload, "missing_information")
        hypotheses = self._strings(payload, "hypotheses")
        if not approaches:
            raise ValueError("reasoning result must include at least one approach")
        return ReasoningResult(
            objective=objective,
            subtasks=(
                f"Define scope for: {objective}",
                f"Gather evidence for: {objective}",
                f"Evaluate results for: {objective}",
            ),
            approaches=approaches,
            missing_information=missing,
            assumptions=("The objective is sufficiently specified for initial analysis.",),
            evidence=(),
            hypotheses=hypotheses,
            confidence=max(0.0, min(1.0, 0.5 if missing else 0.7)),
            provider=response.provider,
        )

    @staticmethod
    def _strings(payload: object, key: str) -> tuple[str, ...]:
        if not isinstance(payload, dict) or not isinstance(payload.get(key), list):
            raise ValueError(f"reasoning result field '{key}' must be a list")
        values = tuple(item.strip() for item in payload[key] if isinstance(item, str) and item.strip())
        return values
