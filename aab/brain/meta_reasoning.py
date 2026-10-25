"""Safety-gated experiments for proposing and evaluating algorithmic changes.

Experiments operate on configuration data supplied by the caller. They do not
edit Python source, import candidate code, or replace a running model.
"""

from __future__ import annotations

import copy
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping


Benchmark = Callable[[Mapping[str, object], Path], float]


@dataclass(frozen=True)
class ImprovementProposal:
    name: str
    changes: dict[str, object]
    rationale: str
    success_criteria: str


@dataclass(frozen=True)
class ExperimentResult:
    proposal: ImprovementProposal
    baseline_score: float
    candidate_score: float
    improved: bool
    sandbox: str
    rationale: str


class ExperimentError(RuntimeError):
    """Raised when an experiment violates its safety contract."""


class MetaReasoningEngine:
    """Propose, isolate, benchmark, and optionally adopt reversible changes."""

    def __init__(
        self,
        baseline_config: Mapping[str, object],
        *,
        allowed_keys: set[str] | None = None,
        max_candidates: int = 3,
        max_seconds: float = 5.0,
    ) -> None:
        if not isinstance(baseline_config, Mapping):
            raise ValueError("baseline_config must be a mapping")
        if max_candidates < 1 or max_seconds <= 0:
            raise ValueError("experiment limits must be positive")
        self._baseline = copy.deepcopy(dict(baseline_config))
        self._active = copy.deepcopy(self._baseline)
        self.allowed_keys = allowed_keys or set(self._baseline)
        self.max_candidates = max_candidates
        self.max_seconds = max_seconds
        self._experiments = 0

    @property
    def active_config(self) -> dict[str, object]:
        return copy.deepcopy(self._active)

    def propose(
        self,
        name: str,
        changes: Mapping[str, object],
        *,
        rationale: str,
        success_criteria: str,
    ) -> ImprovementProposal:
        if not all(isinstance(value, str) and value.strip() for value in (name, rationale, success_criteria)):
            raise ValueError("name, rationale, and success_criteria must be non-empty strings")
        if not isinstance(changes, Mapping) or not changes:
            raise ValueError("changes must be a non-empty mapping")
        unknown = set(changes) - self.allowed_keys
        if unknown:
            raise ExperimentError(f"proposed changes are not allowed: {sorted(unknown)}")
        return ImprovementProposal(
            name.strip(), copy.deepcopy(dict(changes)), rationale.strip(), success_criteria.strip()
        )

    def run(
        self,
        proposals: list[ImprovementProposal],
        benchmark: Benchmark,
        *,
        adopt: bool = False,
    ) -> list[ExperimentResult]:
        if not callable(benchmark):
            raise ValueError("benchmark must be callable")
        if len(proposals) > self.max_candidates:
            raise ExperimentError("candidate limit exceeded")
        results: list[ExperimentResult] = []
        with tempfile.TemporaryDirectory(prefix="aab-experiment-") as sandbox:
            sandbox_path = Path(sandbox)
            baseline_score = self._measure(benchmark, self._baseline, sandbox_path)
            for proposal in proposals:
                candidate = copy.deepcopy(self._baseline)
                candidate.update(copy.deepcopy(proposal.changes))
                started = time.monotonic()
                candidate_score = self._measure(benchmark, candidate, sandbox_path)
                if time.monotonic() - started > self.max_seconds:
                    raise ExperimentError(f"experiment exceeded {self.max_seconds:.3f}s limit")
                improved = candidate_score > baseline_score
                result = ExperimentResult(
                    proposal,
                    baseline_score,
                    candidate_score,
                    improved,
                    str(sandbox_path),
                    (
                        "Candidate improved the predefined benchmark."
                        if improved
                        else "Candidate did not improve the predefined benchmark; baseline retained."
                    ),
                )
                results.append(result)
                self._experiments += 1
                if adopt and improved:
                    self._active = candidate
        return results

    def rollback(self) -> None:
        """Restore the last known accepted baseline configuration."""
        self._active = copy.deepcopy(self._baseline)

    @staticmethod
    def _measure(benchmark: Benchmark, config: Mapping[str, object], sandbox: Path) -> float:
        try:
            score = benchmark(copy.deepcopy(config), sandbox)
        except Exception as exc:
            raise ExperimentError(f"benchmark failed in isolated environment: {exc}") from exc
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ExperimentError("benchmark must return a numeric score")
        if not 0 <= float(score) <= 1:
            raise ExperimentError("benchmark score must be between 0 and 1")
        # Make the sandbox visibly auditable without executing candidate code.
        (sandbox / "experiment.json").write_text(
            json.dumps({"config": dict(config), "score": float(score)}, sort_keys=True),
            encoding="utf-8",
        )
        return float(score)
