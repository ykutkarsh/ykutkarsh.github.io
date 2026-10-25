"""Reversible, explicitly scored decision selection."""

from __future__ import annotations

from dataclasses import dataclass

from aab.memory.database import ExperienceStore


@dataclass(frozen=True)
class ActionOption:
    name: str
    benefit: float
    cost: float
    risk: float
    uncertainty: float
    score: float = 0.0


@dataclass(frozen=True)
class Decision:
    selected: ActionOption
    rationale: str
    alternatives: tuple[ActionOption, ...]


class DecisionEngine:
    def __init__(self, store: ExperienceStore, criteria: dict[str, float] | None = None) -> None:
        self.store = store
        self.criteria = criteria or {"benefit": 1.0, "cost": 1.0, "risk": 1.0, "uncertainty": 1.0}
        if set(self.criteria) != {"benefit", "cost", "risk", "uncertainty"}:
            raise ValueError("criteria must define benefit, cost, risk, and uncertainty")

    def choose(self, objective: str, options: list[ActionOption]) -> Decision:
        if not objective.strip():
            raise ValueError("objective must be a non-empty string")
        if not options:
            raise ValueError("at least one action option is required")
        scored = tuple(
            ActionOption(
                option.name, option.benefit, option.cost, option.risk, option.uncertainty,
                self.criteria["benefit"] * option.benefit
                - self.criteria["cost"] * option.cost
                - self.criteria["risk"] * option.risk
                - self.criteria["uncertainty"] * option.uncertainty,
            )
            for option in options
        )
        selected = max(scored, key=lambda option: option.score)
        rationale = f"Selected '{selected.name}' with highest configured score ({selected.score:.3f})."
        self.store.record_decision(objective, selected.name, rationale)
        return Decision(selected, rationale, scored)
