"""Persistent task tracking and duplicate prevention."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from aab.memory.database import ExperienceStore


@dataclass(frozen=True)
class Task:
    id: int
    objective: str
    question: str
    status: str
    rationale: str


class TaskManager:
    """Create and complete durable tasks while preventing duplicate work."""

    def __init__(self, store: ExperienceStore) -> None:
        self.store = store

    @staticmethod
    def key(objective: str, question: str) -> str:
        value = f"{objective.strip().casefold()}\n{question.strip().casefold()}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def create(self, objective: str, question: str, rationale: str) -> Task | None:
        if not all(isinstance(value, str) and value.strip() for value in (objective, question, rationale)):
            raise ValueError("objective, question, and rationale must be non-empty strings")
        task_key = self.key(objective, question)
        task_id = self.store.create_task(task_key, objective.strip(), question.strip(), rationale.strip())
        if task_id is None:
            return None
        return Task(task_id, objective.strip(), question.strip(), "pending", rationale.strip())

    def mark_running(self, task_id: int) -> None:
        self.store.update_task_status(task_id, "running")

    def complete(self, task_id: int, outcome: str = "completed") -> None:
        if not isinstance(outcome, str) or not outcome.strip():
            raise ValueError("outcome must be a non-empty string")
        self.store.update_task_status(task_id, "completed", outcome.strip())

    def fail(self, task_id: int, outcome: str) -> None:
        if not isinstance(outcome, str) or not outcome.strip():
            raise ValueError("outcome must be a non-empty string")
        self.store.update_task_status(task_id, "failed", outcome.strip())
