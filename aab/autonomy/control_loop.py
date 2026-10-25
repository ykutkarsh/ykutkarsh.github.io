"""Bounded autonomous coordinator for reasoning and curiosity."""

from __future__ import annotations

from dataclasses import dataclass

from aab.brain.curiosity import CuriosityEngine, ResearchQuestion
from aab.brain.reasoning import ReasoningEngine
from aab.autonomy.scheduler import ResourceLimits, Scheduler
from aab.autonomy.task_manager import Task, TaskManager
from aab.memory.database import ExperienceStore


@dataclass(frozen=True)
class IterationResult:
    iteration: int
    task: Task | None
    reasoning_provider: str
    status: str


class ControlLoop:
    """Run the observe-review-question-select cycle with finite limits."""

    def __init__(
        self,
        store: ExperienceStore,
        reasoning: ReasoningEngine | None = None,
        curiosity: CuriosityEngine | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        self.store = store
        self.reasoning = reasoning or ReasoningEngine()
        self.curiosity = curiosity or CuriosityEngine(store)
        self.scheduler = scheduler or Scheduler()
        self.tasks = TaskManager(store)

    def run(self, objective: str, knowledge_gaps: list[str]) -> list[IterationResult]:
        if not isinstance(objective, str) or not objective.strip():
            raise ValueError("objective must be a non-empty string")
        self.scheduler.start()
        results: list[IterationResult] = []
        while self.scheduler.can_continue():
            self.scheduler.begin_iteration()
            reasoning = self.reasoning.analyze(objective)
            question: ResearchQuestion | None = self.curiosity.select(
                objective, knowledge_gaps or list(reasoning.missing_information)
            )
            task = None
            if question is not None:
                task = self.tasks.create(objective, question.text, question.rationale)
                if task is not None:
                    self.scheduler.record_task()
                    self.tasks.mark_running(task.id)
                    self.tasks.complete(task.id, "proposal recorded; investigation not performed")
            results.append(
                IterationResult(
                    self.scheduler.iterations,
                    task,
                    reasoning.provider,
                    "completed" if task else "no-new-task",
                )
            )
            if task is None:
                break
        return results

    def pause(self) -> None:
        self.scheduler.pause()

    def resume(self) -> None:
        self.scheduler.resume()

    def shutdown(self) -> None:
        self.scheduler.shutdown()
