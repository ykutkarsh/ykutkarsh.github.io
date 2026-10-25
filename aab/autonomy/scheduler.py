"""Execution limits and cooperative lifecycle state."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceLimits:
    max_iterations: int = 10
    max_tasks: int = 10
    max_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.max_iterations < 1 or self.max_tasks < 1 or self.max_seconds <= 0:
            raise ValueError("resource limits must be positive")


class Scheduler:
    """Tracks pause, shutdown, and finite execution budgets."""

    def __init__(self, limits: ResourceLimits | None = None, clock=time.monotonic) -> None:
        self.limits = limits or ResourceLimits()
        self._clock = clock
        self._started_at: float | None = None
        self._iterations = 0
        self._tasks = 0
        self._paused = False
        self._shutdown = False

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown

    def start(self) -> None:
        self._started_at = self._clock()
        self._paused = False
        self._shutdown = False
        self._iterations = 0
        self._tasks = 0

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def shutdown(self) -> None:
        self._shutdown = True

    def can_continue(self) -> bool:
        if self._started_at is None or self._paused or self._shutdown:
            return False
        return (
            self._iterations < self.limits.max_iterations
            and self._tasks < self.limits.max_tasks
            and self._clock() - self._started_at < self.limits.max_seconds
        )

    def begin_iteration(self) -> None:
        if not self.can_continue():
            raise RuntimeError("scheduler cannot begin another iteration")
        self._iterations += 1

    def record_task(self) -> None:
        if self._tasks >= self.limits.max_tasks:
            raise RuntimeError("maximum task limit reached")
        self._tasks += 1

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def tasks(self) -> int:
        return self._tasks
