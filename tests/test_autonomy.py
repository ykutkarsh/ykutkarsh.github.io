import pytest

from aab.autonomy.control_loop import ControlLoop
from aab.autonomy.scheduler import ResourceLimits, Scheduler
from aab.autonomy.task_manager import TaskManager
from aab.memory.database import ExperienceStore


def test_control_loop_coordinates_reasoning_and_persists_one_task(tmp_path):
    with ExperienceStore(tmp_path / "memory.db") as store:
        loop = ControlLoop(
            store,
            scheduler=Scheduler(ResourceLimits(max_iterations=3, max_tasks=3, max_seconds=5)),
        )
        results = loop.run("learn the system", ["unknown mechanism"])
        assert len(results) == 2
        assert results[0].task is not None
        assert results[0].reasoning_provider == "local-mock"
        assert results[1].status == "no-new-task"
        assert store.task_count() == 1


def test_duplicate_tasks_are_not_repeated_across_runs(tmp_path):
    with ExperienceStore(tmp_path / "memory.db") as store:
        loop = ControlLoop(store, scheduler=Scheduler(ResourceLimits(2, 2, 5)))
        assert loop.run("learn", ["gap"])[0].task is not None
        assert loop.run("learn", ["gap"])[0].task is None
        assert store.task_count() == 1


def test_scheduler_limits_and_lifecycle_hooks():
    scheduler = Scheduler(ResourceLimits(max_iterations=1, max_tasks=1, max_seconds=5))
    scheduler.start()
    scheduler.begin_iteration()
    assert not scheduler.can_continue()
    scheduler.pause()
    assert scheduler.paused
    scheduler.resume()
    scheduler.shutdown()
    assert scheduler.shutdown_requested


def test_task_manager_rejects_invalid_input_and_duplicate(tmp_path):
    with ExperienceStore(tmp_path / "memory.db") as store:
        manager = TaskManager(store)
        with pytest.raises(ValueError):
            manager.create("", "question", "why")
        assert manager.create("objective", "question", "why") is not None
        assert manager.create("OBJECTIVE", "QUESTION", "different rationale") is None


def test_resource_limits_reject_unbounded_values():
    with pytest.raises(ValueError):
        ResourceLimits(max_iterations=0)
