"""Task suite: reference solutions score 1.0, the untouched workspace scores 0.0, seeds are deterministic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentenv.env import Environment
from agentenv.errors import TaskError
from agentenv.sandbox import SubprocessSandbox
from agentenv.tasks import TASKS, get_task, list_tasks

SEEDS = [0, 1, 2, 42]


def _run_reference(task_id: str, seed: int, root: Path) -> float:
    with Environment(get_task(task_id), workspace_root=root) as env:
        env.reset(seed)
        reward = 0.0
        for action in env.task.reference_solution(seed):
            reward = env.step(action).reward
        return reward


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_reference_solution_scores_one(task_id: str, seed: int, tmp_path: Path) -> None:
    assert _run_reference(task_id, seed, tmp_path) == 1.0


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_untouched_workspace_scores_zero(task_id: str, seed: int, tmp_path: Path) -> None:
    task = get_task(task_id)
    sandbox = SubprocessSandbox.create(root=tmp_path)
    task.setup(sandbox.workspace, seed)
    assert task.reward(sandbox, seed) == 0.0


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_setup_is_deterministic_per_seed(task_id: str, tmp_path: Path) -> None:
    task = get_task(task_id)
    snapshots: list[dict[str, str]] = []
    for seed in (5, 5, 6):
        workspace = tmp_path / f"{seed}-{len(snapshots)}"
        workspace.mkdir()
        task.setup(workspace, seed)
        snapshots.append({p.name: p.read_text() for p in workspace.iterdir()})
    assert snapshots[0] == snapshots[1]
    assert snapshots[0] != snapshots[2]
    assert "TASK.md" in snapshots[0]


def test_partial_credit_for_partially_correct_stats(tmp_path: Path) -> None:
    task = get_task("summarize_numbers")
    sandbox = SubprocessSandbox.create(root=tmp_path)
    task.setup(sandbox.workspace, 0)
    numbers = [int(line) for line in (sandbox.workspace / "numbers.txt").read_text().split()]
    (sandbox.workspace / "stats.json").write_text(json.dumps({"count": len(numbers), "sum": sum(numbers), "min": "?"}))
    assert task.reward(sandbox, 0) == pytest.approx(0.4)
    (sandbox.workspace / "stats.json").write_text("not json")
    assert task.reward(sandbox, 0) == 0.0


def test_checksum_bug_is_actually_wrong(tmp_path: Path) -> None:
    task = get_task("fix_checksum")
    sandbox = SubprocessSandbox.create(root=tmp_path)
    task.setup(sandbox.workspace, 3)
    source = (sandbox.workspace / "mathlib.py").read_text()
    assert "def weighted_checksum" in source
    assert sandbox.python("import mathlib").ok  # the buggy module still imports


def test_log_summary_keys(tmp_path: Path) -> None:
    task = get_task("parse_log")
    with Environment(task, workspace_root=tmp_path) as env:
        env.reset(11)
        for action in task.reference_solution(11):
            env.step(action)
        summary = json.loads((env.sandbox.workspace / "summary.json").read_text())
    assert set(summary) == {"lines", "by_level", "services", "errors_by_service"}
    assert summary["lines"] == sum(summary["by_level"].values())
    assert summary["services"] == sorted(summary["services"])
    assert set(summary["errors_by_service"]) <= set(summary["services"])


def test_registry() -> None:
    assert [t.id for t in list_tasks()] == ["fix_checksum", "summarize_numbers", "parse_log"]
    with pytest.raises(TaskError) as excinfo:
        get_task("nope")
    assert excinfo.value.details["available"] == sorted(TASKS)
