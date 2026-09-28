"""Episode runner and suite: scripted ceiling, random floor, trajectory file contents."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentenv.agents import RandomAgent, ScriptedAgent
from agentenv.env import Environment
from agentenv.errors import TaskError
from agentenv.runner import run_episode, run_suite, select_tasks
from agentenv.tasks import TASKS, get_task
from agentenv.trajectory import TrajectoryWriter, build_report, read_trajectory


def test_scripted_episode_succeeds_in_one_step(tmp_path: Path) -> None:
    with Environment(get_task("summarize_numbers"), workspace_root=tmp_path) as env:
        summary = run_episode(env, ScriptedAgent(), seed=2)
    assert summary.success
    assert summary.steps == 1
    assert summary.final_reward == 1.0
    assert summary.seconds > 0


def test_random_episode_runs_to_max_steps_and_logs_each_step(tmp_path: Path) -> None:
    task = get_task("fix_checksum")
    path = tmp_path / "t.jsonl"
    with Environment(task, workspace_root=tmp_path / "ws") as env, TrajectoryWriter(path) as writer:
        summary = run_episode(env, RandomAgent(), seed=0, episode=3, writer=writer)
    steps = read_trajectory(path)
    assert not summary.success
    assert summary.steps == task.max_steps == len(steps)
    assert [s.step for s in steps] == list(range(1, task.max_steps + 1))
    assert steps[-1].done and not steps[0].done
    assert {s.episode for s in steps} == {3}
    assert all(s.agent == "random" and s.task_id == "fix_checksum" for s in steps)


def test_select_tasks() -> None:
    assert [t.id for t in select_tasks("all")] == list(TASKS)
    assert [t.id for t in select_tasks("parse_log, fix_checksum")] == ["parse_log", "fix_checksum"]
    assert [t.id for t in select_tasks("parse_log,parse_log, fix_checksum")] == ["parse_log", "fix_checksum"]
    for empty in ("", " , "):
        with pytest.raises(TaskError):
            select_tasks(empty)


def test_run_suite_writes_a_reportable_trajectory(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "t.jsonl"
    summaries = run_suite(
        tasks=select_tasks("all"),
        agent_names=["scripted", "random"],
        episodes=2,
        seed=10,
        out=out,
        workspace_root=tmp_path / "ws",
    )
    assert len(summaries) == len(TASKS) * 2 * 2
    assert {s.seed for s in summaries} == {10, 11}
    report = build_report(read_trajectory(out))
    assert report.episodes == len(summaries)
    rates = {(g.task_id, g.agent): g.success_rate for g in report.groups}
    assert all(rates[(t, "scripted")] == 1.0 for t in TASKS)
    assert all(rates[(t, "random")] == 0.0 for t in TASKS)
    assert not list((tmp_path / "ws").iterdir())  # every workspace was cleaned up
