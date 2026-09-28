"""JSONL trajectory writer/reader and the report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentenv.errors import ConfigError
from agentenv.trajectory import TrajectoryStep, TrajectoryWriter, build_report, excerpt, read_trajectory


def _step(**overrides: object) -> TrajectoryStep:
    base: dict[str, object] = {
        "task_id": "t",
        "seed": 0,
        "agent": "a",
        "episode": 0,
        "step": 1,
        "action": {"kind": "shell", "command": "ls"},
        "observation_excerpt": "x",
        "reward": 0.0,
        "done": False,
    }
    return TrajectoryStep.model_validate({**base, **overrides})


def test_writer_writes_one_json_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "out" / "t.jsonl"
    with TrajectoryWriter(path) as writer:
        writer.write(_step())
        writer.write(_step(step=2, reward=1.0, done=True))
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["done"] is True
    assert read_trajectory(path) == [_step(), _step(step=2, reward=1.0, done=True)]


def test_writer_must_be_open(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not open"):
        TrajectoryWriter(tmp_path / "t.jsonl").write(_step())


def test_reader_reports_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text(_step().model_dump_json() + "\n\n{not json\n")
    with pytest.raises(ConfigError) as excinfo:
        read_trajectory(path)
    assert excinfo.value.details == {"line": 3}
    with pytest.raises(ConfigError, match="no such"):
        read_trajectory(tmp_path / "missing.jsonl")


def test_report_groups_by_task_and_agent() -> None:
    steps = [
        _step(agent="scripted", step=1, reward=1.0, done=True),
        _step(agent="scripted", seed=1, episode=1, step=1, reward=1.0, done=True),
        _step(agent="random", step=1),
        _step(agent="random", step=2, reward=0.5, done=True),
        _step(agent="random", seed=1, episode=1, step=1, done=True),
    ]
    report = build_report(steps)
    assert (report.steps, report.episodes) == (5, 4)
    by_agent = {g.agent: g for g in report.groups}
    assert by_agent["scripted"].success_rate == 1.0
    assert by_agent["scripted"].mean_steps == 1.0
    assert by_agent["random"].success_rate == 0.0
    assert by_agent["random"].mean_steps == 1.5
    assert by_agent["random"].mean_reward == 0.25
    table = report.render()
    assert "t                    random" in table
    assert table.endswith("4 episodes, 5 steps")
    totals = {a.agent: a for a in report.agents}
    assert set(totals) == {"scripted", "random"}
    assert (totals["scripted"].tasks, totals["scripted"].episodes, totals["scripted"].success_rate) == (1, 2, 1.0)
    assert totals["random"].mean_steps == 1.5 and totals["random"].mean_reward == 0.25
    assert totals["random"].mean_action_seconds == 0.0
    assert "all (1 tasks)        scripted          2     100%       1.00       1.000  0.000s/action" in table


def test_agent_totals_span_tasks_and_average_action_time() -> None:
    steps = [
        _step(task_id="a", agent="x", step=1, reward=1.0, done=True, info={"duration_seconds": 0.2}),
        _step(task_id="b", agent="x", step=1, reward=0.0, done=False, info={"duration_seconds": 0.4}),
        _step(task_id="b", agent="x", step=2, reward=0.5, done=True),
    ]
    totals = build_report(steps).agents
    assert len(totals) == 1 and totals[0].agent == "x"
    assert (totals[0].tasks, totals[0].episodes, totals[0].successes) == (2, 2, 1)
    assert totals[0].success_rate == 0.5 and totals[0].mean_steps == 1.5 and totals[0].mean_reward == 0.75
    assert totals[0].mean_action_seconds == pytest.approx(0.2)
    assert build_report([]).agents == []


def test_excerpt() -> None:
    assert excerpt("short") == "short"
    assert excerpt("x" * 300) == "x" * 200 + "..."
