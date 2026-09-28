"""Environment protocol: reset/step, actions, rewards and termination."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from agentenv.env import (
    Environment,
    PythonAction,
    ReadFileAction,
    ShellAction,
    StepResult,
    WriteFileAction,
    parse_action,
)
from agentenv.errors import EpisodeError
from agentenv.sandbox import SandboxLimits
from agentenv.tasks import get_task


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Environment]:
    with Environment(
        get_task("summarize_numbers"), workspace_root=tmp_path, limits=SandboxLimits(timeout_seconds=5)
    ) as e:
        yield e


def test_step_before_reset_raises(env: Environment) -> None:
    with pytest.raises(EpisodeError, match="before reset"):
        env.step(ShellAction(command="ls"))


def test_reset_returns_description_and_listing(env: Environment) -> None:
    obs = env.reset(seed=3)
    assert obs.step == 0
    assert obs.seed == 3
    assert obs.task_id == "summarize_numbers"
    assert obs.output == "workspace files: TASK.md, numbers.txt"
    assert "stats.json" in obs.description


def test_every_action_kind_produces_an_observation(env: Environment) -> None:
    env.reset(seed=1)
    listing = env.step(ShellAction(command="ls"))
    assert "numbers.txt" in listing.observation.output
    read = env.step(ReadFileAction(path="TASK.md"))
    assert read.observation.output.startswith("numbers.txt holds")
    written = env.step(WriteFileAction(path="note.txt", content="hi"))
    assert written.observation.output == "wrote 2 chars to note.txt"
    python = env.step(PythonAction(code="print(open('note.txt').read())"))
    assert python.observation.output == "hi"
    assert [r.reward for r in (listing, read, written, python)] == [0.0, 0.0, 0.0, 0.0]
    assert not python.done


def test_sandbox_errors_become_observations(env: Environment) -> None:
    env.reset(seed=1)
    result = env.step(ReadFileAction(path="../outside"))
    assert result.observation.returncode == 1
    assert "escapes the workspace" in result.observation.output
    assert result.info["error"] == "sandbox_error"


def test_reference_solution_finishes_with_reward_one(env: Environment) -> None:
    env.reset(seed=7)
    results: list[StepResult] = [env.step(action) for action in env.task.reference_solution(7)]
    assert results[-1].reward == 1.0
    assert results[-1].done
    with pytest.raises(EpisodeError, match="finished"):
        env.step(ShellAction(command="ls"))


def test_episode_ends_at_max_steps(tmp_path: Path) -> None:
    task = get_task("parse_log").model_copy(update={"max_steps": 2})
    with Environment(task, workspace_root=tmp_path) as env:
        env.reset(seed=0)
        first = env.step(ShellAction(command="true"))
        second = env.step(ShellAction(command="true"))
    assert (first.done, second.done) == (False, True)
    assert second.info["max_steps"] == 2


def test_reset_replaces_the_workspace(env: Environment) -> None:
    env.reset(seed=1)
    old = env.sandbox.workspace
    env.step(WriteFileAction(path="junk", content="x"))
    env.reset(seed=1)
    assert env.sandbox.workspace != old
    assert not old.exists()
    assert not (env.sandbox.workspace / "junk").exists()


def test_parse_action_discriminates_on_kind() -> None:
    assert parse_action({"kind": "python", "code": "print(1)"}) == PythonAction(code="print(1)")
    with pytest.raises(ValueError):
        parse_action({"kind": "teleport"})
