"""Scripted and random agents."""

from __future__ import annotations

import pytest

from agentenv.agents import RandomAgent, ScriptedAgent, make_agent
from agentenv.env import Observation, ShellAction
from agentenv.errors import ConfigError
from agentenv.tasks import get_task


def _obs() -> Observation:
    return Observation(task_id="parse_log", seed=0, step=0, output="")


def test_scripted_agent_replays_the_reference_then_idles() -> None:
    task = get_task("parse_log")
    agent = ScriptedAgent()
    agent.start(task, 4)
    assert agent.act(_obs()) == task.reference_solution(4)[0]
    assert agent.act(_obs()) == ShellAction(command="ls")


def test_random_agent_is_deterministic_per_seed() -> None:
    task = get_task("parse_log")
    runs = []
    for seed in (1, 1, 2):
        agent = RandomAgent()
        agent.start(task, seed)
        runs.append([agent.act(_obs()) for _ in range(8)])
    assert runs[0] == runs[1]
    assert runs[0] != runs[2]
    assert len({a.kind for a in runs[0] + runs[2]}) >= 3


def test_make_agent() -> None:
    assert make_agent("random").name == "random"
    assert make_agent("scripted").name == "scripted"
    with pytest.raises(ConfigError) as excinfo:
        make_agent("oracle")
    assert excinfo.value.details["available"] == ["random", "scripted"]
