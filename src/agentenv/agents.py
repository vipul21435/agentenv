"""Baseline agents: a scripted agent that applies the reference solution and a random floor."""

from __future__ import annotations

import random
from collections import deque
from typing import Protocol

from agentenv.actions import Action, PythonAction, ReadFileAction, ShellAction, WriteFileAction
from agentenv.env import Observation
from agentenv.errors import ConfigError
from agentenv.tasks import Task


class Agent(Protocol):
    """What the episode runner needs from an agent."""

    name: str

    def start(self, task: Task, seed: int) -> None:
        """Called once per episode before the first ``act``."""

    def act(self, observation: Observation) -> Action:
        """Choose the next action given the latest observation."""


class ScriptedAgent:
    """Applies the task's reference solution step by step (the expected ceiling)."""

    name = "scripted"

    def __init__(self) -> None:
        self._queue: deque[Action] = deque()

    def start(self, task: Task, seed: int) -> None:
        self._queue = deque(task.reference_solution(seed))

    def act(self, observation: Observation) -> Action:
        if self._queue:
            return self._queue.popleft()
        return ShellAction(command="ls")


class RandomAgent:
    """Picks seeded random actions from a small pool (the expected floor)."""

    name = "random"
    _FILES = ("TASK.md", "numbers.txt", "app.log", "mathlib.py", "missing.txt")

    def __init__(self) -> None:
        self._rng = random.Random(0)
        self._writes = 0

    def start(self, task: Task, seed: int) -> None:
        self._rng = random.Random(f"random-agent:{task.id}:{seed}")
        self._writes = 0

    def act(self, observation: Observation) -> Action:
        choice = self._rng.randrange(5)
        if choice == 0:
            return ShellAction(command=self._rng.choice(["ls -la", "cat TASK.md", "wc -l *", "head -n 3 *"]))
        if choice == 1:
            return ReadFileAction(path=self._rng.choice(self._FILES))
        if choice == 2:
            self._writes += 1
            return WriteFileAction(path=f"note_{self._writes}.txt", content=f"random note {self._rng.randint(0, 999)}")
        if choice == 3:
            return PythonAction(code=f"print({self._rng.randint(0, 999)} * {self._rng.randint(0, 999)})")
        return ShellAction(command=f"echo {self._rng.randint(0, 999)} > scratch.txt")


AGENTS: dict[str, type[ScriptedAgent] | type[RandomAgent]] = {
    ScriptedAgent.name: ScriptedAgent,
    RandomAgent.name: RandomAgent,
}


def make_agent(name: str) -> Agent:
    """Instantiate an agent by name; raise ``ConfigError`` naming the known agents otherwise."""
    try:
        return AGENTS[name]()
    except KeyError:
        raise ConfigError(f"unknown agent: {name!r}", details={"available": sorted(AGENTS)}) from None
