"""JSONL trajectory logging (one line per step) and the success-rate report built from it."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Self

from pydantic import BaseModel, Field, ValidationError

from agentenv.errors import ConfigError

EXCERPT_CHARS = 200


class TrajectoryStep(BaseModel):
    """One environment step as written to the JSONL file."""

    task_id: str
    seed: int
    agent: str
    episode: int = Field(ge=0)
    step: int = Field(ge=1)
    action: dict[str, Any]
    observation_excerpt: str
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)

    @property
    def episode_key(self) -> tuple[str, str, int, int]:
        return (self.task_id, self.agent, self.seed, self.episode)


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


class TrajectoryWriter:
    """Append-only JSONL writer; each ``write`` is flushed so partial runs stay readable."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.count = 0
        self._handle: IO[str] | None = None

    def open(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        return self

    def write(self, step: TrajectoryStep) -> None:
        if self._handle is None:
            raise ConfigError("trajectory writer is not open", details={"path": str(self.path)})
        self._handle.write(step.model_dump_json() + "\n")
        self._handle.flush()
        self.count += 1

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()


def read_trajectory(path: Path) -> list[TrajectoryStep]:
    """Parse a JSONL trajectory file; raise ``ConfigError`` with the line number on bad input."""
    if not path.is_file():
        raise ConfigError(f"no such trajectory file: {path}", details={"path": str(path)})
    steps: list[TrajectoryStep] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                steps.append(TrajectoryStep.model_validate(json.loads(line)))
            except (ValueError, ValidationError) as exc:
                raise ConfigError(f"bad trajectory line {number}: {exc}", details={"line": number}) from exc
    return steps


class GroupReport(BaseModel):
    """Aggregate over the episodes of one (task, agent) pair."""

    task_id: str
    agent: str
    episodes: int
    successes: int
    success_rate: float
    mean_steps: float
    mean_reward: float


class AgentReport(BaseModel):
    """One agent over every task it ran: the number to compare agents by."""

    agent: str
    tasks: int
    episodes: int
    successes: int
    success_rate: float
    mean_steps: float
    mean_reward: float
    mean_action_seconds: float


class Report(BaseModel):
    steps: int
    episodes: int
    groups: list[GroupReport]
    agents: list[AgentReport] = Field(default_factory=list)

    def render(self) -> str:
        """Fixed-width table for the terminal: one row per (task, agent), then one per agent over all tasks."""
        header = f"{'task':<20} {'agent':<10} {'episodes':>8} {'success':>8} {'mean_steps':>10} {'mean_reward':>11}"
        lines = [header, "-" * len(header)]
        for g in self.groups:
            lines.append(
                f"{g.task_id:<20} {g.agent:<10} {g.episodes:>8} {g.success_rate:>8.0%} "
                f"{g.mean_steps:>10.2f} {g.mean_reward:>11.3f}"
            )
        if self.agents:
            lines.append("-" * len(header))
        for a in self.agents:
            lines.append(
                f"{f'all ({a.tasks} tasks)':<20} {a.agent:<10} {a.episodes:>8} {a.success_rate:>8.0%} "
                f"{a.mean_steps:>10.2f} {a.mean_reward:>11.3f}  {a.mean_action_seconds:.3f}s/action"
            )
        lines.append(f"{self.episodes} episodes, {self.steps} steps")
        return "\n".join(lines)


def build_report(steps: Iterable[TrajectoryStep]) -> Report:
    """Success = the episode reached reward 1.0; steps and reward are per-episode means."""
    by_episode: dict[tuple[str, str, int, int], list[TrajectoryStep]] = defaultdict(list)
    total = 0
    for step in steps:
        by_episode[step.episode_key].append(step)
        total += 1
    by_group: dict[tuple[str, str], list[list[TrajectoryStep]]] = defaultdict(list)
    for key, episode_steps in by_episode.items():
        by_group[(key[0], key[1])].append(episode_steps)
    groups = []
    for (task_id, agent), episodes in sorted(by_group.items()):
        finals = [max(e, key=lambda s: s.step) for e in episodes]
        successes = sum(any(s.reward >= 1.0 for s in e) for e in episodes)
        groups.append(
            GroupReport(
                task_id=task_id,
                agent=agent,
                episodes=len(episodes),
                successes=successes,
                success_rate=successes / len(episodes),
                mean_steps=statistics.fmean(f.step for f in finals),
                mean_reward=statistics.fmean(f.reward for f in finals),
            )
        )
    agents = []
    for agent in sorted({key[1] for key in by_group}):
        episodes = [e for (_, name), group in by_group.items() if name == agent for e in group]
        finals = [max(e, key=lambda s: s.step) for e in episodes]
        durations = [float(s.info.get("duration_seconds", 0.0)) for e in episodes for s in e]
        successes = sum(any(s.reward >= 1.0 for s in e) for e in episodes)
        agents.append(
            AgentReport(
                agent=agent,
                tasks=sum(name == agent for _, name in by_group),
                episodes=len(episodes),
                successes=successes,
                success_rate=successes / len(episodes),
                mean_steps=statistics.fmean(f.step for f in finals),
                mean_reward=statistics.fmean(f.reward for f in finals),
                mean_action_seconds=statistics.fmean(durations) if durations else 0.0,
            )
        )
    return Report(steps=total, episodes=len(by_episode), groups=groups, agents=agents)
