"""Run episodes of an agent on a task and log every step to a trajectory file."""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, Field

from agentenv.agents import Agent, make_agent
from agentenv.env import Environment
from agentenv.errors import TaskError
from agentenv.logs import get_logger, log_event
from agentenv.sandbox import SandboxLimits
from agentenv.tasks import Task, get_task, list_tasks
from agentenv.trajectory import TrajectoryStep, TrajectoryWriter, excerpt

log = get_logger("runner")


class EpisodeSummary(BaseModel):
    task_id: str
    agent: str
    seed: int
    episode: int = Field(ge=0)
    steps: int = Field(ge=0)
    final_reward: float = Field(ge=0.0, le=1.0)
    success: bool
    seconds: float = Field(ge=0.0)


def run_episode(
    env: Environment, agent: Agent, *, seed: int, episode: int = 0, writer: TrajectoryWriter | None = None
) -> EpisodeSummary:
    """Reset ``env`` with ``seed`` and let ``agent`` act until the episode is done."""
    started = time.perf_counter()
    observation = env.reset(seed)
    agent.start(env.task, seed)
    steps, reward, done = 0, 0.0, False
    while not done:
        action = agent.act(observation)
        result = env.step(action)
        steps, reward, done, observation = steps + 1, result.reward, result.done, result.observation
        if writer is not None:
            writer.write(
                TrajectoryStep(
                    task_id=env.task.id,
                    seed=seed,
                    agent=agent.name,
                    episode=episode,
                    step=steps,
                    action=action.model_dump(),
                    observation_excerpt=excerpt(observation.output),
                    reward=reward,
                    done=done,
                    info=result.info,
                )
            )
    summary = EpisodeSummary(
        task_id=env.task.id,
        agent=agent.name,
        seed=seed,
        episode=episode,
        steps=steps,
        final_reward=reward,
        success=reward >= 1.0,
        seconds=time.perf_counter() - started,
    )
    log_event(log, "episode.done", **summary.model_dump())
    return summary


def select_tasks(spec: str) -> list[Task]:
    """``all`` or a comma-separated list of task ids; duplicates are dropped, an empty selection is a TaskError."""
    if spec.strip() == "all":
        return list_tasks()
    ids = list(dict.fromkeys(part.strip() for part in spec.split(",") if part.strip()))
    if not ids:
        raise TaskError("no tasks selected", details={"spec": spec})
    return [get_task(task_id) for task_id in ids]


def run_suite(
    *,
    tasks: list[Task],
    agent_names: list[str],
    episodes: int,
    seed: int,
    out: Path | None,
    limits: SandboxLimits | None = None,
    workspace_root: Path | None = None,
) -> list[EpisodeSummary]:
    """Every agent on every task for ``episodes`` seeds starting at ``seed``; steps go to ``out``."""
    summaries: list[EpisodeSummary] = []
    writer = TrajectoryWriter(out).open() if out is not None else None
    try:
        for task in tasks:
            with Environment(task, limits=limits, workspace_root=workspace_root) as env:
                for name in agent_names:
                    agent = make_agent(name)
                    for episode in range(episodes):
                        summaries.append(run_episode(env, agent, seed=seed + episode, episode=episode, writer=writer))
    finally:
        if writer is not None:
            writer.close()
    return summaries
