"""The gym-style environment: ``reset(seed) -> Observation`` and ``step(Action) -> StepResult``.

Actions are a discriminated union (``shell``, ``read_file``, ``write_file``, ``python``)
executed in a per-episode :class:`~agentenv.sandbox.SubprocessSandbox`. After every step
the task's reward function grades the workspace; the episode ends when the reward reaches
1.0 or ``task.max_steps`` is exhausted.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any, Self

from pydantic import BaseModel, Field

from agentenv.actions import (
    ACTION_ADAPTER,
    Action,
    PythonAction,
    ReadFileAction,
    ShellAction,
    WriteFileAction,
    parse_action,
)
from agentenv.errors import EpisodeError, SandboxError
from agentenv.sandbox import CommandResult, SandboxLimits, SubprocessSandbox
from agentenv.tasks import Task


class Observation(BaseModel):
    task_id: str
    seed: int
    step: int = Field(ge=0)
    output: str
    returncode: int | None = None
    timed_out: bool = False
    description: str = ""


class StepResult(BaseModel):
    observation: Observation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class Environment:
    """One task, one workspace per episode, graded after every step."""

    def __init__(self, task: Task, *, limits: SandboxLimits | None = None, workspace_root: Path | None = None) -> None:
        self.task = task
        self.limits = limits or SandboxLimits()
        self.workspace_root = workspace_root
        self._sandbox: SubprocessSandbox | None = None
        self._seed = 0
        self._step = 0
        self._done = False

    @property
    def sandbox(self) -> SubprocessSandbox:
        if self._sandbox is None:
            raise EpisodeError("step() called before reset()", details={"task_id": self.task.id})
        return self._sandbox

    @property
    def step_count(self) -> int:
        return self._step

    def reset(self, seed: int) -> Observation:
        """Discard any previous workspace, run the seeded task setup and return the first observation."""
        self.close()
        self._sandbox = SubprocessSandbox.create(root=self.workspace_root, limits=self.limits)
        self._seed, self._step, self._done = seed, 0, False
        self.task.setup(self._sandbox.workspace, seed)
        listing = ", ".join(sorted(p.name for p in self._sandbox.workspace.iterdir()))
        return Observation(
            task_id=self.task.id,
            seed=seed,
            step=0,
            output=f"workspace files: {listing}",
            description=self.task.description,
        )

    def step(self, action: Action) -> StepResult:
        """Execute ``action``, grade the workspace and report reward/done."""
        sandbox = self.sandbox
        if self._done:
            raise EpisodeError("episode is finished; call reset()", details={"task_id": self.task.id})
        self._step += 1
        info: dict[str, Any] = {"action": action.kind, "max_steps": self.task.max_steps}
        try:
            result = self._execute(sandbox, action)
        except SandboxError as exc:
            result = CommandResult(stdout="", stderr=f"error: {exc.message}", returncode=1)
            info["error"] = exc.code
        info["duration_seconds"] = round(result.duration_seconds, 4)
        reward = min(1.0, max(0.0, float(self.task.reward(sandbox, self._seed))))
        self._done = reward >= 1.0 or self._step >= self.task.max_steps
        observation = Observation(
            task_id=self.task.id,
            seed=self._seed,
            step=self._step,
            output=result.render(),
            returncode=result.returncode,
            timed_out=result.timed_out,
        )
        return StepResult(observation=observation, reward=reward, done=self._done, info=info)

    @staticmethod
    def _execute(sandbox: SubprocessSandbox, action: Action) -> CommandResult:
        if isinstance(action, ShellAction):
            return sandbox.shell(action.command)
        if isinstance(action, PythonAction):
            return sandbox.python(action.code)
        if isinstance(action, ReadFileAction):
            return CommandResult(stdout=sandbox.read_file(action.path), stderr="", returncode=0)
        written = sandbox.write_file(action.path, action.content)
        return CommandResult(stdout=f"wrote {written} chars to {action.path}", stderr="", returncode=0)

    def close(self) -> None:
        if self._sandbox is not None:
            self._sandbox.close()
            self._sandbox = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()


__all__ = [
    "ACTION_ADAPTER",
    "Action",
    "Environment",
    "Observation",
    "PythonAction",
    "ReadFileAction",
    "ShellAction",
    "StepResult",
    "WriteFileAction",
    "parse_action",
]
