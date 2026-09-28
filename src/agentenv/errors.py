"""Error hierarchy for agentenv.

Every error carries a stable machine-readable ``code`` and a ``details`` mapping so that
the CLI, the API and trajectory logs can report failures uniformly through ``to_dict()``.
"""

from __future__ import annotations

from typing import Any


class AgentEnvError(Exception):
    """Base class for every error raised by agentenv."""

    code: str = "agentenv_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the error."""
        return {"code": self.code, "message": self.message, "details": dict(self.details)}

    def __str__(self) -> str:
        return self.message


class ConfigError(AgentEnvError):
    """Invalid settings, task definitions or command-line arguments."""

    code = "config_error"


class SandboxError(AgentEnvError):
    """A sandbox backend failed to start, execute or clean up."""

    code = "sandbox_error"


class SandboxTimeoutError(SandboxError):
    """A sandboxed command exceeded its time limit."""

    code = "sandbox_timeout"


class TaskError(AgentEnvError):
    """A task definition or its reward function is invalid."""

    code = "task_error"


class EpisodeError(AgentEnvError):
    """The reset/step protocol was violated, for example step() before reset()."""

    code = "episode_error"
