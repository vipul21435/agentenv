"""AgentEnv: a gym-style environment for tool-using LLM agents with verifiable rewards.

The package is built on top of mini-swe-agent (https://github.com/SWE-agent/mini-swe-agent);
the upstream ``minisweagent`` package ships unchanged next to this one and provides the
subprocess and Docker command executors that the sandbox tools wrap.
"""

from agentenv.errors import (
    AgentEnvError,
    ConfigError,
    EpisodeError,
    SandboxError,
    SandboxTimeoutError,
    TaskError,
)

__version__ = "0.1.0"

__all__ = [
    "AgentEnvError",
    "ConfigError",
    "EpisodeError",
    "SandboxError",
    "SandboxTimeoutError",
    "TaskError",
    "__version__",
]
