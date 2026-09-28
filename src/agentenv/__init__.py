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
from agentenv.settings import Settings, get_settings, load_settings

__version__ = "0.1.0"

__all__ = [
    "AgentEnvError",
    "ConfigError",
    "EpisodeError",
    "SandboxError",
    "SandboxTimeoutError",
    "Settings",
    "TaskError",
    "__version__",
    "get_settings",
    "load_settings",
]
