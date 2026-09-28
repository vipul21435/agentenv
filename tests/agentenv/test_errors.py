from __future__ import annotations

import pytest

from agentenv.errors import (
    AgentEnvError,
    ConfigError,
    EpisodeError,
    SandboxError,
    SandboxTimeoutError,
    TaskError,
)

ALL_ERRORS = [AgentEnvError, ConfigError, SandboxError, SandboxTimeoutError, TaskError, EpisodeError]


def test_hierarchy() -> None:
    for cls in ALL_ERRORS:
        assert issubclass(cls, AgentEnvError)
        assert issubclass(cls, Exception)
    assert issubclass(SandboxTimeoutError, SandboxError)


def test_codes_are_unique_and_stable() -> None:
    codes = [cls.code for cls in ALL_ERRORS]
    assert len(set(codes)) == len(codes)
    assert ConfigError.code == "config_error"
    assert SandboxTimeoutError.code == "sandbox_timeout"


def test_str_and_to_dict() -> None:
    err = SandboxError("container failed", details={"image": "python:3.12-slim"})
    assert str(err) == "container failed"
    assert err.to_dict() == {
        "code": "sandbox_error",
        "message": "container failed",
        "details": {"image": "python:3.12-slim"},
    }


def test_details_default_and_copy() -> None:
    assert TaskError("bad task").details == {}
    original = {"task_id": "t1"}
    err = TaskError("bad task", details=original)
    original["task_id"] = "changed"
    assert err.details == {"task_id": "t1"}
    err.to_dict()["details"]["task_id"] = "mutated"
    assert err.details == {"task_id": "t1"}


def test_raise_and_catch_by_base_class() -> None:
    with pytest.raises(AgentEnvError) as info:
        raise EpisodeError("step() called before reset()")
    assert info.value.code == "episode_error"
    assert info.value.message == "step() called before reset()"
