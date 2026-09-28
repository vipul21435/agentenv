"""Subprocess sandbox: confinement, timeouts and output caps."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from agentenv.errors import SandboxError
from agentenv.sandbox import SandboxLimits, SubprocessSandbox, truncate


@pytest.fixture
def sandbox(tmp_path: Path) -> Iterator[SubprocessSandbox]:
    box = SubprocessSandbox.create(root=tmp_path / "ws", limits=SandboxLimits(timeout_seconds=2, max_output_chars=100))
    yield box
    box.close()


def test_create_makes_a_fresh_workspace_under_root(sandbox: SubprocessSandbox, tmp_path: Path) -> None:
    assert sandbox.workspace.is_dir()
    assert sandbox.workspace.parent == (tmp_path / "ws").resolve()


def test_write_then_read_roundtrip(sandbox: SubprocessSandbox) -> None:
    assert sandbox.write_file("sub/dir/a.txt", "hello") == 5
    assert sandbox.read_file("sub/dir/a.txt") == "hello"


@pytest.mark.parametrize("path", ["../escape.txt", "/etc/passwd", "sub/../../x"])
def test_paths_outside_the_workspace_are_rejected(sandbox: SubprocessSandbox, path: str) -> None:
    with pytest.raises(SandboxError) as excinfo:
        sandbox.resolve(path)
    assert excinfo.value.details == {"path": path}


def test_read_missing_file_is_an_error(sandbox: SubprocessSandbox) -> None:
    with pytest.raises(SandboxError, match="no such file"):
        sandbox.read_file("nope.txt")


def test_shell_runs_in_the_workspace(sandbox: SubprocessSandbox) -> None:
    sandbox.write_file("marker", "x")
    result = sandbox.shell('ls && basename "$(pwd)"')
    assert result.ok
    assert "marker" in result.stdout
    assert result.stdout.strip().endswith(sandbox.workspace.name)


def test_python_sees_workspace_modules(sandbox: SubprocessSandbox) -> None:
    sandbox.write_file("mod.py", "VALUE = 41 + 1\n")
    result = sandbox.python("import mod; print(mod.VALUE)")
    assert result.ok
    assert result.stdout.strip() == "42"


def test_nonzero_exit_and_stderr_are_rendered(sandbox: SubprocessSandbox) -> None:
    result = sandbox.shell("echo out; echo err >&2; exit 3")
    assert not result.ok
    assert result.returncode == 3
    assert result.render() == "out\n[stderr]\nerr\n[exit code 3]"


def test_timeout_is_enforced(sandbox: SubprocessSandbox) -> None:
    result = sandbox.shell("sleep 5", timeout=0.3)
    assert result.timed_out
    assert result.returncode is None
    assert result.duration_seconds < 3
    assert "[timed out after" in result.render()


def test_output_is_capped(sandbox: SubprocessSandbox) -> None:
    result = sandbox.python("print('x' * 1000)")
    assert len(result.stdout) < 200
    assert "[truncated 901 chars]" in result.stdout


def test_truncate_keeps_short_text() -> None:
    assert truncate("abc", 10) == "abc"


def test_close_is_idempotent(sandbox: SubprocessSandbox) -> None:
    sandbox.close()
    sandbox.close()
    assert not sandbox.workspace.exists()
