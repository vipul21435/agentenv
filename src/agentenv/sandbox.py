"""Per-episode subprocess sandbox.

Every episode gets a fresh temporary workspace directory. Commands run as child processes
with that directory as their working directory, a minimal environment, a wall-clock
timeout and a cap on captured output. File actions resolve paths relative to the
workspace and refuse anything that escapes it (``../``, absolute paths, symlinks out).

This backend isolates *state* (one directory per episode) but not the host: a command can
still read the host file system or open network connections. The next backend is the
Docker sandbox (``docker run --network=none`` with CPU, memory and pids limits) built on
the upstream ``minisweagent.environments.docker`` executor; the ``Settings.sandbox_backend``
switch already exists for it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pydantic import BaseModel, Field

from agentenv.errors import SandboxError

TRUNCATION_MARKER = "\n...[truncated {dropped} chars]"


class SandboxLimits(BaseModel):
    """Resource limits applied to every command and file action."""

    timeout_seconds: float = Field(default=10.0, gt=0)
    max_output_chars: int = Field(default=4000, ge=100)
    max_file_chars: int = Field(default=20000, ge=100)


class CommandResult(BaseModel):
    """Captured outcome of one sandboxed command."""

    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool = False
    duration_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def render(self) -> str:
        """Combine stdout and stderr into the text an agent observes."""
        parts = [self.stdout.rstrip("\n")]
        if self.stderr.strip():
            parts.append("[stderr]\n" + self.stderr.rstrip("\n"))
        if self.timed_out:
            parts.append(f"[timed out after {self.duration_seconds:.1f}s]")
        elif self.returncode not in (0, None):
            parts.append(f"[exit code {self.returncode}]")
        return "\n".join(part for part in parts if part)


def truncate(text: str, limit: int) -> str:
    """Cut ``text`` to ``limit`` characters and say how much was dropped."""
    if len(text) <= limit:
        return text
    return text[:limit] + TRUNCATION_MARKER.format(dropped=len(text) - limit)


def _as_text(data: str | bytes | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


class SubprocessSandbox:
    """Run shell commands, Python snippets and file operations confined to one workspace."""

    def __init__(self, workspace: Path, limits: SandboxLimits | None = None) -> None:
        self.workspace = workspace.resolve()
        self.limits = limits or SandboxLimits()

    @classmethod
    def create(cls, root: Path | None = None, limits: SandboxLimits | None = None) -> SubprocessSandbox:
        """Create a fresh temporary workspace (under ``root`` if given) and wrap it."""
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
        try:
            workspace = Path(tempfile.mkdtemp(prefix="agentenv-", dir=root))
        except OSError as exc:
            raise SandboxError(f"cannot create workspace: {exc}", details={"root": str(root)}) from exc
        return cls(workspace, limits)

    def resolve(self, path: str) -> Path:
        """Resolve ``path`` inside the workspace; raise ``SandboxError`` if it escapes."""
        candidate = (self.workspace / path).resolve()
        if candidate != self.workspace and not candidate.is_relative_to(self.workspace):
            raise SandboxError(f"path escapes the workspace: {path!r}", details={"path": path})
        return candidate

    def read_file(self, path: str) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise SandboxError(f"no such file: {path!r}", details={"path": path})
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise SandboxError(f"cannot read {path!r}: {exc}", details={"path": path}) from exc
        return truncate(text, self.limits.max_file_chars)

    def write_file(self, path: str, content: str) -> int:
        """Write ``content`` (creating parent directories) and return the number of characters."""
        target = self.resolve(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise SandboxError(f"cannot write {path!r}: {exc}", details={"path": path}) from exc
        return len(content)

    def run(self, argv: list[str], *, timeout: float | None = None, stdin: str | None = None) -> CommandResult:
        """Run ``argv`` in the workspace with a wall-clock timeout and capped output.

        ``stdin`` is fed to the child's standard input (an empty pipe when ``None``), so
        the child never inherits the parent's terminal.
        """
        limit = timeout or self.limits.timeout_seconds
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.workspace),
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                argv,
                cwd=self.workspace,
                env=env,
                input=stdin if stdin is not None else "",
                capture_output=True,
                text=True,
                errors="replace",
                timeout=limit,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                stdout=truncate(_as_text(exc.stdout), self.limits.max_output_chars),
                stderr=truncate(_as_text(exc.stderr), self.limits.max_output_chars),
                returncode=None,
                timed_out=True,
                duration_seconds=time.perf_counter() - start,
            )
        except OSError as exc:
            raise SandboxError(f"cannot start {argv[0]!r}: {exc}", details={"argv": argv}) from exc
        return CommandResult(
            stdout=truncate(completed.stdout, self.limits.max_output_chars),
            stderr=truncate(completed.stderr, self.limits.max_output_chars),
            returncode=completed.returncode,
            duration_seconds=time.perf_counter() - start,
        )

    def shell(self, command: str, *, timeout: float | None = None) -> CommandResult:
        return self.run(["bash", "-c", command], timeout=timeout)

    def python(self, code: str, *, timeout: float | None = None) -> CommandResult:
        """Run ``code`` with the interpreter that runs agentenv; the workspace is on ``sys.path``.

        The code goes in on stdin (``python -``) rather than ``-c``, so it never appears in
        the child's argv where anything the code imports could read it back.
        """
        return self.run([sys.executable, "-"], timeout=timeout, stdin=code)

    def close(self) -> None:
        """Delete the workspace. Safe to call more than once."""
        shutil.rmtree(self.workspace, ignore_errors=True)
