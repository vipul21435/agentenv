"""Local toolchain report: Python, docker availability and the effective sandbox backend."""

from __future__ import annotations

import platform
import shutil
import subprocess

from pydantic import BaseModel

from agentenv import __version__
from agentenv.settings import Settings


class DockerProbe(BaseModel):
    """Result of probing the docker (or compatible) executable."""

    executable: str
    found: bool
    available: bool
    version: str | None = None
    error: str | None = None


class DoctorReport(BaseModel):
    """What ``agentenv doctor`` prints."""

    agentenv_version: str
    python_version: str
    platform: str
    sandbox_backend: str
    model_provider: str
    docker: DockerProbe
    ok: bool
    problems: list[str]


def probe_docker(executable: str = "docker", timeout: float = 5.0) -> DockerProbe:
    """Check whether ``executable`` exists and can reach a running daemon."""
    path = shutil.which(executable)
    if path is None:
        return DockerProbe(
            executable=executable, found=False, available=False, error=f"{executable!r} not found on PATH"
        )
    try:
        result = subprocess.run(
            [path, "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return DockerProbe(executable=executable, found=True, available=False, error=f"{type(exc).__name__}: {exc}")
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip() or f"exit code {result.returncode}"
        return DockerProbe(executable=executable, found=True, available=False, error=message)
    return DockerProbe(executable=executable, found=True, available=True, version=result.stdout.strip() or None)


def build_report(settings: Settings) -> DoctorReport:
    """Probe the machine and list configuration problems for the given settings."""
    docker = probe_docker(settings.docker_executable)
    problems: list[str] = []
    if settings.sandbox_backend == "docker" and not docker.available:
        problems.append(
            "AGENTENV_SANDBOX_BACKEND=docker but no docker daemon is reachable; "
            "start docker or use AGENTENV_SANDBOX_BACKEND=subprocess"
        )
    if settings.model_provider == "openai" and settings.openai_api_key is None and settings.openai_base_url is None:
        problems.append("AGENTENV_MODEL_PROVIDER=openai needs OPENAI_API_KEY (or OPENAI_BASE_URL for a local server)")
    return DoctorReport(
        agentenv_version=__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        sandbox_backend=settings.sandbox_backend,
        model_provider=settings.model_provider,
        docker=docker,
        ok=not problems,
        problems=problems,
    )
