from __future__ import annotations

import shutil
import subprocess
from typing import Any

import pytest

from agentenv import __version__
from agentenv.doctor import DockerProbe, build_report, probe_docker
from agentenv.settings import load_settings


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["docker"], returncode, stdout=stdout, stderr=stderr)


def test_probe_reports_missing_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    probe = probe_docker("nope-docker")
    assert probe == DockerProbe(
        executable="nope-docker", found=False, available=False, error="'nope-docker' not found on PATH"
    )


def test_probe_reports_running_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/local/bin/docker")
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return _completed(0, stdout="29.3.1\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    probe = probe_docker()
    assert probe.available is True
    assert probe.version == "29.3.1"
    assert probe.error is None
    assert calls == [["/usr/local/bin/docker", "version", "--format", "{{.Server.Version}}"]]


def test_probe_reports_daemon_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: _completed(1, stderr="Cannot connect to the Docker daemon\n")
    )
    probe = probe_docker()
    assert probe.found is True
    assert probe.available is False
    assert probe.error == "Cannot connect to the Docker daemon"


def test_probe_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/docker")

    def slow_run(*_a: Any, **_k: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="docker", timeout=5)

    monkeypatch.setattr(subprocess, "run", slow_run)
    probe = probe_docker(timeout=5)
    assert probe.available is False
    assert probe.error is not None
    assert probe.error.startswith("TimeoutExpired")


def _unavailable(*_a: Any, **_k: Any) -> DockerProbe:
    return DockerProbe(executable="docker", found=False, available=False, error="not found")


def _available(*_a: Any, **_k: Any) -> DockerProbe:
    return DockerProbe(executable="docker", found=True, available=True, version="29.3.1")


def test_report_is_ok_with_offline_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentenv.doctor.probe_docker", _unavailable)
    report = build_report(load_settings(env_file=None))
    assert report.ok is True
    assert report.problems == []
    assert report.agentenv_version == __version__
    assert report.sandbox_backend == "subprocess"
    assert report.docker.available is False


def test_report_flags_docker_backend_without_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentenv.doctor.probe_docker", _unavailable)
    report = build_report(load_settings(env_file=None, sandbox_backend="docker"))
    assert report.ok is False
    assert len(report.problems) == 1
    assert "AGENTENV_SANDBOX_BACKEND=docker" in report.problems[0]


def test_report_accepts_docker_backend_with_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentenv.doctor.probe_docker", _available)
    report = build_report(load_settings(env_file=None, sandbox_backend="docker"))
    assert report.ok is True


def test_report_flags_openai_provider_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentenv.doctor.probe_docker", _unavailable)
    report = build_report(load_settings(env_file=None, model_provider="openai"))
    assert report.ok is False
    assert "OPENAI_API_KEY" in report.problems[0]
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    assert build_report(load_settings(env_file=None, model_provider="openai")).ok is True


@pytest.mark.docker
def test_probe_against_the_real_docker_daemon() -> None:
    probe = probe_docker()
    if not probe.available:
        pytest.skip(f"docker unavailable: {probe.error}")
    assert probe.found is True
    assert probe.version
