from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agentenv import __version__
from agentenv.cli import EXIT_CONFIG, EXIT_PROBLEMS, app
from agentenv.doctor import DockerProbe

runner = CliRunner()


def _probe(available: bool) -> Any:
    def fake(*_a: Any, **_k: Any) -> DockerProbe:
        if available:
            return DockerProbe(executable="docker", found=True, available=True, version="1.2.3")
        return DockerProbe(executable="docker", found=False, available=False, error="not found")

    return fake


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"agentenv {__version__}"


def test_no_arguments_prints_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output
    assert "doctor" in result.output


def test_doctor_json_with_offline_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentenv.doctor.probe_docker", _probe(available=False))
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["sandbox_backend"] == "subprocess"
    assert report["docker"]["available"] is False


def test_doctor_text_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agentenv.doctor.probe_docker", _probe(available=True))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "docker (docker): available (server 1.2.3)" in result.stdout
    assert result.stdout.strip().endswith("ok")


def test_doctor_fails_when_docker_backend_has_no_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTENV_SANDBOX_BACKEND", "docker")
    monkeypatch.setattr("agentenv.doctor.probe_docker", _probe(available=False))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == EXIT_PROBLEMS
    assert "problem: AGENTENV_SANDBOX_BACKEND=docker" in result.stdout
    assert result.stdout.strip().endswith("not ok")


def test_doctor_logs_a_structured_event(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTENV_LOG_FORMAT", "json")
    monkeypatch.setattr("agentenv.doctor.probe_docker", _probe(available=False))
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    events = [json.loads(line) for line in result.stderr.splitlines() if line.startswith("{")]
    assert [event["event"] for event in events] == ["doctor.report"]
    assert events[0]["ok"] is True


def test_settings_command_masks_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-do-not-print")
    monkeypatch.setenv("AGENTENV_SEED", "11")
    result = runner.invoke(app, ["settings"])
    assert result.exit_code == 0, result.output
    assert "sk-do-not-print" not in result.output
    payload = json.loads(result.stdout)
    assert payload["openai_api_key"] == "***"
    assert payload["seed"] == 11


def test_invalid_settings_exit_with_config_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTENV_SEED", "-3")
    result = runner.invoke(app, ["settings"])
    assert result.exit_code == EXIT_CONFIG
    error = json.loads(result.stderr.strip().splitlines()[-1])
    assert error["code"] == "config_error"
    assert error["details"]["errors"][0]["field"] == "seed"


def test_python_module_entry_point() -> None:
    env = {**os.environ, "MSWEA_SILENT_STARTUP": "1"}
    completed = subprocess.run(
        [sys.executable, "-m", "agentenv", "version"], capture_output=True, text=True, timeout=60, env=env, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == f"agentenv {__version__}"


def test_tasks_command_lists_the_suite() -> None:
    result = runner.invoke(app, ["tasks"])
    assert result.exit_code == 0
    assert [line.split()[0] for line in result.stdout.splitlines()] == [
        "fix_checksum",
        "summarize_numbers",
        "parse_log",
    ]


def test_run_and_report_commands(tmp_path: Path) -> None:
    out = tmp_path / "t.jsonl"
    result = runner.invoke(
        app,
        [
            "run",
            "--task",
            "parse_log",
            "--agent",
            "scripted",
            "--agent",
            "random",
            "--episodes",
            "2",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "parse_log            scripted          2     100%" in result.stdout
    assert "parse_log            random            2       0%" in result.stdout
    assert "per episode" in result.stdout
    report = runner.invoke(app, ["report", str(out), "--json"])
    assert report.exit_code == 0
    assert json.loads(report.stdout)["episodes"] == 4


def test_run_rejects_unknown_task(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "--task", "nope", "--out", str(tmp_path / "t.jsonl")])
    assert result.exit_code == 2
    assert "unknown task" in result.output


def test_report_rejects_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["report", str(tmp_path / "missing.jsonl")])
    assert result.exit_code == 2
    assert "no such trajectory file" in result.output
