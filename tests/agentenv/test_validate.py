"""Fail-to-pass validator: the shipped suite passes; broken, too-easy, flaky and non-deterministic tasks are caught."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentenv.actions import Action, ShellAction, WriteFileAction
from agentenv.cli import EXIT_CONFIG, EXIT_PROBLEMS, app
from agentenv.sandbox import SubprocessSandbox
from agentenv.tasks import Task, get_task, list_tasks
from agentenv.validate import ValidationReport, validate_task, validate_tasks, workspace_digest

runner = CliRunner()


def _setup(workspace: Path, seed: int) -> None:
    (workspace / "TASK.md").write_text(f"seed {seed}\n")


def _reference(seed: int) -> list[Action]:
    return [WriteFileAction(path="done.txt", content="done")]


def _reward_done(sandbox: SubprocessSandbox, seed: int) -> float:
    return 1.0 if (sandbox.workspace / "done.txt").is_file() else 0.0


def _task(task_id: str, **overrides: object) -> Task:
    fields: dict[str, object] = {
        "id": task_id,
        "description": "synthetic",
        "setup": _setup,
        "reward": _reward_done,
        "reference_solution": _reference,
    }
    fields.update(overrides)
    return Task.model_validate(fields)


def test_shipped_suite_passes_with_repeats(tmp_path: Path) -> None:
    report = validate_tasks(list_tasks(), seeds=[0, 7], repeats=2, workspace_root=tmp_path)
    assert report.ok
    assert report.counts() == {"pass": len(list_tasks()), "fail": 0, "flaky": 0}
    assert all(task.checks == {"reference": "pass", "baseline": "pass", "setup": "pass"} for task in report.tasks)
    for task in report.tasks:
        assert len(task.runs) == 2 * 2 * 2 + 2  # reference and baseline per seed and repeat, setup per seed
    text = report.render()
    assert "10 tasks, 2 seeds x 2 repeats: 10 pass, 0 fail, 0 flaky" in text
    assert "data_pipeline               4/4       4/4       2/2  pass" in text


def test_synthetic_task_passes(tmp_path: Path) -> None:
    verdict = validate_task(_task("good"), seeds=[0], workspace_root=tmp_path)
    assert verdict.verdict == "pass"
    assert verdict.failures == []


def test_broken_reference_fails(tmp_path: Path) -> None:
    def never(sandbox: SubprocessSandbox, seed: int) -> float:
        return 0.0

    verdict = validate_task(_task("broken", reward=never), seeds=[0, 1], workspace_root=tmp_path)
    assert verdict.verdict == "fail"
    assert verdict.checks == {"reference": "fail", "baseline": "pass", "setup": "pass"}
    assert [run.detail for run in verdict.failures] == ["reference ended with reward 0.000 after 1 of 1 actions"] * 2


def test_too_easy_task_fails_the_baseline(tmp_path: Path) -> None:
    def always(sandbox: SubprocessSandbox, seed: int) -> float:
        return 1.0

    verdict = validate_task(_task("easy", reward=always), seeds=[0], workspace_root=tmp_path)
    assert verdict.verdict == "fail"
    assert verdict.checks["baseline"] == "fail"
    assert verdict.checks["reference"] == "pass"
    assert "untouched workspace scored 1.000" in verdict.failures[0].detail


def test_flaky_reward_is_reported_as_flaky(tmp_path: Path) -> None:
    calls: list[int] = []

    def flaky(sandbox: SubprocessSandbox, seed: int) -> float:
        if not (sandbox.workspace / "done.txt").is_file():
            return 0.0
        calls.append(1)
        return float(len(calls) % 2)  # alternates on every solved run

    verdict = validate_task(_task("flaky", reward=flaky), seeds=[0], repeats=4, workspace_root=tmp_path)
    assert verdict.verdict == "flaky"
    assert verdict.checks["reference"] == "flaky"
    assert verdict.checks["baseline"] == "pass"


def test_nondeterministic_setup_fails(tmp_path: Path) -> None:
    counter: list[int] = []

    def drifting_setup(workspace: Path, seed: int) -> None:
        counter.append(1)
        (workspace / "TASK.md").write_text(f"run {len(counter)}\n")

    verdict = validate_task(_task("drift", setup=drifting_setup), seeds=[0], repeats=2, workspace_root=tmp_path)
    assert verdict.checks["setup"] == "fail"
    assert verdict.verdict == "fail"
    assert "setup differed across 4 resets" in verdict.failures[0].detail


def test_nondeterministic_setup_fails_at_one_repeat(tmp_path: Path) -> None:
    """The default --repeats 1 still compares two resets per seed (reference and baseline)."""
    counter: list[int] = []

    def drifting_setup(workspace: Path, seed: int) -> None:
        counter.append(1)
        (workspace / "TASK.md").write_text(f"run {len(counter)}\n")

    verdict = validate_task(_task("drift", setup=drifting_setup), seeds=[0, 1], repeats=1, workspace_root=tmp_path)
    assert verdict.checks["setup"] == "fail"
    assert "setup differed across 2 resets" in verdict.failures[0].detail


def test_reference_that_raises_sandbox_error_is_a_failure(tmp_path: Path) -> None:
    def escaping(seed: int) -> list[Action]:
        return [WriteFileAction(path="../escape.txt", content="x"), ShellAction(command="true")]

    verdict = validate_task(_task("escape", reference_solution=escaping), seeds=[0], workspace_root=tmp_path)
    assert verdict.checks["reference"] == "fail"
    assert verdict.failures[0].steps == 2  # the escape becomes an observation, the episode continues


def test_reference_longer_than_max_steps_fails(tmp_path: Path) -> None:
    def two_writes(seed: int) -> list[Action]:
        return [WriteFileAction(path="first.txt", content="1"), WriteFileAction(path="done.txt", content="2")]

    verdict = validate_task(
        _task("short", max_steps=1, reference_solution=two_writes), seeds=[0], workspace_root=tmp_path
    )
    assert verdict.checks["reference"] == "fail"
    assert verdict.failures[0].detail == "reference ended with reward 0.000 after 1 of 2 actions"


def test_workspace_digest_covers_nested_files(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.txt").write_text("1")
    before = workspace_digest(tmp_path)
    assert before == workspace_digest(tmp_path)
    (tmp_path / "a" / "x.txt").write_text("2")
    assert before != workspace_digest(tmp_path)


def test_cli_validate_tasks_text_and_json(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["validate-tasks", "--task", "fix_checksum,rename_files", "--seeds", "1", "--seed", "5"]
    )
    assert result.exit_code == 0, result.output
    assert "2 tasks, 1 seeds x 1 repeats: 2 pass, 0 fail, 0 flaky" in result.stdout
    result = runner.invoke(app, ["validate-tasks", "--task", "parse_log", "--seeds", "2", "--repeats", "2", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["seeds"] == [0, 1] and payload["repeats"] == 2
    assert payload["tasks"][0]["verdict"] == "pass"


def test_cli_validate_tasks_exit_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    result = runner.invoke(app, ["validate-tasks", "--task", "nope"])
    assert result.exit_code == EXIT_CONFIG
    assert json.loads(result.stderr.strip().splitlines()[-1])["code"] == "task_error"

    def never(sandbox: SubprocessSandbox, seed: int) -> float:
        return 0.0

    broken = get_task("fix_checksum").model_copy(update={"reward": never})
    monkeypatch.setattr("agentenv.cli.select_tasks", lambda spec: [broken])
    result = runner.invoke(app, ["validate-tasks", "--seeds", "1"])
    assert result.exit_code == EXIT_PROBLEMS
    assert "fix_checksum                0/1       1/1       1/1  fail" in result.stdout


def test_report_with_no_tasks_is_not_ok() -> None:
    assert ValidationReport(seeds=[0], repeats=1, tasks=[]).ok is False
