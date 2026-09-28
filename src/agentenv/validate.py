"""Fail-to-pass validation of the task suite through the real :class:`~agentenv.env.Environment`.

For every task, seed and repeat the validator runs two episodes: the reference solution
(must end with reward 1.0 within ``max_steps``) and the empty baseline (one read-only
``true`` command on the untouched workspace, must score 0.0). It also checks that
``reset(seed)`` writes a byte-identical workspace on every repeat. Per task and check the
verdict is ``pass`` (every run ok), ``fail`` (every run failed) or ``flaky`` (mixed).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agentenv.actions import ShellAction
from agentenv.env import Environment
from agentenv.logs import get_logger, log_event
from agentenv.sandbox import SandboxLimits
from agentenv.tasks import Task

log = get_logger("validate")

CheckKind = Literal["reference", "baseline", "setup"]
Verdict = Literal["pass", "fail", "flaky"]
CHECK_KINDS: tuple[CheckKind, ...] = ("reference", "baseline", "setup")


class CheckRun(BaseModel):
    """One episode (or one determinism comparison) of one check."""

    kind: CheckKind
    seed: int = Field(ge=0)
    repeat: int = Field(ge=0)
    ok: bool
    reward: float = Field(ge=0.0, le=1.0)
    steps: int = Field(ge=0)
    detail: str = ""


class TaskVerdict(BaseModel):
    task_id: str
    verdict: Verdict
    checks: dict[CheckKind, Verdict]
    runs: list[CheckRun]

    @property
    def failures(self) -> list[CheckRun]:
        return [run for run in self.runs if not run.ok]


class ValidationReport(BaseModel):
    seeds: list[int]
    repeats: int = Field(ge=1)
    tasks: list[TaskVerdict]

    @property
    def ok(self) -> bool:
        return bool(self.tasks) and all(task.verdict == "pass" for task in self.tasks)

    def counts(self) -> dict[Verdict, int]:
        return {verdict: sum(task.verdict == verdict for task in self.tasks) for verdict in ("pass", "fail", "flaky")}

    def render(self) -> str:
        """Fixed-width table with one line per task plus a summary line."""
        header = f"{'task':<21}{'reference':>10}{'baseline':>10}{'setup':>10}  verdict"
        lines = [header, "-" * len(header)]
        for task in self.tasks:
            cells = []
            for kind in CHECK_KINDS:
                runs = [run for run in task.runs if run.kind == kind]
                cells.append(f"{sum(run.ok for run in runs)}/{len(runs)}")
            lines.append(f"{task.task_id:<21}{cells[0]:>10}{cells[1]:>10}{cells[2]:>10}  {task.verdict}")
            lines.extend(f"  {run.kind} seed={run.seed} repeat={run.repeat}: {run.detail}" for run in task.failures)
        counts = self.counts()
        lines.append(
            f"{len(self.tasks)} tasks, {len(self.seeds)} seeds x {self.repeats} repeats: "
            f"{counts['pass']} pass, {counts['fail']} fail, {counts['flaky']} flaky"
        )
        return "\n".join(lines)


def workspace_digest(workspace: Path) -> str:
    """SHA-256 over every file path and content under ``workspace`` (order-independent)."""
    digest = hashlib.sha256()
    for path in sorted(p for p in workspace.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(workspace)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _reference_run(env: Environment, seed: int, repeat: int) -> tuple[CheckRun, str]:
    env.reset(seed)
    digest = workspace_digest(env.sandbox.workspace)
    actions = env.task.reference_solution(seed)
    reward, steps, done = 0.0, 0, False
    for action in actions:
        if done:  # max_steps reached (or solved early): the remaining actions are never sent
            break
        result = env.step(action)
        reward, steps, done = result.reward, steps + 1, result.done
    ok = reward >= 1.0 and done
    detail = "" if ok else f"reference ended with reward {reward:.3f} after {steps} of {len(actions)} actions"
    return CheckRun(
        kind="reference", seed=seed, repeat=repeat, ok=ok, reward=reward, steps=steps, detail=detail
    ), digest


def _baseline_run(env: Environment, seed: int, repeat: int) -> tuple[CheckRun, str]:
    env.reset(seed)
    digest = workspace_digest(env.sandbox.workspace)
    result = env.step(ShellAction(command="true"))
    ok = result.reward == 0.0
    detail = "" if ok else f"untouched workspace scored {result.reward:.3f}"
    return CheckRun(
        kind="baseline", seed=seed, repeat=repeat, ok=ok, reward=result.reward, steps=1, detail=detail
    ), digest


def _verdict(runs: Iterable[CheckRun]) -> Verdict:
    results = [run.ok for run in runs]
    if all(results):
        return "pass"
    return "fail" if not any(results) else "flaky"


def validate_task(
    task: Task,
    *,
    seeds: Sequence[int],
    repeats: int = 1,
    limits: SandboxLimits | None = None,
    workspace_root: Path | None = None,
) -> TaskVerdict:
    """Run the reference, baseline and setup checks for one task over ``seeds`` x ``repeats``."""
    runs: list[CheckRun] = []
    with Environment(task, limits=limits, workspace_root=workspace_root) as env:
        for seed in seeds:
            digests: list[str] = []  # one per reset: 2 x repeats, so never vacuous at repeats=1
            for repeat in range(repeats):
                reference, digest = _reference_run(env, seed, repeat)
                runs.append(reference)
                baseline, baseline_digest = _baseline_run(env, seed, repeat)
                runs.append(baseline)
                digests.extend((digest, baseline_digest))
            same = len(set(digests)) == 1
            runs.append(
                CheckRun(
                    kind="setup",
                    seed=seed,
                    repeat=0,
                    ok=same,
                    reward=0.0,
                    steps=0,
                    detail="" if same else f"setup differed across {len(digests)} resets",
                )
            )
    checks: dict[CheckKind, Verdict] = {}
    for kind in CHECK_KINDS:
        per_seed = [_verdict(run for run in runs if run.kind == kind and run.seed == seed) for seed in seeds]
        checks[kind] = "fail" if "fail" in per_seed else ("flaky" if "flaky" in per_seed else "pass")
    verdicts = list(checks.values())
    verdict: Verdict = "fail" if "fail" in verdicts else ("flaky" if "flaky" in verdicts else "pass")
    log_event(
        log,
        "validate.task",
        task_id=task.id,
        verdict=verdict,
        reference=checks["reference"],
        baseline=checks["baseline"],
        setup=checks["setup"],
    )
    return TaskVerdict(task_id=task.id, verdict=verdict, checks=checks, runs=runs)


def validate_tasks(
    tasks: Sequence[Task],
    *,
    seeds: Sequence[int],
    repeats: int = 1,
    limits: SandboxLimits | None = None,
    workspace_root: Path | None = None,
) -> ValidationReport:
    """Validate every task; ``report.ok`` is False when any task fails or is flaky."""
    verdicts = [
        validate_task(task, seeds=seeds, repeats=repeats, limits=limits, workspace_root=workspace_root)
        for task in tasks
    ]
    return ValidationReport(seeds=list(seeds), repeats=repeats, tasks=verdicts)
