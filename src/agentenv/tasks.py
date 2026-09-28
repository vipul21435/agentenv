"""Task model and the built-in task suite.

A :class:`Task` bundles a seeded ``setup`` (writes the starting workspace), a ``reward``
function that grades the workspace in ``[0, 1]`` and a ``reference_solution`` (the actions
a scripted agent applies). Every task derives its parameters from ``seed`` through
``random.Random``, so the same seed always produces the same files and expected values.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentenv.actions import Action, PythonAction, WriteFileAction
from agentenv.errors import TaskError
from agentenv.sandbox import SubprocessSandbox

SetupFn = Callable[[Path, int], None]
RewardFn = Callable[[SubprocessSandbox, int], float]
ReferenceFn = Callable[[int], list[Action]]


class Task(BaseModel):
    """A seeded, verifiable task."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    max_steps: int = Field(default=6, ge=1)
    setup: SetupFn
    reward: RewardFn
    reference_solution: ReferenceFn


def task_rng(task_id: str, seed: int) -> random.Random:
    """Deterministic generator for one task and seed (string seeds hash stably across processes)."""
    return random.Random(f"{task_id}:{seed}")


def _load_json(sandbox: SubprocessSandbox, path: str) -> Any:
    target = sandbox.workspace / path
    if not target.is_file():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _fraction_matching(actual: Any, expected: dict[str, Any], *, tolerance: float = 1e-3) -> float:
    """Partial credit: the share of expected keys whose value matches."""
    if not isinstance(actual, dict):
        return 0.0
    hits = 0
    for key, want in expected.items():
        got = actual.get(key)
        if isinstance(want, float) and isinstance(got, int | float) and not isinstance(got, bool):
            hits += abs(float(got) - want) <= tolerance
        else:
            hits += got == want
    return hits / len(expected)


# --------------------------------------------------------------------------- fix_checksum

_CHECKSUM_TEMPLATE = '''"""Utility functions for the release pipeline."""


def weighted_checksum(values):
    """Return sum((i + 1) * v for i, v in enumerate(values)) modulo {modulus}."""
    return {body}
'''
_CHECKSUM_BUGS = (
    "sum(i * v for i, v in enumerate(values)) % {modulus}",
    "sum((i + 1) * v for i, v in enumerate(values))",
    "sum((i + 1) * v for i, v in enumerate(values)) % ({modulus} - 1)",
)


def _checksum_params(seed: int) -> tuple[int, str, list[list[int]], list[list[int]]]:
    """Modulus, bug variant, hidden test inputs and two visible example inputs."""
    rng = task_rng("fix_checksum", seed)
    modulus = rng.choice([97, 101, 251, 1009])
    bug = rng.choice(_CHECKSUM_BUGS)
    cases = [[rng.randint(1, 1000) for _ in range(rng.randint(3, 8))] for _ in range(5)]
    examples = [[rng.randint(1, 1000) for _ in range(rng.randint(3, 5))] for _ in range(2)]
    return modulus, bug, cases, examples


def _checksum_expected(values: list[int], modulus: int) -> int:
    return sum((i + 1) * v for i, v in enumerate(values)) % modulus


def _checksum_setup(workspace: Path, seed: int) -> None:
    modulus, bug, _, examples = _checksum_params(seed)
    (workspace / "mathlib.py").write_text(_CHECKSUM_TEMPLATE.format(modulus=modulus, body=bug.format(modulus=modulus)))
    examples_text = "".join(
        f"weighted_checksum({values!r}) == {_checksum_expected(values, modulus)}\n" for values in examples
    )
    (workspace / "TASK.md").write_text(
        "mathlib.weighted_checksum(values) must return\n"
        f"sum((i + 1) * v for i, v in enumerate(values)) % {modulus}\n"
        "for any list of integers. It currently returns wrong values. Fix mathlib.py;\n"
        "hidden tests import it and compare against the formula. Worked examples:\n\n" + examples_text
    )


def _checksum_reward(sandbox: SubprocessSandbox, seed: int) -> float:
    modulus, _, cases, _ = _checksum_params(seed)
    checks = [(values, _checksum_expected(values, modulus)) for values in cases]
    code = (
        "from mathlib import weighted_checksum\n"
        f"for values, expected in {checks!r}:\n"
        "    assert weighted_checksum(values) == expected, (values, expected)\n"
    )
    return 1.0 if sandbox.python(code).ok else 0.0


def _checksum_reference(seed: int) -> list[Action]:
    modulus, _, _, _ = _checksum_params(seed)
    body = f"sum((i + 1) * v for i, v in enumerate(values)) % {modulus}"
    return [WriteFileAction(path="mathlib.py", content=_CHECKSUM_TEMPLATE.format(modulus=modulus, body=body))]


# ---------------------------------------------------------------------- summarize_numbers

_STATS_KEYS = ("count", "sum", "min", "max", "mean")


def _numbers(seed: int) -> list[int]:
    rng = task_rng("summarize_numbers", seed)
    return [rng.randint(-500, 500) for _ in range(rng.randint(20, 40))]


def _numbers_expected(seed: int) -> dict[str, Any]:
    numbers = _numbers(seed)
    return {
        "count": len(numbers),
        "sum": sum(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "mean": round(statistics.fmean(numbers), 3),
    }


def _numbers_setup(workspace: Path, seed: int) -> None:
    (workspace / "numbers.txt").write_text("\n".join(str(n) for n in _numbers(seed)) + "\n")
    (workspace / "TASK.md").write_text(
        "numbers.txt holds one integer per line. Write stats.json, a JSON object with the\n"
        'keys "count", "sum", "min", "max" and "mean" (mean rounded to 3 decimals).\n'
    )


def _numbers_reward(sandbox: SubprocessSandbox, seed: int) -> float:
    return _fraction_matching(_load_json(sandbox, "stats.json"), _numbers_expected(seed))


_NUMBERS_SOLUTION = """import json
numbers = [int(line) for line in open("numbers.txt") if line.strip()]
stats = {
    "count": len(numbers),
    "sum": sum(numbers),
    "min": min(numbers),
    "max": max(numbers),
    "mean": round(sum(numbers) / len(numbers), 3),
}
json.dump(stats, open("stats.json", "w"))
print(stats)
"""


def _numbers_reference(seed: int) -> list[Action]:
    return [PythonAction(code=_NUMBERS_SOLUTION)]


# ----------------------------------------------------------------------------- parse_log

_SERVICES = ("auth", "billing", "gateway", "search", "worker")
_MESSAGES = ("request handled", "cache miss", "retrying upstream", "connection reset", "slow query")


def _log_lines(seed: int) -> list[tuple[str, str]]:
    rng = task_rng("parse_log", seed)
    services = rng.sample(_SERVICES, rng.randint(3, 5))
    lines: list[tuple[str, str]] = []
    for index in range(rng.randint(30, 60)):
        level = rng.choices(["INFO", "WARN", "ERROR"], weights=[6, 2, 2])[0]
        service = rng.choice(services)
        stamp = f"2026-03-{1 + index // 20:02d}T{index % 24:02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"
        lines.append((level, f"{stamp} {level} {service}: {rng.choice(_MESSAGES)}"))
    return [(level, line) for level, line in lines]


def _log_expected(seed: int) -> dict[str, Any]:
    entries = _log_lines(seed)
    by_level = Counter(level for level, _ in entries)
    services = sorted({line.split()[2].rstrip(":") for _, line in entries})
    errors_by_service = Counter(line.split()[2].rstrip(":") for level, line in entries if level == "ERROR")
    return {
        "lines": len(entries),
        "by_level": {level: by_level.get(level, 0) for level in ("ERROR", "INFO", "WARN")},
        "services": services,
        "errors_by_service": dict(sorted(errors_by_service.items())),
    }


def _log_setup(workspace: Path, seed: int) -> None:
    (workspace / "app.log").write_text("".join(line + "\n" for _, line in _log_lines(seed)))
    (workspace / "TASK.md").write_text(
        "app.log has lines of the form '<timestamp> <LEVEL> <service>: <message>'.\n"
        'Write summary.json, a JSON object with: "lines" (total line count),\n'
        '"by_level" (object with counts for ERROR, INFO and WARN, always all three),\n'
        '"services" (sorted list of distinct service names) and "errors_by_service"\n'
        "(object mapping each service that logged at least one ERROR to its ERROR count).\n"
    )


def _log_reward(sandbox: SubprocessSandbox, seed: int) -> float:
    return _fraction_matching(_load_json(sandbox, "summary.json"), _log_expected(seed))


_LOG_SOLUTION = """import json
from collections import Counter
levels, errors, services = Counter(), Counter(), set()
lines = 0
for line in open("app.log"):
    if not line.strip():
        continue
    lines += 1
    _, level, service, *_ = line.split()
    service = service.rstrip(":")
    levels[level] += 1
    services.add(service)
    if level == "ERROR":
        errors[service] += 1
summary = {
    "lines": lines,
    "by_level": {level: levels.get(level, 0) for level in ("ERROR", "INFO", "WARN")},
    "services": sorted(services),
    "errors_by_service": dict(sorted(errors.items())),
}
json.dump(summary, open("summary.json", "w"))
print(summary)
"""


def _log_reference(seed: int) -> list[Action]:
    return [PythonAction(code=_LOG_SOLUTION)]


# ------------------------------------------------------------------------------ registry

TASKS: dict[str, Task] = {
    task.id: task
    for task in (
        Task(
            id="fix_checksum",
            description="Fix the buggy weighted_checksum() in mathlib.py so the hidden tests pass (see TASK.md).",
            setup=_checksum_setup,
            reward=_checksum_reward,
            reference_solution=_checksum_reference,
        ),
        Task(
            id="summarize_numbers",
            description="Compute count/sum/min/max/mean of numbers.txt and write them to stats.json (see TASK.md).",
            setup=_numbers_setup,
            reward=_numbers_reward,
            reference_solution=_numbers_reference,
        ),
        Task(
            id="parse_log",
            description="Parse app.log into summary.json with per-level counts, services and errors per service.",
            setup=_log_setup,
            reward=_log_reward,
            reference_solution=_log_reference,
        ),
    )
}


def list_tasks() -> list[Task]:
    return list(TASKS.values())


def get_task(task_id: str) -> Task:
    """Look a task up by id; raise ``TaskError`` naming the known ids otherwise."""
    try:
        return TASKS[task_id]
    except KeyError:
        raise TaskError(f"unknown task: {task_id!r}", details={"available": sorted(TASKS)}) from None
