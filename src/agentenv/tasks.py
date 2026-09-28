"""Task model and the built-in task suite.

A :class:`Task` bundles a seeded ``setup`` (writes the starting workspace), a ``reward``
function that grades the workspace in ``[0, 1]`` and a ``reference_solution`` (the actions
a scripted agent applies). Every task derives its parameters from ``seed`` through
``random.Random``, so the same seed always produces the same files and expected values.
"""

from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentenv.actions import Action, PythonAction, ShellAction, WriteFileAction
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
        if isinstance(want, bool) or isinstance(got, bool):
            # True == 1 in Python; a boolean leaf must stay a boolean (and an int must not become one)
            hits += isinstance(got, bool) and isinstance(want, bool) and got == want
        elif isinstance(want, float) and isinstance(got, int | float):
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


_PROBE_PRELUDE = "import io as _io, json as _json, sys as _sys\n_out = _sys.stdout\n_sys.stdout = _io.StringIO()\n"
_PROBE_EPILOGUE = "_out.write('\\n' + _json.dumps(observed) + '\\n')\n_out.flush()\n"


def _hidden_probe(sandbox: SubprocessSandbox, code: str, expected: Any) -> bool:
    """True only when the probe ``code`` reports exactly ``expected``.

    The probe imports the agent's module, calls it on the hidden inputs and prints the
    observed results as one JSON line on its own stdout handle (captured before the import,
    so a module that prints or swaps ``sys.stdout`` cannot hide it). The expected values
    stay in this process: they are never in the child's argv, environment or stdin, so a
    module that exits cleanly at import (``os._exit(0)``) or reads its own command line
    back has nothing to echo. Any exception, timeout, non-zero exit, truncated or
    malformed output counts as a failure.
    """
    result = sandbox.python(_PROBE_PRELUDE + code + _PROBE_EPILOGUE)
    if not result.ok:
        return False
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return False
    try:
        observed = json.loads(lines[-1])
    except ValueError:
        return False
    return bool(observed == expected)


def _checksum_reward(sandbox: SubprocessSandbox, seed: int) -> float:
    modulus, _, cases, _ = _checksum_params(seed)
    expected = [_checksum_expected(values, modulus) for values in cases]
    code = f"from mathlib import weighted_checksum\nobserved = [weighted_checksum(values) for values in {cases!r}]\n"
    return 1.0 if _hidden_probe(sandbox, code, expected) else 0.0


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


# ----------------------------------------------------------------------------- fix_slugify

_SLUG_TEMPLATE = '''"""Text helpers for the content pipeline."""

import re


def slugify(text):
    """Lowercase ``text``, replace every run of characters outside a-z0-9 with '-',
    strip leading and trailing '-', cut the result to at most {limit} characters and
    strip trailing '-' again."""
    return {body}
'''
_SLUG_BUGS = (
    're.sub(r"[^a-z0-9]+", "-", text).strip("-")[:{limit}].rstrip("-")',
    're.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:{limit}].rstrip("_")',
    're.sub(r"[^a-z0-9]+", "-", text.lower())[:{limit}]',
)
_SLUG_FIX = 're.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:{limit}].rstrip("-")'
_SLUG_WORDS = ("release", "notes", "Hello", "World", "API", "v2", "beta", "launch", "Q3", "roadmap", "fix", "docs")
_SLUG_SEPARATORS = (" ", "  ", "_", "!", ", ", " -- ", "/")


def _slug_params(seed: int) -> tuple[int, str, list[str]]:
    """Length limit, bug variant and five hidden inputs (the first one exercises every bug)."""
    rng = task_rng("fix_slugify", seed)
    limit = rng.choice([12, 16, 20, 32])
    bug = rng.choice(_SLUG_BUGS)
    cases: list[str] = []
    for index in range(5):
        words = [rng.choice(_SLUG_WORDS) for _ in range(rng.randint(2, 5))]
        if index == 0:
            words[0] = words[0].upper()
        text = "".join(word + rng.choice(_SLUG_SEPARATORS) for word in words).rstrip()
        prefix = "!" if index == 0 else rng.choice(["", " ", "!"])
        cases.append(prefix + text + rng.choice(["", "!", "  "]))
    return limit, bug, cases


def _slug_expected(text: str, limit: int) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit].rstrip("-")


def _slug_setup(workspace: Path, seed: int) -> None:
    limit, bug, _ = _slug_params(seed)
    (workspace / "textutil.py").write_text(_SLUG_TEMPLATE.format(limit=limit, body=bug.format(limit=limit)))
    example = "Hello, World! Release notes"
    (workspace / "TASK.md").write_text(
        "textutil.slugify(text) must lowercase the text, replace every run of characters\n"
        f"outside a-z0-9 with '-', strip leading and trailing '-', cut to {limit} characters\n"
        "and strip trailing '-' again. It currently returns wrong values for some inputs.\n"
        "Fix textutil.py; hidden tests import it. Worked example:\n\n"
        f"slugify({example!r}) == {_slug_expected(example, limit)!r}\n"
    )


def _slug_reward(sandbox: SubprocessSandbox, seed: int) -> float:
    limit, _, cases = _slug_params(seed)
    expected = [_slug_expected(text, limit) for text in cases]
    code = f"from textutil import slugify\nobserved = [slugify(text) for text in {cases!r}]\n"
    return 1.0 if _hidden_probe(sandbox, code, expected) else 0.0


def _slug_reference(seed: int) -> list[Action]:
    limit, _, _ = _slug_params(seed)
    return [
        WriteFileAction(
            path="textutil.py", content=_SLUG_TEMPLATE.format(limit=limit, body=_SLUG_FIX.format(limit=limit))
        )
    ]


# ---------------------------------------------------------------------- implement_ringbuffer


def _ring_params(seed: int) -> tuple[int, list[list[int]]]:
    """Capacity and three push sequences: shorter than, equal to and longer than the capacity."""
    rng = task_rng("implement_ringbuffer", seed)
    capacity = rng.randint(3, 8)
    lengths = [rng.randint(1, capacity - 1), capacity, capacity + rng.randint(1, 6)]
    return capacity, [[rng.randint(-99, 99) for _ in range(length)] for length in lengths]


def _ring_setup(workspace: Path, seed: int) -> None:
    capacity, _ = _ring_params(seed)
    (workspace / "TASK.md").write_text(
        "Create ringbuffer.py with a class RingBuffer(capacity):\n"
        "- the constructor raises ValueError when capacity < 1\n"
        "- push(item) appends item and drops the oldest item once more than capacity are held\n"
        "- items() returns the held items as a list, oldest first\n"
        "- len(buffer) is the number of held items\n"
        f"Hidden tests use capacity {capacity} and check items() and len() after each push.\n"
    )


def _ring_expected(capacity: int, sequences: list[list[int]]) -> list[Any]:
    """What the probe must observe: ``ValueError`` for capacity 0, then items and len after every push."""
    trace: list[Any] = ["ValueError"]
    for sequence in sequences:
        steps: list[Any] = [[[], 0]]
        for index in range(len(sequence)):
            held = sequence[: index + 1][-capacity:]
            steps.append([held, len(held)])
        trace.append(steps)
    return trace


def _ring_reward(sandbox: SubprocessSandbox, seed: int) -> float:
    capacity, sequences = _ring_params(seed)
    code = (
        "from ringbuffer import RingBuffer\n"
        "try:\n"
        "    RingBuffer(0)\n"
        "except ValueError:\n"
        "    observed = ['ValueError']\n"
        "else:\n"
        "    observed = ['capacity 0 accepted']\n"
        "def _snapshot(buffer):\n"
        "    held = buffer.items()\n"
        "    return [held if isinstance(held, list) else repr(held), len(buffer)]\n"
        f"for sequence in {sequences!r}:\n"
        f"    buffer = RingBuffer({capacity})\n"
        "    steps = [_snapshot(buffer)]\n"
        "    for item in sequence:\n"
        "        buffer.push(item)\n"
        "        steps.append(_snapshot(buffer))\n"
        "    observed.append(steps)\n"
    )
    return 1.0 if _hidden_probe(sandbox, code, _ring_expected(capacity, sequences)) else 0.0


_RING_SOLUTION = '''"""Fixed-capacity FIFO buffer."""


class RingBuffer:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._items = []

    def push(self, item):
        self._items.append(item)
        if len(self._items) > self.capacity:
            del self._items[0]

    def items(self):
        return list(self._items)

    def __len__(self):
        return len(self._items)
'''


def _ring_reference(seed: int) -> list[Action]:
    return [WriteFileAction(path="ringbuffer.py", content=_RING_SOLUTION)]


# ----------------------------------------------------------------------------- csv_totals

_CUSTOMERS = ("acme", "globex", "initech", "umbrella", "hooli", "vandelay")
_STATUSES = ("paid", "paid", "paid", "refunded", "pending")


def _orders(seed: int) -> list[tuple[int, str, int, str]]:
    """Rows of (order_id, customer, amount_cents, status); every sampled customer appears at least once."""
    rng = task_rng("csv_totals", seed)
    customers = rng.sample(_CUSTOMERS, rng.randint(3, 5))
    rows: list[tuple[int, str, int, str]] = []
    for index in range(rng.randint(15, 30)):
        customer = customers[index] if index < len(customers) else rng.choice(customers)
        rows.append((1000 + index, customer, rng.randint(500, 99999), rng.choice(_STATUSES)))
    return rows


def _orders_expected(seed: int) -> dict[str, Any]:
    totals: dict[str, int] = {}
    for _, customer, cents, status in _orders(seed):
        totals.setdefault(customer, 0)
        if status == "paid":
            totals[customer] += cents
    return {customer: round(cents / 100, 2) for customer, cents in sorted(totals.items())}


def _orders_setup(workspace: Path, seed: int) -> None:
    lines = ["order_id,customer,amount,status"]
    lines += [
        f"{order_id},{customer},{cents // 100}.{cents % 100:02d},{status}"
        for order_id, customer, cents, status in _orders(seed)
    ]
    (workspace / "orders.csv").write_text("\n".join(lines) + "\n")
    (workspace / "TASK.md").write_text(
        "orders.csv has the columns order_id, customer, amount (decimal) and status.\n"
        "Write totals.json: a JSON object mapping every customer in the file to the sum of\n"
        "amount over its rows with status 'paid', rounded to 2 decimals (0.0 if none).\n"
    )


def _orders_reward(sandbox: SubprocessSandbox, seed: int) -> float:
    return _fraction_matching(_load_json(sandbox, "totals.json"), _orders_expected(seed), tolerance=0.005)


_ORDERS_SOLUTION = """import csv, json
totals = {}
with open("orders.csv", newline="") as handle:
    for row in csv.DictReader(handle):
        totals.setdefault(row["customer"], 0)
        if row["status"] == "paid":
            totals[row["customer"]] += round(float(row["amount"]) * 100)
result = {customer: round(cents / 100, 2) for customer, cents in sorted(totals.items())}
json.dump(result, open("totals.json", "w"))
print(result)
"""


def _orders_reference(seed: int) -> list[Action]:
    return [PythonAction(code=_ORDERS_SOLUTION)]


# ---------------------------------------------------------------------------- json_flatten

_SECTIONS = ("database", "cache", "server", "auth", "metrics", "queue")
_KEYS = ("host", "port", "enabled", "timeout", "name", "retries", "tls", "pool")


def _config(seed: int) -> dict[str, Any]:
    rng = task_rng("json_flatten", seed)
    config: dict[str, Any] = {"version": rng.randint(1, 9)}
    for section in rng.sample(_SECTIONS, rng.randint(3, 5)):
        block: dict[str, Any] = {}
        for key in rng.sample(_KEYS, rng.randint(2, 4)):
            if key in ("enabled", "tls"):
                block[key] = rng.random() < 0.5
            elif key in ("host", "name"):
                block[key] = f"{section}-{rng.randint(1, 99)}"
            else:
                block[key] = rng.randint(1, 9000)
        if rng.random() < 0.5:
            low = rng.randint(1, 50)
            block["limits"] = {"min": low, "max": low + rng.randint(1, 500)}
        config[section] = block
    return config


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flat: dict[str, Any] = {}
    for key, item in value.items():
        flat.update(_flatten(item, f"{prefix}.{key}" if prefix else key))
    return flat


def _flatten_setup(workspace: Path, seed: int) -> None:
    (workspace / "config.json").write_text(json.dumps(_config(seed), indent=2) + "\n")
    (workspace / "TASK.md").write_text(
        "config.json is a nested JSON object. Write flat.json: one JSON object whose keys\n"
        'are the dotted paths to every leaf value (for example "database.limits.max")\n'
        "and whose values are the leaves unchanged.\n"
    )


def _flatten_reward(sandbox: SubprocessSandbox, seed: int) -> float:
    return _fraction_matching(_load_json(sandbox, "flat.json"), _flatten(_config(seed)))


_FLATTEN_SOLUTION = """import json
def flatten(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: value}
    out = {}
    for key, item in value.items():
        out.update(flatten(item, f"{prefix}.{key}" if prefix else key))
    return out
flat = flatten(json.load(open("config.json")))
json.dump(flat, open("flat.json", "w"), indent=2)
print(flat)
"""


def _flatten_reference(seed: int) -> list[Action]:
    return [PythonAction(code=_FLATTEN_SOLUTION)]


# ----------------------------------------------------------------------------- grep_report

_SOURCE_DIRS = ("src/core", "src/api", "src/util", "docs")
_SOURCE_LINES = (
    "def handler(request):",
    "    return render(request)",
    "# TODO: handle the timeout case",
    "# TODO: remove after the migration",
    "import json",
    "VALUE = 42",
    "print('ready')",
    "See the README for details.",
)


def _source_tree(seed: int) -> dict[str, str]:
    """Relative path -> file content for a small seeded source tree; some files carry TODO lines."""
    rng = task_rng("grep_report", seed)
    files: dict[str, str] = {}
    for index in range(rng.randint(8, 14)):
        folder = rng.choice(_SOURCE_DIRS)
        suffix = ".md" if folder == "docs" else ".py"
        lines = [rng.choice(_SOURCE_LINES) for _ in range(rng.randint(3, 8))]
        if index == 0:
            lines.append("# TODO: first marker")
        files[f"{folder}/file_{index}{suffix}"] = "\n".join(lines) + "\n"
    return files


def _grep_expected(seed: int) -> set[str]:
    return {
        f"{path}:{count}"
        for path, content in _source_tree(seed).items()
        if (count := sum("TODO" in line for line in content.splitlines()))
    }


def _grep_setup(workspace: Path, seed: int) -> None:
    for path, content in _source_tree(seed).items():
        target = workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    (workspace / "TASK.md").write_text(
        "The tree under src/ and docs/ contains files with TODO markers. Write report.txt\n"
        "with one line per file that contains at least one line with 'TODO', in the form\n"
        "'<relative path>:<number of lines containing TODO>', sorted by path.\n"
    )


def _jaccard(actual: set[str], expected: set[str]) -> float:
    union = actual | expected
    return len(actual & expected) / len(union) if union else 0.0


def _report_lines(sandbox: SubprocessSandbox, path: str) -> set[str]:
    target = sandbox.workspace / path
    if not target.is_file():
        return set()
    return {line.strip() for line in target.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()}


def _grep_reward(sandbox: SubprocessSandbox, seed: int) -> float:
    return _jaccard(_report_lines(sandbox, "report.txt"), _grep_expected(seed))


def _grep_reference(seed: int) -> list[Action]:
    return [ShellAction(command="grep -rc TODO src docs | grep -v ':0$' | sort > report.txt")]


# ---------------------------------------------------------------------------- rename_files


def _incoming(seed: int) -> tuple[dict[str, str], list[str]]:
    """Files in incoming/ (name -> content) and the names that end in .tmp."""
    rng = task_rng("rename_files", seed)
    files: dict[str, str] = {}
    for index in range(rng.randint(6, 12)):
        suffix = ".tmp" if index == 0 or rng.random() < 0.6 else rng.choice([".txt", ".log"])
        files[f"report_{rng.randint(100, 999)}_{index}{suffix}"] = f"payload {rng.randint(0, 10**6)}\n"
    return files, sorted(name for name in files if name.endswith(".tmp"))


def _rename_setup(workspace: Path, seed: int) -> None:
    files, _ = _incoming(seed)
    (workspace / "incoming").mkdir()
    for name, content in files.items():
        (workspace / "incoming" / name).write_text(content)
    (workspace / "TASK.md").write_text(
        "Rename every .tmp file in incoming/ to the same name with the extension .dat,\n"
        "keeping its content; leave the other files alone. No .tmp file may remain.\n"
    )


def _rename_reward(sandbox: SubprocessSandbox, seed: int) -> float:
    files, tmp_names = _incoming(seed)
    folder = sandbox.workspace / "incoming"
    renamed = 0
    for name in tmp_names:
        target = folder / (name[: -len(".tmp")] + ".dat")
        renamed += target.is_file() and target.read_text() == files[name]
    # the clean-up term is only paid once at least one file was renamed, so deleting beats nothing
    none_left = renamed > 0 and folder.is_dir() and not any(folder.glob("*.tmp"))
    return (renamed + none_left) / (len(tmp_names) + 1)


def _rename_reference(seed: int) -> list[Action]:
    return [ShellAction(command='for f in incoming/*.tmp; do mv "$f" "${f%.tmp}.dat"; done')]


# --------------------------------------------------------------------------- data_pipeline


def _parts(seed: int) -> list[list[tuple[int, str]]]:
    """Per raw part file: rows of (id, value) where value may be negative or 'n/a'."""
    rng = task_rng("data_pipeline", seed)
    parts: list[list[tuple[int, str]]] = []
    next_id = 1
    for _ in range(rng.randint(3, 5)):
        rows: list[tuple[int, str]] = []
        for _ in range(rng.randint(4, 8)):
            roll = rng.random()
            value = "n/a" if roll < 0.1 else str(rng.randint(-50, -1) if roll < 0.25 else rng.randint(0, 500))
            rows.append((next_id, value))
            next_id += 1
        parts.append(rows)
    parts[0][0] = (parts[0][0][0], "n/a")
    return parts


def _pipeline_expected(seed: int) -> tuple[list[str], list[str], dict[str, Any]]:
    merged = [f"{row_id},{value}" for rows in _parts(seed) for row_id, value in rows]
    clean = [line for line in merged if line.split(",")[1].isdigit()]
    total = sum(int(line.split(",")[1]) for line in clean)
    return merged, clean, {"rows": len(clean), "dropped": len(merged) - len(clean), "total": total}


def _pipeline_setup(workspace: Path, seed: int) -> None:
    (workspace / "raw").mkdir()
    for index, rows in enumerate(_parts(seed), start=1):
        body = "".join(f"{row_id},{value}\n" for row_id, value in rows)
        (workspace / "raw" / f"part_{index}.csv").write_text("id,value\n" + body)
    (workspace / "TASK.md").write_text(
        "raw/part_<n>.csv files share the header 'id,value'. Three steps, each graded:\n"
        "1. merged.csv: the header once, then all rows of part_1, part_2, ... in order.\n"
        "2. clean.csv: the header, then the rows of merged.csv whose value is a\n"
        "   non-negative integer, in the same order.\n"
        '3. summary.json: {"rows": clean row count, "dropped": merged minus clean row\n'
        '   count, "total": sum of the clean values}.\n'
    )


def _csv_rows(sandbox: SubprocessSandbox, path: str) -> list[str] | None:
    target = sandbox.workspace / path
    if not target.is_file():
        return None
    lines = [line.strip() for line in target.read_text(encoding="utf-8", errors="replace").splitlines()]
    if not lines or lines[0] != "id,value":
        return None
    return [line for line in lines[1:] if line]


def _pipeline_reward(sandbox: SubprocessSandbox, seed: int) -> float:
    merged, clean, summary = _pipeline_expected(seed)
    stages = [
        float(_csv_rows(sandbox, "merged.csv") == merged),
        float(_csv_rows(sandbox, "clean.csv") == clean),
        _fraction_matching(_load_json(sandbox, "summary.json"), summary),
    ]
    return sum(stages) / len(stages)


_PIPELINE_MERGE = """import glob
rows = []
for path in sorted(glob.glob("raw/part_*.csv")):
    rows.extend(open(path).read().splitlines()[1:])
open("merged.csv", "w").write("id,value\\n" + "".join(row + "\\n" for row in rows))
print(len(rows), "rows merged")
"""
_PIPELINE_CLEAN = """rows = open("merged.csv").read().splitlines()[1:]
keep = [row for row in rows if row.split(",")[1].isdigit()]
open("clean.csv", "w").write("id,value\\n" + "".join(row + "\\n" for row in keep))
print(len(keep), "rows kept")
"""
_PIPELINE_SUMMARY = """import json
merged = open("merged.csv").read().splitlines()[1:]
clean = open("clean.csv").read().splitlines()[1:]
summary = {"rows": len(clean), "dropped": len(merged) - len(clean), "total": sum(int(r.split(",")[1]) for r in clean)}
json.dump(summary, open("summary.json", "w"))
print(summary)
"""


def _pipeline_reference(seed: int) -> list[Action]:
    return [
        PythonAction(code=_PIPELINE_MERGE),
        PythonAction(code=_PIPELINE_CLEAN),
        PythonAction(code=_PIPELINE_SUMMARY),
    ]


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
        Task(
            id="fix_slugify",
            description="Fix the buggy slugify() in textutil.py so the hidden tests pass (see TASK.md).",
            setup=_slug_setup,
            reward=_slug_reward,
            reference_solution=_slug_reference,
        ),
        Task(
            id="implement_ringbuffer",
            description="Implement ringbuffer.RingBuffer from the spec in TASK.md; hidden tests check push/items/len.",
            setup=_ring_setup,
            reward=_ring_reward,
            reference_solution=_ring_reference,
        ),
        Task(
            id="csv_totals",
            description="Sum the paid amounts per customer in orders.csv and write totals.json.",
            setup=_orders_setup,
            reward=_orders_reward,
            reference_solution=_orders_reference,
        ),
        Task(
            id="json_flatten",
            description="Flatten the nested config.json into flat.json keyed by dotted paths.",
            setup=_flatten_setup,
            reward=_flatten_reward,
            reference_solution=_flatten_reference,
        ),
        Task(
            id="grep_report",
            description="Write report.txt listing every file under src/ and docs/ with TODO lines and their count.",
            setup=_grep_setup,
            reward=_grep_reward,
            reference_solution=_grep_reference,
        ),
        Task(
            id="rename_files",
            description="Rename every incoming/*.tmp file to .dat keeping its content; no .tmp may remain.",
            setup=_rename_setup,
            reward=_rename_reward,
            reference_solution=_rename_reference,
        ),
        Task(
            id="data_pipeline",
            description="Three graded steps: merge raw/part_*.csv, drop bad rows into clean.csv, write summary.json.",
            max_steps=9,
            setup=_pipeline_setup,
            reward=_pipeline_reward,
            reference_solution=_pipeline_reference,
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
