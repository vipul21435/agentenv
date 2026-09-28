"""Task suite: reference solutions score 1.0, the untouched workspace scores 0.0, seeds are deterministic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentenv.env import Environment, PythonAction, ShellAction, WriteFileAction
from agentenv.errors import TaskError
from agentenv.sandbox import SubprocessSandbox
from agentenv.tasks import _SLUG_BUGS, TASKS, _config, _flatten, get_task, list_tasks

SEEDS = [0, 1, 2, 42]
TASK_IDS = [
    "fix_checksum",
    "summarize_numbers",
    "parse_log",
    "fix_slugify",
    "implement_ringbuffer",
    "csv_totals",
    "json_flatten",
    "grep_report",
    "rename_files",
    "data_pipeline",
]


def _run_reference(task_id: str, seed: int, root: Path) -> float:
    with Environment(get_task(task_id), workspace_root=root) as env:
        env.reset(seed)
        reward = 0.0
        for action in env.task.reference_solution(seed):
            reward = env.step(action).reward
        return reward


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_reference_solution_scores_one(task_id: str, seed: int, tmp_path: Path) -> None:
    assert _run_reference(task_id, seed, tmp_path) == 1.0


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_untouched_workspace_scores_zero(task_id: str, seed: int, tmp_path: Path) -> None:
    task = get_task(task_id)
    sandbox = SubprocessSandbox.create(root=tmp_path)
    task.setup(sandbox.workspace, seed)
    assert task.reward(sandbox, seed) == 0.0


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_setup_is_deterministic_per_seed(task_id: str, tmp_path: Path) -> None:
    task = get_task(task_id)
    snapshots: list[dict[str, str]] = []
    for seed in (5, 5, 6):
        workspace = tmp_path / f"{seed}-{len(snapshots)}"
        workspace.mkdir()
        task.setup(workspace, seed)
        snapshots.append({str(p.relative_to(workspace)): p.read_text() for p in workspace.rglob("*") if p.is_file()})
    assert snapshots[0] == snapshots[1]
    assert snapshots[0] != snapshots[2]
    assert "TASK.md" in snapshots[0]


def test_partial_credit_for_partially_correct_stats(tmp_path: Path) -> None:
    task = get_task("summarize_numbers")
    sandbox = SubprocessSandbox.create(root=tmp_path)
    task.setup(sandbox.workspace, 0)
    numbers = [int(line) for line in (sandbox.workspace / "numbers.txt").read_text().split()]
    (sandbox.workspace / "stats.json").write_text(json.dumps({"count": len(numbers), "sum": sum(numbers), "min": "?"}))
    assert task.reward(sandbox, 0) == pytest.approx(0.4)
    (sandbox.workspace / "stats.json").write_text("not json")
    assert task.reward(sandbox, 0) == 0.0


def test_checksum_bug_is_actually_wrong(tmp_path: Path) -> None:
    task = get_task("fix_checksum")
    sandbox = SubprocessSandbox.create(root=tmp_path)
    task.setup(sandbox.workspace, 3)
    source = (sandbox.workspace / "mathlib.py").read_text()
    assert "def weighted_checksum" in source
    assert sandbox.python("import mathlib").ok  # the buggy module still imports


def test_log_summary_keys(tmp_path: Path) -> None:
    task = get_task("parse_log")
    with Environment(task, workspace_root=tmp_path) as env:
        env.reset(11)
        for action in task.reference_solution(11):
            env.step(action)
        summary = json.loads((env.sandbox.workspace / "summary.json").read_text())
    assert set(summary) == {"lines", "by_level", "services", "errors_by_service"}
    assert summary["lines"] == sum(summary["by_level"].values())
    assert summary["services"] == sorted(summary["services"])
    assert set(summary["errors_by_service"]) <= set(summary["services"])


def test_registry() -> None:
    assert [t.id for t in list_tasks()] == TASK_IDS
    assert len(TASK_IDS) >= 10
    with pytest.raises(TaskError) as excinfo:
        get_task("nope")
    assert excinfo.value.details["available"] == sorted(TASKS)


def _prepared(task_id: str, seed: int, root: Path) -> SubprocessSandbox:
    sandbox = SubprocessSandbox.create(root=root)
    get_task(task_id).setup(sandbox.workspace, seed)
    return sandbox


@pytest.mark.parametrize("seed", SEEDS)
def test_slugify_bug_variant_fails_and_every_variant_is_caught(seed: int, tmp_path: Path) -> None:
    task = get_task("fix_slugify")
    sandbox = _prepared("fix_slugify", seed, tmp_path)
    source = (sandbox.workspace / "textutil.py").read_text()
    assert sandbox.python("import textutil").ok
    assert task.reward(sandbox, seed) == 0.0
    for bug in _SLUG_BUGS:
        limit = int(source.split("at most ")[1].split()[0])
        body = bug.format(limit=limit)
        (sandbox.workspace / "textutil.py").write_text(source.split("    return ")[0] + f"    return {body}\n")
        assert task.reward(sandbox, seed) == 0.0, bug


def test_ringbuffer_rejects_wrong_eviction_order(tmp_path: Path) -> None:
    task = get_task("implement_ringbuffer")
    sandbox = _prepared("implement_ringbuffer", 1, tmp_path)
    (sandbox.workspace / "ringbuffer.py").write_text(
        "class RingBuffer:\n"
        "    def __init__(self, capacity):\n"
        "        if capacity < 1:\n"
        "            raise ValueError\n"
        "        self.capacity, self._items = capacity, []\n"
        "    def push(self, item):\n"
        "        if len(self._items) < self.capacity:\n"
        "            self._items.append(item)\n"  # drops the newest instead of the oldest
        "    def items(self):\n"
        "        return list(self._items)\n"
        "    def __len__(self):\n"
        "        return len(self._items)\n"
    )
    assert task.reward(sandbox, 1) == 0.0


def test_csv_totals_partial_credit_and_tolerance(tmp_path: Path) -> None:
    task = get_task("csv_totals")
    sandbox = _prepared("csv_totals", 0, tmp_path)
    for action in task.reference_solution(0):
        assert isinstance(action, PythonAction)
        assert sandbox.python(action.code).ok
    totals = json.loads((sandbox.workspace / "totals.json").read_text())
    assert len(totals) >= 3
    first = next(iter(totals))
    totals[first] += 0.004  # inside the half-cent tolerance
    (sandbox.workspace / "totals.json").write_text(json.dumps(totals))
    assert task.reward(sandbox, 0) == 1.0
    totals[first] += 1.0
    (sandbox.workspace / "totals.json").write_text(json.dumps(totals))
    assert task.reward(sandbox, 0) == pytest.approx((len(totals) - 1) / len(totals))


def test_json_flatten_keys_are_dotted_paths(tmp_path: Path) -> None:
    task = get_task("json_flatten")
    sandbox = _prepared("json_flatten", 2, tmp_path)
    config = json.loads((sandbox.workspace / "config.json").read_text())
    for action in task.reference_solution(2):
        assert isinstance(action, PythonAction)
        assert sandbox.python(action.code).ok
    flat = json.loads((sandbox.workspace / "flat.json").read_text())
    assert flat["version"] == config["version"]
    assert all(not isinstance(value, dict) for value in flat.values())
    assert any("." in key for key in flat)
    (sandbox.workspace / "flat.json").write_text(json.dumps({"version": config["version"]}))
    assert 0.0 < task.reward(sandbox, 2) < 0.5


def test_grep_report_partial_credit_is_jaccard(tmp_path: Path) -> None:
    task = get_task("grep_report")
    sandbox = _prepared("grep_report", 0, tmp_path)
    assert task.reward(sandbox, 0) == 0.0
    assert sandbox.shell("grep -rc TODO src docs | grep -v ':0$' | sort > report.txt").ok
    expected = (sandbox.workspace / "report.txt").read_text().splitlines()
    assert expected and all(":" in line for line in expected)
    (sandbox.workspace / "report.txt").write_text("\n".join(expected[:1] + ["bogus.py:9"]) + "\n")
    assert task.reward(sandbox, 0) == pytest.approx(1 / (len(expected) + 1))
    (sandbox.workspace / "report.txt").write_text("\n".join(expected) + "\n")
    assert task.reward(sandbox, 0) == 1.0


def test_rename_files_needs_every_tmp_gone(tmp_path: Path) -> None:
    task = get_task("rename_files")
    sandbox = _prepared("rename_files", 3, tmp_path)
    tmp_files = sorted((sandbox.workspace / "incoming").glob("*.tmp"))
    others = {p.name for p in (sandbox.workspace / "incoming").iterdir()} - {p.name for p in tmp_files}
    assert tmp_files
    first = tmp_files[0]
    first.with_suffix(".dat").write_text(first.read_text())  # copied, not moved: the .tmp remains
    assert task.reward(sandbox, 3) == pytest.approx(1 / (len(tmp_files) + 1))
    first.unlink()
    for path in tmp_files[1:]:
        path.rename(path.with_suffix(".dat"))
    assert task.reward(sandbox, 3) == 1.0
    assert others <= {p.name for p in (sandbox.workspace / "incoming").iterdir()}


def test_data_pipeline_gives_one_third_per_stage(tmp_path: Path) -> None:
    task = get_task("data_pipeline")
    with Environment(task, workspace_root=tmp_path) as env:
        env.reset(4)
        rewards = [env.step(action).reward for action in task.reference_solution(4)]
        assert rewards == pytest.approx([1 / 3, 2 / 3, 1.0])
        summary = json.loads((env.sandbox.workspace / "summary.json").read_text())
        merged = (env.sandbox.workspace / "merged.csv").read_text().splitlines()
    assert summary["rows"] + summary["dropped"] == len(merged) - 1
    assert summary["dropped"] >= 1


def test_shell_reference_solutions_run_through_the_environment(tmp_path: Path) -> None:
    for task_id in ("grep_report", "rename_files"):
        task = get_task(task_id)
        assert all(isinstance(action, ShellAction) for action in task.reference_solution(0))
        assert _run_reference(task_id, 0, tmp_path) == 1.0


HIDDEN_TEST_MODULES = [
    ("fix_checksum", "mathlib.py"),
    ("fix_slugify", "textutil.py"),
    ("implement_ringbuffer", "ringbuffer.py"),
]

# A module that reads the grader's own command line back (procfs on Linux, ps on macOS)
# and echoes anything that looks like a pass marker before exiting cleanly.
_CMDLINE_ECHO = (
    "import os, re, subprocess, sys\n"
    "try:\n"
    "    cmd = open('/proc/self/cmdline', 'rb').read().decode(errors='replace')\n"
    "except OSError:\n"
    "    cmd = subprocess.run(['ps', '-o', 'command=', '-p', str(os.getpid())], capture_output=True, text=True).stdout\n"
    "found = re.findall(r'AGENTENV[A-Z_]*[0-9a-f]+|observed|expected', cmd + ' '.join(sys.argv))\n"
    "sys.__stdout__.write(' '.join(found) + '\\n')\n"
    "sys.__stdout__.flush()\n"
    "os._exit(0)\n"
)


@pytest.mark.parametrize(("task_id", "module"), HIDDEN_TEST_MODULES)
@pytest.mark.parametrize(
    "content",
    ["import os\nos._exit(0)\n", "import sys\nsys.exit(0)\n", "raise SystemExit(0)\n", _CMDLINE_ECHO],
    ids=["os_exit", "sys_exit", "system_exit", "cmdline_echo"],
)
def test_hidden_tests_reject_module_that_exits_cleanly_at_import(
    task_id: str, module: str, content: str, tmp_path: Path
) -> None:
    """Exiting 0 at import, or echoing whatever the grader put on its command line, earns nothing."""
    with Environment(get_task(task_id), workspace_root=tmp_path) as env:
        env.reset(1)
        result = env.step(WriteFileAction(path=module, content=content))
        assert result.reward == 0.0
        assert not result.done


@pytest.mark.parametrize(("task_id", "module"), HIDDEN_TEST_MODULES)
def test_hidden_tests_expected_values_never_reach_the_child(task_id: str, module: str, tmp_path: Path) -> None:
    """The probe child sees only the inputs: its argv, environment and stdin carry no expected values."""
    spy = (
        "import json, os, sys\n"
        "leak = {'argv': sys.argv, 'env': dict(os.environ)}\n"
        "open('leak.json', 'w').write(json.dumps(leak))\n"
    )
    with Environment(get_task(task_id), workspace_root=tmp_path) as env:
        env.reset(1)
        assert env.step(WriteFileAction(path=module, content=spy)).reward == 0.0
        leak = json.loads((env.sandbox.workspace / "leak.json").read_text())
    assert leak["argv"] == ["-"]
    extra = set(leak["env"]) - {"PATH", "HOME", "LANG", "PYTHONDONTWRITEBYTECODE", "PYTHONIOENCODING"}
    assert all(key.startswith("__CF_") for key in extra), extra  # macOS injects __CF_USER_TEXT_ENCODING


@pytest.mark.parametrize(("task_id", "module"), HIDDEN_TEST_MODULES)
def test_hidden_tests_ignore_module_output_and_stdout_swaps(task_id: str, module: str, tmp_path: Path) -> None:
    """A correct module that prints at import or replaces sys.stdout still scores 1.0."""
    task = get_task(task_id)
    reference = task.reference_solution(1)
    assert len(reference) == 1 and isinstance(reference[0], WriteFileAction)
    noisy = 'import io, sys\nprint("[]")\nsys.stdout = io.StringIO()\n' + reference[0].content
    with Environment(task, workspace_root=tmp_path) as env:
        env.reset(1)
        assert env.step(WriteFileAction(path=module, content=noisy)).reward == 1.0


def test_json_flatten_rejects_booleans_written_as_ints(tmp_path: Path) -> None:
    flat = _flatten(_config(0))
    bools = sum(isinstance(value, bool) for value in flat.values())
    assert bools >= 1
    bogus = {key: (int(value) if isinstance(value, bool) else value) for key, value in flat.items()}
    with Environment(get_task("json_flatten"), workspace_root=tmp_path) as env:
        env.reset(0)
        reward = env.step(WriteFileAction(path="flat.json", content=json.dumps(bogus))).reward
    assert reward == pytest.approx((len(flat) - bools) / len(flat))


def test_rename_files_pays_nothing_for_deleting(tmp_path: Path) -> None:
    with Environment(get_task("rename_files"), workspace_root=tmp_path) as env:
        env.reset(0)
        assert env.step(ShellAction(command="rm incoming/*.tmp")).reward == 0.0
