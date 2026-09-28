# Developer entry points. Every target runs through uv and works offline.
.PHONY: install lint format typecheck test test-all ci demo clean

UV ?= uv
export MSWEA_SILENT_STARTUP = 1

install:
	$(UV) sync --group dev

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck:
	$(UV) run mypy

# Fast suite for local iteration: everything except tests marked slow.
test:
	$(UV) run pytest -q -n auto -m "not slow"

test-all:
	$(UV) run pytest -q -n auto

# What CI runs: lint, typecheck, full suite with a coverage report.
ci: lint typecheck
	$(UV) run pytest -q -n auto --cov --cov-report=term-missing --cov-report=xml

# Demo: scripted and random agents over every task for 3 seeds, JSONL trajectory
# plus the success-rate table. Offline, well under a minute.
DEMO_OUT ?= trajectories.jsonl
demo:
	$(UV) run agentenv version
	$(UV) run agentenv tasks
	$(UV) run agentenv run --task all --agent scripted --agent random --episodes 3 --seed 0 --out $(DEMO_OUT)
	$(UV) run agentenv report $(DEMO_OUT)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov trajectories.jsonl
