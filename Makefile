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

# Smoke demo: check the local toolchain and print the effective settings.
demo:
	$(UV) run agentenv version
	$(UV) run agentenv doctor
	$(UV) run agentenv settings

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
