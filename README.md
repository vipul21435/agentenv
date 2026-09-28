Forked from https://github.com/SWE-agent/mini-swe-agent.

# AgentEnv

A gym-style environment for tool-using LLM agents with verifiable rewards.
`reset()` hands an agent a seeded task inside a sandbox (subprocess by default, Docker
with no network and resource limits when available), `step()` runs one tool call
(shell, file or python) and returns an observation, a reward from a pytest-checked
reward function and a `done` flag, and every episode is written as JSONL for RL
training. It is built for a laptop: no GPU, no paid API key, deterministic episodes.

Status: slice 1 of 7. What exists today is the fork baseline (upstream tests green on
Python 3.12), modern tooling with CI, and the `agentenv` package skeleton: typed
settings, an error hierarchy, structured logging and an `agentenv doctor` command.
The environment API, sandbox tools, task suite, trajectory logging, agents, reports
and the FastAPI service follow in the next slices.

## Why this fork

mini-swe-agent (upstream, MIT) is a deliberately small agent scaffold: a ~100-line
agent loop, command executors for subprocess, Docker/Podman, Singularity and
bubblewrap, litellm-based model adapters, YAML config templates, a trajectory
inspector TUI and a ~600-test suite. That is exactly the thin base an RL environment
needs underneath: the executors become the sandbox layer and the agent loop becomes
the optional LLM agent, while the environment API, tasks and rewards are new work.
Upstream code lives untouched in `src/minisweagent`; everything new lives in
`src/agentenv`.

## Quickstart

```bash
uv sync --group dev          # Python 3.12, locked dependencies, offline
make ci                      # ruff, ruff format --check, mypy, pytest with coverage
make demo                    # agentenv version / doctor / settings
cp .env.example .env         # optional: every variable has an offline default
uv run agentenv doctor --json
```

`make test` runs the suite in parallel without the tests marked `slow`; tests that
need a docker daemon carry the `docker` marker and skip cleanly when none is running.

## What I built on top

- Tooling baseline: `pyproject.toml` for Python 3.12 with a committed `uv.lock`, ruff
  lint + format, strict mypy for the new package, pytest + pytest-cov + pytest-xdist,
  pre-commit hooks, a Makefile (`lint`, `format`, `typecheck`, `test`, `test-all`,
  `ci`, `demo`, `clean`), `.env.example`, and a single offline GitHub Actions
  workflow; upstream's hosted-service integrations (Portkey, Modal, Contree, SWE-ReX)
  moved behind optional extras so the default install and test run need no account.
- `agentenv` package skeleton: typed settings (`pydantic-settings`, `AGENTENV_`
  prefix, `.env` support, secrets masked), an error hierarchy with stable machine
  readable codes (`ConfigError`, `SandboxError`, `SandboxTimeoutError`, `TaskError`,
  `EpisodeError`), structured text/JSON logging with `log_event`, a docker probe, and
  the `agentenv version | doctor | settings` CLI (Typer) with tests at 98% coverage.

## Layout

```
src/minisweagent/   upstream mini-swe-agent (unchanged apart from ruff formatting)
src/agentenv/       new package: settings, errors, logs, doctor, cli
tests/agentenv/     tests for the new package (unit + a subprocess integration test)
tests/...           upstream test suite (kept green)
docs/               upstream docs; docs/UPSTREAM_README.md is the original README
```

## Design decisions

- Python 3.12 only, managed by uv; `uv.lock` is committed and CI installs with
  `uv sync --locked`.
- Defaults are offline: subprocess sandbox, deterministic `stub` model provider, no
  key. Real providers are opt-in through `AGENTENV_MODEL_PROVIDER=openai` plus
  `OPENAI_API_KEY` / `OPENAI_BASE_URL` (a local OpenAI-compatible server works).
- Hosted or paid services stay behind extras (`portkey`, `swerex`, `modal`,
  `contree`, `docs`, `full`); their tests `importorskip` the SDKs. `portkey-ai` is
  installed in the dev group because its tests are fully mocked.
- Upstream's workflows (docs build, PyPI release, link checks, pylint, pytest with
  API secrets, podman and apptainer) were replaced by one workflow that runs the same
  checks as `make ci` without secrets.
- mypy runs strict on `src/agentenv` and `tests/agentenv`. Upstream `minisweagent`
  is not strictly typed (36 errors at fork time) and is imported with
  `follow_imports = "silent"` rather than patched.
- The upstream startup banner is silenced in tests through `MSWEA_SILENT_STARTUP=1`
  (set in `tests/conftest.py`, the Makefile and CI).

## License

MIT. The upstream copyright notice is kept in `LICENSE.md`; new code is
Copyright 2026 Vipul Raj Jha under the same license.
