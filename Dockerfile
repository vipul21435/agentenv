# syntax=docker/dockerfile:1
# Digest of python:3.12-slim as pulled on 2026-09-29 (docker image inspect --format "{{index .RepoDigests 0}}").
FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f

# uv, pinned to the version that produced uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    MSWEA_SILENT_STARTUP=1 \
    AGENTENV_WORKSPACE_ROOT=/tmp/agentenv/workspaces \
    PATH="/app/.venv/bin:${PATH}"

RUN groupadd --system agentenv \
    && useradd --system --gid agentenv --create-home --home-dir /home/agentenv agentenv

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable \
    && rm -rf /root/.cache \
    && chown -R agentenv:agentenv /app

USER agentenv

# Default: the offline demo (both agents, every task, 3 seeds). Override with e.g. `--help`.
ENTRYPOINT ["agentenv"]
CMD ["run", "--task", "all", "--agent", "scripted", "--agent", "random", "--episodes", "3", "--seed", "0", "--out", "/tmp/agentenv/trajectories.jsonl"]
