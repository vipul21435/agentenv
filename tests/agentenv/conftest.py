"""Hermetic defaults for the agentenv tests: no AGENTENV_ variables, no .env, fresh caches."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from agentenv.logs import reset_logging
from agentenv.settings import reset_settings_cache

_ALIASED = {"OPENAI_API_KEY", "OPENAI_BASE_URL"}


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for name in list(os.environ):
        if name.startswith("AGENTENV_") or name in _ALIASED:
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)  # no stray .env from the repository root
    reset_settings_cache()
    reset_logging()
    yield
    reset_logging()
    reset_settings_cache()
