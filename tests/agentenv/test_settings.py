from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentenv.errors import ConfigError
from agentenv.settings import Settings, get_settings, load_settings, reset_settings_cache


def test_defaults_are_offline() -> None:
    settings = load_settings(env_file=None)
    assert settings.sandbox_backend == "subprocess"
    assert settings.model_provider == "stub"
    assert settings.log_level == "INFO"
    assert settings.log_format == "text"
    assert settings.seed == 0
    assert settings.workspace_root is None
    assert settings.openai_api_key is None
    assert settings.openai_base_url is None
    assert settings.docker_image == "python:3.12-slim"


def test_environment_overrides_with_case_normalisation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTENV_LOG_LEVEL", "debug")
    monkeypatch.setenv("AGENTENV_LOG_FORMAT", "JSON")
    monkeypatch.setenv("AGENTENV_SANDBOX_BACKEND", " Docker ")
    monkeypatch.setenv("AGENTENV_SEED", "42")
    monkeypatch.setenv("AGENTENV_WORKSPACE_ROOT", "/tmp/agentenv-ws")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    settings = load_settings(env_file=None)
    assert settings.log_level == "DEBUG"
    assert settings.log_format == "json"
    assert settings.sandbox_backend == "docker"
    assert settings.seed == 42
    assert settings.workspace_root == Path("/tmp/agentenv-ws")
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-test-123"
    assert settings.openai_base_url == "http://localhost:8000/v1"


def test_empty_environment_values_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTENV_WORKSPACE_ROOT", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    settings = load_settings(env_file=None)
    assert settings.workspace_root is None
    assert settings.openai_api_key is None


def test_env_file_is_read(tmp_path: Path) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text("AGENTENV_MODEL_NAME=local-model\nAGENTENV_SEED=3\n")
    settings = load_settings(env_file=env_file)
    assert settings.model_name == "local-model"
    assert settings.seed == 3


def test_keyword_overrides_win() -> None:
    settings = load_settings(env_file=None, seed=9, model_provider="openai")
    assert settings.seed == 9
    assert settings.model_provider == "openai"


def test_invalid_values_raise_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTENV_SANDBOX_BACKEND", "firecracker")
    monkeypatch.setenv("AGENTENV_SEED", "-1")
    with pytest.raises(ConfigError) as info:
        load_settings(env_file=None)
    fields = {error["field"] for error in info.value.details["errors"]}
    assert fields == {"sandbox_backend", "seed"}
    json.dumps(info.value.to_dict())  # details stay JSON serialisable


def test_public_dict_masks_the_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_settings(env_file=None).public_dict()["openai_api_key"] is None
    monkeypatch.setenv("OPENAI_API_KEY", "sk-very-secret")
    public = load_settings(env_file=None).public_dict()
    assert public["openai_api_key"] == "***"
    assert "sk-very-secret" not in json.dumps(public)
    assert public["sandbox_backend"] == "subprocess"


def test_settings_are_frozen() -> None:
    settings = load_settings(env_file=None)
    with pytest.raises(ValidationError):
        settings.seed = 1


def test_get_settings_is_cached_until_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    first = get_settings()
    assert get_settings() is first
    monkeypatch.setenv("AGENTENV_SEED", "5")
    assert get_settings().seed == 0
    reset_settings_cache()
    assert get_settings().seed == 5


def test_settings_class_is_directly_usable() -> None:
    assert Settings(seed=2).seed == 2  # cwd is a temp dir without .env (see conftest)
