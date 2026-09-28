"""Typed settings for agentenv.

Values come from the process environment (``AGENTENV_`` prefix) and an optional ``.env``
file. Everything defaults to an offline configuration: subprocess sandbox, deterministic
stub model provider, no API key.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, ValidationError, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentenv.errors import ConfigError

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
LogFormat = Literal["text", "json"]
SandboxBackend = Literal["subprocess", "docker"]
ModelProvider = Literal["stub", "openai"]

DEFAULT_ENV_FILE = ".env"


class Settings(BaseSettings):
    """Effective agentenv configuration (immutable once loaded)."""

    model_config = SettingsConfigDict(
        env_prefix="AGENTENV_",
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
    )

    log_level: LogLevel = "INFO"
    log_format: LogFormat = "text"
    sandbox_backend: SandboxBackend = "subprocess"
    docker_image: str = Field(default="python:3.12-slim", min_length=1)
    docker_executable: str = Field(default="docker", min_length=1)
    seed: int = Field(default=0, ge=0)
    workspace_root: Path | None = None
    model_provider: ModelProvider = "stub"
    model_name: str = Field(default="gpt-4o-mini", min_length=1)
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")

    @field_validator("log_level", "log_format", "sandbox_backend", "model_provider", mode="before")
    @classmethod
    def _normalise_case(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, str):
            return value.strip().upper() if info.field_name == "log_level" else value.strip().lower()
        return value

    def public_dict(self) -> dict[str, Any]:
        """Settings as JSON-friendly values with the API key masked."""
        data = self.model_dump(mode="json")
        data["openai_api_key"] = "***" if self.openai_api_key is not None else None
        return data


def load_settings(*, env_file: str | Path | None = DEFAULT_ENV_FILE, **overrides: Any) -> Settings:
    """Build settings from the environment (and ``env_file``); raise ``ConfigError`` on invalid values."""
    try:
        # ``_env_file`` is a pydantic-settings init keyword, not a field; mypy synthesises
        # ``Settings.__init__`` from the fields, so it is passed through a dict unpack.
        return Settings(**{"_env_file": env_file, **overrides})
    except ValidationError as exc:
        errors = [
            {
                "field": ".".join(str(part) for part in error["loc"]),
                "message": error["msg"],
                "input": error.get("input"),
            }
            for error in exc.errors(include_url=False)
        ]
        raise ConfigError("invalid agentenv settings", details={"errors": errors}) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings."""
    return load_settings()


def reset_settings_cache() -> None:
    """Forget the cached settings so the next ``get_settings()`` re-reads the environment."""
    get_settings.cache_clear()
