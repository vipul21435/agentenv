"""Structured logging for agentenv.

``configure_logging`` installs a single handler on the ``agentenv`` logger that writes
either human-readable text lines or one JSON object per line. ``log_event`` attaches
structured fields to a record so that both formats carry them.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import IO, Any

from agentenv.errors import ConfigError

LOGGER_NAME = "agentenv"
LOG_FORMATS = ("text", "json")

# Attribute names that logging.LogRecord already uses. Structured fields must not clash
# with them, otherwise logging raises a KeyError while building the record.
_RECORD_ATTRIBUTES = frozenset(
    logging.LogRecord(LOGGER_NAME, logging.INFO, __file__, 0, "", None, None).__dict__
) | frozenset({"message", "asctime"})

_active_handler: logging.Handler | None = None


def _structured_fields(record: logging.LogRecord) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _RECORD_ATTRIBUTES and key != "event" and not key.startswith("_")
    }


class JsonFormatter(logging.Formatter):
    """One JSON object per line with a UTC timestamp and the structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
            "message": record.getMessage(),
        }
        payload.update(_structured_fields(record))
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)


class TextFormatter(logging.Formatter):
    """Human-readable lines; structured fields are appended as ``key=value`` pairs."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        fields = _structured_fields(record)
        if fields:
            line += " " + " ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))
        return line


def configure_logging(level: str = "INFO", fmt: str = "text", stream: IO[str] | None = None) -> logging.Logger:
    """Configure the ``agentenv`` logger. Calling it again replaces the previous handler."""
    global _active_handler
    if fmt not in LOG_FORMATS:
        raise ConfigError(f"unknown log format: {fmt!r}", details={"allowed": list(LOG_FORMATS)})
    level_name = level.upper()
    if level_name not in logging.getLevelNamesMapping():
        raise ConfigError(f"unknown log level: {level!r}")
    logger = logging.getLogger(LOGGER_NAME)
    if _active_handler is not None:
        logger.removeHandler(_active_handler)
        _active_handler.close()
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    logger.addHandler(handler)
    logger.setLevel(level_name)
    logger.propagate = False
    _active_handler = handler
    return logger


def reset_logging() -> None:
    """Remove the handler installed by ``configure_logging`` (used by tests)."""
    global _active_handler
    logger = logging.getLogger(LOGGER_NAME)
    if _active_handler is not None:
        logger.removeHandler(_active_handler)
        _active_handler.close()
        _active_handler = None
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the ``agentenv`` logger or one of its children (``agentenv.<name>``)."""
    return logging.getLogger(LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}")


def log_event(logger: logging.Logger, event: str, /, *, level: int = logging.INFO, **fields: Any) -> None:
    """Log ``event`` with structured ``fields`` attached to the record."""
    clashes = sorted(key for key in fields if key in _RECORD_ATTRIBUTES or key == "event")
    if clashes:
        raise ValueError(f"reserved log field names: {clashes}")
    logger.log(level, event, extra={"event": event, **fields})
