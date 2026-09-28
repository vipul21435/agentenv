from __future__ import annotations

import json
import logging
from io import StringIO

import pytest

from agentenv.errors import ConfigError
from agentenv.logs import LOGGER_NAME, configure_logging, get_logger, log_event, reset_logging


def test_json_lines_carry_structured_fields() -> None:
    buf = StringIO()
    logger = configure_logging("DEBUG", "json", stream=buf)
    log_event(logger, "episode.start", task_id="t1", seed=7)
    lines = buf.getvalue().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "episode.start"
    assert record["message"] == "episode.start"
    assert record["task_id"] == "t1"
    assert record["seed"] == 7
    assert record["level"] == "INFO"
    assert record["logger"] == LOGGER_NAME
    assert record["ts"].endswith("+00:00")


def test_text_format_appends_sorted_fields() -> None:
    buf = StringIO()
    logger = configure_logging("INFO", "text", stream=buf)
    log_event(logger, "step.done", reward=1.0, action="ls")
    line = buf.getvalue().strip()
    assert "INFO agentenv: step.done" in line
    assert line.endswith("action='ls' reward=1.0")


def test_plain_logging_calls_still_work_in_json_mode() -> None:
    buf = StringIO()
    logger = configure_logging("INFO", "json", stream=buf)
    logger.info("hello %s", "world")
    record = json.loads(buf.getvalue())
    assert record["event"] == "hello world"
    assert record["message"] == "hello world"


def test_child_loggers_share_the_handler_and_level() -> None:
    buf = StringIO()
    configure_logging("WARNING", "json", stream=buf)
    child = get_logger("cli")
    assert child.name == "agentenv.cli"
    log_event(child, "ignored.info")
    log_event(child, "kept.warning", level=logging.WARNING)
    records = [json.loads(line) for line in buf.getvalue().splitlines()]
    assert [r["event"] for r in records] == ["kept.warning"]
    assert records[0]["logger"] == "agentenv.cli"


def test_reconfigure_replaces_previous_handler() -> None:
    first, second = StringIO(), StringIO()
    configure_logging("INFO", "text", stream=first)
    logger = configure_logging("INFO", "json", stream=second)
    log_event(logger, "once")
    assert first.getvalue() == ""
    assert json.loads(second.getvalue())["event"] == "once"
    assert len(logger.handlers) == 1
    assert logger.propagate is False


def test_exception_info_is_included() -> None:
    buf = StringIO()
    logger = configure_logging("INFO", "json", stream=buf)
    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("failed")
    record = json.loads(buf.getvalue())
    assert record["level"] == "ERROR"
    assert "ValueError: boom" in record["exc_info"]


def test_reserved_field_names_are_rejected() -> None:
    logger = configure_logging("INFO", "json", stream=StringIO())
    with pytest.raises(ValueError, match="reserved"):
        log_event(logger, "bad", message="x")
    with pytest.raises(ValueError, match="reserved"):
        log_event(logger, "bad", event="x")


def test_invalid_format_and_level_raise_config_error() -> None:
    with pytest.raises(ConfigError) as info:
        configure_logging("INFO", "xml")
    assert info.value.details == {"allowed": ["text", "json"]}
    with pytest.raises(ConfigError):
        configure_logging("LOUD", "text")


def test_reset_logging_restores_defaults() -> None:
    logger = configure_logging("DEBUG", "json", stream=StringIO())
    reset_logging()
    assert logger.handlers == []
    assert logger.propagate is True
    assert logger.level == logging.NOTSET
    reset_logging()  # idempotent
