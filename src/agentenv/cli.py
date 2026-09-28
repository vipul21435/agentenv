"""Command-line interface: ``agentenv version``, ``agentenv doctor`` and ``agentenv settings``."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from agentenv import __version__
from agentenv.doctor import build_report
from agentenv.errors import AgentEnvError
from agentenv.logs import configure_logging, get_logger, log_event
from agentenv.settings import Settings, load_settings

app = typer.Typer(
    name="agentenv",
    help="Gym-style environment for tool-using LLM agents with verifiable rewards.",
    no_args_is_help=True,
    add_completion=False,
)
log = get_logger("cli")

EXIT_PROBLEMS = 1
EXIT_CONFIG = 2


def _settings() -> Settings:
    """Load settings, configure logging from them, exit with code 2 on invalid config."""
    try:
        settings = load_settings()
    except AgentEnvError as exc:
        typer.echo(json.dumps(exc.to_dict(), default=str), err=True)
        raise typer.Exit(code=EXIT_CONFIG) from exc
    configure_logging(settings.log_level, settings.log_format)
    return settings


@app.command()
def version() -> None:
    """Print the agentenv version."""
    typer.echo(f"agentenv {__version__}")


@app.command()
def doctor(
    json_output: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Check the local toolchain: python, docker, sandbox backend and model provider."""
    settings = _settings()
    report = build_report(settings)
    log_event(
        log,
        "doctor.report",
        ok=report.ok,
        sandbox_backend=report.sandbox_backend,
        docker_available=report.docker.available,
        problems=len(report.problems),
    )
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        typer.echo(f"agentenv {report.agentenv_version} on python {report.python_version} ({report.platform})")
        typer.echo(f"sandbox backend: {report.sandbox_backend}")
        typer.echo(f"model provider: {report.model_provider}")
        if report.docker.available:
            docker_state = f"available (server {report.docker.version})"
        else:
            docker_state = f"unavailable ({report.docker.error})"
        typer.echo(f"docker ({report.docker.executable}): {docker_state}")
        for problem in report.problems:
            typer.echo(f"problem: {problem}")
        typer.echo("ok" if report.ok else "not ok")
    if not report.ok:
        raise typer.Exit(code=EXIT_PROBLEMS)


@app.command("settings")
def show_settings() -> None:
    """Print the effective settings as JSON (secrets masked)."""
    settings = _settings()
    typer.echo(json.dumps(settings.public_dict(), indent=2, sort_keys=True))


def main() -> None:  # pragma: no cover - exercised through ``python -m agentenv`` in a subprocess
    """Console-script entry point."""
    app()
