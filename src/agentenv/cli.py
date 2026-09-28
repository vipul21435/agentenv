"""Command-line interface: ``version``, ``doctor``, ``settings``, ``tasks``, ``run`` and ``report``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from agentenv import __version__
from agentenv.doctor import build_report
from agentenv.errors import AgentEnvError
from agentenv.logs import configure_logging, get_logger, log_event
from agentenv.runner import run_suite, select_tasks
from agentenv.settings import Settings, load_settings
from agentenv.tasks import list_tasks
from agentenv.trajectory import build_report as build_trajectory_report
from agentenv.trajectory import read_trajectory

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


@app.command("tasks")
def show_tasks() -> None:
    """List the built-in tasks."""
    for task in list_tasks():
        typer.echo(f"{task.id:<20} max_steps={task.max_steps:<3} {task.description}")


@app.command()
def run(
    task: Annotated[str, typer.Option("--task", help="Task id, comma-separated ids, or 'all'.")] = "all",
    agent: Annotated[
        list[str] | None, typer.Option("--agent", help="Agent name (repeatable): scripted, random. Default: scripted.")
    ] = None,
    episodes: Annotated[int, typer.Option("--episodes", min=1, help="Episodes per task and agent.")] = 1,
    seed: Annotated[int, typer.Option("--seed", min=0, help="First seed; episode i uses seed + i.")] = 0,
    out: Annotated[Path, typer.Option("--out", help="JSONL trajectory file (one line per step).")] = Path(
        "trajectories.jsonl"
    ),
) -> None:
    """Run agents on tasks, write the JSONL trajectory and print the success-rate table."""
    settings = _settings()
    try:
        summaries = run_suite(
            tasks=select_tasks(task),
            agent_names=agent or ["scripted"],
            episodes=episodes,
            seed=seed,
            out=out,
            workspace_root=settings.workspace_root,
        )
    except AgentEnvError as exc:
        typer.echo(json.dumps(exc.to_dict(), default=str), err=True)
        raise typer.Exit(code=EXIT_CONFIG) from exc
    seconds = sum(s.seconds for s in summaries)
    typer.echo(build_trajectory_report(read_trajectory(out)).render())
    typer.echo(f"wrote {out} ({seconds:.2f}s of episodes, {seconds / len(summaries):.3f}s per episode)")


@app.command()
def report(
    trajectory: Annotated[Path, typer.Argument(help="JSONL file written by 'agentenv run'.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Summarise a trajectory file: success rate and mean steps per task and agent."""
    try:
        result = build_trajectory_report(read_trajectory(trajectory))
    except AgentEnvError as exc:
        typer.echo(json.dumps(exc.to_dict(), default=str), err=True)
        raise typer.Exit(code=EXIT_CONFIG) from exc
    typer.echo(result.model_dump_json(indent=2) if json_output else result.render())


def main() -> None:  # pragma: no cover - exercised through ``python -m agentenv`` in a subprocess
    """Console-script entry point."""
    app()
