"""Typed actions an agent can take: a discriminated union keyed on ``kind``."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter


class ShellAction(BaseModel):
    kind: Literal["shell"] = "shell"
    command: str = Field(min_length=1)


class ReadFileAction(BaseModel):
    kind: Literal["read_file"] = "read_file"
    path: str = Field(min_length=1)


class WriteFileAction(BaseModel):
    kind: Literal["write_file"] = "write_file"
    path: str = Field(min_length=1)
    content: str


class PythonAction(BaseModel):
    kind: Literal["python"] = "python"
    code: str = Field(min_length=1)


Action = Annotated[ShellAction | ReadFileAction | WriteFileAction | PythonAction, Field(discriminator="kind")]
ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)


def parse_action(data: dict[str, Any]) -> Action:
    """Validate a JSON-like mapping into a typed action."""
    return ACTION_ADAPTER.validate_python(data)
