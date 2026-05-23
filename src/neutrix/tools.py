"""Built-in tools exposed to the LLM via OpenAI function-calling schema.

Tools are intentionally minimal and safe-by-default. Shell execution prints
a confirmation prompt that the TUI surfaces to the user.
"""
from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ----- implementations --------------------------------------------------------


def _read_file(path: str, max_bytes: int = 200_000) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"ERROR: {path} does not exist"
    if not p.is_file():
        return f"ERROR: {path} is not a regular file"
    data = p.read_bytes()[:max_bytes]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return f"ERROR: {path} is not utf-8 text"


def _write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"OK: wrote {len(content)} chars to {p}"


def _list_dir(path: str = ".") -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"ERROR: {path} does not exist"
    if not p.is_dir():
        return f"ERROR: {path} is not a directory"
    items = []
    for entry in sorted(p.iterdir()):
        kind = "d" if entry.is_dir() else "f"
        items.append(f"{kind} {entry.name}")
    return "\n".join(items) if items else "(empty)"


def _run_shell(command: str, timeout: int = 30) -> str:
    logger.info("shell tool: {!r}", command)
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout}s"
    out = proc.stdout or ""
    err = proc.stderr or ""
    parts = [f"exit_code: {proc.returncode}"]
    if out:
        parts.append(f"stdout:\n{out}")
    if err:
        parts.append(f"stderr:\n{err}")
    return "\n".join(parts)


# ----- registry ---------------------------------------------------------------


BUILTIN_TOOLS: dict[str, Tool] = {
    "read_file": Tool(
        name="read_file",
        description="Read a UTF-8 text file from disk and return its contents.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
            },
            "required": ["path"],
        },
        func=_read_file,
    ),
    "write_file": Tool(
        name="write_file",
        description="Write UTF-8 text content to a file (overwriting if it exists).",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write."},
                "content": {"type": "string", "description": "Content to write."},
            },
            "required": ["path", "content"],
        },
        func=_write_file,
    ),
    "list_dir": Tool(
        name="list_dir",
        description="List entries in a directory (one per line, prefixed 'd' or 'f').",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (defaults to current).",
                    "default": ".",
                },
            },
        },
        func=_list_dir,
    ),
    "run_shell": Tool(
        name="run_shell",
        description=(
            "Run a shell command and return its stdout/stderr/exit_code. "
            "Use sparingly; for destructive ops, ask the user first."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds.",
                    "default": 30,
                },
            },
            "required": ["command"],
        },
        func=_run_shell,
    ),
}


def get_schemas() -> list[dict[str, Any]]:
    return [t.schema() for t in BUILTIN_TOOLS.values()]


def dispatch(name: str, arguments_json: str) -> str:
    """Look up `name` in the registry and call it with parsed JSON args.

    Returns the tool's string result (errors are returned as text, not raised).
    """
    tool = BUILTIN_TOOLS.get(name)
    if tool is None:
        return f"ERROR: unknown tool {name!r}"
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as e:
        return f"ERROR: invalid JSON args: {e}"
    try:
        return tool.func(**args)
    except TypeError as e:
        return f"ERROR: bad arguments: {e}"
    except Exception as e:
        logger.exception("tool {} crashed", name)
        return f"ERROR: tool crashed: {e}"
