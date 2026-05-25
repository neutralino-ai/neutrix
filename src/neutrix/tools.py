"""Built-in tools exposed to the LLM via OpenAI function-calling schema.

Tools are intentionally minimal and safe-by-default. Shell execution prints
a confirmation prompt that the TUI surfaces to the user.
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from neutrix.store import ChatStore

# Status names the LLM may pass to TaskUpdate. Matches Claude Code's
# `TaskUpdateStatusSchema().or(z.literal("deleted"))` shape — "deleted"
# is the action that removes the task, not a stored status value.
_TASK_UPDATE_STATUSES = ("pending", "in_progress", "completed", "deleted")
_STORE_REQUIRED_TOOLS = frozenset({"TaskCreate", "TaskUpdate", "TaskList"})


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


def _task_create(
    subject: str,
    description: str = "",
    *,
    store: ChatStore | None = None,
) -> str:
    if store is None:
        return "ERROR: TaskCreate requires a ChatStore"
    if not subject:
        return "ERROR: subject is required"
    task = store.add_task(subject, description=description or "")
    return f"ok, created task {task.id}: {task.subject}"


def _task_update(
    taskId: str,  # ruff: noqa - LLM-facing name matches Claude Code's tool
    status: str | None = None,
    subject: str | None = None,
    description: str | None = None,
    *,
    store: ChatStore | None = None,
) -> str:
    if store is None:
        return "ERROR: TaskUpdate requires a ChatStore"
    if status is not None and status not in _TASK_UPDATE_STATUSES:
        allowed = ", ".join(_TASK_UPDATE_STATUSES)
        return f"ERROR: status must be one of {allowed}"

    if status == "deleted":
        removed = store.remove_task(taskId)
        if removed is None:
            return f"task {taskId} not found"
        return f"ok, deleted task {removed.id}: {removed.subject}"

    updated = store.update_task(
        taskId,
        status=status,
        subject=subject,
        description=description,
    )
    if updated is None:
        return f"task {taskId} not found"
    changed: list[str] = []
    if status is not None:
        changed.append(f"status={status}")
    if subject is not None:
        changed.append(f"subject={subject!r}")
    if description is not None:
        changed.append(f"description={description!r}")
    summary = ", ".join(changed) if changed else "no fields changed"
    return f"ok, task {updated.id} updated: {summary}"


def _task_list(*, store: ChatStore | None = None) -> str:
    if store is None:
        return "ERROR: TaskList requires a ChatStore"
    items = [
        {
            "id": task.id,
            "subject": task.subject,
            "status": task.status,
            "description": task.description,
        }
        for task in store.tasks
    ]
    return json.dumps(items)


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
    "TaskCreate": Tool(
        name="TaskCreate",
        description=(
            "Add a new task to the session task list. Use when the user "
            "agrees to track work, or when you want to remember items "
            "that span multiple turns. Returns the new task id."
        ),
        parameters={
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Short title (one line).",
                },
                "description": {
                    "type": "string",
                    "description": "Optional longer detail.",
                },
            },
            "required": ["subject"],
        },
        func=_task_create,
    ),
    "TaskUpdate": Tool(
        name="TaskUpdate",
        description=(
            "Update an existing task's status, subject, or description. "
            "Set status to 'in_progress' when starting work, 'completed' "
            "when done, or 'deleted' to remove the task entirely."
        ),
        parameters={
            "type": "object",
            "properties": {
                "taskId": {
                    "type": "string",
                    "description": "Id returned by TaskCreate.",
                },
                "status": {
                    "type": "string",
                    "enum": list(_TASK_UPDATE_STATUSES),
                    "description": (
                        "New status, or 'deleted' to remove the task."
                    ),
                },
                "subject": {
                    "type": "string",
                    "description": "Replacement subject.",
                },
                "description": {
                    "type": "string",
                    "description": "Replacement description.",
                },
            },
            "required": ["taskId"],
        },
        func=_task_update,
    ),
    "TaskList": Tool(
        name="TaskList",
        description=(
            "Return the full session task list as a JSON array of "
            "{id, subject, status, description}. Use when you need to "
            "re-orient on every tracked item, not just the actionable "
            "subset surfaced by the system reminder."
        ),
        parameters={
            "type": "object",
            "properties": {},
        },
        func=_task_list,
    ),
}


def get_schemas() -> list[dict[str, Any]]:
    return [t.schema() for t in BUILTIN_TOOLS.values()]


def dispatch(
    name: str,
    arguments_json: str,
    *,
    store: ChatStore | None = None,
) -> str:
    """Look up `name` in the registry and call it with parsed JSON args.

    The ``store`` keyword is forwarded only to tools whose implementation
    declares a ``store`` keyword parameter (currently ``TaskCreate``,
    ``TaskUpdate``, ``TaskList``); all other tools see their original
    signature. This keeps existing callers working while letting the
    task tools mutate :class:`neutrix.store.ChatStore` directly.

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
        signature = inspect.signature(tool.func)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and "store" in signature.parameters:
        args.setdefault("store", store)
    try:
        return tool.func(**args)
    except TypeError as e:
        return f"ERROR: bad arguments: {e}"
    except Exception as e:
        logger.exception("tool {} crashed", name)
        return f"ERROR: tool crashed: {e}"
