"""Built-in tools exposed to the LLM via OpenAI function-calling schema.

Tools are intentionally minimal and safe-by-default. Shell execution prints
a confirmation prompt that the TUI surfaces to the user.
"""
from __future__ import annotations

import asyncio
import contextvars
import fnmatch
import inspect
import json
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from neutrix.store import ChatStore

if TYPE_CHECKING:
    from neutrix.config import Slot
    from neutrix.executor import Executor

# v0.10.0 recursion backstop. A contextvar (NOT threading.local): the subagent
# dispatches its own tools via ``asyncio.to_thread``, which copies the current
# context across the thread boundary, so this flag propagates into any nested
# ``Agent`` dispatch even though it runs on a different thread. Schema-scoping
# (the subagent never sees ``Agent``) is primary; this catches the rare model
# that calls an unlisted tool.
_inside_subagent: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "neutrix_inside_subagent", default=False
)

# Status names the LLM may pass to TaskUpdate. Matches Claude Code's
# `TaskUpdateStatusSchema().or(z.literal("deleted"))` shape — "deleted"
# is the action that removes the task, not a stored status value.
_TASK_UPDATE_STATUSES = ("pending", "in_progress", "completed", "deleted")
_STORE_REQUIRED_TOOLS = frozenset({"TaskCreate", "TaskUpdate", "TaskList"})


# Tool descriptions sent to the LLM. Lifted verbatim from Claude Code's
# V2 task tool prompts (cc2/src/tools/{TaskCreateTool,TaskUpdateTool,
# TaskListTool}/prompt.ts) with the agent-swarm-only sections dropped —
# neutrix has no teammate-swarm features. These descriptions are the
# primary mechanism that shapes LLM behavior around the task tools;
# putting the "Mark in_progress BEFORE beginning work" etc. guidance
# here (rather than in result-text nudges) matches the way Claude Code
# itself drives the LLM to actually start, update, and complete tasks.

_TASK_CREATE_DESCRIPTION = """\
Use this tool proactively to capture the user's full multi-step plan as **separate tasks**, not piece by piece. This helps you track progress, organize complex work, and demonstrate thoroughness to the user.
It also helps the user understand the progress of the task and overall progress of their requests.

## When to Use This Tool

Use this tool proactively in these scenarios:

- After receiving new instructions — immediately capture every distinct step the request implies as its own task, BEFORE starting work on any of them. Do NOT add tasks one at a time as you discover them mid-work; the user expects to see the full plan upfront.
- Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
- Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
- Plan mode - When using plan mode, create a task list to track the work
- User explicitly requests todo list - When the user directly asks you to use the todo list
- User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
- When you start working on a task - Mark it as in_progress BEFORE beginning work
- After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation

## When NOT to Use This Tool

Skip using this tool when:
- There is only a single, straightforward task
- The task is trivial and tracking it provides no organizational benefit
- The task can be completed in less than 3 trivial steps
- The task is purely conversational or informational

NOTE that you should not use this tool if there is only one trivial task to do. In this case you are better off just doing the task directly.

## Task Fields

- **subject**: A brief, actionable title in imperative form (e.g., "Fix authentication bug in login flow")
- **description**: What needs to be done

All tasks are created with status `pending`.

## Tips

- Create tasks with clear, specific subjects that describe the outcome
- Check TaskList first to avoid creating duplicate tasks
"""

_TASK_UPDATE_DESCRIPTION = """\
Use this tool to update a task in the task list.

## When to Use This Tool

**Mark tasks as resolved:**
- When you have completed the work described in a task
- When a task is no longer needed or has been superseded
- IMPORTANT: Always mark your assigned tasks as resolved when you finish them
- After resolving, call TaskList to find your next task

- ONLY mark a task as completed when you have FULLY accomplished it
- If you encounter errors, blockers, or cannot finish, keep the task as in_progress
- When blocked, create a new task describing what needs to be resolved
- Never mark a task as completed if:
  - Tests are failing
  - Implementation is partial
  - You encountered unresolved errors
  - You couldn't find necessary files or dependencies

**Delete tasks:**
- When a task is no longer relevant or was created in error
- Setting status to `deleted` permanently removes the task

**Update task details:**
- When requirements change or become clearer

## Fields You Can Update

- **status**: The task status (see Status Workflow below)
- **subject**: Change the task title (imperative form, e.g., "Run tests")
- **description**: Change the task description

## Status Workflow

Status progresses: `pending` → `in_progress` → `completed`

Use `deleted` to permanently remove a task.

## Examples

Mark task as in progress when starting work:
```json
{"taskId": "1", "status": "in_progress"}
```

Mark task as completed after finishing work:
```json
{"taskId": "1", "status": "completed"}
```

Delete a task:
```json
{"taskId": "1", "status": "deleted"}
```
"""

_TASK_LIST_DESCRIPTION = """\
Use this tool to list all tasks in the task list.

## When to Use This Tool

- To see what tasks are available to work on (status: 'pending', not blocked)
- To check overall progress on the project
- After completing a task, to check for newly unblocked work or the next available task
- **Prefer working on tasks in ID order** (lowest ID first) when multiple tasks are available, as earlier tasks often set up context for later ones

## Output

Returns a JSON array. Each entry has:
- **id**: Task identifier (use with TaskUpdate)
- **subject**: Brief description of the task
- **status**: 'pending', 'in_progress', or 'completed'
- **description**: Full description of what needs to be done
"""


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


_READ_DEFAULT_LIMIT = 2000


def _resolved(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _read(
    path: str,
    offset: int = 0,
    limit: int = _READ_DEFAULT_LIMIT,
    *,
    executor: Executor | None = None,
) -> str:
    """Read a UTF-8 text file as ``cat -n`` lines; window via offset/limit.

    Records the path as read on the executor (v1.1.0 read-before-edit). The
    ``limit`` cap keeps a huge file from blowing the round (feeds compaction).
    """
    p = Path(path).expanduser()
    if not p.exists():
        return f"ERROR: {path} does not exist"
    if not p.is_file():
        return f"ERROR: {path} is not a regular file"
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"ERROR: {path} is not utf-8 text (binary/multimodal Read is a v2.x stretch)"
    if executor is not None:
        executor.mark_read(_resolved(path))
    lines = text.splitlines()
    start = max(0, int(offset))
    end = start + max(1, int(limit))
    window = lines[start:end]
    if not window:
        return f"(no lines at offset {start}; file has {len(lines)} lines)"
    body = "\n".join(f"{start + i + 1:6d}\t{ln}" for i, ln in enumerate(window))
    if end < len(lines):
        body += f"\n… [{len(lines) - end} more lines; pass offset={end} to continue]"
    return body


def _edit(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    *,
    executor: Executor | None = None,
) -> str:
    """Exact, unique str-replace (or replace_all). Requires a prior Read."""
    p = Path(path).expanduser()
    if not p.exists():
        return f"ERROR: {path} does not exist"
    if executor is not None and _resolved(path) not in executor.read_paths:
        return f"ERROR: Read {path} before editing it (read-before-edit)"
    if old_string == new_string:
        return "ERROR: old_string and new_string are identical"
    try:
        content = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"ERROR: {path} is not utf-8 text"
    count = content.count(old_string)
    if count == 0:
        return f"ERROR: old_string not found in {path}"
    if count > 1 and not replace_all:
        return (
            f"ERROR: old_string appears {count} times in {path}; add surrounding "
            "context to make it unique, or pass replace_all=true"
        )
    new_content = (
        content.replace(old_string, new_string)
        if replace_all
        else content.replace(old_string, new_string, 1)
    )
    p.write_text(new_content, encoding="utf-8")
    n = count if replace_all else 1
    before, after = content.count("\n") + 1, new_content.count("\n") + 1
    return f"OK: edited {p} · {n} replacement{'s' if n != 1 else ''} · {before}→{after} lines"


def _write(
    path: str,
    content: str,
    *,
    executor: Executor | None = None,
) -> str:
    """Write/overwrite a file. Overwriting an existing file requires a Read."""
    p = Path(path).expanduser()
    if (
        p.exists()
        and executor is not None
        and _resolved(path) not in executor.read_paths
    ):
        return f"ERROR: {path} exists — Read it before overwriting"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    if executor is not None:
        executor.mark_read(_resolved(path))  # now known
    return f"OK: wrote {len(content)} chars to {p}"


def _glob(pattern: str, path: str = ".") -> str:
    """Glob files under ``path`` (supports ``**``), newest first, capped at 100."""
    base = Path(path).expanduser()
    if not base.exists():
        return f"ERROR: {path} does not exist"
    try:
        matches = [m for m in base.glob(pattern) if m.is_file()]
    except (ValueError, OSError) as exc:
        return f"ERROR: bad glob pattern: {exc}"
    matches.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    files = [str(m) for m in matches[:100]]
    if not files:
        return "(no matches)"
    out = "\n".join(files)
    if len(matches) > 100:
        out += f"\n… [{len(matches) - 100} more; refine the pattern]"
    return out


_GREP_OUTPUT_MODES = ("content", "files_with_matches", "count")


def _grep(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    output_mode: str = "files_with_matches",
    case_insensitive: bool = False,
) -> str:
    """Search file contents by regex. Uses ripgrep when on PATH, else Python.

    ``output_mode``: files_with_matches (default) | content | count.
    """
    if output_mode not in _GREP_OUTPUT_MODES:
        return f"ERROR: output_mode must be one of {', '.join(_GREP_OUTPUT_MODES)}"
    rg = shutil.which("rg")
    if rg:
        return _grep_rg(rg, pattern, path, glob, output_mode, case_insensitive)
    return _grep_python(pattern, path, glob, output_mode, case_insensitive)


def _grep_rg(
    rg: str, pattern: str, path: str, glob: str | None, output_mode: str, ci: bool
) -> str:
    args = [rg, "--no-heading", "--color", "never"]
    if ci:
        args.append("-i")
    if glob:
        args += ["--glob", glob]
    if output_mode == "files_with_matches":
        args.append("-l")
    elif output_mode == "count":
        args.append("-c")
    else:
        args.append("-n")
    args += ["-e", pattern, path]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return f"ERROR: rg failed: {exc}"
    if proc.returncode not in (0, 1):  # 1 = no matches (not an error)
        return f"ERROR: rg: {proc.stderr.strip()}"
    out = proc.stdout.strip()
    return out or "(no matches)"


def _grep_python(
    pattern: str, path: str, glob: str | None, output_mode: str, ci: bool
) -> str:
    try:
        rx = re.compile(pattern, re.IGNORECASE if ci else 0)
    except re.error as exc:
        return f"ERROR: bad regex: {exc}"
    base = Path(path).expanduser()
    files: list[Path] = (
        [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
    )
    skip_dirs = {".git", ".svn", ".hg", "__pycache__", ".venv", "node_modules"}
    matched_files: list[str] = []
    content_lines: list[str] = []
    counts: list[str] = []
    for f in files:
        if any(part in skip_dirs for part in f.parts):
            continue
        if glob and not fnmatch.fnmatch(f.name, glob):
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits = [(i + 1, ln) for i, ln in enumerate(text.splitlines()) if rx.search(ln)]
        if not hits:
            continue
        matched_files.append(str(f))
        counts.append(f"{f}:{len(hits)}")
        for lineno, ln in hits:
            content_lines.append(f"{f}:{lineno}:{ln}")
        if len(content_lines) > 500:
            content_lines.append("… [truncated; refine the search]")
            break
    if output_mode == "files_with_matches":
        return "\n".join(matched_files) or "(no matches)"
    if output_mode == "count":
        return "\n".join(counts) or "(no matches)"
    return "\n".join(content_lines) or "(no matches)"


def _run_shell(
    command: str,
    timeout: int = 30,
    *,
    executor: Executor | None = None,
) -> str:
    """Run a shell command in a fresh process group so it can be tree-killed.

    ``start_new_session=True`` puts the child (and any grandchildren
    it spawns through e.g. a shell pipeline) into its own process
    group, so :func:`neutrix.executor._tree_kill` reaches the whole
    tree with one ``killpg``. Registers the Popen with the
    ``executor``'s cancellation pool before blocking on
    ``communicate``; unregisters in ``finally`` even on the timeout
    path.
    """
    logger.info("shell tool: {!r}", command)
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.getcwd(),
        start_new_session=True,
    )
    if executor is not None:
        executor.register_cancellable(proc)
    timed_out = False
    try:
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            timed_out = True
    finally:
        if executor is not None:
            executor.unregister_cancellable(proc)
    if timed_out:
        return f"ERROR: command timed out after {timeout}s"
    if (
        executor is not None
        and executor._cancel_requested
        and proc.returncode is not None
        and proc.returncode < 0
    ):
        return "[cancelled by user]"
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
    return f"Task #{task.id} created successfully: {task.subject}"


def _task_update(
    taskId: str,
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
            return f"Task #{taskId} not found"
        return f"Updated task #{removed.id} deleted"

    updated = store.update_task(
        taskId,
        status=status,
        subject=subject,
        description=description,
    )
    if updated is None:
        return f"Task #{taskId} not found"
    changed: list[str] = []
    if status is not None:
        changed.append("status")
    if subject is not None:
        changed.append("subject")
    if description is not None:
        changed.append("description")
    summary = ", ".join(changed) if changed else "no fields changed"
    return f"Updated task #{updated.id} {summary}"


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


_AGENT_DESCRIPTION = """\
Dispatch a fresh-context sub-agent to complete one self-contained task and return its result.

The sub-agent runs with its own conversation and tools, works the task to completion, and returns ONLY its final answer — so your own context grows by just that answer, not by all the intermediate work. Use this to delegate focused, context-heavy sub-tasks (read and summarize many files, explore a part of the codebase, draft something from gathered material) without inflating this conversation.

## When to use
- A sub-task needs to read/inspect a lot of material but you only need the conclusion.
- Exploratory work whose intermediate steps don't need to live in this conversation.

## When NOT to use
- A single quick tool call you can make yourself.
- Work that needs your full conversation context to make sense.

Notes: the sub-agent cannot ask you questions (it runs unattended) and cannot itself dispatch sub-agents (single level). `subagent_type` is `general-purpose` or the name of a custom agent defined in `.claude/agents/`.
"""


def _agent(
    description: str,
    prompt: str,
    subagent_type: str = "general-purpose",
    *,
    executor: Executor | None = None,
    slot: Slot | None = None,
) -> str:
    """Dispatch a subagent and return its final text (v0.10.0).

    Sync (runs in the executor's worker thread); drives the async subagent
    via ``asyncio.run`` on this thread's own event loop. Builds a fresh LLM
    from the parent ``slot`` to avoid sharing the parent's loop-bound client.
    """
    if _inside_subagent.get():
        return (
            "ERROR: Agent cannot be called from inside a sub-agent "
            "(sub-agents are single-level — complete the task with your own tools)"
        )
    if not prompt:
        return "ERROR: prompt is required"
    if slot is None:
        return "ERROR: Agent is unavailable (no slot wired to the executor)"

    # v1.3.0: subagent_type may name a custom agent from .claude/agents/*.md;
    # general-purpose uses the default worker prompt.
    custom_system_prompt: str | None = None
    if subagent_type != "general-purpose":
        from neutrix.skills import discover_agents

        agents = discover_agents(os.getcwd())
        agent_def = agents.get(subagent_type.lower())
        if agent_def is None:
            avail = ", ".join(["general-purpose", *sorted(agents)])
            return f"ERROR: unknown subagent_type {subagent_type!r}; available: {avail}"
        custom_system_prompt = agent_def.system_prompt

    from neutrix.llm import CANCELLED_TOOL_RESULT, OpenAIChatLLM
    from neutrix.subagent import SUBAGENT_SYSTEM_PROMPT, run_subagent

    llm = OpenAIChatLLM(slot)
    cancel_event = threading.Event()
    if executor is not None:
        executor.register_cancel_event(cancel_event)
    token = _inside_subagent.set(True)
    try:
        result = asyncio.run(
            run_subagent(
                user_prompt=prompt,
                slot=slot,
                llm=llm,
                tool_names=subagent_tool_names(),
                cancel_event=cancel_event,
                system_prompt=custom_system_prompt or SUBAGENT_SYSTEM_PROMPT,
            )
        )
    finally:
        _inside_subagent.reset(token)
        if executor is not None:
            executor.unregister_cancel_event(cancel_event)

    if result.cancelled:
        return CANCELLED_TOOL_RESULT
    if result.error:
        return f"[subagent error: {result.error}]"
    return result.final_text


def subagent_tool_names() -> frozenset[str]:
    """The tool allowlist a subagent gets: every builtin except ``Agent``.

    Omitting ``Agent`` makes recursion structurally impossible (v0.10.0 split #3).
    """
    return frozenset(BUILTIN_TOOLS) - {"Agent"}


# ----- registry ---------------------------------------------------------------


BUILTIN_TOOLS: dict[str, Tool] = {
    "Read": Tool(
        name="Read",
        description=(
            "Read a UTF-8 text file as line-numbered text. Use offset/limit to "
            "window large files. You must Read a file before you Edit or "
            "overwrite it. (Images/PDF/notebooks: not yet supported.)"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative path."},
                "offset": {
                    "type": "integer",
                    "description": "0-based first line to show.",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max lines to show.",
                    "default": _READ_DEFAULT_LIMIT,
                },
            },
            "required": ["path"],
        },
        func=_read,
    ),
    "Edit": Tool(
        name="Edit",
        description=(
            "Replace an exact string in a file. old_string must match exactly "
            "and be unique (else pass replace_all). Requires a prior Read of the "
            "file. Strip the line-number prefix from Read output before matching."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
                "old_string": {"type": "string", "description": "Exact text to replace."},
                "new_string": {"type": "string", "description": "Replacement (must differ)."},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every occurrence.",
                    "default": False,
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
        func=_edit,
    ),
    "Write": Tool(
        name="Write",
        description=(
            "Write content to a file, overwriting if it exists. Prefer Edit for "
            "changes; use Write for new files or full rewrites. Overwriting an "
            "existing file requires a prior Read."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write."},
                "content": {"type": "string", "description": "Full file content."},
            },
            "required": ["path", "content"],
        },
        func=_write,
    ),
    "Grep": Tool(
        name="Grep",
        description=(
            "Search file contents by regular expression (ripgrep when available). "
            "Prefer this over Bash grep/rg."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex to search for."},
                "path": {
                    "type": "string",
                    "description": "Dir or file to search (default cwd).",
                    "default": ".",
                },
                "glob": {
                    "type": "string",
                    "description": "Filter files by glob, e.g. '*.py'.",
                },
                "output_mode": {
                    "type": "string",
                    "enum": list(_GREP_OUTPUT_MODES),
                    "description": "files_with_matches (default) | content | count.",
                    "default": "files_with_matches",
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive match.",
                    "default": False,
                },
            },
            "required": ["pattern"],
        },
        func=_grep,
    ),
    "Glob": Tool(
        name="Glob",
        description=(
            "Find files by glob pattern (supports **), newest first. Prefer this "
            "over Bash find/ls."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob, e.g. '**/*.py'."},
                "path": {
                    "type": "string",
                    "description": "Base directory (default cwd).",
                    "default": ".",
                },
            },
            "required": ["pattern"],
        },
        func=_glob,
    ),
    "Bash": Tool(
        name="Bash",
        description=(
            "Run a shell command; returns stdout/stderr/exit_code. Use sparingly. "
            "Prefer the dedicated tools — Read (not cat/head/tail), Edit (not sed), "
            "Grep (not grep/rg), Glob (not find/ls). For destructive ops, ask first."
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
        description=_TASK_CREATE_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "A brief title for the task",
                },
                "description": {
                    "type": "string",
                    "description": "What needs to be done",
                },
            },
            "required": ["subject"],
        },
        func=_task_create,
    ),
    "TaskUpdate": Tool(
        name="TaskUpdate",
        description=_TASK_UPDATE_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "taskId": {
                    "type": "string",
                    "description": "The ID of the task to update",
                },
                "status": {
                    "type": "string",
                    "enum": list(_TASK_UPDATE_STATUSES),
                    "description": "New status for the task",
                },
                "subject": {
                    "type": "string",
                    "description": "New subject for the task",
                },
                "description": {
                    "type": "string",
                    "description": "New description for the task",
                },
            },
            "required": ["taskId"],
        },
        func=_task_update,
    ),
    "TaskList": Tool(
        name="TaskList",
        description=_TASK_LIST_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {},
        },
        func=_task_list,
    ),
    "Agent": Tool(
        name="Agent",
        description=_AGENT_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "A short (3-5 word) description of the task",
                },
                "prompt": {
                    "type": "string",
                    "description": "The task for the sub-agent to perform",
                },
                "subagent_type": {
                    "type": "string",
                    "description": (
                        "'general-purpose', or the name of a custom agent in "
                        ".claude/agents/"
                    ),
                    "default": "general-purpose",
                },
            },
            "required": ["description", "prompt"],
        },
        func=_agent,
    ),
}


def get_schemas(names: Collection[str] | None = None) -> list[dict[str, Any]]:
    """Return tool schemas, optionally scoped to ``names``.

    ``names=None`` (the default) returns every builtin — the main chat's
    behavior. A subagent passes an allowlist that omits ``Agent`` so it
    never sees the tool that would spawn another (v0.10.0 split #3).
    Unknown names in the set are ignored.
    """
    if names is None:
        return [t.schema() for t in BUILTIN_TOOLS.values()]
    return [t.schema() for name, t in BUILTIN_TOOLS.items() if name in names]


def dispatch(
    name: str,
    arguments_json: str,
    *,
    store: ChatStore | None = None,
    executor: Executor | None = None,
    slot: Slot | None = None,
) -> str:
    """Look up ``name`` in the registry and call it with parsed JSON args.

    The ``store`` keyword is forwarded only to tools whose
    implementation declares a ``store`` keyword parameter (currently
    ``TaskCreate``, ``TaskUpdate``, ``TaskList``). The ``executor``
    keyword is forwarded only to tools that declare it (currently
    ``run_shell``, which registers its Popen with the executor's
    cancellation pool). Tools that declare neither see their original
    signature, so the LLM-facing JSON schema stays free of any
    plumbing kwargs.

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
    if signature is not None:
        if "store" in signature.parameters:
            args.setdefault("store", store)
        if "executor" in signature.parameters:
            args.setdefault("executor", executor)
        if "slot" in signature.parameters:
            args.setdefault("slot", slot)
    try:
        return tool.func(**args)
    except TypeError as e:
        return f"ERROR: bad arguments: {e}"
    except Exception as e:
        logger.exception("tool {} crashed", name)
        return f"ERROR: tool crashed: {e}"
