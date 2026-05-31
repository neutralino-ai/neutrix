"""Session persistence for resume (v1.5.2).

Every turn is appended as one JSONL line under
``~/.cache/neutrix/sessions/<sanitized-cwd>/<uuid>.jsonl`` — neutrix's OWN cache
dir, **never** ``~/.claude`` (user-directed, 2026-05-29: "don't write to ~/.claude
folder"). The line *shape* still mirrors Claude Code's
(``{"type": <role>, "message": <openai dict>, "timestamp": <iso>, …}``) so the
format is familiar, but the files live outside ``~/.claude`` — the tradeoff is
``ccusage`` (which scans ``~/.claude/projects``) won't natively see neutrix
sessions; the resume value is fully preserved. Distinct from
:mod:`neutrix.transcript`'s single-file ``/save``·``/load`` export.

Per-turn neutrix-namespaced extras (``llm_ms`` / ``tool_ms``) record the timing
the v1.5.0 status bar showed but never persisted. A ``{"type": "tasks", "tasks":
[...]}`` line snapshots the task list; the last one wins on load. Writes are
best-effort (a logging failure never breaks the turn). ``load_session`` returns
the ``(raw_messages, records, tasks)`` triple the
:class:`~neutrix.context_manager.ReplaceHistoryEvent` already consumes.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from neutrix.store import MessageRecord, Task, openai_to_record, record_to_openai
from neutrix.transcript import _task_from_dict, _task_to_dict

if TYPE_CHECKING:  # only for the annotation — keep the heavy llm import out of runtime
    from neutrix.llm import Usage

_MESSAGE_TYPES = frozenset({"user", "assistant", "tool", "system"})
_MAX_DIR_NAME = 200


def _sanitize_cwd(cwd: str | Path) -> str:
    """Map a working dir to a filesystem-safe project-dir name (CC-style).

    ``/a/b/c`` → ``-a-b-c``; if that exceeds ``_MAX_DIR_NAME`` chars, truncate and
    append a short hash so it stays unique and bounded (mirrors CC's
    ``{prefix}-{hash}``).
    """
    raw = str(Path(cwd).expanduser())
    name = raw.replace("/", "-").replace("\\", "-")
    if len(name) <= _MAX_DIR_NAME:
        return name
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{name[: _MAX_DIR_NAME - 13]}-{digest}"


def session_dir(cwd: str | Path, home: str | Path | None = None) -> Path:
    """``<home>/.cache/neutrix/sessions/<sanitized-cwd>/`` — created on demand.

    neutrix's own cache dir, deliberately NOT ``~/.claude`` (user-directed).
    Precedence for the base: explicit ``home`` arg > ``$NEUTRIX_SESSION_HOME``
    (a user/test override) > the real home. The env override lets tests redirect
    every write to a tmp dir without threading ``home`` through every call.
    """
    base = home if home is not None else os.environ.get("NEUTRIX_SESSION_HOME")
    home_dir = Path(base) if base else Path.home()
    return home_dir / ".cache" / "neutrix" / "sessions" / _sanitize_cwd(cwd)


def new_session_id() -> str:
    return str(uuid.uuid4())


def session_path(cwd: str | Path, session_id: str, home: str | Path | None = None) -> Path:
    return session_dir(cwd, home) / f"{session_id}.jsonl"


@dataclass(frozen=True)
class SessionInfo:
    path: Path
    session_id: str
    mtime: float
    first_prompt: str
    n_messages: int


class SessionWriter:
    """Append-per-turn JSONL writer for one live session (best-effort)."""

    def __init__(self, cwd: str | Path, session_id: str, home: str | Path | None = None):
        self.cwd = str(Path(cwd).expanduser())
        self.session_id = session_id
        self.path = session_path(cwd, session_id, home)
        self._dir_ready = False

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _write_line(self, obj: dict[str, Any]) -> None:
        try:
            if not self._dir_ready:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._dir_ready = True
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except OSError as exc:  # never break a turn over a logging failure
            logger.warning("session log append failed: {}", exc)

    def append_message(
        self, record: MessageRecord, *, llm_ms: float | None = None,
        tool_ms: float | None = None,
    ) -> None:
        line: dict[str, Any] = {
            "type": record.role,
            "message": record_to_openai(record),
            "timestamp": self._now(),
            "sessionId": self.session_id,
            "cwd": self.cwd,
        }
        if record.tool_name:
            # The OpenAI role:tool shape has no name field, so the tool name
            # would be lost on the round-trip — keep it as a namespaced extra.
            line["tool_name"] = record.tool_name
        if llm_ms is not None:
            line["llm_ms"] = round(llm_ms, 1)
        if tool_ms is not None:
            line["tool_ms"] = round(tool_ms, 1)
        self._write_line(line)

    def append_tasks(self, tasks: tuple[Task, ...]) -> None:
        self._write_line({
            "type": "tasks",
            "tasks": [_task_to_dict(t) for t in tasks],
            "timestamp": self._now(),
        })

    def append_usage(
        self,
        *,
        model: str,
        usage: Usage,
        llm_ms: float | None = None,
        tool_ms: float | None = None,
    ) -> None:
        """Append one per-turn usage line (v1.7.0, Split #11).

        A dedicated ``{"type": "usage", …}`` line — not message-line extras —
        bundling the model, the normalized 4 token classes, the untouched
        provider ``raw`` payload (source of truth for repricing), and the v1.5.2
        ``llm_ms``/``tool_ms`` timing. ``load_session`` ignores this line type;
        :meth:`CostLedger.from_jsonl` consumes it on resume. **Dollars are never
        stored** — the ledger computes them on read so a price-table change
        reprices past sessions.
        """
        line: dict[str, Any] = {
            "type": "usage",
            "model": model,
            "input": usage.input,
            "output": usage.output,
            "cache_read": usage.cache_read,
            "cache_write": usage.cache_write,
            "raw": usage.raw,
            "timestamp": self._now(),
            "sessionId": self.session_id,
        }
        if llm_ms is not None:
            line["llm_ms"] = round(llm_ms, 1)
        if tool_ms is not None:
            line["tool_ms"] = round(tool_ms, 1)
        self._write_line(line)


def _read_lines(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        text = Path(path).expanduser().read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def load_session(
    path: str | Path,
) -> tuple[list[dict[str, Any]], tuple[MessageRecord, ...], tuple[Task, ...]]:
    """Rebuild the ``(raw_messages, records, tasks)`` triple from a session log.

    Matches what :class:`~neutrix.context_manager.ReplaceHistoryEvent` consumes,
    so resume reuses the same restore path as ``/load``.
    """
    raw_messages: list[dict[str, Any]] = []
    records: list[MessageRecord] = []
    tasks: tuple[Task, ...] = ()
    for obj in _read_lines(path):
        kind = obj.get("type")
        if kind in _MESSAGE_TYPES:
            msg = obj.get("message")
            if isinstance(msg, dict):
                raw_messages.append(msg)
                rec = openai_to_record(msg)
                tool_name = obj.get("tool_name")
                if tool_name and not rec.tool_name:
                    rec = replace(rec, tool_name=str(tool_name))
                records.append(rec)
        elif kind == "tasks":
            raw = obj.get("tasks") or []
            tasks = tuple(_task_from_dict(t) for t in raw if isinstance(t, dict))
    return raw_messages, tuple(records), tasks


def _first_prompt(objs: list[dict[str, Any]]) -> str:
    for obj in objs:
        if obj.get("type") == "user":
            msg = obj.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return "(no prompt)"


def list_sessions(cwd: str | Path, home: str | Path | None = None) -> list[SessionInfo]:
    """Sessions for ``cwd``, newest first (by file mtime)."""
    d = session_dir(cwd, home)
    if not d.is_dir():
        return []
    infos: list[SessionInfo] = []
    for p in d.glob("*.jsonl"):
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        objs = _read_lines(p)
        n_msgs = sum(1 for o in objs if o.get("type") in _MESSAGE_TYPES)
        infos.append(
            SessionInfo(
                path=p,
                session_id=p.stem,
                mtime=mtime,
                first_prompt=_first_prompt(objs),
                n_messages=n_msgs,
            )
        )
    infos.sort(key=lambda s: s.mtime, reverse=True)
    return infos


def most_recent(cwd: str | Path, home: str | Path | None = None) -> SessionInfo | None:
    sessions = list_sessions(cwd, home)
    return sessions[0] if sessions else None
