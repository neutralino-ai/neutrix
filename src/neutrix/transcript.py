"""Save / load chat transcripts as JSON.

This module operates on a :class:`neutrix.store.ChatStore`. The on-disk
format started life in v0.6.x as ``{version: 1, saved_at, provider,
model, messages}``; v0.8.0 bumps it to v2 by adding a ``tasks`` array
that mirrors :py:attr:`ChatStore.tasks`. The reminder messages injected
by :mod:`neutrix.agent_loop` ride along as ordinary user-role entries
in ``messages``, so they round-trip without a dedicated key. v1 files
load cleanly with ``tasks=()``; saves always write v2.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from neutrix.store import ChatStore, Task, openai_to_record, record_to_openai

TRANSCRIPT_VERSION = 2
_SUPPORTED_VERSIONS = frozenset({1, 2})


def save(
    path: str | Path,
    store: ChatStore,
    *,
    provider: str,
    model: str,
) -> Path:
    """Write the store's settled messages and tasks to ``path``.

    Transient state (queued messages, pending tool calls, in-progress
    assistant stream text) is intentionally not persisted — those exist
    only within a live session.
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": TRANSCRIPT_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "provider": provider,
        "model": model,
        "messages": [record_to_openai(m) for m in store.messages],
        "tasks": [_task_to_dict(task) for task in store.tasks],
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def load(path: str | Path) -> tuple[ChatStore, dict[str, Any]]:
    """Read ``path`` and return ``(store, metadata)``.

    ``metadata`` holds ``provider``, ``model``, ``saved_at``, and
    ``raw_messages`` — the unprocessed OpenAI-format messages list, so
    callers that still maintain a separate ``Agent.messages`` list for
    LLM dispatch (v0.7.x) can populate it directly without going through
    the store conversion.

    The returned ``ChatStore`` is fully populated, including ``tasks``.
    Callers that maintain their own renderer-owned store must copy the
    tasks across (``self.store.replace_tasks(loaded.tasks)``) — the v1
    loader returns an empty task tuple.
    """
    p = Path(path).expanduser()
    payload = json.loads(p.read_text(encoding="utf-8"))
    version = payload.get("version")
    if version not in _SUPPORTED_VERSIONS:
        raise ValueError(
            f"unsupported transcript version {version!r}; "
            f"this neutrix expects {TRANSCRIPT_VERSION}"
        )
    store = ChatStore()
    raw_messages = payload.get("messages") or []
    for raw in raw_messages:
        if isinstance(raw, dict):
            store.append_message(openai_to_record(raw))
    raw_tasks = payload.get("tasks") or []
    loaded_tasks = [_task_from_dict(item) for item in raw_tasks if isinstance(item, dict)]
    if loaded_tasks:
        store.replace_tasks(loaded_tasks)
    metadata: dict[str, Any] = {
        "provider": payload.get("provider", ""),
        "model": payload.get("model", ""),
        "saved_at": payload.get("saved_at", ""),
        "raw_messages": raw_messages,
    }
    return store, metadata


def _task_to_dict(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "subject": task.subject,
        "description": task.description,
        "status": task.status,
        "created_at": task.created_at.isoformat(timespec="seconds"),
        "updated_at": task.updated_at.isoformat(timespec="seconds"),
    }


def _task_from_dict(item: dict[str, Any]) -> Task:
    return Task(
        id=str(item.get("id", "")),
        subject=str(item.get("subject", "")),
        description=str(item.get("description", "") or ""),
        status=item.get("status", "pending"),
        created_at=_parse_ts(item.get("created_at")),
        updated_at=_parse_ts(item.get("updated_at")),
    )


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.now()
