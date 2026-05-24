"""Save / load chat transcripts as JSON.

This module operates on a :class:`neutrix.store.ChatStore`. The on-disk
format stays compatible with the v0.6.x ``session.py`` files: an object
with ``version``, ``saved_at``, ``provider``, ``model``, and
``messages`` (OpenAI Chat Completions format). Files written by older
neutrix versions load cleanly; files written here remain readable by
any tool that parsed the older format.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from neutrix.store import ChatStore, openai_to_record, record_to_openai

TRANSCRIPT_VERSION = 1


def save(
    path: str | Path,
    store: ChatStore,
    *,
    provider: str,
    model: str,
) -> Path:
    """Write the store's settled messages to ``path`` and return it.

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
    """
    p = Path(path).expanduser()
    payload = json.loads(p.read_text(encoding="utf-8"))
    if payload.get("version") != TRANSCRIPT_VERSION:
        raise ValueError(
            f"unsupported transcript version {payload.get('version')!r}; "
            f"this neutrix expects {TRANSCRIPT_VERSION}"
        )
    store = ChatStore()
    raw_messages = payload.get("messages") or []
    for raw in raw_messages:
        if isinstance(raw, dict):
            store.append_message(openai_to_record(raw))
    metadata: dict[str, Any] = {
        "provider": payload.get("provider", ""),
        "model": payload.get("model", ""),
        "saved_at": payload.get("saved_at", ""),
        "raw_messages": raw_messages,
    }
    return store, metadata
