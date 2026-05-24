"""Tests for :mod:`neutrix.transcript`.

Cover the on-disk format guarantees:

- save + load round-trips a populated ChatStore lossly enough that the
  raw_messages and the typed store both reconstruct;
- a v0.6.x session.py-format file (manually hand-crafted) still loads
  without an error;
- a transcript with an unknown version raises ValueError.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from neutrix import transcript
from neutrix.store import ChatStore, MessageRecord


def test_save_load_round_trip_preserves_messages(tmp_path: Path):
    store = ChatStore()
    store.append_message(MessageRecord(role="system", content="be terse"))
    store.append_message(MessageRecord(role="user", content="hi"))
    store.append_message(MessageRecord(role="assistant", content="hello"))

    out = transcript.save(
        tmp_path / "t.json",
        store,
        provider="ihep",
        model="anthropic/claude-haiku-4-5",
    )
    assert out.exists()

    reloaded, metadata = transcript.load(out)
    assert metadata["provider"] == "ihep"
    assert metadata["model"] == "anthropic/claude-haiku-4-5"
    assert metadata["saved_at"]  # non-empty ISO timestamp
    assert len(reloaded.messages) == 3
    assert [(m.role, m.content) for m in reloaded.messages] == [
        ("system", "be terse"),
        ("user", "hi"),
        ("assistant", "hello"),
    ]


def test_save_preserves_openai_extras(tmp_path: Path):
    """An assistant message with tool_calls survives round-trip."""
    store = ChatStore()
    raw_tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "list_dir", "arguments": "{}"},
        }
    ]
    store.append_message(
        MessageRecord(
            role="assistant",
            content=None,
            extra={"tool_calls": raw_tool_calls},
        )
    )
    out = transcript.save(
        tmp_path / "t.json", store, provider="p", model="m"
    )

    on_disk = json.loads(out.read_text())
    assert on_disk["messages"][0]["content"] is None
    assert on_disk["messages"][0]["tool_calls"] == raw_tool_calls

    reloaded, metadata = transcript.load(out)
    assert reloaded.messages[0].content is None
    assert reloaded.messages[0].extra == {"tool_calls": raw_tool_calls}
    assert metadata["raw_messages"][0]["tool_calls"] == raw_tool_calls


def test_v06_session_format_loads_cleanly(tmp_path: Path):
    """A v0.6.x session.py file (same format, no transcript-only fields)
    must load via :func:`transcript.load` without error."""
    legacy_payload = {
        "version": 1,
        "saved_at": "2025-01-01T00:00:00",
        "provider": "ihep",
        "model": "anthropic/claude-haiku-4-5",
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "what's 2+2"},
            {"role": "assistant", "content": "4"},
        ],
    }
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(legacy_payload), encoding="utf-8")

    store, metadata = transcript.load(p)
    assert metadata["provider"] == "ihep"
    assert len(store.messages) == 3
    assert metadata["raw_messages"] == legacy_payload["messages"]


def test_unknown_version_raises(tmp_path: Path):
    p = tmp_path / "future.json"
    p.write_text(json.dumps({"version": 99, "messages": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported transcript version"):
        transcript.load(p)


def test_transient_state_is_not_persisted(tmp_path: Path):
    """Queue, pending tool calls, and pending assistant text never go to disk."""
    store = ChatStore()
    store.append_message(MessageRecord(role="user", content="hi"))
    store.enqueue_user("should not persist")
    store.add_pending_tool_call("list_dir", "{}")
    store.start_assistant_stream()
    store.extend_assistant_stream("partial...")

    out = transcript.save(
        tmp_path / "t.json", store, provider="p", model="m"
    )
    on_disk = json.loads(out.read_text())
    # Only the one settled user message survives — nothing else.
    assert len(on_disk["messages"]) == 1
    assert on_disk["messages"][0]["content"] == "hi"
    assert "queued_user_messages" not in on_disk
    assert "pending_tool_calls" not in on_disk
    assert "pending_assistant_text" not in on_disk
