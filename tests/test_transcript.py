"""Tests for :mod:`neutrix.transcript`.

Cover the on-disk format guarantees:

- save + load round-trips a populated ChatStore (messages AND tasks);
- a v0.6.x session.py-format file (manually hand-crafted) still loads
  without an error and reports an empty task list;
- a transcript with an unknown version raises ValueError;
- the saved file is format v2 and carries the ``tasks`` key.
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


# ---- v0.8.0 tasks + reminders ----------------------------------------------


def test_save_writes_version_two(tmp_path: Path):
    store = ChatStore()
    out = transcript.save(tmp_path / "v.json", store, provider="p", model="m")
    assert json.loads(out.read_text())["version"] == 2


def test_save_load_round_trips_tasks(tmp_path: Path):
    """Tasks persist through save/load with all stored fields intact."""
    store = ChatStore()
    a = store.add_task("first")
    b = store.add_task("second", description="detail")
    store.update_task(b.id, status="in_progress")

    out = transcript.save(tmp_path / "t.json", store, provider="p", model="m")
    on_disk = json.loads(out.read_text())
    assert [t["subject"] for t in on_disk["tasks"]] == ["first", "second"]
    assert on_disk["tasks"][1]["status"] == "in_progress"

    reloaded, _meta = transcript.load(out)
    loaded = reloaded.tasks
    assert [t.id for t in loaded] == [a.id, b.id]
    assert loaded[0].subject == "first"
    assert loaded[1].description == "detail"
    assert loaded[1].status == "in_progress"
    # Adding a new task after load resumes from max(id)+1.
    fresh = reloaded.add_task("third")
    assert fresh.id == "3"


def test_v1_file_loads_with_empty_tasks(tmp_path: Path):
    """A pre-v0.8.0 file (version 1, no tasks key) loads cleanly."""
    legacy = {
        "version": 1,
        "saved_at": "2026-01-01T00:00:00",
        "provider": "ihep",
        "model": "anthropic/claude-haiku-4-5",
        "messages": [{"role": "user", "content": "hi"}],
    }
    p = tmp_path / "v1.json"
    p.write_text(json.dumps(legacy), encoding="utf-8")
    store, metadata = transcript.load(p)
    assert store.tasks == ()
    assert metadata["raw_messages"] == legacy["messages"]
    # Adding a task after loading a v1 file starts from id 1.
    assert store.add_task("first").id == "1"


def test_reminder_message_round_trips_unchanged(tmp_path: Path):
    """The injected <system-reminder> is an ordinary user message; it must
    survive save/load identically (no special-casing, no stripping)."""
    store = ChatStore()
    reminder_text = (
        "<system-reminder>\nThe task tools haven't been used recently.\n"
        "</system-reminder>"
    )
    store.append_message(MessageRecord(role="user", content="real question"))
    store.append_message(MessageRecord(role="assistant", content="answer"))
    store.append_message(MessageRecord(role="user", content=reminder_text))

    out = transcript.save(tmp_path / "t.json", store, provider="p", model="m")
    reloaded, metadata = transcript.load(out)
    contents = [m.content for m in reloaded.messages]
    assert reminder_text in contents
    assert metadata["raw_messages"][-1]["content"] == reminder_text
