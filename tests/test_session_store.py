"""v1.5.1 — Claude-Code-compatible session persist + resume."""
from __future__ import annotations

import json

from neutrix.session_store import (
    SessionWriter,
    _sanitize_cwd,
    list_sessions,
    load_session,
    most_recent,
    new_session_id,
    session_dir,
)
from neutrix.store import MessageRecord, Task


def _user(text: str) -> MessageRecord:
    return MessageRecord(role="user", content=text)


def _assistant(text: str) -> MessageRecord:
    return MessageRecord(role="assistant", content=text)


def _tool(tcid: str, name: str, content: str) -> MessageRecord:
    return MessageRecord(role="tool", content=content, tool_name=name, tool_call_id=tcid)


# ---- layout / sanitize ----------------------------------------------------


def test_session_dir_under_neutrix_cache_not_claude(tmp_path):
    d = session_dir("/home/me/proj", home=tmp_path)
    assert d == tmp_path / ".cache" / "neutrix" / "sessions" / "-home-me-proj"
    assert ".claude" not in str(d)  # user-directed: never write to ~/.claude


def test_sanitize_long_cwd_hashes():
    long = "/" + "/".join(["seg"] * 100)
    name = _sanitize_cwd(long)
    assert len(name) <= 200
    # deterministic
    assert name == _sanitize_cwd(long)


# ---- write → load round-trip ----------------------------------------------


def test_append_and_load_round_trip(tmp_path):
    sid = new_session_id()
    w = SessionWriter("/proj", sid, home=tmp_path)
    w.append_message(_user("hello"))
    w.append_message(_assistant("hi there"), llm_ms=1234.5)
    w.append_message(_tool("call_1", "Bash", "ok"), tool_ms=56.7)
    w.append_tasks((Task(id="1", subject="do it", status="in_progress"),))

    raw, records, tasks = load_session(w.path)
    assert [r.role for r in records] == ["user", "assistant", "tool"]
    assert records[0].content == "hello"
    assert records[2].tool_call_id == "call_1" and records[2].tool_name == "Bash"
    assert raw[1]["role"] == "assistant"
    assert tasks and tasks[0].subject == "do it" and tasks[0].status == "in_progress"


def test_timing_fields_written(tmp_path):
    sid = new_session_id()
    w = SessionWriter("/proj", sid, home=tmp_path)
    w.append_message(_assistant("reply"), llm_ms=2000.0)
    w.append_message(_tool("c", "Bash", "done"), tool_ms=80.0)
    lines = [json.loads(x) for x in w.path.read_text().splitlines()]
    assert lines[0]["llm_ms"] == 2000.0
    assert lines[1]["tool_ms"] == 80.0
    # CC-shaped: type + message + timestamp + sessionId + cwd
    assert lines[0]["type"] == "assistant"
    assert set(lines[0]) >= {"type", "message", "timestamp", "sessionId", "cwd"}


def test_last_tasks_snapshot_wins(tmp_path):
    sid = new_session_id()
    w = SessionWriter("/proj", sid, home=tmp_path)
    w.append_tasks((Task(id="1", subject="first"),))
    w.append_tasks((Task(id="1", subject="first", status="completed"),))
    _raw, _records, tasks = load_session(w.path)
    assert len(tasks) == 1 and tasks[0].status == "completed"


# ---- list / most-recent ---------------------------------------------------


def test_list_sessions_newest_first(tmp_path):
    a = SessionWriter("/proj", "aaaa", home=tmp_path)
    a.append_message(_user("first session question"))
    b = SessionWriter("/proj", "bbbb", home=tmp_path)
    b.append_message(_user("second session question"))
    # bump b's mtime after a
    import os
    import time as _t
    now = _t.time()
    os.utime(a.path, (now, now))
    os.utime(b.path, (now + 10, now + 10))

    sessions = list_sessions("/proj", home=tmp_path)
    assert [s.session_id for s in sessions] == ["bbbb", "aaaa"]
    assert sessions[0].first_prompt == "second session question"
    assert sessions[0].n_messages == 1
    assert most_recent("/proj", home=tmp_path).session_id == "bbbb"


def test_list_sessions_empty(tmp_path):
    assert list_sessions("/nope", home=tmp_path) == []
    assert most_recent("/nope", home=tmp_path) is None


def test_bad_lines_skipped(tmp_path):
    sid = new_session_id()
    w = SessionWriter("/proj", sid, home=tmp_path)
    w.append_message(_user("good"))
    with w.path.open("a") as fh:
        fh.write("not json\n\n")
    _raw, records, _tasks = load_session(w.path)
    assert len(records) == 1 and records[0].content == "good"
