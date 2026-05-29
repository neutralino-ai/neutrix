"""Tests for the Task* tools added in v0.8.0.

Pre-existing tool dispatch behavior is covered in :mod:`tests.test_smoke`;
this module focuses on the store-backed Task tools and the dispatch()
keyword-forwarding rule.
"""
from __future__ import annotations

import json

from neutrix.executor import Executor
from neutrix.store import ChatStore
from neutrix.tools import BUILTIN_TOOLS, dispatch, get_schemas


def test_task_tools_appear_in_registry_with_claude_code_names():
    assert "TaskCreate" in BUILTIN_TOOLS
    assert "TaskUpdate" in BUILTIN_TOOLS
    assert "TaskList" in BUILTIN_TOOLS


def test_task_tool_descriptions_carry_claude_code_v2_lifecycle_guidance():
    """v0.8.1: tool descriptions are lifted from Claude Code's V2
    task tool prompts. Without these lifecycle cues the LLM doesn't
    auto-start work (observed in v0.8.0 manual testing). Regression
    guard: if someone shortens these strings back to the 1-line stubs,
    this test catches it."""
    create_desc = BUILTIN_TOOLS["TaskCreate"].description
    assert "Mark it as in_progress BEFORE beginning work" in create_desc
    assert "When you start working on a task" in create_desc

    update_desc = BUILTIN_TOOLS["TaskUpdate"].description
    assert "Always mark your assigned tasks as resolved" in update_desc
    assert "After resolving, call TaskList to find your next task" in update_desc

    list_desc = BUILTIN_TOOLS["TaskList"].description
    assert "Prefer working on tasks in ID order" in list_desc


def test_task_tool_schemas_match_documented_shape():
    schemas = {schema["function"]["name"]: schema for schema in get_schemas()}
    create = schemas["TaskCreate"]["function"]["parameters"]
    assert "subject" in create["properties"]
    assert create["required"] == ["subject"]

    update = schemas["TaskUpdate"]["function"]["parameters"]
    assert update["required"] == ["taskId"]
    assert "deleted" in update["properties"]["status"]["enum"]

    listing = schemas["TaskList"]["function"]["parameters"]
    assert listing["properties"] == {}


def test_task_create_mutates_store_and_returns_human_message():
    store = ChatStore()
    out = dispatch("TaskCreate", json.dumps({"subject": "first"}), store=store)
    assert out == "Task #1 created successfully: first"
    assert [(t.id, t.subject, t.status) for t in store.tasks] == [
        ("1", "first", "pending")
    ]


def test_task_create_result_matches_claude_code_v2_format():
    """v0.8.1: result strings mirror Claude Code's V2
    TaskCreateTool.mapToolResultToToolResultBlockParam exactly —
    bare ack, no nudge. The LLM is shaped via the rich tool
    description, not via result-text suffixes (which V2 doesn't use)."""
    store = ChatStore()
    out = dispatch("TaskCreate", json.dumps({"subject": "x"}), store=store)
    assert "Please proceed" not in out
    assert out == "Task #1 created successfully: x"


def test_task_create_accepts_optional_description():
    store = ChatStore()
    dispatch(
        "TaskCreate",
        json.dumps({"subject": "first", "description": "detail"}),
        store=store,
    )
    assert store.tasks[0].description == "detail"


def test_task_update_changes_status_field():
    store = ChatStore()
    store.add_task("first")
    out = dispatch(
        "TaskUpdate",
        json.dumps({"taskId": "1", "status": "in_progress"}),
        store=store,
    )
    assert out == "Updated task #1 status"
    assert store.tasks[0].status == "in_progress"


def test_task_update_result_matches_claude_code_v2_format():
    """V2 result text is `Updated task #ID field1, field2` — bare field
    names, no values, no nudge. Match exactly so the LLM gets the same
    cue surface CC's own sessions get."""
    store = ChatStore()
    store.add_task("first")
    out = dispatch(
        "TaskUpdate",
        json.dumps({"taskId": "1", "status": "in_progress", "subject": "renamed"}),
        store=store,
    )
    assert "Please proceed" not in out
    assert out == "Updated task #1 status, subject"


def test_task_update_unknown_id_returns_not_found():
    store = ChatStore()
    out = dispatch(
        "TaskUpdate",
        json.dumps({"taskId": "99", "status": "completed"}),
        store=store,
    )
    assert out == "Task #99 not found"


def test_task_update_with_status_deleted_removes_task():
    store = ChatStore()
    store.add_task("first")
    out = dispatch(
        "TaskUpdate",
        json.dumps({"taskId": "1", "status": "deleted"}),
        store=store,
    )
    # V2 treats deletion as a status change, so the success message has the
    # same `Updated task #N <field>` shape with `deleted` as the field.
    assert out == "Updated task #1 deleted"
    assert store.tasks == ()


def test_task_update_rejects_unknown_status_value():
    store = ChatStore()
    store.add_task("first")
    out = dispatch(
        "TaskUpdate",
        json.dumps({"taskId": "1", "status": "bogus"}),
        store=store,
    )
    assert out.startswith("ERROR: status must be one of")
    assert store.tasks[0].status == "pending"


def test_task_list_returns_json_encoded_array():
    store = ChatStore()
    store.add_task("first")
    store.add_task("second", description="d")
    store.update_task("1", status="in_progress")
    out = dispatch("TaskList", "{}", store=store)
    parsed = json.loads(out)
    assert parsed == [
        {"id": "1", "subject": "first", "status": "in_progress", "description": ""},
        {"id": "2", "subject": "second", "status": "pending", "description": "d"},
    ]


def test_task_tools_without_store_return_explicit_error():
    for name in ("TaskCreate", "TaskUpdate", "TaskList"):
        args = (
            json.dumps({"subject": "x"})
            if name == "TaskCreate"
            else json.dumps({"taskId": "1"})
            if name == "TaskUpdate"
            else "{}"
        )
        result = dispatch(name, args)
        assert result == f"ERROR: {name} requires a ChatStore"


def test_dispatch_does_not_pass_store_to_non_task_tools(tmp_path):
    """File-tools must keep their signature; passing store=... must not regress."""
    target = tmp_path / "out.txt"
    out = dispatch(
        "Write",
        json.dumps({"path": str(target), "content": "hi"}),
        store=ChatStore(),
    )
    assert "OK" in out
    assert target.read_text() == "hi"


# ===== v1.1.0 coding tools (Read/Edit/Write/Grep/Glob/Bash) ==============


def _w(path, content):
    """Helper: create a file directly (bypass the Write tool's read-guard)."""
    path.write_text(content, encoding="utf-8")


def test_read_windows_and_marks_path(tmp_path):
    f = tmp_path / "f.txt"
    _w(f, "\n".join(f"line{i}" for i in range(1, 11)))  # 10 lines
    ex = Executor()
    out = dispatch("Read", json.dumps({"path": str(f), "offset": 2, "limit": 3}), executor=ex)
    assert "line3" in out and "line5" in out
    assert "line2" not in out and "line6" not in out
    assert "3\t" in out  # cat -n numbering reflects offset (1-based line 3)
    assert "more lines" in out  # truncation hint
    assert str(f.resolve()) in ex.read_paths


def test_edit_requires_prior_read(tmp_path):
    f = tmp_path / "f.py"
    _w(f, "x = 1\n")
    ex = Executor()
    # Edit before Read → refused.
    out = dispatch(
        "Edit",
        json.dumps({"path": str(f), "old_string": "x = 1", "new_string": "x = 2"}),
        executor=ex,
    )
    assert out.startswith("ERROR") and "read-before-edit" in out.lower()
    assert f.read_text() == "x = 1\n"  # untouched
    # After Read → allowed.
    dispatch("Read", json.dumps({"path": str(f)}), executor=ex)
    out = dispatch(
        "Edit",
        json.dumps({"path": str(f), "old_string": "x = 1", "new_string": "x = 2"}),
        executor=ex,
    )
    assert out.startswith("OK")
    assert f.read_text() == "x = 2\n"


def test_edit_uniqueness_and_replace_all(tmp_path):
    f = tmp_path / "f.txt"
    _w(f, "a\na\na\n")
    ex = Executor()
    dispatch("Read", json.dumps({"path": str(f)}), executor=ex)
    # Non-unique without replace_all → refused.
    out = dispatch(
        "Edit",
        json.dumps({"path": str(f), "old_string": "a", "new_string": "b"}),
        executor=ex,
    )
    assert out.startswith("ERROR") and "unique" in out
    # replace_all → all replaced.
    out = dispatch(
        "Edit",
        json.dumps({"path": str(f), "old_string": "a", "new_string": "b", "replace_all": True}),
        executor=ex,
    )
    assert out.startswith("OK")
    assert f.read_text() == "b\nb\nb\n"


def test_edit_must_differ(tmp_path):
    f = tmp_path / "f.txt"
    _w(f, "same\n")
    ex = Executor()
    dispatch("Read", json.dumps({"path": str(f)}), executor=ex)
    out = dispatch(
        "Edit",
        json.dumps({"path": str(f), "old_string": "same", "new_string": "same"}),
        executor=ex,
    )
    assert out.startswith("ERROR") and "identical" in out


def test_write_overwrite_requires_read(tmp_path):
    f = tmp_path / "f.txt"
    ex = Executor()
    # New file → no Read needed.
    out = dispatch("Write", json.dumps({"path": str(f), "content": "v1"}), executor=ex)
    assert out.startswith("OK")
    # Overwrite an existing file the agent hasn't Read this session → refused.
    ex2 = Executor()
    out = dispatch("Write", json.dumps({"path": str(f), "content": "v2"}), executor=ex2)
    assert out.startswith("ERROR") and "Read it before overwriting" in out
    assert f.read_text() == "v1"
    # After Read → allowed.
    dispatch("Read", json.dumps({"path": str(f)}), executor=ex2)
    out = dispatch("Write", json.dumps({"path": str(f), "content": "v2"}), executor=ex2)
    assert out.startswith("OK")
    assert f.read_text() == "v2"


def test_glob_finds_files_recursively(tmp_path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y")
    (tmp_path / "c.txt").write_text("z")
    out = dispatch("Glob", json.dumps({"pattern": "**/*.py", "path": str(tmp_path)}))
    assert "a.py" in out and "b.py" in out
    assert "c.txt" not in out


def test_grep_python_fallback(tmp_path, monkeypatch):
    # Force the pure-Python path (no rg).
    monkeypatch.setattr("neutrix.tools.shutil.which", lambda _name: None)
    (tmp_path / "a.py").write_text("import os\nTODO: fix\n")
    (tmp_path / "b.py").write_text("clean\n")
    files = dispatch(
        "Grep",
        json.dumps({"pattern": "TODO", "path": str(tmp_path), "output_mode": "files_with_matches"}),
    )
    assert "a.py" in files and "b.py" not in files
    content = dispatch(
        "Grep",
        json.dumps({"pattern": "TODO", "path": str(tmp_path), "output_mode": "content"}),
    )
    assert "a.py:2:TODO: fix" in content


def test_grep_no_matches_python_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr("neutrix.tools.shutil.which", lambda _name: None)
    (tmp_path / "a.py").write_text("nothing here\n")
    out = dispatch("Grep", json.dumps({"pattern": "ZZZ", "path": str(tmp_path)}))
    assert out == "(no matches)"


def test_new_tools_in_registry_old_ones_gone():
    for name in ("Read", "Edit", "Write", "Grep", "Glob", "Bash"):
        assert name in BUILTIN_TOOLS
    for old in ("read_file", "write_file", "list_dir", "run_shell"):
        assert old not in BUILTIN_TOOLS
