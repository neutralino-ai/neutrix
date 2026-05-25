"""Tests for the Task* tools added in v0.8.0.

Pre-existing tool dispatch behavior is covered in :mod:`tests.test_smoke`;
this module focuses on the store-backed Task tools and the dispatch()
keyword-forwarding rule.
"""
from __future__ import annotations

import json

from neutrix.store import ChatStore
from neutrix.tools import BUILTIN_TOOLS, dispatch, get_schemas


def test_task_tools_appear_in_registry_with_claude_code_names():
    assert "TaskCreate" in BUILTIN_TOOLS
    assert "TaskUpdate" in BUILTIN_TOOLS
    assert "TaskList" in BUILTIN_TOOLS


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
    assert out == "ok, created task 1: first"
    assert [(t.id, t.subject, t.status) for t in store.tasks] == [
        ("1", "first", "pending")
    ]


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
    assert out == "ok, task 1 updated: status=in_progress"
    assert store.tasks[0].status == "in_progress"


def test_task_update_unknown_id_returns_not_found():
    store = ChatStore()
    out = dispatch(
        "TaskUpdate",
        json.dumps({"taskId": "99", "status": "completed"}),
        store=store,
    )
    assert out == "task 99 not found"


def test_task_update_with_status_deleted_removes_task():
    store = ChatStore()
    store.add_task("first")
    out = dispatch(
        "TaskUpdate",
        json.dumps({"taskId": "1", "status": "deleted"}),
        store=store,
    )
    assert out == "ok, deleted task 1: first"
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
    """File-tools must keep their existing signature; passing store=... must
    not regress them."""
    target = tmp_path / "out.txt"
    out = dispatch(
        "write_file",
        json.dumps({"path": str(target), "content": "hi"}),
        store=ChatStore(),
    )
    assert "OK" in out
    assert target.read_text() == "hi"
