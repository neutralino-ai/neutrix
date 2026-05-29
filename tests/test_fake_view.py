"""v0.10.3 swap-test: prove the view layer is a pure ChatStore reader.

Two guarantees:
1. ``FakeView`` renders the scenario set correctly from store state alone
   (conversation, streaming, tool round, reminder, cancel, tasks, folded tray).
2. ``tests/fake_view.py`` imports ONLY ``neutrix.store`` (+ stdlib) — the
   import-boundary proof that a view needs nothing but the store. Scoped to the
   FakeView module, not ``terminal_chat.py`` (which legitimately hosts the
   orchestrator that drives ContextManager — see splits #1).
"""
from __future__ import annotations

import ast
from pathlib import Path

from neutrix.store import ChatStore, MessageRecord
from tests.fake_view import INTERRUPTED_MARKER, FakeView


def test_empty_store_renders_empty() -> None:
    assert FakeView().render(ChatStore()) == ""


def test_user_and_assistant_turns() -> None:
    store = ChatStore()
    store.append_message(MessageRecord(role="system", content="sp"))
    store.append_message(MessageRecord(role="user", content="hello"))
    store.append_message(MessageRecord(role="assistant", content="hi there"))
    out = FakeView().render(store)
    assert "[system] sp" in out
    assert "> hello" in out
    assert "< hi there" in out


def test_streaming_pending_text_renders() -> None:
    store = ChatStore()
    store.start_assistant_stream()
    store.extend_assistant_stream("partial answ")
    assert "partial answ" in FakeView().render(store)


def test_tool_round_and_folded_tray() -> None:
    store = ChatStore()
    store.append_message(
        MessageRecord(
            role="assistant",
            content=None,
            extra={"tool_calls": [{"function": {"name": "read_file"}}]},
        )
    )
    store.append_message(MessageRecord(role="tool", content="file contents here"))
    store.add_folded_tool_result("read_file", "{}", "y" * 300)
    out = FakeView().render(store)
    assert "-> read_file" in out
    assert "<- tool_result file contents" in out
    assert "[tool_result 1] read_file" in out


def test_subagent_folded_result_labelled() -> None:
    store = ChatStore()
    store.add_folded_tool_result("Agent", "{}", "delegated answer")
    assert "[subagent 1] Agent" in FakeView().render(store)


def test_reminder_and_cancel_render_distinctly() -> None:
    store = ChatStore()
    store.append_message(
        MessageRecord(role="user", content="<system-reminder>\ntasks\n</system-reminder>")
    )
    store.append_message(MessageRecord(role="user", content=INTERRUPTED_MARKER))
    out = FakeView().render(store)
    assert "[reminder] (folded)" in out
    assert "[cancelled] interrupted by user" in out
    # The raw reminder XML is not dumped as a plain user turn.
    assert "<system-reminder>" not in out


def test_tasks_render() -> None:
    store = ChatStore()
    store.add_task("ship v1.0")
    assert "[pending] ship v1.0" in FakeView().render(store)


def test_folded_tray_wiped_on_reset() -> None:
    store = ChatStore()
    store.add_folded_tool_result("read_file", "{}", "x")
    store.reset()
    assert store.folded_tool_results == ()


# ---- the import-boundary proof --------------------------------------------


def test_fake_view_imports_only_store() -> None:
    """fake_view.py must import nothing from neutrix except neutrix.store.

    This is the v0.10.3 contract: a view needs only the store. If a future
    edit reaches into context_manager/llm/tools/executor, the store stopped
    being the single state holder — fail loudly.
    """
    source = (Path(__file__).parent / "fake_view.py").read_text()
    tree = ast.parse(source)
    neutrix_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("neutrix"):
            neutrix_imports.add(node.module or "")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("neutrix"):
                    neutrix_imports.add(alias.name)
    assert neutrix_imports == {"neutrix.store"}, (
        f"FakeView must import only neutrix.store, got: {sorted(neutrix_imports)}"
    )
