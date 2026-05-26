"""Tests for the append-only terminal chat renderer."""
from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path
from queue import Queue
from typing import Any

import pytest
from rich.console import Console

from neutrix.agent_loop import AgentEvent
from neutrix.config import Config, Slot
from neutrix.store import ChatStore
from neutrix.terminal_chat import (
    MAX_PANEL_ROWS,
    QuitArmingState,
    TerminalChat,
    ToolRecord,
    apply_enter_or_continuation,
    approximate_token_count,
    build_draft_key_bindings,
    delete_buffer_to_line_end,
    format_task_panel,
    move_buffer_to_line_start,
    result_line_count,
)


def _slot(name: str = "strong") -> Slot:
    return Slot(
        name=name,
        provider="test",
        model=f"{name}-model",
        base_url="https://example.test/v1",
        api_key="sk-test",
    )


def _config(tmp_path: Path) -> Config:
    return Config(
        providers={"test": {"base_url": "https://example.test/v1", "api_key": "sk-test"}},
        slots={
            "fast": {"provider": "test", "model": "fast-model"},
            "strong": {"provider": "test", "model": "strong-model"},
        },
        path=tmp_path / "config.yaml",
    )


class _FakeLLM:
    """Minimal LLM stub for fake agents that never invoke an LLM."""

    def switch(self, slot: Slot) -> None:
        pass

    async def stream_response(self, *, model, messages, tools=None):
        if False:  # pragma: no cover - never iterated
            yield None

    def stop(self) -> None:
        pass


class ToolAgent:
    def __init__(self) -> None:
        self.slot = _slot()
        self.llm = _FakeLLM()
        self.use_tools = True
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": "system prompt"}
        ]
        self.sent: list[str] = []
        self.store: ChatStore | None = None

    def effective_tools_enabled(self) -> bool:
        return self.use_tools

    def switch(self, slot: Slot) -> None:
        self.slot = slot

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": "system prompt"}]

    def rollback_to(self, snapshot_len: int) -> None:
        if 0 <= snapshot_len < len(self.messages):
            del self.messages[snapshot_len:]

    async def stream_reply(self, user_text: str, *, executor: Any = None):
        self.sent.append(user_text)
        self.messages.append({"role": "user", "content": user_text})
        arguments = '{"path": "."}'
        result = "f a.txt\nf b.txt\n"
        yield AgentEvent("tool_call", {"name": "list_dir", "arguments": arguments})
        self.messages.append(
            {"role": "tool", "tool_call_id": "call_1", "content": result}
        )
        yield AgentEvent("tool_result", {"name": "list_dir", "result": result})
        self.messages.append({"role": "assistant", "content": "done"})
        yield AgentEvent("assistant", "done")
        yield AgentEvent("done")


class BlockingAgent(ToolAgent):
    def __init__(self) -> None:
        super().__init__()
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.releases: asyncio.Queue[None] = asyncio.Queue()

    async def stream_reply(self, user_text: str, *, executor: Any = None):
        self.sent.append(user_text)
        self.messages.append({"role": "user", "content": user_text})
        await self.started.put(user_text)
        await self.releases.get()
        self.messages.append({"role": "assistant", "content": f"reply {user_text}"})
        yield AgentEvent("assistant", f"reply {user_text}")
        yield AgentEvent("done")


def _render(value: object) -> str:
    """Flatten a value that may be str or prompt_toolkit FormattedText
    (an iterable of (style, text) tuples) into a single string for
    substring assertions in tests."""
    if isinstance(value, str):
        return value
    if hasattr(value, "__iter__"):
        return "".join(text for _style, text in value)
    return str(value)


def _chat(
    agent: ToolAgent,
    tmp_path: Path,
    inputs: list[str],
) -> tuple[TerminalChat, StringIO, list[str]]:
    output = StringIO()
    input_iter = iter(inputs)
    prompts: list[str] = []
    console = Console(
        file=output,
        force_terminal=False,
        color_system=None,
        width=100,
    )

    def input_func(prompt: str) -> str:
        prompts.append(prompt)
        return next(input_iter)

    chat = TerminalChat(
        agent,
        config=_config(tmp_path),
        render_markdown=False,
        input_func=input_func,
        console=console,
    )
    return chat, output, prompts


class FakeDocument:
    def __init__(self, text: str, cursor_position: int) -> None:
        self.text = text
        self.cursor_position = cursor_position

    @property
    def text_after_cursor(self) -> str:
        return self.text[self.cursor_position :]

    def get_start_of_line_position(self) -> int:
        line_start = self.text.rfind("\n", 0, self.cursor_position) + 1
        return line_start - self.cursor_position

    def get_end_of_line_position(self) -> int:
        line_end = self.text.find("\n", self.cursor_position)
        if line_end == -1:
            line_end = len(self.text)
        return line_end - self.cursor_position


class FakeBuffer:
    def __init__(self, text: str, cursor_position: int) -> None:
        self.text = text
        self.cursor_position = cursor_position

    @property
    def document(self) -> FakeDocument:
        return FakeDocument(self.text, self.cursor_position)

    def delete(self, count: int) -> None:
        self.text = (
            self.text[: self.cursor_position]
            + self.text[self.cursor_position + count :]
        )


def test_tool_result_summary_counts_lines_and_approx_tokens() -> None:
    assert result_line_count("") == 0
    assert result_line_count("one\ntwo\n") == 2
    assert approximate_token_count("one two\nthree") == 3

    record = ToolRecord(
        index=1,
        name="list_dir",
        arguments='{"path": "."}',
        result="one two\nthree",
    )

    assert record.summary == (
        '<- tool_result [tool 1] list_dir {"path": "."} | folded | 2 lines | ~3 tokens'
    )


def test_draft_key_helpers_match_readline_line_editing() -> None:
    buffer = FakeBuffer("abc\ndef", 6)
    move_buffer_to_line_start(buffer)
    assert buffer.cursor_position == 4

    buffer = FakeBuffer("abc\ndef", 5)
    delete_buffer_to_line_end(buffer)
    assert buffer.text == "abc\nd"

    buffer = FakeBuffer("abc\ndef", 3)
    delete_buffer_to_line_end(buffer)
    assert buffer.text == "abcdef"


def test_terminal_chat_folds_tool_results_and_expands_on_command(tmp_path: Path) -> None:
    agent = ToolAgent()
    chat, output, prompts = _chat(
        agent, tmp_path, ["please list", "/tool", "/tool 1", "/quit"]
    )

    chat.run()

    rendered = output.getvalue()
    assert prompts and all(prompt == "" for prompt in prompts)
    assert "you>" not in rendered
    assert "system:" not in rendered
    assert "user:" not in rendered
    assert "assistant:" not in rendered
    assert "assistant is responding" not in rendered
    assert "please list" in rendered
    assert "strong | test/strong-model | tools:on" not in rendered
    assert '-> tool_use    list_dir {"path": "."}' in rendered
    assert (
        '<- tool_result [tool 1] list_dir {"path": "."} | folded | 2 lines | ~4 tokens'
        in rendered
    )
    assert "[tool 1] list_dir full result:" in rendered
    assert "f a.txt\nf b.txt" in rendered
    assert agent.messages[2]["content"] == "f a.txt\nf b.txt\n"


@pytest.mark.asyncio
async def test_terminal_chat_accepts_and_queues_input_while_agent_is_busy(
    tmp_path: Path,
) -> None:
    agent = BlockingAgent()
    output = StringIO()
    input_values: Queue[str] = Queue()
    prompts: list[str] = []

    def input_func(prompt: str) -> str:
        prompts.append(prompt)
        return input_values.get(timeout=2)

    chat = TerminalChat(
        agent,
        config=_config(tmp_path),
        render_markdown=False,
        input_func=input_func,
        console=Console(
            file=output,
            force_terminal=False,
            color_system=None,
            width=100,
        ),
    )
    task = asyncio.create_task(chat.run_async())

    input_values.put("first")
    assert await asyncio.wait_for(agent.started.get(), timeout=1) == "first"

    input_values.put("second")
    for _ in range(20):
        await asyncio.sleep(0.01)
        if chat.store.queued_user_messages:
            break

    assert chat._busy
    queued = chat.store.queued_user_messages
    assert [q.text for q in queued] == ["second"]
    # The visible queue is rendered ABOVE the input via the message
    # supplier. The bottom toolbar is gone in v0.7.0 (it blinked
    # during streaming output); /status is the on-demand replacement.
    assert "queued:" not in chat._status_line()
    rendered_message = _render(chat._above_input())
    assert "› second" in rendered_message  # noqa: RUF001 -- chosen UI glyph

    await agent.releases.put(None)
    assert await asyncio.wait_for(agent.started.get(), timeout=1) == "second"
    await agent.releases.put(None)
    input_values.put("/quit")

    await asyncio.wait_for(task, timeout=2)

    assert prompts and all(prompt == "" for prompt in prompts)
    assert agent.sent == ["first", "second"]
    rendered = output.getvalue()
    assert "first" in rendered
    assert "second" in rendered
    assert "reply first" in rendered
    assert "reply second" in rendered


def test_terminal_chat_tool_toggle_updates_status(tmp_path: Path) -> None:
    agent = ToolAgent()
    chat, output, _prompts = _chat(agent, tmp_path, ["/tools off", "/quit"])

    chat.run()

    rendered = output.getvalue()
    assert "tool calling disabled" in rendered
    assert "tools:off" not in rendered
    assert "tools:off" in chat._status_line()
    assert agent.use_tools is False


def test_terminal_chat_status_command_prints_current_state(tmp_path: Path) -> None:
    """/status prints slot, provider/model, tool state, and msg count."""
    agent = ToolAgent()
    chat, output, _prompts = _chat(agent, tmp_path, ["/status", "/quit"])
    chat.run()
    rendered = output.getvalue()
    assert "strong" in rendered
    assert "test/strong-model" in rendered
    assert "tools:on" in rendered
    assert "msgs:1" in rendered  # only the system message at start


def test_terminal_chat_folds_loaded_tool_messages_with_call_arguments(
    tmp_path: Path,
) -> None:
    agent = ToolAgent()
    agent.messages = [
        {"role": "system", "content": "system prompt"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "README.md"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "hello world\n"},
    ]
    chat, output, _prompts = _chat(agent, tmp_path, ["/quit"])

    chat.run()

    rendered = output.getvalue()
    assert '-> tool_use    read_file {"path": "README.md"}' in rendered
    assert (
        '<- tool_result [tool 1] read_file {"path": "README.md"} | folded | 1 lines | ~2 tokens'
        in rendered
    )


@pytest.mark.asyncio
async def test_terminal_chat_renders_multiple_queued_messages_in_order(
    tmp_path: Path,
) -> None:
    """Covers PRD v0.7.0 acceptance steps 3 + 4 + 5 + 6:

    - two messages submitted while the assistant is busy queue up,
    - both show in the toolbar in submission order with the dim-style
      QUEUED_PREFIX,
    - the `queued:N` counter is absent,
    - both are consumed in order when the assistant frees up.
    """
    agent = BlockingAgent()
    output = StringIO()
    input_values: Queue[str] = Queue()
    prompts: list[str] = []

    def input_func(prompt: str) -> str:
        prompts.append(prompt)
        return input_values.get(timeout=2)

    chat = TerminalChat(
        agent,
        config=_config(tmp_path),
        render_markdown=False,
        input_func=input_func,
        console=Console(
            file=output,
            force_terminal=False,
            color_system=None,
            width=100,
        ),
    )
    task = asyncio.create_task(chat.run_async())

    input_values.put("first")
    assert await asyncio.wait_for(agent.started.get(), timeout=1) == "first"

    # Two more messages typed while the agent is still blocked.
    input_values.put("second")
    input_values.put("third")
    for _ in range(50):
        await asyncio.sleep(0.01)
        if len(chat.store.queued_user_messages) == 2:
            break

    queued = chat.store.queued_user_messages
    assert [q.text for q in queued] == ["second", "third"]

    # Status line: no queue rendering, no `queued:` substring.
    assert "queued:" not in chat._status_line()

    # Queue renders ABOVE the input cursor via the message supplier.
    message = chat._above_input()
    assert not isinstance(
        message, str
    ), "_above_input must be FormattedText when queue is non-empty"
    fragments = list(message)
    queued_lines = [text for _style, text in fragments if text.startswith("› ")]  # noqa: RUF001
    assert queued_lines == ["› second\n", "› third\n"]  # noqa: RUF001
    # No tasks → all fragments are dim queue lines.
    assert all(style == "fg:ansibrightblack" for style, _text in fragments)

    # Release each turn; the queue must drain in submission order.
    await agent.releases.put(None)
    assert await asyncio.wait_for(agent.started.get(), timeout=1) == "second"
    await agent.releases.put(None)
    assert await asyncio.wait_for(agent.started.get(), timeout=1) == "third"
    await agent.releases.put(None)
    input_values.put("/quit")

    await asyncio.wait_for(task, timeout=2)
    assert agent.sent == ["first", "second", "third"]
    assert chat.store.queued_user_messages == ()


def test_terminal_chat_status_line_carries_only_status_no_queue(
    tmp_path: Path,
) -> None:
    """`_status_line` is the on-demand status string used by /status.
    It never contains queue rendering; the queue lives in
    `_above_input` (rendered above the input)."""
    agent = ToolAgent()
    chat, _output, _prompts = _chat(agent, tmp_path, ["/quit"])
    line = chat._status_line()
    assert isinstance(line, str)
    assert "queued:" not in line
    assert "tools:" in line
    assert "msgs:" in line
    # No queue, no tasks → empty supplier output.
    assert chat._above_input() == ""


class TaskCreatingAgent(ToolAgent):
    """Fake agent that emits a TaskCreate tool call and mutates the store
    directly the way the real Agent's dispatch does."""

    def __init__(self, subject: str = "first") -> None:
        super().__init__()
        self.subject = subject

    async def stream_reply(self, user_text: str, *, executor: Any = None):
        self.sent.append(user_text)
        self.messages.append({"role": "user", "content": user_text})
        arguments = f'{{"subject": "{self.subject}"}}'
        yield AgentEvent("tool_call", {"name": "TaskCreate", "arguments": arguments})
        assert self.store is not None
        task = self.store.add_task(self.subject)
        result = f"ok, created task {task.id}: {task.subject}"
        self.messages.append(
            {"role": "tool", "tool_call_id": "call_1", "content": result}
        )
        yield AgentEvent("tool_result", {"name": "TaskCreate", "result": result})
        self.messages.append({"role": "assistant", "content": "tracked"})
        yield AgentEvent("assistant", "tracked")
        yield AgentEvent("done")


def test_terminal_chat_constructor_wires_store_into_agent(tmp_path: Path) -> None:
    """The PRD requires TerminalChat to construct Agent with store=self.store
    so the LLM-callable Task tools can mutate the live store."""
    agent = ToolAgent()
    chat, _output, _prompts = _chat(agent, tmp_path, ["/quit"])
    assert agent.store is chat.store


def test_terminal_chat_tasks_command_prints_no_tasks_when_empty(
    tmp_path: Path,
) -> None:
    agent = ToolAgent()
    chat, output, _prompts = _chat(agent, tmp_path, ["/tasks", "/quit"])
    chat.run()
    assert "no tasks" in output.getvalue()


def test_terminal_chat_tasks_command_lists_seeded_tasks(tmp_path: Path) -> None:
    agent = ToolAgent()
    chat, output, _prompts = _chat(agent, tmp_path, ["/tasks", "/quit"])
    chat.store.add_task("refactor onion")
    chat.store.add_task("ship v0.8.0")
    chat.store.update_task("2", status="in_progress")
    chat.run()
    rendered = output.getvalue()
    flat = " ".join(rendered.split())
    assert "#1 [pending] refactor onion" in flat
    assert "#2 [in_progress] ship v0.8.0" in flat


def test_terminal_chat_taskcreate_tool_populates_store_and_tasks_command(
    tmp_path: Path,
) -> None:
    """A fake agent emits a TaskCreate tool call → the store gains the task
    → a subsequent /tasks shows it."""
    agent = TaskCreatingAgent(subject="refactor onion")
    chat, output, _prompts = _chat(
        agent, tmp_path, ["track that", "/tasks", "/quit"]
    )
    chat.run()
    assert [(t.id, t.subject, t.status) for t in chat.store.tasks] == [
        ("1", "refactor onion", "pending")
    ]
    rendered = output.getvalue()
    flat = " ".join(rendered.split())
    assert "-> tool_use TaskCreate" in flat
    assert "#1 [pending] refactor onion" in flat


def test_terminal_chat_load_preserves_tasks(tmp_path: Path) -> None:
    """The /load path must call replace_tasks; otherwise tasks silently
    disappear (advisor flag during PRD review)."""
    save_path = tmp_path / "session.json"

    agent_a = TaskCreatingAgent(subject="restored-task")
    chat_a, _output_a, _ = _chat(
        agent_a, tmp_path, ["track this", f"/save {save_path}", "/quit"]
    )
    chat_a.run()
    assert chat_a.store.tasks  # sanity

    agent_b = ToolAgent()
    chat_b, output_b, _ = _chat(
        agent_b, tmp_path, [f"/load {save_path}", "/tasks", "/quit"]
    )
    chat_b.run()
    rendered = output_b.getvalue()
    # /load notice line gets soft-wrapped at 100 cols; collapse whitespace.
    flat = " ".join(rendered.split())
    assert "1 tasks" in flat
    assert "#1 [pending] restored-task" in flat
    assert chat_b.store.tasks[0].subject == "restored-task"


def test_terminal_chat_clear_resets_tasks(tmp_path: Path) -> None:
    agent = ToolAgent()
    chat, output, _ = _chat(
        agent, tmp_path, ["/clear", "/tasks", "/quit"]
    )
    chat.store.add_task("will be cleared")
    chat.run()
    assert chat.store.tasks == ()
    assert "no tasks" in output.getvalue()


def test_terminal_chat_save_and_load_round_trip_via_commands(tmp_path: Path) -> None:
    """Covers PRD v0.7.0 acceptance step 7: /save then a fresh chat /load
    reconstructs the conversation visibly and re-seeds the store."""
    save_path = tmp_path / "session.json"

    agent_a = ToolAgent()
    chat_a, _output_a, _prompts_a = _chat(
        agent_a,
        tmp_path,
        ["please list", f"/save {save_path}", "/quit"],
    )
    chat_a.run()
    assert save_path.exists()

    agent_b = ToolAgent()
    chat_b, output_b, _prompts_b = _chat(
        agent_b,
        tmp_path,
        [f"/load {save_path}", "/quit"],
    )
    chat_b.run()

    rendered = output_b.getvalue()
    assert "please list" in rendered
    # Tool results render as folded summaries on load.
    assert "[tool 1]" in rendered
    assert "folded | 2 lines" in rendered
    assert "done" in rendered
    # The store mirrors the agent's loaded messages.
    assert len(chat_b.store.messages) == len(agent_b.messages)
    assert chat_b.store.messages[1].role == "user"
    assert chat_b.store.messages[1].content == "please list"


# ---- v0.8.1: task panel + folded reminder rendering ------------------------


def test_format_task_panel_returns_empty_for_no_tasks():
    """Hidden when no tasks exist — matches Claude TaskListV2 behavior so
    the input cursor sits at its natural position."""
    assert format_task_panel(()) == []


def test_format_task_panel_orders_in_progress_then_pending_then_completed():
    """v0.8.1 panel sort: in_progress → pending → completed, id ascending
    inside each bucket (matches Claude Code's getTaskListSortedByStatus)."""
    store = ChatStore()
    store.add_task("a-pending")
    store.add_task("b-completed")
    store.add_task("c-in-progress")
    store.update_task("2", status="completed")
    store.update_task("3", status="in_progress")
    fragments = format_task_panel(store.tasks)

    flat = "".join(text for _style, text in fragments)
    in_progress_pos = flat.index("c-in-progress")
    pending_pos = flat.index("a-pending")
    completed_pos = flat.index("b-completed")
    assert in_progress_pos < pending_pos < completed_pos


def test_format_task_panel_styles_match_status():
    """in_progress: cyan bold, pending: default, completed: green."""
    store = ChatStore()
    store.add_task("plain")
    store.add_task("active")
    store.add_task("done")
    store.update_task("2", status="in_progress")
    store.update_task("3", status="completed")
    styles_by_subject = {
        text.strip().split()[-1]: style for style, text in format_task_panel(store.tasks)
    }
    assert styles_by_subject["active"] == "fg:ansicyan bold"
    assert styles_by_subject["plain"] == ""
    assert styles_by_subject["done"] == "fg:ansigreen"


def test_format_task_panel_caps_visible_rows_and_emits_overflow_line():
    """Cap at MAX_PANEL_ROWS visible task lines; append a single dim
    overflow line summarizing the truncated bucket sizes."""
    store = ChatStore()
    for i in range(MAX_PANEL_ROWS + 3):
        store.add_task(f"task-{i}")
    store.update_task("1", status="completed")
    store.update_task("2", status="completed")
    fragments = format_task_panel(store.tasks)

    task_rows = [t for s, t in fragments if "task-" in t]
    assert len(task_rows) == MAX_PANEL_ROWS
    overflow = [(s, t) for s, t in fragments if t.startswith("  … +")]
    assert len(overflow) == 1
    style, text = overflow[0]
    assert style == "fg:ansibrightblack"
    # 6 pending + 2 completed total = 8 tasks; visible = top 5 sorted
    # (5 pending), overflow = 1 pending + 2 completed.
    assert "1 pending" in text
    assert "2 done" in text


def test_draft_reader_placeholder_uses_dim_formatted_text():
    """v0.8.1: the empty-input hint renders in fg:ansibrightblack so it
    reads as a hint rather than as foreground text."""
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.formatted_text import FormattedText

    from neutrix.terminal_chat import DraftReader

    reader = DraftReader(message_supplier=lambda: "")
    session = reader._session
    assert session is not None
    placeholder = session.placeholder
    assert isinstance(placeholder, FormattedText)
    styles = {style for style, _text in placeholder}
    assert styles == {"fg:ansibrightblack"}


class RecordingView:
    """Minimal TerminalView stand-in that captures notice/user/etc. calls
    so reminder-folding tests can assert which render path was taken
    without diff'ing rendered text."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    async def print_notice(self, content: str, *, style: str = "dim") -> None:
        self.calls.append(("notice", content, style))

    async def print_system(self, content: str) -> None:
        self.calls.append(("system", content, None))

    async def print_user(self, content: str) -> None:
        self.calls.append(("user", content, None))

    async def print_assistant(self, content: str) -> None:
        self.calls.append(("assistant", content, None))

    async def print_text(self, text) -> None:
        self.calls.append(("text", str(text), None))

    async def write_raw(self, text: str) -> None:
        self.calls.append(("raw", text, None))


def _make_reminder(body_lines: list[str] | None = None) -> dict[str, Any]:
    body = "\n".join(
        body_lines
        or [
            "The task tools haven't been used recently.",
            "",
            "Here are the existing tasks:",
            "",
            "#1. [pending] refactor onion",
        ]
    )
    return {"role": "user", "content": f"<system-reminder>\n{body}\n</system-reminder>"}


def _seed_chat_with_messages(
    tmp_path: Path,
    messages: list[dict[str, Any]],
) -> tuple[TerminalChat, RecordingView]:
    agent = ToolAgent()
    agent.messages = messages
    chat, _output, _prompts = _chat(agent, tmp_path, ["/quit"])
    view = RecordingView()
    chat.view = view  # swap in the recorder for direct introspection
    return chat, view


@pytest.mark.asyncio
async def test_render_transcript_folds_reminder_as_dim_notice(tmp_path: Path):
    """v0.8.1: a <system-reminder> user message in agent.messages renders
    as a single dim notice on replay (after /load or /clear), not as a
    plain user block — and adjacent real turns render normally."""
    chat, view = _seed_chat_with_messages(
        tmp_path,
        [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            _make_reminder(),
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "ok"},
        ],
    )
    chat.store.add_task("refactor onion")

    await chat._render_transcript()

    notice_calls = [c for c in view.calls if c[0] == "notice"]
    user_calls = [c for c in view.calls if c[0] == "user"]
    # The reminder renders as exactly one dim notice.
    assert len(notice_calls) == 1
    assert notice_calls[0][2] == "dim"
    assert notice_calls[0][1].startswith("system reminder: task list injected")
    # Real user messages render normally — the reminder did not eat them.
    assert [c[1] for c in user_calls] == ["hello", "next"]


@pytest.mark.asyncio
async def test_send_message_folds_live_reminder_appended_during_turn(
    tmp_path: Path,
):
    """v0.8.1: when the agent injects a reminder before the LLM call,
    the user sees the folded notice land in the live transcript
    exactly once per reminder, not as a plain user block."""

    class ReminderAgent(ToolAgent):
        async def stream_reply(self, user_text: str, *, executor: Any = None):
            self.messages.append({"role": "user", "content": user_text})
            # Mirror what agent_loop._maybe_inject_task_reminder does:
            # append a Claude-shaped <system-reminder> user message
            # before the assistant turn.
            self.messages.append(_make_reminder())
            self.messages.append({"role": "assistant", "content": "ack"})
            yield AgentEvent("assistant", "ack")
            yield AgentEvent("done")

    agent = ReminderAgent()
    chat, _output, _prompts = _chat(agent, tmp_path, ["/quit"])
    view = RecordingView()
    chat.view = view
    chat.store.add_task("refactor onion")

    await chat._send_message("kickoff")

    notice_calls = [c for c in view.calls if c[0] == "notice"]
    folded_calls = [
        c
        for c in notice_calls
        if c[1].startswith("system reminder: task list injected")
    ]
    assert len(folded_calls) == 1
    assert folded_calls[0][2] == "dim"


@pytest.mark.asyncio
async def test_send_message_skips_folded_notice_when_no_reminder_landed(
    tmp_path: Path,
):
    """A turn that doesn't trigger a reminder must not emit the folded
    notice — guard against false positives in the snapshot-diff logic."""
    agent = ToolAgent()  # base ToolAgent does no reminder injection
    chat, _output, _prompts = _chat(agent, tmp_path, ["/quit"])
    view = RecordingView()
    chat.view = view

    await chat._send_message("hello")

    notice_calls = [c for c in view.calls if c[0] == "notice"]
    assert not any(
        c[1].startswith("system reminder: task list injected") for c in notice_calls
    )


# ---- v0.9.1: keyboard ergonomics (Codex parity + backslash continuation) ----


def test_quit_arming_state_fresh_is_outside_window():
    """A pristine state has ``armed_at=-math.inf`` so the window is
    always closed — no hint, no exit, no chord."""
    state = QuitArmingState()
    assert state.within_window() is False
    assert state.within_window("c-c") is False
    assert state.within_window("c-d") is False
    assert state.hint_text() is None


def test_quit_arming_state_arm_c_c_only_confirms_c_c():
    """After ``arm("c-c")``: the renderer-mode predicate
    (``within_window()``) is True, the c-c gate is True, but a c-d
    press would NOT confirm — that's what makes cross-key presses
    re-arm instead of exit."""
    state = QuitArmingState()
    state.arm("c-c")
    assert state.within_window() is True
    assert state.within_window("c-c") is True
    assert state.within_window("c-d") is False
    assert state.hint_text() == "press Ctrl+C again to exit"


def test_quit_arming_state_arm_c_d_only_confirms_c_d():
    """Symmetric with the c-c case; the hint string flips."""
    state = QuitArmingState()
    state.arm("c-d")
    assert state.within_window("c-d") is True
    assert state.within_window("c-c") is False
    assert state.hint_text() == "press Ctrl+D again to exit"


def test_quit_arming_state_cross_arm_keeps_other_chord_timer_alive(monkeypatch):
    """Independent timers: arm("c-c") then arm("c-d") leaves c-c's
    own window UNTOUCHED. The hint refreshes to the most recent
    chord (c-d), but the c-c gate stays True so a third press of
    Ctrl+C still exits. This is the proof that cross-key presses
    are non-destructive."""
    import time as _time

    fake_now = [1000.0]
    monkeypatch.setattr(_time, "monotonic", lambda: fake_now[0])

    state = QuitArmingState()
    state.arm("c-c")
    assert state.hint_text() == "press Ctrl+C again to exit"

    # Time advances inside both chords' windows. arm c-d.
    fake_now[0] += 0.3
    state.arm("c-d")

    # Hint refreshes to the most-recent chord — c-d.
    assert state.hint_text() == "press Ctrl+D again to exit"
    # BOTH chords are within their own windows now.
    assert state.within_window("c-c") is True
    assert state.within_window("c-d") is True


def test_quit_arming_state_each_chord_expires_independently(monkeypatch):
    """Walk time past c-c's window but still inside c-d's. c-c
    becomes False; c-d stays True; the hint follows the last-armed
    chord (c-d in this scenario)."""
    import time as _time

    fake_now = [1000.0]
    monkeypatch.setattr(_time, "monotonic", lambda: fake_now[0])

    state = QuitArmingState()
    state.arm("c-c")  # at t=0
    fake_now[0] += 0.3
    state.arm("c-d")  # at t=0.3

    # Walk to t=1.05: c-c is expired (1.05 > 1.0), c-d still alive
    # (1.05 - 0.3 = 0.75 < 1.0).
    fake_now[0] = 1000.0 + 1.05
    assert state.within_window("c-c") is False
    assert state.within_window("c-d") is True
    # Renderer-mode follows last_armed_key (c-d, still alive).
    assert state.within_window() is True
    assert state.hint_text() == "press Ctrl+D again to exit"


def test_quit_arming_state_expires_after_window(monkeypatch):
    """Past ``QUIT_WINDOW_S`` the chord's window closes; once the
    last-armed chord is gone, ``hint_text()`` falls to None."""
    import time as _time

    fake_now = [1000.0]
    monkeypatch.setattr(_time, "monotonic", lambda: fake_now[0])

    state = QuitArmingState()
    state.arm("c-c")
    assert state.within_window() is True

    fake_now[0] += QuitArmingState.QUIT_WINDOW_S + 0.05
    assert state.within_window() is False
    assert state.within_window("c-c") is False
    assert state.hint_text() is None


def test_apply_enter_or_continuation_strips_trailing_backslash():
    """Bash- / Claude-style continuation: trailing ``\\`` + Enter →
    newline, backslash consumed, cursor parked at the new end."""
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document

    buf = Buffer()
    buf.document = Document("hello\\", cursor_position=6)

    handled = apply_enter_or_continuation(buf)

    assert handled is True
    assert buf.text == "hello\n"
    assert buf.cursor_position == len(buf.text)


def test_apply_enter_or_continuation_no_op_when_no_trailing_backslash():
    """Plain Enter: the helper returns False so the binding submits
    instead of inserting a newline."""
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document

    buf = Buffer()
    buf.document = Document("hello", cursor_position=5)

    assert apply_enter_or_continuation(buf) is False
    assert buf.text == "hello"


def test_apply_enter_or_continuation_ignores_mid_buffer_backslash():
    """Continuation only fires at end-of-buffer; a backslash with text
    after the cursor must NOT be eaten, because the user is editing
    earlier in the draft and pressing Enter to split a line."""
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document

    buf = Buffer()
    buf.document = Document("hello\\world", cursor_position=6)

    assert apply_enter_or_continuation(buf) is False
    assert buf.text == "hello\\world"


def test_build_draft_key_bindings_registers_newline_keys():
    """Acceptance: c-j is the v0.9.2 newline binding. The v0.9.1
    ``(escape, enter)`` Alt+Enter binding has been DELETED — v0.9.2
    binds ``escape`` directly (eager=True) as the cancel key, which
    forces the meta-prefix to dispatch before any composed sequence
    can match. Users insert newlines via Ctrl+J."""
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.keys import Keys

    bindings = build_draft_key_bindings()
    keys_registered = [kb.keys for kb in bindings.bindings]

    assert (Keys.ControlJ,) in keys_registered
    # Deliberate v0.9.2 regression: the Alt+Enter binding is gone so
    # Esc-as-cancel can claim the meta-prefix unambiguously.
    assert (Keys.Escape, Keys.ControlM) not in keys_registered
    # And the standalone Escape binding IS registered (cancel hook).
    assert (Keys.Escape,) in keys_registered


def test_build_draft_key_bindings_registers_quit_and_suspend_keys():
    """Acceptance: c-c, c-d (quit dance) and c-z (suspend) are bound.

    The c-d binding additionally carries a non-trivial ``filter`` (the
    buffer-empty :py:class:`prompt_toolkit.filters.Condition`) so that
    non-empty-buffer Ctrl+D still forward-deletes. We assert the
    binding exists AND that its filter is not the always-True default.
    """
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.filters import to_filter
    from prompt_toolkit.keys import Keys

    bindings = build_draft_key_bindings()
    keys_registered = [kb.keys for kb in bindings.bindings]

    assert (Keys.ControlC,) in keys_registered
    assert (Keys.ControlD,) in keys_registered
    assert (Keys.ControlZ,) in keys_registered

    # The c-d binding's filter must be something narrower than the
    # always-True default — otherwise non-empty buffers would lose
    # forward-delete.
    always_true = to_filter(True)
    c_d_bindings = [kb for kb in bindings.bindings if kb.keys == (Keys.ControlD,)]
    assert c_d_bindings, "c-d binding missing"
    assert c_d_bindings[0].filter is not always_true


def test_above_input_shows_quit_hint_when_armed(tmp_path: Path):
    """When the DraftReader's quit_state is armed, ``_above_input``
    renders a dim-yellow ``press Ctrl+C again to exit`` line; when
    disarmed it vanishes again. Covers the renderer side of the
    v0.9.1 hint contract."""
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.formatted_text import FormattedText

    agent = ToolAgent()
    chat = TerminalChat(
        agent,
        config=_config(tmp_path),
        render_markdown=False,
        input_func=None,  # force a real DraftReader, not the stub
        console=Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    reader = chat.view._draft_reader
    assert reader is not None  # sanity: prompt_toolkit branch active

    # Disarmed: nothing to render, fall through to "".
    assert chat._above_input() == ""

    # Arm with c-c → the hint names Ctrl+C specifically and renders
    # in the dim hierarchy color (not yellow — see PRD: the
    # affordance is a soft hint, not a warning).
    reader.quit_state.arm("c-c")
    rendered = chat._above_input()
    assert isinstance(rendered, FormattedText)
    fragments = list(rendered)
    assert fragments[-1] == (
        "fg:ansibrightblack",
        "press Ctrl+C again to exit\n",
    )

    # Re-arm with c-d → the hint flips to Ctrl+D (same dim style).
    reader.quit_state.arm("c-d")
    rendered = chat._above_input()
    assert isinstance(rendered, FormattedText)
    fragments = list(rendered)
    assert fragments[-1] == (
        "fg:ansibrightblack",
        "press Ctrl+D again to exit\n",
    )


@pytest.mark.asyncio
async def test_double_ctrl_c_exits_real_prompt_session_quickly():
    """End-to-end: drive a real ``PromptSession`` with our bindings,
    feed two ``\\x03`` bytes back-to-back, and assert ``prompt_async``
    raises ``KeyboardInterrupt`` well within the 1-s quit window.

    Catches regressions where the c-c binding fires but the
    ``within_window()`` predicate disagrees with the timer assumption.
    """
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from neutrix.terminal_chat import QuitArmingState as _QuitArmingState
    from neutrix.terminal_chat import build_draft_key_bindings

    quit_state = _QuitArmingState()
    with create_pipe_input() as pipe_input:
        session = PromptSession(
            input=pipe_input,
            output=DummyOutput(),
            key_bindings=build_draft_key_bindings(quit_state),
        )
        pipe_input.send_text("\x03\x03")
        with pytest.raises(KeyboardInterrupt):
            await asyncio.wait_for(
                session.prompt_async(handle_sigint=False),
                timeout=1.0,
            )


@pytest.mark.asyncio
async def test_single_ctrl_c_alone_does_not_exit():
    """End-to-end counterpart: a single ``\\x03`` arms the hint but
    does NOT exit; ``prompt_async`` keeps running until cancelled.

    Asserts the call blocks past the auto-fade deadline (so we know
    the binding isn't silently exiting on the lone press).
    """
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from neutrix.terminal_chat import QuitArmingState as _QuitArmingState
    from neutrix.terminal_chat import build_draft_key_bindings

    quit_state = _QuitArmingState()
    with create_pipe_input() as pipe_input:
        session = PromptSession(
            input=pipe_input,
            output=DummyOutput(),
            key_bindings=build_draft_key_bindings(quit_state),
        )
        pipe_input.send_text("\x03")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                session.prompt_async(handle_sigint=False),
                timeout=0.3,
            )


@pytest.mark.asyncio
async def test_double_ctrl_d_exits_real_prompt_session_with_eof():
    """Two ``\\x04`` bytes on an empty buffer must raise ``EOFError``,
    not ``KeyboardInterrupt``. The c-d branch shares
    :class:`QuitArmingState` with c-c but exits with a different
    exception so :py:meth:`_input_loop`'s existing ``except EOFError``
    fires."""
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from neutrix.terminal_chat import QuitArmingState as _QuitArmingState
    from neutrix.terminal_chat import build_draft_key_bindings

    quit_state = _QuitArmingState()
    with create_pipe_input() as pipe_input:
        session = PromptSession(
            input=pipe_input,
            output=DummyOutput(),
            key_bindings=build_draft_key_bindings(quit_state),
        )
        pipe_input.send_text("\x04\x04")
        with pytest.raises(EOFError):
            await asyncio.wait_for(
                session.prompt_async(handle_sigint=False),
                timeout=1.0,
            )


@pytest.mark.asyncio
async def test_single_ctrl_d_alone_does_not_exit():
    """A lone ``\\x04`` arms the hint but does NOT exit (symmetric
    with the c-c counterpart)."""
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from neutrix.terminal_chat import QuitArmingState as _QuitArmingState
    from neutrix.terminal_chat import build_draft_key_bindings

    quit_state = _QuitArmingState()
    with create_pipe_input() as pipe_input:
        session = PromptSession(
            input=pipe_input,
            output=DummyOutput(),
            key_bindings=build_draft_key_bindings(quit_state),
        )
        pipe_input.send_text("\x04")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                session.prompt_async(handle_sigint=False),
                timeout=0.3,
            )


@pytest.mark.asyncio
async def test_ctrl_c_then_ctrl_d_no_exit_both_timers_alive():
    """Cross-key: c-c arms, then c-d arms independently. Neither
    keystroke is a second tap of its own chord, so the call must
    NOT exit. Both timers stay alive on their own clocks.

    Asserts the prompt times out AND both chords are still within
    their own windows.
    """
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from neutrix.terminal_chat import QuitArmingState as _QuitArmingState
    from neutrix.terminal_chat import build_draft_key_bindings

    quit_state = _QuitArmingState()
    with create_pipe_input() as pipe_input:
        session = PromptSession(
            input=pipe_input,
            output=DummyOutput(),
            key_bindings=build_draft_key_bindings(quit_state),
        )
        pipe_input.send_text("\x03\x04")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                session.prompt_async(handle_sigint=False),
                timeout=0.3,
            )
        # The hint follows the most-recent press (c-d), but BOTH
        # timers should still be alive — independent clocks.
        assert quit_state.last_armed_key == "c-d"
        assert quit_state.hint_text() == "press Ctrl+D again to exit"
        assert quit_state.within_window("c-c") is True
        assert quit_state.within_window("c-d") is True


@pytest.mark.asyncio
async def test_ctrl_d_then_ctrl_c_no_exit_both_timers_alive():
    """Symmetric: c-d arms then c-c arms; no exit; both timers
    alive; the hint follows c-c (most recent)."""
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from neutrix.terminal_chat import QuitArmingState as _QuitArmingState
    from neutrix.terminal_chat import build_draft_key_bindings

    quit_state = _QuitArmingState()
    with create_pipe_input() as pipe_input:
        session = PromptSession(
            input=pipe_input,
            output=DummyOutput(),
            key_bindings=build_draft_key_bindings(quit_state),
        )
        pipe_input.send_text("\x04\x03")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                session.prompt_async(handle_sigint=False),
                timeout=0.3,
            )
        assert quit_state.last_armed_key == "c-c"
        assert quit_state.hint_text() == "press Ctrl+C again to exit"
        assert quit_state.within_window("c-c") is True
        assert quit_state.within_window("c-d") is True


@pytest.mark.asyncio
async def test_ctrl_c_then_ctrl_d_then_ctrl_c_exits_keyboard_interrupt():
    """Independent-timer lock-in: c-c arms → c-d arms → c-c again
    completes the c-c double-tap within c-c's ORIGINAL window. The
    intervening c-d never touched c-c's clock, so the third tap
    exits via ``KeyboardInterrupt``."""
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from neutrix.terminal_chat import QuitArmingState as _QuitArmingState
    from neutrix.terminal_chat import build_draft_key_bindings

    quit_state = _QuitArmingState()
    with create_pipe_input() as pipe_input:
        session = PromptSession(
            input=pipe_input,
            output=DummyOutput(),
            key_bindings=build_draft_key_bindings(quit_state),
        )
        pipe_input.send_text("\x03\x04\x03")
        with pytest.raises(KeyboardInterrupt):
            await asyncio.wait_for(
                session.prompt_async(handle_sigint=False),
                timeout=1.0,
            )


@pytest.mark.asyncio
async def test_ctrl_d_then_ctrl_c_then_ctrl_d_exits_eof():
    """Symmetric lock-in: c-d arms → c-c arms → c-d again
    completes the c-d double-tap within c-d's ORIGINAL window;
    exits via ``EOFError``."""
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from neutrix.terminal_chat import QuitArmingState as _QuitArmingState
    from neutrix.terminal_chat import build_draft_key_bindings

    quit_state = _QuitArmingState()
    with create_pipe_input() as pipe_input:
        session = PromptSession(
            input=pipe_input,
            output=DummyOutput(),
            key_bindings=build_draft_key_bindings(quit_state),
        )
        pipe_input.send_text("\x04\x03\x04")
        with pytest.raises(EOFError):
            await asyncio.wait_for(
                session.prompt_async(handle_sigint=False),
                timeout=1.0,
            )


@pytest.mark.asyncio
async def test_ctrl_d_on_non_empty_buffer_does_not_arm_quit():
    """When the draft has text, Ctrl+D must keep its default
    forward-delete-character behavior and NOT enter the quit dance.

    Feed ``"hi\\x04"`` — typing ``hi``, then Ctrl+D with the cursor
    at end-of-buffer. With the cursor at end there's nothing to
    forward-delete, but the critical invariant is that the
    QuitArmingState window stays closed (within_window() False)
    so a subsequent lone Ctrl+D / Ctrl+C would not exit. After
    submitting with Enter the call must complete normally with
    text "hi"."""
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from neutrix.terminal_chat import QuitArmingState as _QuitArmingState
    from neutrix.terminal_chat import build_draft_key_bindings

    quit_state = _QuitArmingState()
    with create_pipe_input() as pipe_input:
        session = PromptSession(
            input=pipe_input,
            output=DummyOutput(),
            key_bindings=build_draft_key_bindings(quit_state),
        )
        # ``hi`` then Ctrl+D (no-op at end-of-buffer for default
        # delete-char-forward — important: NOT a quit) then Enter.
        pipe_input.send_text("hi\x04\r")
        result = await asyncio.wait_for(
            session.prompt_async(handle_sigint=False),
            timeout=1.0,
        )
        assert result == "hi"
        assert quit_state.within_window() is False


# ---- v0.9.2: Esc / Ctrl+C cancellation + idle-state contract ---------------


@pytest.mark.asyncio
async def test_escape_invokes_cancel_hook_exactly_once():
    """Drive a real PromptSession with our bindings; feed ``\\x1b``;
    assert the registered ``cancel_hook`` fired once. Esc is the
    universal "stop" key from v0.9.2 onward."""
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from neutrix.terminal_chat import QuitArmingState as _QuitArmingState
    from neutrix.terminal_chat import build_draft_key_bindings

    calls: list[bool] = []

    def hook() -> bool:
        calls.append(True)
        return True  # pretend we cancelled something

    quit_state = _QuitArmingState()
    with create_pipe_input() as pipe_input:
        session = PromptSession(
            input=pipe_input,
            output=DummyOutput(),
            key_bindings=build_draft_key_bindings(quit_state, cancel_hook=hook),
        )
        pipe_input.send_text("\x1b\r")  # Esc then Enter to surface result
        await asyncio.wait_for(
            session.prompt_async(handle_sigint=False),
            timeout=1.0,
        )
        assert calls == [True]


@pytest.mark.asyncio
async def test_escape_while_idle_no_buffer_mutation():
    """Esc with no cancel hook (or a hook that returns False) must
    be a clean no-op: no exception, no buffer change."""
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from neutrix.terminal_chat import QuitArmingState as _QuitArmingState
    from neutrix.terminal_chat import build_draft_key_bindings

    quit_state = _QuitArmingState()
    with create_pipe_input() as pipe_input:
        session = PromptSession(
            input=pipe_input,
            output=DummyOutput(),
            key_bindings=build_draft_key_bindings(quit_state, cancel_hook=None),
        )
        pipe_input.send_text("hello\x1b\r")  # Esc with a non-empty buffer
        result = await asyncio.wait_for(
            session.prompt_async(handle_sigint=False),
            timeout=1.0,
        )
        assert result == "hello"
        assert quit_state.within_window() is False


@pytest.mark.asyncio
async def test_alt_enter_no_longer_inserts_newline():
    """Deliberate v0.9.2 regression: the v0.9.1 ``(escape, enter)``
    binding has been deleted. ``\\x1b\\r`` no longer produces a
    newline in the buffer. ``eager=True`` on the standalone Esc
    binding swallows the meta-prefix so the composed sequence
    never matches."""
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from neutrix.terminal_chat import QuitArmingState as _QuitArmingState
    from neutrix.terminal_chat import build_draft_key_bindings

    quit_state = _QuitArmingState()
    with create_pipe_input() as pipe_input:
        session = PromptSession(
            input=pipe_input,
            output=DummyOutput(),
            key_bindings=build_draft_key_bindings(quit_state),
        )
        # Alt+Enter (`\x1b\r`) then a final Enter to submit.
        pipe_input.send_text("hi\x1b\r\r")
        result = await asyncio.wait_for(
            session.prompt_async(handle_sigint=False),
            timeout=1.0,
        )
        # No newline inserted; only the typed "hi" survives.
        assert result == "hi"


@pytest.mark.asyncio
async def test_ctrl_j_still_inserts_newline():
    """Ctrl+J is the v0.9.2 newline binding. Feed ``hi\\nworld\\r``;
    the buffer composes "hi\\nworld" when submitted."""
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from neutrix.terminal_chat import QuitArmingState as _QuitArmingState
    from neutrix.terminal_chat import build_draft_key_bindings

    quit_state = _QuitArmingState()
    with create_pipe_input() as pipe_input:
        session = PromptSession(
            input=pipe_input,
            output=DummyOutput(),
            key_bindings=build_draft_key_bindings(quit_state),
        )
        pipe_input.send_text("hi\nworld\r")
        result = await asyncio.wait_for(
            session.prompt_async(handle_sigint=False),
            timeout=1.0,
        )
        assert result == "hi\nworld"


@pytest.mark.asyncio
async def test_ctrl_c_while_busy_cancels_and_does_not_arm_quit():
    """First Ctrl+C while a turn is in flight: cancel fires, quit
    is NOT armed (the v0.9.1 hint must not appear during cancel)."""
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from neutrix.terminal_chat import QuitArmingState as _QuitArmingState
    from neutrix.terminal_chat import build_draft_key_bindings

    def busy_hook() -> bool:
        return True  # something was in flight; we cancelled it

    quit_state = _QuitArmingState()
    with create_pipe_input() as pipe_input:
        session = PromptSession(
            input=pipe_input,
            output=DummyOutput(),
            key_bindings=build_draft_key_bindings(
                quit_state, cancel_hook=busy_hook
            ),
        )
        pipe_input.send_text("\x03")  # single Ctrl+C
        # The prompt must NOT exit on the lone Ctrl+C-while-busy;
        # cancellation handles it without arming quit.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                session.prompt_async(handle_sigint=False),
                timeout=0.3,
            )
        # And the v0.9.1 arming window stays closed.
        assert quit_state.within_window("c-c") is False
        assert quit_state.hint_text() is None


@pytest.mark.asyncio
async def test_ctrl_c_while_idle_still_arms_quit():
    """When the cancel hook reports False (nothing in flight) the
    binding falls through to v0.9.1's arm-or-exit dance, which arms
    c-c's window. Symmetric proof that idle-mode preserved."""
    pytest.importorskip("prompt_toolkit")
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from neutrix.terminal_chat import QuitArmingState as _QuitArmingState
    from neutrix.terminal_chat import build_draft_key_bindings

    def idle_hook() -> bool:
        return False

    quit_state = _QuitArmingState()
    with create_pipe_input() as pipe_input:
        session = PromptSession(
            input=pipe_input,
            output=DummyOutput(),
            key_bindings=build_draft_key_bindings(
                quit_state, cancel_hook=idle_hook
            ),
        )
        pipe_input.send_text("\x03")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                session.prompt_async(handle_sigint=False),
                timeout=0.3,
            )
        assert quit_state.within_window("c-c") is True


@pytest.mark.asyncio
async def test_send_message_routes_through_controller_and_recovers_on_cancel(
    tmp_path: Path,
):
    """End-to-end: a fake-LLM agent that suspends mid-stream. Trigger
    cancel via ``try_cancel_current_stream``. ``_send_message``
    catches CancelledError, restores the idle-state contract, and a
    follow-up ``_send_message`` reaches the controller cleanly."""
    from neutrix.agent_loop import Agent

    class SuspendingLLMForChat:
        """LLM that blocks the first request, completes the second."""

        def __init__(self) -> None:
            self.released = asyncio.Event()
            self.calls = 0

        def switch(self, slot: Slot) -> None:
            pass

        def stop(self) -> None:
            self.released.set()

        async def stream_response(self, *, model, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                await self.released.wait()
                return
            from neutrix.llm import LLMEvent, LLMResponse

            yield LLMEvent(
                "assistant",
                LLMResponse(
                    {"role": "assistant", "content": "ok"},
                    finish_reason="stop",
                ),
            )

    slot = _slot()
    fake_llm = SuspendingLLMForChat()
    real_agent = Agent(slot=slot, llm=fake_llm, use_tools=False)
    output = StringIO()

    def input_func(prompt: str) -> str:  # pragma: no cover - never called
        return ""

    chat = TerminalChat(
        real_agent,
        config=_config(tmp_path),
        render_markdown=False,
        input_func=input_func,
        console=Console(
            file=output,
            force_terminal=False,
            color_system=None,
            width=100,
        ),
    )

    send_task = asyncio.create_task(chat._send_message("first"))
    # Wait until the controller has set its in-flight task — that's
    # when ``try_cancel_current_stream`` actually does something.
    for _ in range(100):
        await asyncio.sleep(0.01)
        if chat.controller._current_stream_task is not None:
            break
    assert chat.controller._current_stream_task is not None

    assert chat.try_cancel_current_stream() is True
    await asyncio.wait_for(send_task, timeout=1.0)

    # Idle-state contract.
    assert chat.store.llm_active is False
    assert chat.store.pending_tool_calls == ()
    assert "interrupted" in output.getvalue()
    # Agent.messages rolled back: no orphan user_turn.
    assert real_agent.messages == [
        {"role": "system", "content": real_agent.system_prompt}
    ]

    # Follow-up call must work normally.
    await asyncio.wait_for(chat._send_message("again"), timeout=2.0)
    assert chat.store.llm_active is False
    # second LLM call completed and left "ok" assistant turn.
    assert any(
        msg.get("role") == "assistant" and msg.get("content") == "ok"
        for msg in real_agent.messages
    )


@pytest.mark.asyncio
async def test_try_cancel_when_idle_returns_false(tmp_path: Path):
    """``try_cancel_current_stream`` is the key-binding hook; when
    nothing is in flight it returns False so the c-c handler falls
    through to its v0.9.1 arming."""
    from neutrix.agent_loop import Agent

    class DormantLLM:
        def switch(self, slot: Slot) -> None: pass
        def stop(self) -> None: pass
        async def stream_response(self, **_: Any):  # pragma: no cover
            if False:
                yield None

    slot = _slot()
    agent = Agent(slot=slot, llm=DormantLLM(), use_tools=False)
    chat = TerminalChat(
        agent,
        config=_config(tmp_path),
        render_markdown=False,
        input_func=lambda _: "",
        console=Console(file=StringIO(), force_terminal=False, color_system=None),
    )
    assert chat.try_cancel_current_stream() is False


@pytest.mark.asyncio
async def test_queued_user_messages_survive_cancel(tmp_path: Path):
    """A message queued while busy must outlive the cancel — once the
    worker loop returns to ``queue.get()``, the next message starts
    processing on its own."""
    agent = BlockingAgent()
    output = StringIO()
    input_values: Queue[str] = Queue()

    def input_func(prompt: str) -> str:
        return input_values.get(timeout=2)

    chat = TerminalChat(
        agent,
        config=_config(tmp_path),
        render_markdown=False,
        input_func=input_func,
        console=Console(
            file=output,
            force_terminal=False,
            color_system=None,
            width=100,
        ),
    )
    task = asyncio.create_task(chat.run_async())

    input_values.put("first")
    assert await asyncio.wait_for(agent.started.get(), timeout=1) == "first"

    input_values.put("queued-second")
    # Let the queue accept it.
    for _ in range(50):
        await asyncio.sleep(0.01)
        if chat.store.queued_user_messages:
            break
    assert [q.text for q in chat.store.queued_user_messages] == ["queued-second"]

    # Cancel the first turn. ``try_cancel_current_stream`` from the
    # PRD is the canonical hook — return value confirms the cancel
    # actually fired.
    assert chat.try_cancel_current_stream() is True

    # The worker should now drain the queue: "queued-second" is the
    # next message presented to the agent.
    assert (
        await asyncio.wait_for(agent.started.get(), timeout=2)
        == "queued-second"
    )

    await agent.releases.put(None)
    input_values.put("/quit")
    await asyncio.wait_for(task, timeout=2)
    rendered = output.getvalue()
    assert "interrupted" in rendered
    assert "queued-second" in rendered
