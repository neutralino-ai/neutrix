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
    TerminalChat,
    ToolRecord,
    approximate_token_count,
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


class ToolAgent:
    def __init__(self) -> None:
        self.slot = _slot()
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

    async def stream_reply(self, user_text: str):
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

    async def stream_reply(self, user_text: str):
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

    async def stream_reply(self, user_text: str):
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
        async def stream_reply(self, user_text: str):
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
