"""Tests for the append-only terminal chat renderer wired to ContextManager."""
from __future__ import annotations

import asyncio
import os
from io import StringIO
from pathlib import Path
from queue import Queue
from typing import Any

import pytest
from rich.console import Console

from neutrix.config import Config, Slot
from neutrix.context_manager import (
    ContextManager,
    UserMessageEvent,
)
from neutrix.cost_ledger import CostLedger
from neutrix.executor import Executor
from neutrix.llm import (
    INTERRUPTED_BY_USER_MARKER,
    LLMEvent,
    LLMResponse,
    Usage,
)
from neutrix.session_store import SessionWriter, new_session_id
from neutrix.store import ChatStore, ToolRecord
from neutrix.terminal_chat import (
    HEARTBEAT_GLYPH,
    QuitArmingState,
    TerminalChat,
    apply_enter_or_continuation,
    approximate_token_count,
    delete_buffer_to_line_end,
    format_task_panel,
    move_buffer_to_line_start,
    result_line_count,
    tool_record_summary,
)

QUEUED_PREFIX = "› "  # noqa: RUF001  -- matches the renderer's chosen glyph


# ---- fixtures --------------------------------------------------------------


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
        pricing={
            "currency": "$",
            "models": {
                # priced at the real claude-opus-4-7 USD rates (per Mtok)
                "anthropic/claude-opus-4-7": {
                    "input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_write": 6.25,
                },
            },
        },
    )


class FakeLLM:
    """Yields pre-canned rounds. One ``rounds`` entry per LLM round."""

    def __init__(self, rounds: list[list[LLMEvent]] | None = None) -> None:
        self.rounds = list(rounds or [])
        self.switched_to: Slot | None = None

    def switch(self, slot: Slot) -> None:
        self.switched_to = slot

    def stop(self) -> None:
        pass

    async def stream_response(self, *, model, messages, tools=None):
        if not self.rounds:
            return
        for event in self.rounds.pop(0):
            yield event


class BlockingLLM:
    """Blocks per round; releases when the test puts on ``releases``."""

    def __init__(self) -> None:
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.releases: asyncio.Queue[str] = asyncio.Queue()

    def switch(self, slot: Slot) -> None:
        pass

    def stop(self) -> None:
        pass

    async def stream_response(self, *, model, messages, tools=None):
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        await self.started.put(str(last_user))
        text = await self.releases.get()
        yield LLMEvent(
            "assistant",
            LLMResponse(
                message={"role": "assistant", "content": text},
                finish_reason="stop",
            ),
        )


def _assistant_text(text: str) -> LLMEvent:
    return LLMEvent(
        "assistant",
        LLMResponse(
            message={"role": "assistant", "content": text},
            finish_reason="stop",
        ),
    )


def _assistant_tool(name: str, args: str, call_id: str = "c1") -> LLMEvent:
    return LLMEvent(
        "assistant",
        LLMResponse(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    }
                ],
            },
            finish_reason="tool_calls",
        ),
    )


def _make_ctx(
    llm: Any,
    *,
    use_tools: bool = True,
    seed_messages: list[dict[str, Any]] | None = None,
) -> ContextManager:
    store = ChatStore()
    executor = Executor()
    if seed_messages is None:
        seed_messages = [{"role": "system", "content": "system prompt"}]
    return ContextManager(
        slot=_slot(),
        llm=llm,
        executor=executor,
        store=store,
        system_prompt="system prompt",
        use_tools=use_tools,
        messages=list(seed_messages),
    )


def _render(value: object) -> str:
    """Flatten a value that may be str or prompt_toolkit FormattedText."""
    if isinstance(value, str):
        return value
    if hasattr(value, "__iter__"):
        return "".join(text for _style, text in value)
    return str(value)


def _make_chat(
    ctx: ContextManager,
    tmp_path: Path,
    inputs: list[str],
) -> tuple[TerminalChat, StringIO, list[str]]:
    output = StringIO()
    input_iter = iter(inputs)
    prompts: list[str] = []
    console = Console(file=output, force_terminal=False, color_system=None, width=100)

    def input_func(prompt: str) -> str:
        prompts.append(prompt)
        return next(input_iter)

    chat = TerminalChat(
        ctx,
        config=_config(tmp_path),
        render_markdown=False,
        input_func=input_func,
        console=console,
    )
    # v1.5.2: redirect session logs to tmp so tests never touch the real cache.
    chat._session_home = tmp_path
    return chat, output, prompts


# ---- FakeBuffer / FakeDocument (for key-helper tests) -------------------


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

    def delete_before_cursor(self, count: int = 1) -> None:
        self.text = (
            self.text[: self.cursor_position - count]
            + self.text[self.cursor_position :]
        )
        self.cursor_position -= count

    def newline(self) -> None:
        self.text = (
            self.text[: self.cursor_position] + "\n" + self.text[self.cursor_position :]
        )
        self.cursor_position += 1


# ---- pure-helper tests --------------------------------------------------


def test_tool_result_summary_counts_lines_and_approx_tokens() -> None:
    assert result_line_count("") == 0
    assert result_line_count("one\ntwo\n") == 2
    assert approximate_token_count("one two\nthree") == 3


def test_apply_enter_or_continuation_strips_trailing_backslash():
    buffer = FakeBuffer("hello\\", 6)
    assert apply_enter_or_continuation(buffer) is True
    assert buffer.text == "hello\n"


def test_apply_enter_or_continuation_no_op_when_no_trailing_backslash():
    buffer = FakeBuffer("hello", 5)
    assert apply_enter_or_continuation(buffer) is False
    assert buffer.text == "hello"


def test_apply_enter_or_continuation_ignores_mid_buffer_backslash():
    buffer = FakeBuffer("hel\\lo", 6)
    assert apply_enter_or_continuation(buffer) is False


def test_delete_buffer_to_line_end_collapses_lines():
    """At end-of-line: deletes the newline to merge with the next line."""
    buffer = FakeBuffer("abc\ndef", 3)
    delete_buffer_to_line_end(buffer)
    assert buffer.text == "abcdef"

    """Mid-line: deletes from cursor to end of the logical line."""
    buffer = FakeBuffer("hello world", 5)
    delete_buffer_to_line_end(buffer)
    assert buffer.text == "hello"


def test_move_buffer_to_line_start_uses_document_position():
    buffer = FakeBuffer("abc\ndef", 6)
    move_buffer_to_line_start(buffer)
    assert buffer.cursor_position == 4


def test_format_task_panel_returns_empty_for_no_tasks():
    assert format_task_panel(()) == []


def test_tool_record_summary_format():
    record = ToolRecord(index=2, name="read_file", arguments='{"path":"x"}', result="hi\n")
    summary = tool_record_summary(record)
    assert "[tool 2]" in summary
    assert "read_file" in summary
    assert "folded" in summary


# ---- TerminalChat integration via ContextManager -----------------------


def test_terminal_chat_constructor_wires_store_and_ctx(tmp_path: Path) -> None:
    """Constructor uses ctx.store; executor is wired to the same store."""
    llm = FakeLLM()
    ctx = _make_ctx(llm)
    chat, _output, _prompts = _make_chat(ctx, tmp_path, ["/quit"])
    assert chat.store is ctx.store
    assert chat.ctx.executor.store is ctx.store


def test_terminal_chat_status_command_prints_current_state(tmp_path: Path) -> None:
    """/status prints slot, provider/model, tool state, and msg count."""
    llm = FakeLLM()
    ctx = _make_ctx(llm)
    chat, output, _prompts = _make_chat(ctx, tmp_path, ["/status", "/quit"])
    chat.run()
    rendered = output.getvalue()
    assert "strong" in rendered
    assert "test/strong-model" in rendered
    assert "tools:on" in rendered
    assert "msgs:1" in rendered  # just the system message


def test_terminal_chat_tool_toggle_updates_status(tmp_path: Path) -> None:
    llm = FakeLLM()
    ctx = _make_ctx(llm)
    chat, output, _prompts = _make_chat(ctx, tmp_path, ["/tools off", "/quit"])
    chat.run()
    rendered = output.getvalue()
    assert "tool calling disabled" in rendered
    assert ctx.use_tools is False


def test_terminal_chat_full_round_renders_user_assistant(tmp_path: Path) -> None:
    """A complete user→assistant round shows both messages in the
    transcript, driven by the store.changes() watcher."""
    llm = FakeLLM([[_assistant_text("hello back")]])
    ctx = _make_ctx(llm, use_tools=False)
    chat, output, _prompts = _make_chat(ctx, tmp_path, ["hi there", "/quit"])
    chat.run()
    rendered = output.getvalue()
    assert "hi there" in rendered
    assert "hello back" in rendered


def test_terminal_chat_renders_tool_use_and_result(
    tmp_path: Path, monkeypatch
) -> None:
    """A tool round: ``-> tool_use`` for the call and folded ``<- tool_result``
    for the response. The renderer reads from ChatStore.changes()."""
    monkeypatch.setattr(
        "neutrix.executor.dispatch",
        lambda name, arguments, **_: f"ran {name}\n",
    )
    llm = FakeLLM(
        [
            [_assistant_tool("list_dir", '{"path":"."}')],
            [_assistant_text("done")],
        ]
    )
    ctx = _make_ctx(llm, use_tools=True)
    chat, output, _prompts = _make_chat(ctx, tmp_path, ["please list", "/quit"])
    chat.run()
    rendered = output.getvalue()
    assert "please list" in rendered
    assert "-> tool_use" in rendered
    assert "list_dir" in rendered
    assert "<- tool_result" in rendered
    assert "done" in rendered


def test_terminal_chat_tasks_command_lists_seeded_tasks(tmp_path: Path) -> None:
    llm = FakeLLM()
    ctx = _make_ctx(llm)
    ctx.store.add_task("first")
    ctx.store.add_task("second")
    chat, output, _prompts = _make_chat(ctx, tmp_path, ["/tasks", "/quit"])
    chat.run()
    rendered = output.getvalue()
    assert "#1" in rendered
    assert "first" in rendered
    assert "second" in rendered


def test_terminal_chat_clear_resets_history(tmp_path: Path) -> None:
    """Sending then /clear leaves the messages list at system-only."""
    llm = FakeLLM([[_assistant_text("hello")]])
    ctx = _make_ctx(llm, use_tools=False)
    chat, _output, _prompts = _make_chat(ctx, tmp_path, ["hi", "/clear", "/quit"])
    chat.run()
    assert [m["role"] for m in ctx.messages] == ["system"]


def _seeded_body(n_pairs: int) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "system prompt"}]
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": f"user message number {i}"})
        msgs.append({"role": "assistant", "content": f"assistant reply number {i}"})
    return msgs


def test_terminal_chat_compact_notice_and_no_reprint(tmp_path: Path) -> None:
    """/compact prints a dim notice, shrinks history, and does NOT
    re-print the kept tail (already in scrollback — split #9)."""
    llm = FakeLLM()
    ctx = _make_ctx(llm, use_tools=False, seed_messages=_seeded_body(10))
    chat, output, _prompts = _make_chat(ctx, tmp_path, ["/compact", "/quit"])
    chat.run()

    text = output.getvalue()
    assert "compacted 10 turns" in text
    assert "tokens dropped" in text
    # History was compacted: system + one marker + 10 kept body messages.
    assert [m["role"] for m in ctx.messages][:2] == ["system", "user"]
    markers = [
        m
        for m in ctx.messages
        if isinstance(m.get("content"), str)
        and m["content"].startswith("<system-compact>")
    ]
    assert len(markers) == 1
    assert len(ctx.messages) == 12
    # The newest kept message was rendered once (at startup) and NOT
    # reprinted after compaction.
    assert text.count("user message number 9") == 1


def test_terminal_chat_compact_too_short_is_noop_notice(tmp_path: Path) -> None:
    """/compact on a trivially short conversation reports nothing to do."""
    llm = FakeLLM()
    ctx = _make_ctx(llm, use_tools=False, seed_messages=_seeded_body(0))
    chat, output, _prompts = _make_chat(ctx, tmp_path, ["/compact", "/quit"])
    chat.run()
    assert "nothing to compact" in output.getvalue()
    assert [m["role"] for m in ctx.messages] == ["system"]


# ---- cancel-as-steer (the v0.9.3 contract) -----------------------------


@pytest.mark.asyncio
async def test_save_while_busy_is_rejected_with_notice(tmp_path: Path) -> None:
    """``/save`` during a non-IDLE turn shows a notice; chat state
    unchanged. The check lives in TerminalChat's command shim (the
    only consumer of v0.9.3's ``ctx.is_busy()``)."""
    llm = BlockingLLM()
    ctx = _make_ctx(llm, use_tools=False)
    output = StringIO()
    input_values: Queue[str] = Queue()

    def input_func(_p: str) -> str:
        return input_values.get(timeout=5)

    chat = TerminalChat(
        ctx,
        config=_config(tmp_path),
        render_markdown=False,
        input_func=input_func,
        console=Console(file=output, force_terminal=False, color_system=None, width=100),
    )
    task = asyncio.create_task(chat.run_async())

    input_values.put("hi")
    await asyncio.wait_for(llm.started.get(), timeout=2.0)

    # While busy, /save should be rejected.
    input_values.put(f"/save {tmp_path / 'session.json'}")
    for _ in range(50):
        await asyncio.sleep(0.01)
        if "waits for the assistant" in output.getvalue():
            break
    assert "waits for the assistant" in output.getvalue()
    assert not (tmp_path / "session.json").exists()

    # Release the LLM so the chat can shut down.
    llm.releases.put_nowait("done")
    input_values.put("/quit")
    await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_try_cancel_when_idle_returns_false(tmp_path: Path) -> None:
    llm = FakeLLM()
    ctx = _make_ctx(llm)
    chat, _output, _prompts = _make_chat(ctx, tmp_path, ["/quit"])
    assert chat.try_cancel_current_stream() is False


@pytest.mark.asyncio
async def test_cancel_appends_marker_visible_in_transcript(tmp_path: Path) -> None:
    """Esc fires CancelEvent → marker visible in ChatStore + transcript."""
    llm = BlockingLLM()
    ctx = _make_ctx(llm, use_tools=False)
    output = StringIO()
    input_values: Queue[str] = Queue()
    prompts: list[str] = []

    def input_func(prompt: str) -> str:
        prompts.append(prompt)
        return input_values.get(timeout=5)

    chat = TerminalChat(
        ctx,
        config=_config(tmp_path),
        render_markdown=False,
        input_func=input_func,
        console=Console(file=output, force_terminal=False, color_system=None, width=120),
    )
    task = asyncio.create_task(chat.run_async())

    input_values.put("write a story")
    # Wait until the LLM is in flight.
    await asyncio.wait_for(llm.started.get(), timeout=2.0)

    # Fire cancel through the same hook the key binding uses.
    assert chat.try_cancel_current_stream() is True

    # Wait briefly so the renderer paints the marker before we shut down.
    for _ in range(50):
        await asyncio.sleep(0.01)
        if not ctx.is_busy():
            break

    input_values.put("/quit")
    await asyncio.wait_for(task, timeout=2.0)

    rendered = output.getvalue()
    assert INTERRUPTED_BY_USER_MARKER in rendered
    # And in messages/store.
    assert ctx.messages[-1]["content"] == INTERRUPTED_BY_USER_MARKER


@pytest.mark.asyncio
async def test_mirror_new_agent_messages_method_is_gone(tmp_path: Path) -> None:
    """v0.9.3 contract: TerminalChat no longer reaches into agent
    messages — store.changes() is the canonical source."""
    llm = FakeLLM()
    ctx = _make_ctx(llm)
    chat, _output, _prompts = _make_chat(ctx, tmp_path, ["/quit"])
    assert not hasattr(chat, "_mirror_new_agent_messages")


@pytest.mark.asyncio
async def test_steering_smoke_pairing_sees_cancelled_tool_result(
    tmp_path: Path, monkeypatch
) -> None:
    """The v0.9.3 steering smoke test.

    Sequence:
    1. User asks for a slow shell.
    2. LLM emits a ``run_shell sleep`` tool_call.
    3. We cancel mid-tool — drive task is cancelled, marker appended,
       the tool_finished event never lands (its result is dropped per
       PRD's pure-compute tool non-goal).
    4. User asks for a different command.
    5. The next LLM call's CM-owned messages contain
       ``[..., assistant(tool_calls=[c1]), user("[interrupted by user]"),
       user("instead, just ls")]`` — orphan tool_call with no matching
       tool message.
    6. The :func:`_ensure_tool_result_pairing` layer in
       :mod:`neutrix.llm` runs on this payload before the SDK sees it
       and synthesizes ``tool(c1, "[cancelled by user]")`` immediately
       after the orphan, so the outbound payload the LLM actually
       receives has the cancellation as a real tool result it can read.
    """
    from neutrix.llm import CANCELLED_TOOL_RESULT, _ensure_tool_result_pairing

    tool_entered = asyncio.Event()
    tool_release = asyncio.Event()

    def slow_dispatch(name, arguments, **_):
        import time

        tool_entered.set()
        for _ in range(300):
            if tool_release.is_set():
                break
            time.sleep(0.01)
        return f"ran {name}"

    monkeypatch.setattr("neutrix.executor.dispatch", slow_dispatch)

    captured_messages: list[list[dict[str, Any]]] = []

    class CapturingLLM:
        def __init__(self) -> None:
            self.calls = 0

        def switch(self, slot: Slot) -> None:
            pass

        def stop(self) -> None:
            pass

        async def stream_response(self, *, model, messages, tools=None):
            captured_messages.append([dict(m) for m in messages])
            self.calls += 1
            if self.calls == 1:
                yield _assistant_tool("run_shell", '{"command":"sleep 30"}', call_id="c1")
                return
            yield _assistant_text("ok, just ls")

    llm = CapturingLLM()
    ctx = _make_ctx(llm, use_tools=True)

    # Send first user message; wait until tool dispatch is in flight.
    msg_task = asyncio.create_task(ctx.handle_event(UserMessageEvent("run sleep")))
    while not tool_entered.is_set():
        await asyncio.sleep(0.01)

    # Cancel.
    assert ctx.cancel() is True
    tool_release.set()  # let the daemon thread complete in background
    await asyncio.wait_for(msg_task, timeout=2.0)

    # Send the steering prompt.
    await ctx.handle_event(UserMessageEvent("instead, just ls"))

    assert len(captured_messages) >= 2
    second_round_input = captured_messages[1]
    # The CM-owned messages going into the LLM should have the orphan +
    # marker + steering prompt, without any tool message for c1 (the
    # background thread's result was dropped).
    assert not any(
        m.get("role") == "tool" and m.get("tool_call_id") == "c1"
        for m in second_round_input
    )
    # Apply pairing — this is what OpenAIChatLLM does before the SDK
    # sees the payload.
    paired = _ensure_tool_result_pairing(second_round_input)
    assistant_idx = next(
        i
        for i, m in enumerate(paired)
        if m.get("role") == "assistant"
        and isinstance(m.get("tool_calls"), list)
        and m["tool_calls"]
    )
    # Synthetic tool_result is inserted immediately after the orphan.
    assert paired[assistant_idx + 1]["role"] == "tool"
    assert paired[assistant_idx + 1]["tool_call_id"] == "c1"
    assert paired[assistant_idx + 1]["content"] == CANCELLED_TOOL_RESULT
    # The marker and steering prompt are still in the paired output.
    assert any(
        m.get("role") == "user" and m.get("content") == INTERRUPTED_BY_USER_MARKER
        for m in paired
    )
    assert paired[-1] == {"role": "user", "content": "instead, just ls"}


# ---- queue display while busy ------------------------------------------


@pytest.mark.asyncio
async def test_queue_display_while_busy_shows_queued_user_messages(tmp_path: Path) -> None:
    """User types while LLM is busy → message lands in store.queued + queue display."""
    llm = BlockingLLM()
    ctx = _make_ctx(llm, use_tools=False)
    output = StringIO()
    input_values: Queue[str] = Queue()

    def input_func(_p: str) -> str:
        return input_values.get(timeout=5)

    chat = TerminalChat(
        ctx,
        config=_config(tmp_path),
        render_markdown=False,
        input_func=input_func,
        console=Console(file=output, force_terminal=False, color_system=None, width=100),
    )
    task = asyncio.create_task(chat.run_async())

    input_values.put("first")
    await asyncio.wait_for(llm.started.get(), timeout=2.0)
    input_values.put("second")

    # Wait until the queue display picks it up.
    for _ in range(50):
        await asyncio.sleep(0.01)
        if chat.store.queued_user_messages:
            break
    queued = chat.store.queued_user_messages
    assert [q.text for q in queued] == ["second"]
    rendered_above = _render(chat._above_input())
    assert f"{QUEUED_PREFIX}second" in rendered_above

    # Release first, then second.
    llm.releases.put_nowait("done first")
    await asyncio.wait_for(llm.started.get(), timeout=2.0)
    llm.releases.put_nowait("done second")
    input_values.put("/quit")
    await asyncio.wait_for(task, timeout=2.0)

    rendered = output.getvalue()
    assert "first" in rendered
    assert "second" in rendered


# ---- heartbeat above-input integration (v0.9.4) -------------------------


@pytest.mark.asyncio
async def test_heartbeat_renders_above_input_while_busy(tmp_path: Path) -> None:
    """During AWAITING_LLM, the heartbeat sits at the TOP of the
    above-input stack, ahead of the task panel and the queue (split #9).
    """
    llm = BlockingLLM()
    ctx = _make_ctx(llm, use_tools=False)
    ctx.store.add_task("first task")
    output = StringIO()
    input_values: Queue[str] = Queue()

    def input_func(_p: str) -> str:
        return input_values.get(timeout=5)

    chat = TerminalChat(
        ctx,
        config=_config(tmp_path),
        render_markdown=False,
        input_func=input_func,
        console=Console(file=output, force_terminal=False, color_system=None, width=100),
    )
    task = asyncio.create_task(chat.run_async())

    input_values.put("hi")
    await asyncio.wait_for(llm.started.get(), timeout=2.0)
    # Enqueue a second user message so all three above-input rows have content.
    input_values.put("second")
    for _ in range(50):
        await asyncio.sleep(0.01)
        if chat.store.queued_user_messages:
            break

    # v0.9.8: the dot winks on/off by tick parity; pin an even (visible)
    # tick so this layout/stack-order assertion is deterministic. The wink
    # timing itself is covered by tests/test_heartbeat_format.py.
    chat._heartbeat_tick = 0
    rendered = _render(chat._above_input())
    assert HEARTBEAT_GLYPH in rendered
    assert "LLM" in rendered
    assert "first task" in rendered
    assert f"{QUEUED_PREFIX}second" in rendered

    # Stack order: heartbeat → task panel → queue.
    heartbeat_pos = rendered.index(HEARTBEAT_GLYPH)
    task_pos = rendered.index("first task")
    queue_pos = rendered.index(f"{QUEUED_PREFIX}second")
    assert heartbeat_pos < task_pos < queue_pos

    llm.releases.put_nowait("done first")
    await asyncio.wait_for(llm.started.get(), timeout=2.0)
    llm.releases.put_nowait("done second")
    input_values.put("/quit")
    await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_heartbeat_absent_above_input_when_idle(tmp_path: Path) -> None:
    """At IDLE, the heartbeat renders nothing — only tasks/queue/hint."""
    llm = FakeLLM()
    ctx = _make_ctx(llm)
    ctx.store.add_task("idle task")
    chat, _output, _prompts = _make_chat(ctx, tmp_path, ["/quit"])

    rendered = _render(chat._above_input())
    assert HEARTBEAT_GLYPH not in rendered
    assert "LLM" not in rendered
    assert "idle task" in rendered


# ---- QuitArmingState (unchanged semantics) ------------------------------


def test_quit_arming_state_fresh_is_outside_window():
    state = QuitArmingState()
    assert state.hint_text() is None
    assert state.within_window("c-c") is False
    assert state.within_window("c-d") is False


def test_quit_arming_state_arm_c_c_only_confirms_c_c():
    state = QuitArmingState()
    state.arm("c-c")
    assert state.within_window("c-c") is True
    # c-d does NOT confirm c-c.
    assert state.within_window("c-d") is False


def test_quit_arming_state_arm_c_d_only_confirms_c_d():
    state = QuitArmingState()
    state.arm("c-d")
    assert state.within_window("c-d") is True
    assert state.within_window("c-c") is False


def test_quit_arming_state_expires_after_window(monkeypatch):
    import time

    state = QuitArmingState()
    base = time.monotonic()
    monkeypatch.setattr("time.monotonic", lambda: base)
    state.arm("c-c")
    # Within the window.
    monkeypatch.setattr("time.monotonic", lambda: base + 0.5)
    assert state.within_window("c-c") is True
    # Past the window.
    monkeypatch.setattr("time.monotonic", lambda: base + 2.0)
    assert state.within_window("c-c") is False
    assert state.hint_text() is None


# ---- save/load round trip ---------------------------------------------


def test_terminal_chat_save_and_load_round_trip_via_commands(tmp_path: Path) -> None:
    """A session with a user turn + assistant survives /save → /load."""
    save_path = tmp_path / "session.json"
    llm = FakeLLM([[_assistant_text("hello")]])
    ctx = _make_ctx(llm, use_tools=False)
    chat, _output, _prompts = _make_chat(
        ctx, tmp_path, ["hi", f"/save {save_path}", "/quit"]
    )
    chat.run()
    assert save_path.exists()

    # Load into a fresh chat.
    llm2 = FakeLLM()
    ctx2 = _make_ctx(llm2, use_tools=False)
    chat2, output2, _prompts2 = _make_chat(
        ctx2, tmp_path, [f"/load {save_path}", "/quit"]
    )
    chat2.run()
    assert "hi" in output2.getvalue()
    assert "hello" in output2.getvalue()


@pytest.mark.asyncio
async def test_cmd_init_enqueues_survey_prompt(tmp_path):
    """v1.2.0: /init drives the agent — it enqueues a non-empty survey prompt."""
    ctx = _make_ctx(FakeLLM())
    chat, _output, _prompts = _make_chat(ctx, tmp_path, inputs=[])
    chat._input_queue = asyncio.Queue()
    await chat._cmd_init([])
    queued = [q.text for q in chat.store.queued_user_messages]
    assert queued and "CLAUDE.md" in queued[0]
    assert chat._input_queue.qsize() == 1


@pytest.mark.asyncio
async def test_markdown_skill_dispatches_rendered_body(tmp_path):
    """v1.3.0: typing /name for a markdown skill enqueues its rendered body."""
    from neutrix.skills import SkillDef

    ctx = _make_ctx(FakeLLM())
    chat, _output, _prompts = _make_chat(ctx, tmp_path, inputs=[])
    chat._input_queue = asyncio.Queue()
    chat._skills = {
        "fixbug": SkillDef(
            name="fixbug", description="d", body="Investigate $ARGUMENTS thoroughly.", source="x"
        )
    }
    await chat._run_command("/fixbug the parser")
    queued = [q.text for q in chat.store.queued_user_messages]
    assert queued == ["Investigate the parser thoroughly."]
    assert chat._input_queue.qsize() == 1


@pytest.mark.asyncio
async def test_unknown_command_still_errors(tmp_path):
    ctx = _make_ctx(FakeLLM())
    chat, output, _prompts = _make_chat(ctx, tmp_path, inputs=[])
    chat._input_queue = asyncio.Queue()
    chat._skills = {}
    await chat._run_command("/definitelynotacommand")
    assert "unknown command" in output.getvalue()


@pytest.mark.asyncio
async def test_cmd_allow_toggles_permission_mode(tmp_path):
    """v1.4.0: /allow toggles auto ↔ allow-all on the executor."""
    ctx = _make_ctx(FakeLLM())
    chat, _output, _prompts = _make_chat(ctx, tmp_path, inputs=[])
    assert ctx.executor.permission_mode == "auto"  # default
    await chat._cmd_allow([])
    assert ctx.executor.permission_mode == "allow-all"
    await chat._cmd_allow([])
    assert ctx.executor.permission_mode == "auto"


def test_stream_preview_bounds_to_last_lines():
    from neutrix.terminal_chat import STREAM_PREVIEW_LINES, _stream_preview

    assert _stream_preview(None) == ""
    assert _stream_preview("") == ""
    assert _stream_preview("one line") == "one line"
    many = "\n".join(f"L{i}" for i in range(1, STREAM_PREVIEW_LINES + 5))
    out = _stream_preview(many)
    assert out.startswith("… ")  # earlier lines elided
    assert out.endswith(f"L{STREAM_PREVIEW_LINES + 4}")  # last line shown
    assert "L1\n" not in out  # first lines dropped


def test_above_input_shows_streaming_preview(tmp_path):
    """v1.4.7: pending assistant text renders in the live region (not scrollback)."""
    ctx = _make_ctx(FakeLLM())
    chat, _output, _prompts = _make_chat(ctx, tmp_path, inputs=[])
    ctx.store.start_assistant_stream()
    ctx.store.extend_assistant_stream("hello streaming world")
    rendered = _render(chat._above_input())
    assert "hello streaming world" in rendered


# ---- v1.7.0: cost ledger surface + persistence -----------------------------


def _priced_ctx(llm: Any) -> ContextManager:
    """A ctx whose slot model is in the price table, so cost renders a dollar
    figure (not "(cost unknown)")."""
    return ContextManager(
        slot=Slot(
            name="strong",
            provider="ihep",
            model="anthropic/claude-opus-4-7",
            base_url="https://example.test/v1",
            api_key="sk-test",
        ),
        llm=llm,
        executor=Executor(),
        store=ChatStore(),
        system_prompt="system prompt",
        use_tools=False,
        messages=[{"role": "system", "content": "system prompt"}],
    )


def _assistant_usage(text: str, usage: Usage) -> LLMEvent:
    return LLMEvent(
        "assistant",
        LLMResponse(message={"role": "assistant", "content": text},
                    finish_reason="stop", usage=usage),
    )


def test_persist_new_usage_flushes_and_cursor_is_idempotent(tmp_path: Path) -> None:
    ctx = _priced_ctx(FakeLLM())
    chat, _out, _ = _make_chat(ctx, tmp_path, [])
    chat._setup_session_writer()
    assert chat._ledger is not None
    assert chat.ctx.ledger is chat._ledger  # injected into the CM
    chat._ledger.record("claude-opus-4-7", Usage(input=100, output=50), 10.0, 0.0)
    chat._persist_new_usage()
    chat._persist_new_usage()  # cursor guards against re-writing
    led = CostLedger.from_jsonl(chat._session_writer.path)
    assert len(led.entries) == 1
    assert led.entries[0].usage.input == 100


@pytest.mark.asyncio
async def test_turn_completion_flush_persists_final_turn_without_render_loop(
    tmp_path: Path, monkeypatch
) -> None:
    """The explicit turn-completion flush persists the final turn's usage even
    when the changes() render loop never runs — the advisor's stranding guard
    (Split #11). Drives one turn directly, with no render watcher and no
    shutdown flush, so only _process_user_turn's explicit flush could persist it.
    """
    llm = FakeLLM([[_assistant_usage("hi", Usage(input=120, output=30))]])
    ctx = _priced_ctx(llm)
    chat, _out, _ = _make_chat(ctx, tmp_path, [])
    chat._setup_session_writer()

    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr(chat, "_maybe_run_advisor", _noop)  # don't build a real client
    chat._input_queue = asyncio.Queue()
    chat.store.enqueue_user("hi")
    chat._input_queue.put_nowait("hi")
    await chat._input_queue.get()  # balance the task_done() in _process_user_turn

    await chat._process_user_turn("hi")  # render watcher is NOT running

    led = CostLedger.from_jsonl(chat._session_writer.path)
    led.price_table = chat.config.price_table()  # from_jsonl rebuilds entries, not the table
    assert len(led.entries) == 1  # only the explicit turn-completion flush could write this
    assert led.entries[0].usage.input == 120
    assert led.cost() is not None


def test_setup_writer_rebuilds_ledger_on_resume(tmp_path: Path) -> None:
    sid = new_session_id()
    seed = SessionWriter(os.getcwd(), sid, home=tmp_path)
    seed.append_usage(
        model="anthropic/claude-opus-4-7", usage=Usage(input=1_000_000, output=0), llm_ms=100.0
    )
    ctx = _priced_ctx(FakeLLM())
    chat, _out, _ = _make_chat(ctx, tmp_path, [])
    chat._resume_session_id = sid
    chat._setup_session_writer()
    assert chat._ledger is not None
    assert len(chat._ledger.entries) == 1
    assert chat._ledger.cost() == 5.0  # 1M input * $5/Mtok (real opus-4-7 rate, from _config)
    assert chat._usage_written_count == 1  # won't re-write the loaded entry
    assert chat.ctx.ledger is chat._ledger


def test_cost_readout_shows_dollars_and_hides_when_unknown(tmp_path: Path) -> None:
    ctx = _priced_ctx(FakeLLM())
    chat, _out, _ = _make_chat(ctx, tmp_path, [])
    chat._ledger = CostLedger()
    chat._ledger.price_table = chat.config.price_table()
    assert chat._cost_readout() is None  # no usage yet → hidden
    chat._ledger.record("anthropic/claude-opus-4-7", Usage(input=12_400, output=3_100), 0, 0)
    readout = chat._cost_readout()
    assert readout is not None and readout.startswith("$")
    assert "hit" in readout and "miss" in readout and "out" in readout
    # An unpriced model → cost unknown → readout hidden.
    chat._ledger = CostLedger()
    chat._ledger.price_table = chat.config.price_table()
    chat._ledger.record("mystery-model", Usage(input=1000, output=500), 0, 0)
    assert chat._cost_readout() is None


@pytest.mark.asyncio
async def test_cmd_cost_renders_totals_and_empty(tmp_path: Path) -> None:
    ctx = _priced_ctx(FakeLLM())
    chat, output, _ = _make_chat(ctx, tmp_path, [])
    chat._ledger = CostLedger()
    chat._ledger.price_table = chat.config.price_table()
    await chat._cmd_cost([])
    assert "no usage recorded" in output.getvalue()
    chat._ledger.record("anthropic/claude-opus-4-7", Usage(input=1_000_000, output=0), 1000.0, 0.0)
    await chat._cmd_cost([])
    out = output.getvalue()
    assert "$5.0000" in out  # 1M input * $5/Mtok
    assert "1,000,000 miss" in out
