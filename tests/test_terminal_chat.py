"""Tests for the append-only terminal chat renderer wired to ContextManager."""
from __future__ import annotations

import asyncio
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
from neutrix.executor import Executor
from neutrix.llm import (
    INTERRUPTED_BY_USER_MARKER,
    LLMEvent,
    LLMResponse,
)
from neutrix.store import ChatStore
from neutrix.terminal_chat import (
    QuitArmingState,
    TerminalChat,
    ToolRecord,
    apply_enter_or_continuation,
    approximate_token_count,
    delete_buffer_to_line_end,
    format_task_panel,
    move_buffer_to_line_start,
    result_line_count,
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
    assert "[tool 2]" in record.summary
    assert "read_file" in record.summary
    assert "folded" in record.summary


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
