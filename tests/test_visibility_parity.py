"""Visibility-parity invariant tests (v0.10.2).

Asserts that every channel the LLM receives in a round
(``ContextManager.round_bundle()``) is surfaced in the user-visible transcript —
folded summaries count (expand-by-append is the append-only fold model). The
``no_hidden_channel`` test iterates the bundle's fields dynamically so a future
input channel that isn't rendered trips it.
"""
from __future__ import annotations

import dataclasses
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from neutrix.config import Config, Slot
from neutrix.context_manager import ContextManager
from neutrix.executor import Executor
from neutrix.llm import LLMEvent, LLMResponse
from neutrix.store import ChatStore, MessageRecord, ToolRecord
from neutrix.terminal_chat import TerminalChat, tool_record_summary

_LONG_SYSTEM = "You are a meticulous assistant. " * 12  # > 200 B → folds


def _slot() -> Slot:
    return Slot(
        name="fast",
        provider="test",
        model="test-model",
        base_url="https://example.test/v1",
        api_key="sk-test",
    )


def _config(tmp_path: Path) -> Config:
    return Config(
        providers={"test": {"base_url": "https://example.test/v1", "api_key": "sk-test"}},
        slots={"fast": {"provider": "test", "model": "test-model"}},
        path=tmp_path / "config.yaml",
    )


class _FakeLLM:
    def switch(self, slot: Slot) -> None:  # pragma: no cover
        pass

    def stop(self) -> None:  # pragma: no cover
        pass

    async def stream_response(self, *, model, messages, tools=None):
        yield LLMEvent(
            "assistant", LLMResponse({"role": "assistant", "content": "ack"}, "stop")
        )


def _ctx(messages: list[dict[str, Any]], *, use_tools: bool = True) -> ContextManager:
    return ContextManager(
        slot=_slot(),
        llm=_FakeLLM(),
        executor=Executor(),
        store=ChatStore(),
        system_prompt=str(messages[0]["content"]),
        use_tools=use_tools,
        messages=list(messages),
    )


def _chat(ctx: ContextManager, tmp_path: Path) -> tuple[TerminalChat, StringIO]:
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=100)
    chat = TerminalChat(ctx, config=_config(tmp_path), render_markdown=False, console=console)
    return chat, output


async def _render_session(chat: TerminalChat) -> None:
    await chat._render_initial_transcript()
    await chat._render_tool_schemas_block()


# ---- focused channel tests ------------------------------------------------


@pytest.mark.asyncio
async def test_tool_schemas_block_renders_when_enabled(tmp_path: Path) -> None:
    ctx = _ctx([{"role": "system", "content": "sp"}, {"role": "user", "content": "hi"}])
    chat, output = _chat(ctx, tmp_path)
    await _render_session(chat)
    assert "[tools]" in output.getvalue()


@pytest.mark.asyncio
async def test_tool_schemas_block_absent_when_tools_off(tmp_path: Path) -> None:
    ctx = _ctx(
        [{"role": "system", "content": "sp"}, {"role": "user", "content": "hi"}],
        use_tools=False,
    )
    chat, output = _chat(ctx, tmp_path)
    await _render_session(chat)
    assert "[tools]" not in output.getvalue()


@pytest.mark.asyncio
async def test_long_system_prompt_folds(tmp_path: Path) -> None:
    ctx = _ctx([{"role": "system", "content": _LONG_SYSTEM}])
    chat, output = _chat(ctx, tmp_path)
    await _render_session(chat)
    text = output.getvalue()
    assert "[system]" in text and "folded" in text
    # Full text reachable via /show system (stash captured).
    assert chat._system_full == _LONG_SYSTEM
    await chat._cmd_show(["system"])
    assert "meticulous assistant" in output.getvalue()


@pytest.mark.asyncio
async def test_short_system_prompt_stays_inline(tmp_path: Path) -> None:
    ctx = _ctx([{"role": "system", "content": "Be brief."}])
    chat, output = _chat(ctx, tmp_path)
    await _render_session(chat)
    text = output.getvalue()
    assert "Be brief." in text
    assert "[system]" not in text


def test_subagent_label_on_agent_tool_result() -> None:
    """A folded result for the Agent tool reads 'subagent', not 'tool_result'."""
    agent_rec = ToolRecord(index=1, name="Agent", arguments="{}", result="x" * 500)
    plain_rec = ToolRecord(index=2, name="read_file", arguments="{}", result="y" * 500)
    assert "subagent" in tool_record_summary(agent_rec)
    assert "tool_result" not in tool_record_summary(agent_rec)
    assert "tool_result" in tool_record_summary(plain_rec)


@pytest.mark.asyncio
async def test_show_tools_lists_schemas(tmp_path: Path) -> None:
    ctx = _ctx([{"role": "system", "content": "sp"}])
    chat, output = _chat(ctx, tmp_path)
    await _render_session(chat)
    await chat._cmd_show(["tools"])
    text = output.getvalue()
    assert "read_file" in text  # a builtin schema name appears on expand


# ---- the invariant --------------------------------------------------------


@pytest.mark.asyncio
async def test_no_hidden_channel(tmp_path: Path) -> None:
    """Every populated channel of the round bundle produces ≥1 render call.

    Iterates the bundle's dataclass fields dynamically, so a future input
    channel that isn't given a render hook trips this test.
    """
    messages = [
        {"role": "system", "content": _LONG_SYSTEM},
        {"role": "user", "content": "summarize the readme"},
        {"role": "assistant", "content": "Here is the summary."},
    ]
    ctx = _ctx(messages)
    chat, output = _chat(ctx, tmp_path)
    await _render_session(chat)
    text = output.getvalue()

    bundle = ctx.round_bundle()
    for field in dataclasses.fields(bundle):
        value = getattr(bundle, field.name)
        if field.name == "system":
            assert value  # populated
            assert "[system]" in text  # folded summary rendered
        elif field.name == "tools":
            assert value is not None
            assert "[tools]" in text  # folded schemas block rendered
        elif field.name == "messages":
            # Each non-system message's content surfaces (folded forms count,
            # but these are plain user/assistant turns → verbatim).
            for msg in value:
                if msg.get("role") == "system":
                    continue
                content = msg.get("content")
                if isinstance(content, str) and content:
                    assert content in text, f"hidden message channel: {content!r}"
        else:  # pragma: no cover - guards against a new unrendered field
            raise AssertionError(f"new bundle channel {field.name!r} has no parity check")


@pytest.mark.asyncio
async def test_store_shrink_realigns_cursor_and_keeps_rendering(tmp_path: Path) -> None:
    """A CM-internal compaction shrinks the store; the renderer must not go silent.

    Reproduces the v0.10.5 render-desync: the monotonic cursor exceeds
    len(records) after an auto-compaction (no manual realign), so the next
    assistant turn would never render. The shrink-aware watcher realigns.
    """
    ctx = _ctx([{"role": "system", "content": "sp"}])
    # Simulate a long, already-rendered transcript.
    for i in range(12):
        ctx.store.append_message(MessageRecord(role="user", content=f"u{i}"))
        ctx.store.append_message(MessageRecord(role="assistant", content=f"a{i}"))
    chat, output = _chat(ctx, tmp_path)
    await chat._render_initial_transcript()
    high_water = chat._rendered_message_count
    assert high_water > 10

    # A CM-internal compaction shrinks the store (no cursor realign by anyone).
    ctx.store.reset(system_prompt="sp")
    ctx.store.append_message(
        MessageRecord(role="user", content="<system-summary>did a, b, c</system-summary>")
    )
    ctx.store.append_message(MessageRecord(role="assistant", content="recent reply"))

    await chat._render_new_records()  # shrink-aware: realign + summary notice
    assert chat._rendered_message_count == len(ctx.store.messages)
    assert "[summary]" in output.getvalue()

    # A subsequent assistant turn must still render (the bug made it silent).
    ctx.store.append_message(MessageRecord(role="assistant", content="AFTER-COMPACT"))
    await chat._render_new_records()
    assert "AFTER-COMPACT" in output.getvalue()
    assert chat._summary_full == "did a, b, c"


@pytest.mark.asyncio
async def test_summary_renders_folded_and_expands(tmp_path: Path) -> None:
    """A <system-summary> compaction marker renders folded; /show summary expands."""
    summary = "Earlier: set up the parser and fixed the off-by-one."
    ctx = _ctx([{"role": "system", "content": "sp"}])
    ctx.store.append_message(
        MessageRecord(role="user", content=f"<system-summary>{summary}</system-summary>")
    )
    chat, output = _chat(ctx, tmp_path)
    await chat._render_initial_transcript()
    text = output.getvalue()
    assert "[summary]" in text and "folded" in text
    assert "<system-summary>" not in text  # not dumped raw
    await chat._cmd_show(["summary"])
    assert "off-by-one" in output.getvalue()


@pytest.mark.asyncio
async def test_reminder_renders_as_distinct_notice(tmp_path: Path) -> None:
    """An injected <system-reminder> turn renders as a notice, not a plain user turn."""
    reminder = (
        "<system-reminder>\nThe task tools haven't been used recently. "
        "Here are the existing tasks:\n#1. [pending] x\n</system-reminder>"
    )
    ctx = _ctx([{"role": "system", "content": "sp"}])
    ctx.store.append_message(MessageRecord(role="user", content=reminder))
    chat, output = _chat(ctx, tmp_path)
    await chat._render_initial_transcript()
    text = output.getvalue()
    # The raw reminder XML is NOT dumped as a user turn; the dim notice is shown.
    assert "system reminder" in text.lower()
    assert "<system-reminder>" not in text
