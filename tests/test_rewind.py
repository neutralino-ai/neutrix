"""Tests for ContextManager.rewind_to (v0.9.7 core).

Covers the PRD's two automated-acceptance cases — suffix drop and
tool-round-boundary safety — plus no-op / clamp / store-resync edges.
The UI layer (Up/Down recall, ``/rewind N``, the forward notice) lives in
``terminal_chat.py`` and is tested separately once the v0.9.7 split-point
forks are settled.
"""

from __future__ import annotations

from typing import Any

import pytest

from neutrix.config import Slot
from neutrix.context_manager import ContextManager
from neutrix.executor import Executor
from neutrix.store import ChatStore
from neutrix.terminal_chat import (
    RecallState,
    recallable_user_turns,
    user_turn_indices,
)


class _StubLLM:
    """rewind_to never calls the LLM (it cancel-and-waits, a no-op when IDLE)."""

    def switch(self, slot: Slot) -> None:  # pragma: no cover - unused
        pass

    def stop(self) -> None:  # pragma: no cover - unused
        pass


def _slot() -> Slot:
    return Slot(
        name="fast",
        provider="test",
        model="test-model",
        base_url="https://example.test/v1",
        api_key="sk-test",
    )


def _make_ctx(seed_messages: list[dict[str, Any]]) -> ContextManager:
    return ContextManager(
        slot=_slot(),
        llm=_StubLLM(),
        executor=Executor(),
        store=ChatStore(),
        system_prompt="sp",
        use_tools=True,
        messages=list(seed_messages),
    )


def _sys() -> dict[str, Any]:
    return {"role": "system", "content": "sp"}


def _user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def _asst(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def _asst_tool(call_id: str, name: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}
        ],
    }


def _tool_result(call_id: str, text: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": text}


# A clean 10-message conversation (no tool rounds) — index 5 is a turn boundary.
_CLEAN_10 = [
    _sys(),
    _user("u1"), _asst("a1"),
    _user("u2"), _asst("a2"),
    _user("u3"), _asst("a3"),
    _user("u4"), _asst("a4"),
    _user("u5"),
]

# A conversation containing one tool round (indices 2-4).
_WITH_ROUND = [
    _sys(),            # 0
    _user("u1"),       # 1
    _asst_tool("c1", "run"),  # 2  assistant tool_calls
    _tool_result("c1", "out"),  # 3  tool result
    _asst("answer"),   # 4  assistant final text
    _user("u2"),       # 5
    _asst("a2"),       # 6
]


@pytest.mark.asyncio
async def test_rewind_to_drops_suffix() -> None:
    """cm.rewind_to(5) against a clean 10-message fixture leaves 5."""
    ctx = _make_ctx(_CLEAN_10)
    dropped = await ctx.rewind_to(5)
    assert dropped == 5
    assert len(ctx.messages) == 5
    assert [m["content"] for m in ctx.messages] == ["sp", "u1", "a1", "u2", "a2"]


@pytest.mark.asyncio
async def test_rewind_respects_tool_round_boundary() -> None:
    """A rewind that would split a tool round drops the WHOLE round."""
    ctx = _make_ctx(_WITH_ROUND)
    # index 4 would keep [sys, u1, asst_tool, tool] — a head ending mid-round.
    dropped = await ctx.rewind_to(4)
    # Snaps back past the tool result and the orphaned tool_calls to index 2.
    assert len(ctx.messages) == 2
    assert dropped == 5
    assert [m["role"] for m in ctx.messages] == ["system", "user"]
    # No dangling tool_use / tool_result remains in the kept head.
    assert not any(m["role"] == "tool" for m in ctx.messages)
    assert not any(m.get("tool_calls") for m in ctx.messages)


@pytest.mark.asyncio
async def test_rewind_at_turn_boundary_keeps_whole_round() -> None:
    """Cutting at a user-turn boundary AFTER a round keeps the round intact."""
    ctx = _make_ctx(_WITH_ROUND)
    dropped = await ctx.rewind_to(5)  # messages[4] = assistant text → clean
    assert dropped == 2
    assert len(ctx.messages) == 5
    assert any(m["role"] == "tool" for m in ctx.messages)  # round preserved


@pytest.mark.asyncio
async def test_rewind_noop_at_or_beyond_len() -> None:
    ctx = _make_ctx(_CLEAN_10)
    assert await ctx.rewind_to(10) == 0
    assert await ctx.rewind_to(99) == 0
    assert len(ctx.messages) == 10


@pytest.mark.asyncio
async def test_rewind_clamps_into_system_prefix() -> None:
    """An index inside the system prefix clamps to keep the prefix."""
    ctx = _make_ctx(_CLEAN_10)
    dropped = await ctx.rewind_to(0)
    assert len(ctx.messages) == 1
    assert ctx.messages[0]["role"] == "system"
    assert dropped == 9


@pytest.mark.asyncio
async def test_rewind_rebuilds_store_and_preserves_tasks() -> None:
    """Store is re-synced to the trimmed messages; tasks survive the rewind."""
    ctx = _make_ctx(_CLEAN_10)
    ctx.store.add_task("keep me")
    await ctx.rewind_to(5)
    assert len(ctx.store.messages) == 5
    assert [r.content for r in ctx.store.messages] == ["sp", "u1", "a1", "u2", "a2"]
    assert any(t.subject == "keep me" for t in ctx.store.tasks)


# ---- recall logic (Up/Down) + turn-index helpers (v0.9.7 UI) --------------


def test_up_arrow_walks_user_turns() -> None:
    """Up walks from the most-recent prior turn backward, clamping at oldest."""
    turns = ["a", "b", "c"]  # oldest first
    rs = RecallState()
    assert rs.up(turns) == "c"
    assert rs.up(turns) == "b"
    assert rs.up(turns) == "a"
    assert rs.up(turns) == "a"  # clamped
    assert rs.active


def test_recall_down_returns_to_fresh() -> None:
    turns = ["a", "b", "c"]
    rs = RecallState()
    rs.up(turns)
    rs.up(turns)  # cursor=2 → "b"
    assert rs.down(turns) == "c"  # forward toward most-recent
    assert rs.down(turns) == ""  # back to the fresh (empty) draft
    assert not rs.active


def test_recall_then_esc_clears() -> None:
    """Esc -> reset(): cursor returns to 0 (fresh), recall inactive."""
    rs = RecallState()
    rs.up(["x", "y"])
    assert rs.active
    rs.reset()
    assert not rs.active
    assert rs.cursor == 0


def test_recall_empty_history_is_noop() -> None:
    rs = RecallState()
    assert rs.up([]) is None
    assert rs.down([]) is None
    assert not rs.active


def test_user_turn_indices_and_recall_source_skip_markers() -> None:
    """Injected role:user markers (compact placeholder) aren't recallable turns."""
    compact = "<system-compact>3 earlier turns removed by /compact (no summary)</system-compact>"
    msgs = [
        {"role": "system", "content": "sp"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": compact},
        {"role": "user", "content": "second"},
    ]
    assert user_turn_indices(msgs) == [1, 4]
    assert recallable_user_turns(msgs) == ["hello", "second"]
