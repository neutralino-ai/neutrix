"""Tests for the v0.9.6 mechanical /compact (no-LLM oldest-half drop)."""
from __future__ import annotations

from typing import Any

import pytest

from neutrix.compaction import (
    COMPACT_MARKER_OPEN,
    CompactionOutcome,
    compact_messages,
    is_compact_marker,
)
from neutrix.config import Slot
from neutrix.context_manager import ContextManager
from neutrix.executor import Executor
from neutrix.llm import (
    INTERRUPTED_BY_USER_MARKER,
    MISSING_TOOL_RESULT,
    _ensure_tool_result_pairing,
)
from neutrix.store import ChatStore

# ---- fixtures --------------------------------------------------------------


def _sys() -> dict[str, Any]:
    return {"role": "system", "content": "system prompt"}


def _user(i: int) -> dict[str, Any]:
    return {"role": "user", "content": f"user message number {i}"}


def _asst(i: int) -> dict[str, Any]:
    return {"role": "assistant", "content": f"assistant reply number {i}"}


def _asst_tools(*call_ids: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": "echo", "arguments": "{}"},
            }
            for cid in call_ids
        ],
    }


def _tool(cid: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": cid, "content": f"tool result for {cid}"}


def _body_pairs(n: int) -> list[dict[str, Any]]:
    """``n`` user/assistant pairs (2n body messages) after the system."""
    msgs: list[dict[str, Any]] = [_sys()]
    for i in range(n):
        msgs.append(_user(i))
        msgs.append(_asst(i))
    return msgs


def _markers(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in messages if is_compact_marker(m.get("content"))]


def _validate_pairing(messages: list[dict[str, Any]]) -> None:
    """Assert the OpenAI tool_use/tool_result invariant holds."""
    seen_call_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                seen_call_ids.add(str(tc.get("id")))
        elif msg.get("role") == "tool":
            tcid = str(msg.get("tool_call_id"))
            assert tcid in seen_call_ids, f"orphan tool_result {tcid!r} (no preceding tool_call)"
    # Every tool_call must have a matching tool_result somewhere.
    result_ids = {
        str(m.get("tool_call_id")) for m in messages if m.get("role") == "tool"
    }
    assert seen_call_ids <= result_ids, f"orphan tool_use: {seen_call_ids - result_ids}"


# ---- pure function: oldest-half drop ---------------------------------------


def test_drops_oldest_half():
    msgs = _body_pairs(10)  # system + 20 body messages
    new, outcome = compact_messages(msgs)

    assert outcome.did_compact is True
    assert outcome.turns_dropped == 10
    # System prefix preserved verbatim at [0].
    assert new[0] == _sys()
    # Exactly one placeholder, immediately after the system prefix.
    assert is_compact_marker(new[1].get("content"))
    assert len(_markers(new)) == 1
    # Newest 10 body messages preserved verbatim as the tail
    # (body index 10 == msgs index 11).
    assert new[2:] == msgs[11:]
    assert len(new[2:]) == 10
    # Input not mutated.
    assert len(msgs) == 21


def test_marker_text_states_no_summary():
    new, _ = compact_messages(_body_pairs(10))
    marker = new[1]["content"]
    assert marker.startswith(COMPACT_MARKER_OPEN)
    assert "10 earlier turns removed by /compact (no summary)" in marker


# ---- pure function: tool-round boundary ------------------------------------


def test_respects_tool_round_boundary():
    # L=8 body; naive drop=4 lands on the SECOND tool result of a
    # multi-call round, which would orphan it. Forward-snap must drop
    # the whole round so the first kept message is not a tool result.
    body = [
        _user(0),            # 0
        _asst(0),            # 1
        _asst_tools("c1", "c2"),  # 2  (round start)
        _tool("c1"),         # 3
        _tool("c2"),         # 4  <- naive cut index
        _asst(1),            # 5
        _user(1),            # 6
        _asst(2),            # 7
    ]
    msgs = [_sys(), *body]
    new, outcome = compact_messages(msgs)

    # First kept message (after system + marker) is NOT an orphan tool_result.
    assert new[0]["role"] == "system"
    assert is_compact_marker(new[1].get("content"))
    assert new[2]["role"] != "tool"
    # Snapped forward from 4 to 5 → 5 body messages dropped.
    assert outcome.turns_dropped == 5
    _validate_pairing(new)


def test_kept_tail_tooluse_keeps_its_results():
    # Kept tail STARTS with an assistant-with-tool_calls; its result must
    # be inside the kept region.
    body = [
        _user(0),            # 0
        _asst(0),            # 1
        _user(1),            # 2
        _asst(1),            # 3
        _asst_tools("c5"),   # 4  <- naive cut (drop=4) lands here (not a tool)
        _tool("c5"),         # 5
        _asst(2),            # 6
        _user(2),            # 7
    ]
    msgs = [_sys(), *body]
    new, _ = compact_messages(msgs)

    assert new[2] == _asst_tools("c5")  # first kept is the tool_call round
    assert _tool("c5") in new            # its result survived in the kept tail
    _validate_pairing(new)


def test_pairing_synthesizes_after_compaction():
    # Kept tail's LAST message is an orphan assistant-with-tool_calls and
    # no [interrupted] follows → the llm.py pairing layer must synthesize
    # a MISSING_TOOL_RESULT at API-send time.
    body = [
        _user(0),
        _asst(0),
        _user(1),
        _asst(1),
        _user(2),
        _asst(2),
        _user(3),
        _asst_tools("c9"),   # orphan tool_use as the final kept message
    ]
    msgs = [_sys(), *body]
    new, _ = compact_messages(msgs)
    assert new[-1] == _asst_tools("c9")

    paired = _ensure_tool_result_pairing(new)
    synth = [
        m
        for m in paired
        if m.get("role") == "tool"
        and m.get("tool_call_id") == "c9"
        and m.get("content") == MISSING_TOOL_RESULT
    ]
    assert len(synth) == 1
    _validate_pairing(paired)
    # No cancel marker was present, so it is "missing", not "cancelled".
    assert all(
        m.get("content") != INTERRUPTED_BY_USER_MARKER for m in paired
    )


# ---- pure function: idempotency / no-op floor ------------------------------


def test_idempotent_no_marker_stacking():
    msgs = _body_pairs(10)
    new1, o1 = compact_messages(msgs)
    new2, o2 = compact_messages(new1)

    assert o1.did_compact is True
    assert o2.did_compact is True
    # The prior placeholder is the oldest body message, so it is dropped
    # and replaced — never stacked.
    assert len(_markers(new2)) == 1
    assert len(new2) < len(new1)


def test_no_op_when_too_short():
    msgs = [_sys(), _user(0)]  # body of 1 → floor(1*0.5)=0
    new, outcome = compact_messages(msgs)
    assert outcome == CompactionOutcome(False, 0, 0)
    assert new == msgs
    assert _markers(new) == []


def test_no_op_system_only():
    msgs = [_sys()]
    new, outcome = compact_messages(msgs)
    assert outcome.did_compact is False
    assert new == msgs


def test_token_estimate_monotonic():
    msgs = _body_pairs(10)
    _, few = compact_messages(msgs, keep_ratio=0.75)  # drops 5
    _, many = compact_messages(msgs, keep_ratio=0.25)  # drops 15
    assert many.turns_dropped > few.turns_dropped
    assert many.approx_tokens_dropped >= few.approx_tokens_dropped > 0


# ---- ContextManager.compact(): store + tasks -------------------------------


class _NullLLM:
    def switch(self, slot: Slot) -> None:  # pragma: no cover - trivial
        pass

    def stop(self) -> None:  # pragma: no cover - trivial
        pass

    async def stream_response(self, **_: Any):  # pragma: no cover - never called
        raise AssertionError("compact must not call the LLM")
        yield None


def _ctx_with_body(n_pairs: int) -> ContextManager:
    slot = Slot(
        name="fast",
        provider="test",
        model="m",
        base_url="https://example.test/v1",
        api_key="sk-test",
    )
    return ContextManager(
        slot=slot,
        llm=_NullLLM(),
        executor=Executor(),
        store=ChatStore(),
        system_prompt="system prompt",
        messages=_body_pairs(n_pairs),
    )


@pytest.mark.asyncio
async def test_cm_compact_preserves_tasks_and_compacts_store():
    ctx = _ctx_with_body(10)  # system + 20 body messages
    ctx.store.add_task("first task")
    ctx.store.add_task("second task")
    assert len(ctx.store.tasks) == 2

    before_msgs = len(ctx.messages)
    outcome = await ctx.compact()

    assert outcome.did_compact is True
    # Tasks survive the context trim.
    subjects = [t.subject for t in ctx.store.tasks]
    assert subjects == ["first task", "second task"]
    # Both the payload and the store shrank and stay in lock-step.
    assert len(ctx.messages) < before_msgs
    assert len(ctx.store.messages) == len(ctx.messages)
    assert ctx.store.messages[0].role == "system"
    assert is_compact_marker(ctx.store.messages[1].content)
    # The id counter resumes past the preserved tasks — no collision.
    new_task = ctx.store.add_task("third task")
    assert new_task.id not in {"1", "2"}


@pytest.mark.asyncio
async def test_cm_compact_no_op_when_short_returns_unchanged():
    ctx = _ctx_with_body(0)  # system only
    outcome = await ctx.compact()
    assert outcome.did_compact is False
    assert [m["role"] for m in ctx.messages] == ["system"]
    assert _markers(ctx.messages) == []
