"""Tests for the v0.9.6 mechanical /compact (no-LLM oldest-half drop)."""
from __future__ import annotations

from typing import Any

import pytest

from neutrix.compaction import (
    COMPACT_MARKER_OPEN,
    SUMMARY_MARKER_OPEN,
    CompactionOutcome,
    compact_messages,
    compact_to_token_budget,
    estimate_tokens,
    is_compact_marker,
    is_summary_marker,
    should_compact,
    smart_compact,
    truncate_large_tool_results,
)
from neutrix.config import Slot
from neutrix.context_manager import ContextManager
from neutrix.executor import Executor
from neutrix.llm import (
    INTERRUPTED_BY_USER_MARKER,
    MISSING_TOOL_RESULT,
    LLMEvent,
    LLMResponse,
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
    """Stub that yields an empty summary → compact() falls back to mechanical.

    v0.9.6's "compact must not call the LLM" contract changed in v0.10.5:
    compact() now attempts a summary first; an empty/failed summary falls back
    to the mechanical oldest-drop, which the two CM-compact tests below now
    exercise via this stub.
    """

    def switch(self, slot: Slot) -> None:  # pragma: no cover - trivial
        pass

    def stop(self) -> None:  # pragma: no cover - trivial
        pass

    async def stream_response(self, **_: Any):
        yield LLMEvent("assistant", LLMResponse({"role": "assistant", "content": ""}, "stop"))


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


# ===== v0.10.5: smart compaction + >1M hardening =========================


class _SummaryLLM:
    """Yields a fixed summary string as the assistant content."""

    def __init__(self, summary: str = "SUMMARY: did X, next Y") -> None:
        self.summary = summary
        self.calls = 0

    def switch(self, slot: Slot) -> None:  # pragma: no cover
        pass

    def stop(self) -> None:  # pragma: no cover
        pass

    async def stream_response(self, **_: Any):
        self.calls += 1
        yield LLMEvent("assistant", LLMResponse({"role": "assistant", "content": self.summary}, "stop"))


def _big_messages(n: int, words_per_msg: int = 200) -> list[dict[str, Any]]:
    blob = " ".join(["word"] * words_per_msg)
    msgs: list[dict[str, Any]] = [_sys()]
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"{i} {blob}"})
    return msgs


def test_estimate_tokens_grows_with_payload():
    small = estimate_tokens([_sys(), _user(1)])
    big = estimate_tokens(_big_messages(20))
    assert big > small > 0


def test_should_compact_threshold():
    msgs = _big_messages(40)  # plenty of tokens
    assert should_compact(msgs, max_context_tokens=100) is True
    assert should_compact([_sys()], max_context_tokens=100000) is False


def test_should_compact_disabled_without_max():
    assert should_compact(_big_messages(40), max_context_tokens=None) is False


def test_compact_to_token_budget_gets_under_budget():
    msgs = _big_messages(30)
    budget = estimate_tokens(msgs) // 3
    out, outcome = compact_to_token_budget(msgs, budget=budget)
    assert outcome.did_compact is True
    assert estimate_tokens(out) <= budget
    assert out[0]["role"] == "system"  # prefix preserved
    _validate_pairing(out)


def test_truncate_large_tool_results_caps_body():
    huge = "x" * 50000
    msgs = [_sys(), _user(1), _asst_tools("c1"), _tool("c1")]
    msgs[-1]["content"] = huge
    out, n = truncate_large_tool_results(msgs, cap=8000)
    assert n == 1
    assert len(out[-1]["content"]) < len(huge)
    assert "truncated" in out[-1]["content"]


def test_truncate_leaves_small_results_untouched():
    msgs = [_sys(), _asst_tools("c1"), _tool("c1")]
    msgs[-1]["content"] = "small"
    out, n = truncate_large_tool_results(msgs, cap=8000)
    assert n == 0
    assert out[-1]["content"] == "small"


@pytest.mark.asyncio
async def test_smart_compact_replaces_segment_with_summary():
    msgs = _big_messages(20)
    llm = _SummaryLLM("SUMMARY: earlier work")
    out, outcome = await smart_compact(msgs, llm=llm, model="m", max_context_tokens=2000)
    assert outcome.did_compact is True
    assert llm.calls == 1
    summary_markers = [m for m in out if is_summary_marker(m.get("content"))]
    assert len(summary_markers) == 1
    assert summary_markers[0]["role"] == "user"
    assert "earlier work" in summary_markers[0]["content"]
    assert out[0]["role"] == "system"  # prefix preserved
    _validate_pairing(out)


@pytest.mark.asyncio
async def test_smart_compact_empty_summary_leaves_unchanged():
    msgs = _big_messages(20)

    class _Empty(_SummaryLLM):
        async def stream_response(self, **_: Any):
            yield LLMEvent("assistant", LLMResponse({"role": "assistant", "content": ""}, "stop"))

    out, outcome = await smart_compact(msgs, llm=_Empty(), model="m", max_context_tokens=2000)
    assert outcome.did_compact is False
    assert out == msgs  # unchanged


@pytest.mark.asyncio
async def test_smart_compact_keeps_recent_tail():
    msgs = _big_messages(20)
    out, outcome = await smart_compact(
        msgs, llm=_SummaryLLM(), model="m", max_context_tokens=4000
    )
    assert outcome.did_compact is True
    # The most-recent message survives verbatim after the summary.
    assert out[-1]["content"] == msgs[-1]["content"]


def test_is_summary_marker():
    assert is_summary_marker(f"{SUMMARY_MARKER_OPEN}x</system-summary>")
    assert not is_summary_marker("plain")
    assert not is_summary_marker(None)


@pytest.mark.asyncio
async def test_cm_compact_smart_records_summary_event():
    slot = Slot(
        name="fast", provider="test", model="m",
        base_url="https://example.test/v1", api_key="sk-test",
        max_context_tokens=2000,
    )
    ctx = ContextManager(
        slot=slot, llm=_SummaryLLM("SUMMARY: progress"), executor=Executor(),
        store=ChatStore(), system_prompt="system prompt",
        messages=_big_messages(16),
    )
    outcome = await ctx.compact()
    assert outcome.did_compact is True
    assert any(is_summary_marker(m.get("content")) for m in ctx.messages)
    events = ctx.store.compaction_events
    assert len(events) == 1 and events[0].kind == "summary"
