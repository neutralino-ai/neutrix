"""Tests for v0.9.5 LLM-timeout watchdog behavior.

The watchdog spawns on every ``AWAITING_LLM`` entry, sleeps
``slot.llm_timeout_s``, then calls ``cm.cancel(reason='timeout')``.
The drive loop's CancelledError handler appends
``[LLM timeout after Ns]`` as an assistant message and returns CM to
IDLE. User-Esc during AWAITING_LLM still produces the v0.9.3
``[interrupted by user]`` marker (not the timeout marker), and the
watchdog gets cancelled cleanly so it never fires late.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from neutrix.config import Slot
from neutrix.context_manager import (
    ContextManager,
    State,
    UserMessageEvent,
)
from neutrix.executor import Executor
from neutrix.llm import INTERRUPTED_BY_USER_MARKER, LLMEvent, LLMResponse
from neutrix.store import ChatStore


def _slot(timeout_s: float = 60.0) -> Slot:
    return Slot(
        name="fast",
        provider="test",
        model="test-model",
        base_url="https://example.test/v1",
        api_key="sk-test",
        llm_timeout_s=timeout_s,
    )


class SuspendingLLM:
    """Blocks forever on each round until ``stop()`` cancels it."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.stop_calls = 0
        self._task: asyncio.Task[Any] | None = None

    def switch(self, slot: Slot) -> None:
        pass

    def stop(self) -> None:
        self.stop_calls += 1
        task = self._task
        if task is not None and not task.done():
            task.cancel()

    async def stream_response(self, **_: Any):
        self.entered.set()
        self._task = asyncio.ensure_future(asyncio.Event().wait())
        try:
            await self._task
        finally:
            self._task = None
        yield LLMEvent(  # pragma: no cover - never reached after cancel
            "assistant", LLMResponse({"role": "assistant", "content": None}, "stop")
        )


def _make_ctx(llm: Any, *, timeout_s: float = 60.0) -> ContextManager:
    return ContextManager(
        slot=_slot(timeout_s=timeout_s),
        llm=llm,
        executor=Executor(),
        store=ChatStore(),
        system_prompt="sp",
        use_tools=False,
        messages=[{"role": "system", "content": "sp"}],
    )


# ---- watchdog fires on timeout -------------------------------------------


@pytest.mark.asyncio
async def test_watchdog_fires_appends_timeout_marker_and_idles():
    """A 0.1 s timeout against a suspending LLM fires the watchdog and
    appends ``[LLM timeout after 0s]`` as an assistant message."""
    llm = SuspendingLLM()
    ctx = _make_ctx(llm, timeout_s=0.1)

    task = asyncio.create_task(ctx.handle_event(UserMessageEvent("hi")))
    await llm.entered.wait()
    # Wait past the timeout — watchdog should fire and unwind the drive.
    await asyncio.wait_for(task, timeout=2.0)

    assert ctx.state == State.IDLE
    last = ctx.messages[-1]
    assert last["role"] == "assistant"
    assert last["content"] == "[LLM timeout after 0s]"
    # No user-marker — this was a timeout, not a user cancel.
    assert not any(
        m.get("role") == "user" and m.get("content") == INTERRUPTED_BY_USER_MARKER
        for m in ctx.messages
    )
    # cancel_reason reset for the next turn.
    assert ctx.cancel_reason == "user"
    # LLM was actually asked to stop.
    assert llm.stop_calls >= 1


@pytest.mark.asyncio
async def test_watchdog_fires_with_integer_elapsed_in_marker():
    """``llm_timeout_s = 2.7`` → marker reads ``[LLM timeout after 2s]``."""
    llm = SuspendingLLM()
    ctx = _make_ctx(llm, timeout_s=2.7)

    task = asyncio.create_task(ctx.handle_event(UserMessageEvent("hi")))
    await llm.entered.wait()
    await asyncio.wait_for(task, timeout=5.0)

    assert ctx.messages[-1]["content"] == "[LLM timeout after 2s]"


# ---- user-Esc does not trigger timeout marker ----------------------------


@pytest.mark.asyncio
async def test_user_cancel_does_not_trigger_timeout_marker_late():
    """A user-Esc during AWAITING_LLM produces the v0.9.3
    ``[interrupted by user]`` marker; the watchdog gets cancelled
    cleanly so no timeout marker ever appears, even after sleeping
    past ``llm_timeout_s``.
    """
    llm = SuspendingLLM()
    ctx = _make_ctx(llm, timeout_s=0.1)

    task = asyncio.create_task(ctx.handle_event(UserMessageEvent("hi")))
    await llm.entered.wait()
    # User cancels before the watchdog has a chance.
    fired = ctx.cancel(reason="user")
    assert fired is True
    await asyncio.wait_for(task, timeout=1.0)

    # Sleep well beyond llm_timeout_s — if the watchdog late-fires
    # this is where the bug would surface as a second appended marker.
    await asyncio.sleep(0.3)

    assert ctx.state == State.IDLE
    # User marker is present.
    assert any(
        m.get("role") == "user" and m.get("content") == INTERRUPTED_BY_USER_MARKER
        for m in ctx.messages
    )
    # No timeout marker — even after waiting past the threshold.
    assert not any(
        m.get("role") == "assistant"
        and isinstance(m.get("content"), str)
        and m["content"].startswith("[LLM timeout after ")
        for m in ctx.messages
    )
    assert ctx.cancel_reason == "user"


# ---- watchdog cancelled on normal LLM success ----------------------------


class FakeLLM:
    """Yields one round immediately — used to confirm the watchdog gets
    cancelled when the LLM succeeds promptly."""

    def __init__(self) -> None:
        self.calls = 0

    def switch(self, slot: Slot) -> None:
        pass

    def stop(self) -> None:
        pass

    async def stream_response(self, **_: Any):
        self.calls += 1
        yield LLMEvent(
            "assistant",
            LLMResponse({"role": "assistant", "content": "hi"}, "stop"),
        )


@pytest.mark.asyncio
async def test_watchdog_does_not_fire_on_normal_success():
    """A fast LLM response cancels the watchdog before it ever fires."""
    llm = FakeLLM()
    # Even with an aggressively-short timeout, the immediate LLM
    # response must beat the watchdog.
    ctx = _make_ctx(llm, timeout_s=0.05)

    await ctx.handle_event(UserMessageEvent("hi"))
    # Give a wall-clock tick beyond the timeout window — a late-firing
    # watchdog would clobber state here.
    await asyncio.sleep(0.15)

    assert ctx.state == State.IDLE
    assert ctx.messages[-1]["content"] == "hi"
    # No spurious timeout marker.
    assert not any(
        m.get("role") == "assistant"
        and isinstance(m.get("content"), str)
        and m["content"].startswith("[LLM timeout after ")
        for m in ctx.messages
    )
