"""Tests for the ContextManager state machine."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from neutrix.config import Slot
from neutrix.context_manager import (
    ClearEvent,
    ContextManager,
    ReplaceHistoryEvent,
    SlotSwitchEvent,
    State,
    UserMessageEvent,
    build_task_reminder,
    is_task_reminder,
)
from neutrix.executor import Executor
from neutrix.llm import (
    INTERRUPTED_BY_USER_MARKER,
    LLMEvent,
    LLMResponse,
)
from neutrix.store import ChatStore, MessageRecord

# ---- fixtures --------------------------------------------------------------


def _slot(name: str = "fast", model: str = "test-model") -> Slot:
    return Slot(
        name=name,
        provider="test",
        model=model,
        base_url="https://example.test/v1",
        api_key="sk-test",
    )


class FakeLLM:
    """Yields pre-canned rounds. One ``rounds`` entry per LLM round."""

    def __init__(self, rounds: list[list[LLMEvent]]) -> None:
        self.rounds = list(rounds)
        self.calls: list[dict[str, Any]] = []
        self.switched_to: Slot | None = None

    def switch(self, slot: Slot) -> None:
        self.switched_to = slot

    def stop(self) -> None:
        pass

    async def stream_response(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ):
        self.calls.append(
            {
                "model": model,
                "messages": [dict(m) for m in messages],
                "tools": tools,
            }
        )
        for event in self.rounds.pop(0):
            yield event


class SuspendingLLM:
    """Blocks forever on each round until ``release()`` is called.

    Used to test cancel-mid-LLM. Honors ``stop()`` by cancelling the
    parked task.
    """

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

    async def stream_response(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ):
        self.entered.set()
        self._task = asyncio.ensure_future(asyncio.Event().wait())
        try:
            await self._task
        except asyncio.CancelledError:
            raise
        finally:
            self._task = None
        # Unreachable, but satisfies the async generator contract.
        yield LLMEvent("assistant", LLMResponse({"role": "assistant", "content": None}, "stop"))  # pragma: no cover


class RaisingLLM:
    """Raises immediately on stream_response — exercises the error path."""

    def switch(self, slot: Slot) -> None:
        pass

    def stop(self) -> None:
        pass

    async def stream_response(self, **_: Any):
        raise RuntimeError("upstream boom")
        yield None  # pragma: no cover - satisfies async-generator typing


def _make_ctx(
    llm: Any,
    *,
    use_tools: bool = True,
    seed_messages: list[dict[str, Any]] | None = None,
) -> ContextManager:
    store = ChatStore()
    executor = Executor()
    if seed_messages is None:
        seed_messages = [{"role": "system", "content": "sp"}]
    ctx = ContextManager(
        slot=_slot(),
        llm=llm,
        executor=executor,
        store=store,
        system_prompt="sp",
        use_tools=use_tools,
        messages=list(seed_messages),
    )
    return ctx


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


# ---- happy path -----------------------------------------------------------


@pytest.mark.asyncio
async def test_user_message_drives_idle_to_awaiting_llm_to_idle():
    llm = FakeLLM([[_assistant_text("hello")]])
    ctx = _make_ctx(llm, use_tools=False)
    await ctx.handle_event(UserMessageEvent("hi"))
    assert ctx.state == State.IDLE
    roles = [m["role"] for m in ctx.messages]
    assert roles == ["system", "user", "assistant"]
    assert ctx.messages[-1]["content"] == "hello"
    # Store mirrors the same shape.
    store_roles = [m.role for m in ctx.store.messages]
    assert store_roles == ["system", "user", "assistant"]


@pytest.mark.asyncio
async def test_user_message_drives_tool_round_then_final(monkeypatch):
    """assistant→tool_call → executor → assistant follow-up → IDLE."""
    monkeypatch.setattr(
        "neutrix.executor.dispatch",
        lambda name, arguments, **_: f"ran {name}",
    )
    llm = FakeLLM(
        [
            [_assistant_tool("echo", "{}")],
            [_assistant_text("done")],
        ]
    )
    ctx = _make_ctx(llm, use_tools=True)
    await ctx.handle_event(UserMessageEvent("run echo"))
    assert ctx.state == State.IDLE
    roles = [m["role"] for m in ctx.messages]
    # system, user, assistant(tool_call), tool, assistant("done")
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert ctx.messages[-2]["tool_call_id"] == "c1"
    assert ctx.messages[-2]["content"] == "ran echo"
    assert ctx.messages[-1]["content"] == "done"


# ---- cancel from AWAITING_LLM ---------------------------------------------


@pytest.mark.asyncio
async def test_cancel_from_awaiting_llm_appends_marker_and_returns_to_idle():
    llm = SuspendingLLM()
    ctx = _make_ctx(llm, use_tools=False)

    task = asyncio.create_task(ctx.handle_event(UserMessageEvent("hi")))
    await llm.entered.wait()
    # Confirm state is AWAITING_LLM.
    assert ctx.state == State.AWAITING_LLM

    fired = ctx.cancel()
    assert fired is True
    await asyncio.wait_for(task, timeout=0.5)

    assert ctx.state == State.IDLE
    assert llm.stop_calls == 1
    # Marker is in messages and store.
    assert ctx.messages[-1] == {
        "role": "user",
        "content": INTERRUPTED_BY_USER_MARKER,
    }
    last_record = ctx.store.messages[-1]
    assert last_record.role == "user"
    assert last_record.content == INTERRUPTED_BY_USER_MARKER


@pytest.mark.asyncio
async def test_cancel_idempotent_during_cancelling():
    """A second CancelEvent while already CANCELLING is a no-op."""
    llm = SuspendingLLM()
    ctx = _make_ctx(llm, use_tools=False)
    task = asyncio.create_task(ctx.handle_event(UserMessageEvent("hi")))
    await llm.entered.wait()
    first = ctx.cancel()
    second = ctx.cancel()
    await asyncio.wait_for(task, timeout=0.5)
    assert first is True
    assert second is False
    markers = [
        m
        for m in ctx.messages
        if m.get("role") == "user" and m.get("content") == INTERRUPTED_BY_USER_MARKER
    ]
    assert len(markers) == 1


@pytest.mark.asyncio
async def test_cancel_when_idle_is_noop():
    llm = FakeLLM([])
    ctx = _make_ctx(llm, use_tools=False)
    assert ctx.cancel() is False
    assert ctx.state == State.IDLE
    # No marker appended.
    assert not any(
        m.get("role") == "user" and m.get("content") == INTERRUPTED_BY_USER_MARKER
        for m in ctx.messages
    )


# ---- cancel from AWAITING_EXECUTOR ---------------------------------------


@pytest.mark.asyncio
async def test_cancel_from_awaiting_executor_appends_marker_and_idles(monkeypatch):
    """Cancel while a tool is mid-dispatch. The marker lands; the
    orphan tool_call assistant turn stays unrepaired in messages (the
    pairing layer handles it at next API send)."""
    tool_entered = asyncio.Event()
    tool_release = asyncio.Event()

    def slow_dispatch(name, arguments, **_):
        tool_entered.set()
        # Block the worker thread until the test releases it.
        asyncio.run(asyncio.sleep(0))  # noop in thread context
        # We need a thread-safe wait — use the executor pool isn't
        # applicable for pure-compute. Instead, busy-wait briefly.
        import time

        for _ in range(200):  # ~2s budget
            if tool_release.is_set():
                break
            time.sleep(0.01)
        return f"ran {name}"

    monkeypatch.setattr("neutrix.executor.dispatch", slow_dispatch)
    llm = FakeLLM(
        [
            [_assistant_tool("slow", "{}")],
            # Second round won't execute because we cancel mid-tool.
            [_assistant_text("never")],
        ]
    )
    ctx = _make_ctx(llm, use_tools=True)

    task = asyncio.create_task(ctx.handle_event(UserMessageEvent("hi")))
    # Wait until the tool is mid-dispatch.
    while not tool_entered.is_set():
        await asyncio.sleep(0.01)
    assert ctx.state == State.AWAITING_EXECUTOR

    fired = ctx.cancel()
    assert fired is True
    # Release the tool so the background thread completes (its result is
    # discarded — the marker was already appended).
    tool_release.set()
    await asyncio.wait_for(task, timeout=2.0)

    assert ctx.state == State.IDLE
    # Marker is appended at the cancel point. The orphan assistant
    # tool_calls message is preserved (the LLM-pairing layer will
    # synthesize the tool_result at next API send).
    roles = [m["role"] for m in ctx.messages]
    assert "assistant" in roles  # the tool_call turn survived
    assert ctx.messages[-1]["content"] == INTERRUPTED_BY_USER_MARKER


# ---- LLM error path -------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_error_appends_synthetic_assistant_and_returns_to_idle():
    llm = RaisingLLM()
    ctx = _make_ctx(llm, use_tools=False)
    await ctx.handle_event(UserMessageEvent("hi"))
    assert ctx.state == State.IDLE
    last = ctx.messages[-1]
    assert last["role"] == "assistant"
    assert last["content"].startswith("[LLM error: ")
    assert "upstream boom" in last["content"]


# ---- system reminder injection -------------------------------------------


def _user(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content}


def _assistant(content: str = "ok") -> dict[str, Any]:
    return {"role": "assistant", "content": content}


@pytest.mark.asyncio
async def test_system_reminder_injected_when_due():
    """≥10 assistant turns + actionable task → reminder appended to
    messages before the LLM call."""
    seed = [{"role": "system", "content": "sp"}, _user("hi")] + [_assistant("a")] * 10
    llm = FakeLLM([[_assistant_text("ok")]])
    ctx = _make_ctx(llm, use_tools=True, seed_messages=seed)
    ctx.store.add_task("first")

    await ctx.handle_event(UserMessageEvent("now what"))

    # Find reminder in messages.
    reminders = [
        m
        for m in ctx.messages
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and is_task_reminder(m["content"])
    ]
    assert len(reminders) == 1


@pytest.mark.asyncio
async def test_system_reminder_not_injected_when_no_tasks():
    seed = [{"role": "system", "content": "sp"}, _user("hi")] + [_assistant("a")] * 20
    llm = FakeLLM([[_assistant_text("ok")]])
    ctx = _make_ctx(llm, use_tools=True, seed_messages=seed)

    await ctx.handle_event(UserMessageEvent("hi"))
    reminders = [
        m
        for m in ctx.messages
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and is_task_reminder(m["content"])
    ]
    assert reminders == []


# ---- clear / replace history ---------------------------------------------


@pytest.mark.asyncio
async def test_clear_when_idle_resets_messages_to_system_only():
    llm = FakeLLM([[_assistant_text("hello")]])
    ctx = _make_ctx(llm, use_tools=False)
    await ctx.handle_event(UserMessageEvent("hi"))

    await ctx.handle_event(ClearEvent())
    assert [m["role"] for m in ctx.messages] == ["system"]
    assert ctx.messages[0]["content"] == "sp"


@pytest.mark.asyncio
async def test_clear_during_awaiting_llm_cancels_first_then_clears():
    llm = SuspendingLLM()
    ctx = _make_ctx(llm, use_tools=False)

    task = asyncio.create_task(ctx.handle_event(UserMessageEvent("hi")))
    await llm.entered.wait()

    # /clear should cancel-and-clear.
    await ctx.handle_event(ClearEvent())
    await asyncio.wait_for(task, timeout=0.5)
    assert ctx.state == State.IDLE
    assert [m["role"] for m in ctx.messages] == ["system"]


@pytest.mark.asyncio
async def test_replace_history_seeds_messages_and_tasks():
    llm = FakeLLM([])
    ctx = _make_ctx(llm, use_tools=False)
    new_messages = [
        {"role": "system", "content": "sp"},
        {"role": "user", "content": "old turn"},
        {"role": "assistant", "content": "old reply"},
    ]
    new_records = (
        MessageRecord(role="system", content="sp"),
        MessageRecord(role="user", content="old turn"),
        MessageRecord(role="assistant", content="old reply"),
    )
    await ctx.handle_event(
        ReplaceHistoryEvent(
            raw_messages=new_messages,
            records=new_records,
            tasks=(),
        )
    )
    assert ctx.messages == new_messages
    assert [m.role for m in ctx.store.messages] == ["system", "user", "assistant"]


# ---- slot switch ---------------------------------------------------------


@pytest.mark.asyncio
async def test_slot_switch_event_updates_slot_and_calls_llm_switch():
    llm = FakeLLM([])
    ctx = _make_ctx(llm, use_tools=False)
    new_slot = _slot(name="strong", model="strong-model")
    await ctx.handle_event(SlotSwitchEvent(slot=new_slot))
    assert ctx.slot is new_slot
    assert llm.switched_to is new_slot


# ---- build_task_reminder regression (helpers moved from agent_loop) ------


def test_build_task_reminder_emitted_when_both_thresholds_met():
    store = ChatStore()
    store.add_task("first")
    store.update_task("1", status="in_progress")
    store.add_task("second")
    messages = [{"role": "user", "content": "hi"}] + [
        {"role": "assistant", "content": "a"} for _ in range(10)
    ]
    reminder = build_task_reminder(messages, store.tasks)
    assert reminder is not None
    assert reminder["role"] == "user"
    assert "#1. [in_progress] first" in reminder["content"]
    assert "#2. [pending] second" in reminder["content"]
