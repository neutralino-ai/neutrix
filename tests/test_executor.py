"""Tests for the per-turn Executor + tree-kill helper."""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time

import pytest

from neutrix.agent_loop import Agent, AgentEvent
from neutrix.config import Slot
from neutrix.executor import Executor
from neutrix.llm import LLMEvent, LLMResponse
from neutrix.tools import _run_shell


def _slot() -> Slot:
    return Slot(
        name="fast",
        provider="test",
        model="test-model",
        base_url="https://example.test/v1",
        api_key="sk-test",
    )


class FakeLLM:
    """Tiny LLM stub: one round per item in ``rounds``."""

    def __init__(self, rounds: list[list[LLMEvent]]) -> None:
        self.rounds = list(rounds)

    def switch(self, slot: Slot) -> None:
        pass

    def stop(self) -> None:
        pass

    async def stream_response(self, *, model, messages, tools=None):
        for event in self.rounds.pop(0):
            yield event


class SuspendingLLM:
    """Blocks indefinitely on the first request so the test can cancel
    mid-stream — the v0.9.2 happy path."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()

    def switch(self, slot: Slot) -> None:
        pass

    def stop(self) -> None:
        self.stopped.set()

    async def stream_response(self, *, model, messages, tools=None):
        self.started.set()
        await self.stopped.wait()
        # Emit nothing — the wrapping ``async for`` exits because the
        # iterator ran out of yields.
        if False:  # pragma: no cover - never iterated
            yield None


@pytest.mark.asyncio
async def test_stream_turn_sets_snapshot_during_turn():
    """While a turn is iterating, ``_turn_snapshot_len`` carries the
    pre-turn message count. After normal unwind it's None again. The
    invariant matters because ``cancel()`` uses ``snapshot_len is
    None`` as the in-flight predicate."""
    llm = FakeLLM(
        [
            [
                LLMEvent(
                    "assistant",
                    LLMResponse(
                        {"role": "assistant", "content": "ok"},
                        finish_reason="stop",
                    ),
                )
            ]
        ]
    )
    agent = Agent(slot=_slot(), llm=llm, use_tools=False)
    executor = Executor(agent=agent)

    snapshot_during: list[int | None] = []
    async for _ in executor.stream_turn("hi"):
        snapshot_during.append(executor._turn_snapshot_len)
        break  # one yield is enough

    assert snapshot_during == [1]  # only the system message at start
    # Drain the rest so the generator unwinds.
    async for _ in executor.stream_turn("hi"):
        pass
    assert executor._turn_snapshot_len is None


@pytest.mark.asyncio
async def test_cancel_rolls_messages_back_to_snapshot():
    """Cancel mid-turn drops the in-flight user_turn (and the orphan
    assistant turn if any) so the conversation stays valid. Pool is
    empty after cancel."""
    llm = SuspendingLLM()
    agent = Agent(slot=_slot(), llm=llm, use_tools=False)
    executor = Executor(agent=agent)

    pre_messages = list(agent.messages)
    agen = executor.stream_turn("hi")

    consumer_started = asyncio.Event()

    async def consume() -> list[AgentEvent]:
        events: list[AgentEvent] = []
        consumer_started.set()
        async for event in agen:
            events.append(event)
        return events

    task = asyncio.create_task(consume())
    await consumer_started.wait()
    # Wait until the snapshot is set (i.e. stream_turn has actually
    # started executing).
    for _ in range(50):
        await asyncio.sleep(0.01)
        if executor._turn_snapshot_len is not None:
            break
    assert executor._turn_snapshot_len == len(pre_messages)

    executor.cancel()
    # Agen still alive at this point; surface the cancel result.
    llm.stopped.set()  # also let any suspending llm unwind
    await asyncio.wait_for(task, timeout=0.5)

    assert agent.messages == pre_messages
    assert executor._pool == []


@pytest.mark.asyncio
async def test_cancel_drops_orphan_assistant_tool_calls_message():
    """Regression gate for the OpenAI 400. After the LLM emits a
    tool_call (which appends an ``assistant`` message with
    ``tool_calls``) but before the tool result comes back, cancelling
    must drop that orphan assistant turn — otherwise the next request
    400s with "messages with tool_calls must be followed by tool
    messages."
    """
    cancelled_during_dispatch = asyncio.Event()

    class HangingDispatch:
        """Replace the dispatch shim with one that suspends, so we can
        cancel mid-tool with the orphan assistant still in place."""

        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def __call__(self, name: str, arguments: str, store, executor) -> str:
            self.calls.append((name, arguments))
            # Block until cancel happens. Use an event so we can flip
            # it from the test.
            cancelled_during_dispatch.wait()
            return "never"

    llm = FakeLLM(
        [
            [
                LLMEvent(
                    "assistant",
                    LLMResponse(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": "echo",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        finish_reason="tool_calls",
                    ),
                )
            ]
        ]
    )

    import neutrix.agent_loop as agent_loop

    original = agent_loop._dispatch_injected
    hanging = HangingDispatch()
    agent_loop._dispatch_injected = hanging  # type: ignore[assignment]

    try:
        agent = Agent(slot=_slot(), llm=llm, use_tools=True)
        executor = Executor(agent=agent)

        pre_messages = list(agent.messages)
        agen = executor.stream_turn("hi")

        events: list[AgentEvent] = []

        async def consume() -> None:
            async for event in agen:
                events.append(event)
                if event.kind == "tool_call":
                    # We've now appended the assistant tool_calls turn
                    # to agent.messages — the orphan-window is open.
                    executor.cancel()
                    cancelled_during_dispatch.set()

        task = asyncio.create_task(consume())
        # Use a thread-event since the synchronous dispatch is the
        # one waiting on it; tests run the event loop and the dispatch
        # is run via to_thread.
        cancelled_during_dispatch_native = threading.Event()
        cancelled_during_dispatch.wait = cancelled_during_dispatch_native.wait  # type: ignore[assignment]
        cancelled_during_dispatch.set = cancelled_during_dispatch_native.set  # type: ignore[assignment]
        # Drive the consumer.
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            cancelled_during_dispatch_native.set()
            await asyncio.wait_for(task, timeout=1.0)

        # No assistant message with tool_calls survives.
        orphans = [
            m
            for m in agent.messages
            if isinstance(m, dict)
            and m.get("role") == "assistant"
            and m.get("tool_calls")
        ]
        assert orphans == []
        # Roll-back means we're back to the pre-turn message list.
        assert agent.messages == pre_messages
    finally:
        agent_loop._dispatch_injected = original  # type: ignore[assignment]


def test_cancel_idle_is_noop():
    """A pristine executor's ``cancel()`` returns silently and does
    not touch anything — it's the contract that lets the controller
    broadcast without first asking 'are you busy?'."""
    llm = FakeLLM([])
    agent = Agent(slot=_slot(), llm=llm, use_tools=False)
    executor = Executor(agent=agent)

    executor.cancel()  # must not raise
    assert executor._turn_snapshot_len is None
    assert executor._pool == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only: killpg")
def test_run_shell_registers_popen_and_tree_kill_unblocks_communicate():
    """The executor-pool kill test. Launch ``sleep 30`` via ``_run_shell``
    on a thread; assert the executor sees the Popen registered;
    cancel; assert the process tree dies within the tree-kill grace
    + some headroom, and the pool is empty afterward."""
    llm = FakeLLM([])
    agent = Agent(slot=_slot(), llm=llm, use_tools=False)
    executor = Executor(agent=agent)
    # Open the snapshot window so cancel() actually does something.
    executor._turn_snapshot_len = len(agent.messages)

    result: list[str] = []

    def runner() -> None:
        result.append(_run_shell("sleep 30", timeout=60, executor=executor))

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    # Poll for Popen registration.
    for _ in range(50):
        if executor._pool:
            break
        time.sleep(0.02)
    assert len(executor._pool) == 1
    pid = executor._pool[0].pid

    start = time.monotonic()
    executor.cancel()
    t.join(timeout=2.0)
    elapsed = time.monotonic() - start

    assert not t.is_alive(), f"_run_shell thread still alive after {elapsed:.2f}s"
    # The grace + SIGKILL should land within a second.
    assert elapsed < 1.5, f"cancel took {elapsed:.2f}s; expected < 1.5s"
    assert executor._pool == []
    # The process is gone — kill(0) raises ProcessLookupError.
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    # And the tool result either reflects the cancel placeholder OR
    # the raw negative-exit (the process group race can pre-empt the
    # _cancel_requested check); both are valid for v0.9.2's scope.
    assert result, "runner thread produced no result"


@pytest.mark.asyncio
async def test_uncancellable_tool_completes_in_background_but_history_rolls_back():
    """A pure-compute tool that doesn't register with the pool keeps
    running after cancel (we can't stop it). The acceptance gate is
    just that ``agent.messages`` is rolled back so the orphan
    assistant turn is gone — the tool's eventual result is dropped."""
    sentinel = threading.Event()

    def slow_dispatch(name: str, arguments: str, store, executor) -> str:
        time.sleep(0.3)
        sentinel.set()
        return "done"

    llm = FakeLLM(
        [
            [
                LLMEvent(
                    "assistant",
                    LLMResponse(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": "slow",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        finish_reason="tool_calls",
                    ),
                )
            ]
        ]
    )

    import neutrix.agent_loop as agent_loop

    original = agent_loop._dispatch_injected
    agent_loop._dispatch_injected = slow_dispatch  # type: ignore[assignment]

    try:
        agent = Agent(slot=_slot(), llm=llm, use_tools=True)
        executor = Executor(agent=agent)
        pre_messages = list(agent.messages)

        agen = executor.stream_turn("hi")
        events: list[AgentEvent] = []

        async def consume() -> None:
            async for event in agen:
                events.append(event)
                if event.kind == "tool_call":
                    executor.cancel()

        task = asyncio.create_task(consume())
        await asyncio.wait_for(task, timeout=2.0)

        # The slow dispatch eventually completes in a daemon thread —
        # we don't kill it; we just drop its result.
        assert sentinel.wait(timeout=1.0)
        assert agent.messages == pre_messages
    finally:
        agent_loop._dispatch_injected = original  # type: ignore[assignment]
