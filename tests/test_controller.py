"""Tests for the Controller — single command surface that broadcasts cancel."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from neutrix.agent_loop import AgentEvent
from neutrix.controller import Controller
from neutrix.store import ChatStore


class StopRecordingLLM:
    """Records ``stop()`` calls; otherwise inert."""

    def __init__(self) -> None:
        self.stop_calls = 0

    def switch(self, slot: Any) -> None:  # pragma: no cover - irrelevant here
        pass

    async def stream_response(self, **_: Any):  # pragma: no cover
        if False:
            yield None

    def stop(self) -> None:
        self.stop_calls += 1


class RecordingExecutor:
    """Just enough Executor surface for the controller tests."""

    def __init__(self, events: list[AgentEvent] | None = None) -> None:
        self.events = events or []
        self.cancel_calls = 0

    def cancel(self) -> None:
        self.cancel_calls += 1

    async def stream_turn(self, user_text: str):
        self.received = user_text
        for event in self.events:
            yield event


def _make_controller(
    *,
    events: list[AgentEvent] | None = None,
) -> tuple[Controller, StopRecordingLLM, RecordingExecutor, ChatStore, list[AgentEvent]]:
    store = ChatStore()
    llm = StopRecordingLLM()
    executor = RecordingExecutor(events=events)
    sunk: list[AgentEvent] = []

    async def sink(event: AgentEvent) -> None:
        sunk.append(event)

    controller = Controller(
        agent=object(),  # type: ignore[arg-type] - controller doesn't read this directly
        executor=executor,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        store=store,
        event_sink=sink,
    )
    return controller, llm, executor, store, sunk


@pytest.mark.asyncio
async def test_send_streams_events_to_store_and_sink_in_order():
    """``send`` applies every event to the store reducer AND
    forwards it to ``event_sink`` in stream order. The reducer flips
    ``llm_active`` on start/end, so the sink sees a consistent store
    by the time it runs."""
    controller, _llm, _executor, store, sunk = _make_controller(
        events=[
            AgentEvent("llm_request_start"),
            AgentEvent("token", "hi"),
            AgentEvent("llm_request_end"),
            AgentEvent("done"),
        ]
    )

    await controller.send("hello")

    assert [e.kind for e in sunk] == [
        "llm_request_start",
        "token",
        "llm_request_end",
        "done",
    ]
    # llm_active flipped on then off via store.apply.
    assert store.llm_active is False


@pytest.mark.asyncio
async def test_send_clears_current_stream_task_on_completion():
    """After normal unwind, the in-flight slot is None — so the next
    ``cancel()`` on an idle controller returns False."""
    controller, _llm, _executor, _store, _sunk = _make_controller(
        events=[AgentEvent("done")]
    )
    await controller.send("hello")
    assert controller._current_stream_task is None


@pytest.mark.asyncio
async def test_cancel_when_idle_returns_false_and_broadcasts_nothing():
    """No in-flight task → cancel is a clean no-op. The LLM is not
    asked to stop, the executor is not cancelled, no task is touched."""
    controller, llm, executor, _store, _sunk = _make_controller()

    assert controller.cancel() is False
    assert llm.stop_calls == 0
    assert executor.cancel_calls == 0


@pytest.mark.asyncio
async def test_cancel_broadcasts_in_order_to_llm_executor_and_task():
    """The release-blocking PRD contract: cancel() calls
    ``llm.stop()`` THEN ``executor.cancel()`` THEN ``task.cancel()``
    each exactly once. The order matters because closing the HTTP
    stream first gets the iterator out of its blocking read before
    the executor's rollback runs, so by the time the task awakens
    with CancelledError the snapshot rollback has already happened."""
    controller, _llm, _executor, _store, _sunk = _make_controller()

    order: list[str] = []

    class OrderedLLM(StopRecordingLLM):
        def stop(self) -> None:
            order.append("llm")
            super().stop()

    class OrderedExecutor(RecordingExecutor):
        def cancel(self) -> None:
            order.append("executor")
            super().cancel()

    controller.llm = OrderedLLM()
    controller.executor = OrderedExecutor()  # type: ignore[assignment]

    class TaskStub:
        def __init__(self) -> None:
            self.cancelled = 0

        def done(self) -> bool:
            return False

        def cancel(self) -> bool:
            order.append("task")
            self.cancelled += 1
            return True

    stub = TaskStub()
    controller._current_stream_task = stub  # type: ignore[assignment]

    assert controller.cancel() is True
    assert order == ["llm", "executor", "task"]
    assert stub.cancelled == 1


@pytest.mark.asyncio
async def test_cancel_on_done_task_returns_false():
    """A task that already finished doesn't get re-cancelled; cancel
    returns False so the caller's busy-check stays accurate."""
    controller, llm, executor, _store, _sunk = _make_controller()

    class DoneTask:
        def done(self) -> bool:
            return True

        def cancel(self) -> bool:  # pragma: no cover - should not be called
            raise AssertionError("must not be cancelled")

    controller._current_stream_task = DoneTask()  # type: ignore[assignment]
    assert controller.cancel() is False
    assert llm.stop_calls == 0
    assert executor.cancel_calls == 0


@pytest.mark.asyncio
async def test_cancel_documents_future_advisor_broadcast_target_extension():
    """Documents the v0.11.0 extension point: the controller's
    cancel() is designed so adding ``self.advisor.cancel()`` is a
    one-line change. We exercise that by injecting an extra target
    via subclass and asserting it would be called in the same
    broadcast."""

    class ExtendedController(Controller):
        def __init__(self, *args: Any, advisor: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.advisor = advisor

        def cancel(self) -> bool:
            task = self._current_stream_task
            if task is None or task.done():
                return False
            self.llm.stop()
            self.executor.cancel()
            self.advisor.cancel()  # v0.11.0 fourth target
            task.cancel()
            return True

    class RecordingAdvisor:
        def __init__(self) -> None:
            self.cancels = 0

        def cancel(self) -> None:
            self.cancels += 1

    store = ChatStore()
    llm = StopRecordingLLM()
    executor = RecordingExecutor()
    advisor = RecordingAdvisor()
    controller = ExtendedController(
        agent=object(),  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        store=store,
        advisor=advisor,
    )

    class TaskStub:
        def __init__(self) -> None:
            self.cancelled = False

        def done(self) -> bool:
            return False

        def cancel(self) -> bool:
            self.cancelled = True
            return True

    controller._current_stream_task = TaskStub()  # type: ignore[assignment]
    assert controller.cancel() is True
    assert advisor.cancels == 1
    assert llm.stop_calls == 1
    assert executor.cancel_calls == 1


@pytest.mark.asyncio
async def test_cancel_during_send_propagates_cancelled_error_to_caller():
    """End-to-end: a long-running stream is cancellable by the
    controller from another task; the awaiter sees CancelledError
    and the in-flight slot is cleared."""
    store = ChatStore()
    llm = StopRecordingLLM()

    class HangingExecutor:
        def __init__(self) -> None:
            self.cancel_calls = 0

        def cancel(self) -> None:
            self.cancel_calls += 1

        async def stream_turn(self, user_text: str):
            await asyncio.Event().wait()  # blocks forever
            if False:  # pragma: no cover
                yield None

    executor = HangingExecutor()
    controller = Controller(
        agent=object(),  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        store=store,
    )

    task = asyncio.create_task(controller.send("hi"))
    # Let the inner await park.
    for _ in range(50):
        await asyncio.sleep(0.01)
        if controller._current_stream_task is not None:
            break
    assert controller._current_stream_task is task

    assert controller.cancel() is True
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.5)
    assert llm.stop_calls == 1
    assert executor.cancel_calls == 1
    assert controller._current_stream_task is None
