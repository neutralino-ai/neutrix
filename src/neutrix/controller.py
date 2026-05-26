"""Single command surface the view drives.

The :class:`Controller` is what the UI talks to. It owns the in-flight
turn's asyncio task and broadcasts cancel to its subordinates
(``LLM.stop()``, ``Executor.cancel()``, future ``Advisor.cancel()`` in
v0.11.0) without polling their state. The view never reaches around
the controller to touch ``llm`` or ``executor`` directly.

Data-flow contract (per v0.9.2 PRD):

- Commands flow UI → Controller → {LLM, Executor, …}. Always method
  calls; always idempotent; never read state back.
- State flows {LLM, Executor, Agent} → ChatStore → UI. Always via the
  v0.9.0 ``store.apply(event)`` reducer; never polled.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from neutrix.agent_loop import Agent, AgentEvent, ChatLLM
from neutrix.executor import Executor
from neutrix.store import ChatStore


@dataclass
class Controller:
    """Broadcaster of commands to LLM + Executor; never polls them.

    The view supplies ``event_sink`` so the controller can hand each
    :class:`~neutrix.agent_loop.AgentEvent` off for rendering without
    having to know what rendering means. The store mutation is
    applied first, on every event, so a UI subscribed to
    ``ChatStore.changes()`` sees consistent state before any
    view-side print.
    """

    agent: Agent
    executor: Executor
    llm: ChatLLM
    store: ChatStore
    event_sink: Callable[[AgentEvent], Awaitable[None]] | None = None
    _current_stream_task: asyncio.Task | None = field(
        default=None, init=False, repr=False
    )

    async def send(self, user_text: str) -> None:
        """Drive one turn: stream events from the executor, fan them
        out to the store reducer + the view event sink, and clear the
        in-flight task slot on unwind.

        Uses :func:`asyncio.current_task` rather than spawning an
        inner ``create_task`` so the cancellation path crosses no
        extra scheduling layers — the view's outer task and the
        controller's tracked task are the same ``Task`` object. The
        PRD's "small redundancy" between view and controller holds
        (both fields point at the same in-flight thing) without
        introducing the scheduling lag that would let inline slash
        commands race ahead of the worker.
        """
        self._current_stream_task = asyncio.current_task()
        try:
            agen = self.executor.stream_turn(user_text)
            try:
                async for event in agen:
                    self.store.apply(event)
                    if self.event_sink is not None:
                        await self.event_sink(event)
            finally:
                await agen.aclose()
        finally:
            self._current_stream_task = None

    def cancel(self) -> bool:
        """Broadcast cancel to subordinates. Return True iff anything
        was actually in flight.

        Each subordinate's cancel is independently idempotent — we
        never ask "are you busy?", we just broadcast and trust them
        to be no-ops when nothing's there. The v0.11.0 Advisor plugs
        in by adding one extra line to this list (per the PRD's
        future-extension test).
        """
        task = self._current_stream_task
        if task is None or task.done():
            return False
        self.llm.stop()
        self.executor.cancel()
        task.cancel()
        return True
