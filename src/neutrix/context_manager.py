"""ContextManager — state machine that owns conversation history.

v0.9.3 dissolves the v0.9.2 ``Agent`` + ``Controller`` pair into one
class. The :class:`ContextManager` is the *only* layer that mutates
``messages`` and ``ChatStore``. UI emits events; CM dispatches; LLM and
Executor do their narrow I/O jobs and report back.

State machine
=============

::

    IDLE ──user_message──▶ AWAITING_LLM ─assistant w/o tool_calls─▶ IDLE
                                │
                                │ assistant w/ tool_calls
                                ▼
                       AWAITING_EXECUTOR ─all tools done─▶ AWAITING_LLM

    Any non-IDLE ──cancel──▶ CANCELLING ──cleanup──▶ IDLE

Cancel semantics
================

Cancel is a **sync** side-effect operation (the UI's key binding is
sync; awaiting through the event loop would add scheduling latency).
:py:meth:`ContextManager.cancel` is the sync convenience wrapper for
:class:`CancelEvent`. On cancel, CM:

1. Sets ``state = CANCELLING``.
2. Calls :py:meth:`Executor.cancel` (tree-kill subprocesses).
3. Calls :py:meth:`LLM.stop` (cancel awaiting create task).
4. Appends a ``role:user`` message
   :data:`~neutrix.llm.INTERRUPTED_BY_USER_MARKER` so the next LLM
   call sees the cancel signal.
5. Cancels the drive task so any ``asyncio.to_thread`` parked on a
   pure-compute tool is abandoned (PRD non-goal — the daemon thread
   runs to completion; result discarded).

Orphan ``tool_use`` blocks in the latest assistant message are LEFT
unrepaired in ``messages``. :func:`neutrix.llm._ensure_tool_result_pairing`
synthesizes ``tool_result`` placeholders at API-send time on a copy of
the outgoing payload, preserving the "only CM mutates messages" rule.

Why the broader refactor
========================

The cancel-steer change alone could have landed as a small patch on
v0.9.2. Bundling the Controller→ContextManager reshape into v0.9.3 is
user-directed (PRD §"Scope of ContextManager refactor"): the
:py:meth:`Agent.stream_reply` async generator that drove the v0.9.2
loop has no producer once CM owns the loop, so AgentEvents +
``TerminalChat._mirror_new_agent_messages`` lose their reason to
exist. UI rendering switches to subscribing to
:py:meth:`ChatStore.changes`. The refactor is forced by the design,
not optional.
"""
from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from loguru import logger

from neutrix.config import Slot
from neutrix.executor import Executor, ToolEvent
from neutrix.llm import (
    INTERRUPTED_BY_USER_MARKER,
    LLMEvent,
    LLMResponse,
)
from neutrix.store import ChatStore, MessageRecord, Task
from neutrix.tools import get_schemas

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Keep it simple."

# Matches Claude Code's TODO_REMINDER_CONFIG exactly.
TURNS_SINCE_WRITE = 10
TURNS_BETWEEN_REMINDERS = 10
TASK_REMINDER_TAG_OPEN = "<system-reminder>"
TASK_REMINDER_TAG_CLOSE = "</system-reminder>"
TASK_REMINDER_MARKER = "Here are the existing tasks:"
TASK_MANAGEMENT_TOOLS = frozenset({"TaskCreate", "TaskUpdate"})

LLM_ERROR_PREFIX = "[LLM error: "


class State(str, Enum):
    """Conversation-loop state. See module docstring for transitions."""

    IDLE = "IDLE"
    AWAITING_LLM = "AWAITING_LLM"
    AWAITING_EXECUTOR = "AWAITING_EXECUTOR"
    CANCELLING = "CANCELLING"


# ---- events ----------------------------------------------------------------


@dataclass(frozen=True)
class UserMessageEvent:
    """User typed a non-slash message — append and drive a turn."""

    text: str


@dataclass(frozen=True)
class CancelEvent:
    """User pressed Esc / Ctrl+C-while-busy.

    Handled synchronously by :py:meth:`ContextManager.handle_event` —
    the UI can also call :py:meth:`ContextManager.cancel` directly
    from a sync key binding.
    """


@dataclass(frozen=True)
class SlotSwitchEvent:
    """/fast or /strong — point the LLM at a different slot."""

    slot: Slot


@dataclass(frozen=True)
class ClearEvent:
    """/clear — reset messages + tasks back to a fresh state."""


@dataclass(frozen=True)
class ReplaceHistoryEvent:
    """/load — replace messages with a loaded transcript + tasks.

    The pair shape mirrors :mod:`neutrix.transcript`'s output:
    ``raw_messages`` are OpenAI-format dicts, ``records`` are the
    typed view, ``tasks`` are the typed Tasks. CM rebuilds the store
    from the records and the messages list from the raw dicts.
    """

    raw_messages: list[dict[str, Any]]
    records: tuple[MessageRecord, ...]
    tasks: tuple[Task, ...]


# ---- LLM protocol ---------------------------------------------------------


class ChatLLM(Protocol):
    def switch(self, slot: Slot) -> None: ...

    def stream_response(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ): ...

    def stop(self) -> None: ...


# ---- ContextManager -------------------------------------------------------


@dataclass
class ContextManager:
    """State-machine owner of one chat session.

    Single mutator of ``messages`` and ``store``. UI emits events via
    :py:meth:`handle_event`; CM decides and dispatches. ``store`` is
    seeded with the system prompt at construction so renderers
    subscribed to :py:meth:`ChatStore.changes` see consistent state
    from the very first frame.
    """

    slot: Slot
    llm: ChatLLM
    executor: Executor
    store: ChatStore
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    use_tools: bool = True
    messages: list[dict[str, Any]] = field(default_factory=list)
    state: State = field(default=State.IDLE, init=False)
    _drive_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # The system prompt anchors message[0]. Seed both messages and
        # the typed store so a renderer wired to ChatStore.changes()
        # sees the system turn from frame one.
        if not self.messages:
            self.messages = [{"role": "system", "content": self.system_prompt}]
        if not self.store.messages:
            for msg in self.messages:
                self.store.append_message(_record_from_openai(msg))
        # Executor sees the store too — Task* tools dispatched on a
        # background thread need it.
        self.executor.store = self.store

    # ------------------------------------------------------------------ queries

    def is_busy(self) -> bool:
        return self.state != State.IDLE

    def supports_tools(self) -> bool:
        return supports_openai_tools(self.slot)

    def effective_tools_enabled(self) -> bool:
        return self.use_tools and self.supports_tools()

    # ------------------------------------------------------------------ events

    async def handle_event(self, event: Any) -> None:
        """Single async entry point for UI-emitted events.

        :class:`CancelEvent` is handled synchronously (its work is all
        side effects; nothing to await). Other events drive async
        work — :class:`UserMessageEvent` drives a full turn; the
        slot/clear/load events mutate state, possibly after awaiting
        a cancel of an in-flight turn.

        CancelledError raised inside a user-message turn is swallowed
        here so the UI contract stays clean: ``handle_event`` never
        raises for a normal user-initiated cancel.
        """
        if isinstance(event, CancelEvent):
            self._do_cancel()
            return
        if isinstance(event, UserMessageEvent):
            try:
                await self._handle_user_message(event.text)
            except asyncio.CancelledError:
                # Drive task was cancelled mid-turn. Marker already
                # appended; drive's finally cleared state to IDLE.
                pass
            return
        if isinstance(event, SlotSwitchEvent):
            self.slot = event.slot
            self.llm.switch(event.slot)
            return
        if isinstance(event, ClearEvent):
            await self._handle_clear()
            return
        if isinstance(event, ReplaceHistoryEvent):
            await self._handle_replace_history(event)
            return
        logger.warning("ContextManager: unknown event {!r}", event)

    def cancel(self) -> bool:
        """Sync convenience for :class:`CancelEvent`. Returns True iff
        something was actually in flight (state was non-IDLE)."""
        if self.state in (State.IDLE, State.CANCELLING):
            return False
        self._do_cancel()
        return True

    # ------------------------------------------------------------------ slots

    def switch(self, slot: Slot) -> None:
        self.slot = slot
        self.llm.switch(slot)

    # -------------------------------------------------- internal handlers

    async def _handle_user_message(self, text: str) -> None:
        # Always append the user turn — even if busy, the typed store
        # shows it in the transcript and the next drive cycle picks it
        # up. The "queue" lives in store.queued_user_messages for UI
        # display; CM just drains messages list in order.
        self._append_user_message(text)
        if self.state != State.IDLE:
            # A turn is already in flight. The new user message is
            # appended; it'll be sent on the next LLM round (it sits
            # in messages and the next iteration of _drive will pick
            # it up automatically since the LLM payload is built from
            # the live list each round).
            return
        self._drive_task = asyncio.current_task()
        try:
            await self._drive()
        finally:
            self._drive_task = None

    def _do_cancel(self) -> None:
        if self.state in (State.IDLE, State.CANCELLING):
            return
        self.state = State.CANCELLING
        # Side effects first — broadcast to subordinates so their
        # parked I/O wakes up.
        self.executor.cancel()
        self.llm.stop()
        # Append the marker. Orphan tool_use in the latest assistant
        # message is intentionally NOT repaired here — the pairing
        # layer in llm.py handles it on the next API send.
        self._append_user_message(INTERRUPTED_BY_USER_MARKER)
        # Cancel the drive task so any asyncio.to_thread parked on a
        # pure-compute tool is abandoned (PRD non-goal — the daemon
        # thread runs to completion; its result is silently dropped).
        # The drive coroutine catches the resulting CancelledError and
        # calls task.uncancel() so the awaiting handle_event coroutine
        # returns normally.
        task = self._drive_task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def _handle_clear(self) -> None:
        await self._cancel_and_wait()
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.store.reset(system_prompt=self.system_prompt)

    async def _handle_replace_history(self, event: ReplaceHistoryEvent) -> None:
        await self._cancel_and_wait()
        self.messages = list(event.raw_messages)
        self.store.reset()
        for record in event.records:
            self.store.append_message(record)
        if event.tasks:
            self.store.replace_tasks(event.tasks)

    async def _cancel_and_wait(self) -> None:
        """If a turn is in flight, cancel it and wait for unwind."""
        if self.state in (State.IDLE, State.CANCELLING):
            return
        self._do_cancel()
        task = self._drive_task
        if task is None or task.done():
            return
        if task is asyncio.current_task():
            return
        try:
            await task
        except (asyncio.CancelledError, Exception) as exc:
            logger.debug("drive task unwound during cancel-and-wait: {}", exc)

    # --------------------------------------------------------------- driver

    async def _drive(self) -> None:
        """Drive turns until messages settle into a final assistant.

        Loops over LLM rounds — each round may end with tool_calls,
        in which case the executor dispatches them and the loop
        continues. The CANCELLING state short-circuits the loop;
        ``finally`` always returns state to IDLE.

        :class:`asyncio.CancelledError` is caught here on the LLM /
        executor awaits. When :py:meth:`_do_cancel` calls
        ``task.cancel()`` on the drive task, the catch consumes the
        cancellation via :py:meth:`asyncio.Task.uncancel` so the
        awaiting :py:meth:`_handle_user_message` returns normally
        rather than propagating the cancellation up to the UI.
        """
        try:
            while True:
                self.state = State.AWAITING_LLM
                self._maybe_inject_system_reminder()
                try:
                    assistant_msg = await self._call_llm()
                except asyncio.CancelledError:
                    self._consume_cancel()
                    return
                except Exception as exc:
                    err = _compact_error(exc)
                    logger.warning("LLM call failed: {}", err)
                    self._append_assistant_message(
                        {
                            "role": "assistant",
                            "content": f"{LLM_ERROR_PREFIX}{err}]",
                        }
                    )
                    return

                if self.state == State.CANCELLING:
                    return

                self._append_assistant_message(assistant_msg)

                tool_calls = _extract_tool_calls(assistant_msg)
                if not tool_calls:
                    return

                self.state = State.AWAITING_EXECUTOR
                try:
                    await self._dispatch_tools(tool_calls)
                except asyncio.CancelledError:
                    self._consume_cancel()
                    return

                if self.state == State.CANCELLING:
                    return
                # Loop back for the next LLM round.
        finally:
            self.store.clear_pending_tool_calls()
            self.state = State.IDLE

    @staticmethod
    def _consume_cancel() -> None:
        """Decrement the task's cancellation count so it returns normally."""
        task = asyncio.current_task()
        if task is None:
            return
        try:
            cancelling = task.cancelling()
        except AttributeError:  # pragma: no cover - <3.11
            return
        if cancelling > 0:
            task.uncancel()

    async def _call_llm(self) -> dict[str, Any]:
        """Make one LLM round; return the assistant message dict."""
        tools = get_schemas() if self.effective_tools_enabled() else None
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": None}
        async for event in self.llm.stream_response(
            model=self.slot.model,
            messages=self.messages,
            tools=tools,
        ):
            if not isinstance(event, LLMEvent):  # pragma: no cover - defensive
                continue
            if event.kind == "assistant":
                payload = event.data
                if isinstance(payload, LLMResponse):
                    assistant_msg = payload.message
                elif isinstance(payload, dict):
                    assistant_msg = payload
        return assistant_msg

    async def _dispatch_tools(self, tool_calls: list[dict[str, str]]) -> None:
        """Iterate executor events and apply each to messages + store.

        Sequential — the executor yields tool_started/tool_finished in
        order. We append a ``role:tool`` message on each tool_finished
        so the next LLM round sees the results in their proper place.
        """
        async for event in self.executor.dispatch_all(tool_calls):
            if not isinstance(event, ToolEvent):  # pragma: no cover - defensive
                continue
            data = event.data
            if event.kind == "tool_started":
                self.store.add_pending_tool_call(
                    str(data.get("tool_name") or ""),
                    str(data.get("args") or ""),
                )
            elif event.kind == "tool_finished":
                tool_name = str(data.get("tool_name") or "")
                tool_call_id = str(data.get("tool_call_id") or "")
                content = str(data.get("content") or "")
                self.store.remove_pending_tool_call(tool_name)
                tool_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                }
                self.messages.append(tool_msg)
                self.store.append_message(
                    MessageRecord(
                        role="tool",
                        content=content,
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                    )
                )
            if self.state == State.CANCELLING:
                # Stop processing further events; abandoning the
                # generator is OK because executor.cancel() already
                # tree-killed subprocesses.
                return

    # --------------------------------------------------- mutation helpers

    def _append_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self.store.append_message(MessageRecord(role="user", content=text))

    def _append_assistant_message(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        self.store.append_message(_record_from_openai(message))

    # ---------------------------------------------- system reminder rule

    def _maybe_inject_system_reminder(self) -> None:
        """Append a Claude-shaped task reminder if both thresholds met.

        Called at the top of each AWAITING_LLM entry. The v0.9.3 design
        moves this from per-turn (v0.8.0's
        :py:meth:`Agent.stream_reply` only injected once per call) to
        per-LLM-round. The behavior is identical: the reminder gate
        requires ``TURNS_BETWEEN_REMINDERS`` assistant turns since the
        last reminder, which a tool-driven follow-up round will not
        satisfy, so no second reminder is appended within one turn.
        """
        reminder = build_task_reminder(self.messages, self.store.tasks)
        if reminder is None:
            return
        self.messages.append(reminder)
        self.store.append_message(
            MessageRecord(role="user", content=reminder["content"])
        )


# ---- task reminder algorithm (moved from v0.8.0 agent_loop.py) -----------


def build_task_reminder(
    messages: list[dict[str, Any]],
    tasks: tuple[Task, ...],
) -> dict[str, Any] | None:
    """Return a Claude-shaped ``<system-reminder>`` user message if due.

    Conditions (all must hold):

    1. At least one task is currently ``pending`` or ``in_progress``.
    2. ``TURNS_SINCE_WRITE`` or more assistant turns have elapsed since
       the LLM last called ``TaskCreate`` or ``TaskUpdate``.
    3. ``TURNS_BETWEEN_REMINDERS`` or more assistant turns have elapsed
       since the previous reminder was injected.
    """
    actionable = [t for t in tasks if t.status in ("pending", "in_progress")]
    if not actionable:
        return None
    if assistant_turns_since_task_management(messages) < TURNS_SINCE_WRITE:
        return None
    if assistant_turns_since_reminder(messages) < TURNS_BETWEEN_REMINDERS:
        return None
    body = _build_task_reminder_body(tasks)
    return {
        "role": "user",
        "content": f"{TASK_REMINDER_TAG_OPEN}\n{body}\n{TASK_REMINDER_TAG_CLOSE}",
    }


def assistant_turns_since_task_management(messages: list[dict[str, Any]]) -> int:
    """Count assistant messages scanning backwards until one is found whose
    ``tool_calls`` includes ``TaskCreate`` or ``TaskUpdate``.

    Returns the total assistant-turn count when no such call exists.
    """
    seen = 0
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        seen += 1
        if _message_calls_task_management(message):
            return seen - 1
    return seen


def assistant_turns_since_reminder(messages: list[dict[str, Any]]) -> int:
    """Count assistant messages scanning backwards until one whose preceding
    or following user message is an already-injected reminder."""
    seen = 0
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            seen += 1
            continue
        if role == "user" and is_task_reminder(message.get("content")):
            return seen
    return seen


def _message_calls_task_management(message: dict[str, Any]) -> bool:
    raw_tool_calls = message.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return False
    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, dict):
            continue
        function = raw_tool_call.get("function")
        if not isinstance(function, dict):
            continue
        if str(function.get("name") or "") in TASK_MANAGEMENT_TOOLS:
            return True
    return False


def is_task_reminder(content: Any) -> bool:
    """Whether ``content`` is the body of a v0.8.0-shape task reminder."""
    if not isinstance(content, str):
        return False
    if not content.startswith(TASK_REMINDER_TAG_OPEN):
        return False
    return TASK_REMINDER_MARKER in content


def format_reminder_notice(tasks: tuple[Task, ...]) -> str:
    """Render the dim one-line notice shown in place of a folded reminder."""
    n_done = sum(1 for t in tasks if t.status == "completed")
    n_inprog = sum(1 for t in tasks if t.status == "in_progress")
    n_todo = sum(1 for t in tasks if t.status == "pending")
    return (
        "system reminder: task list injected "
        f"({n_done} done, {n_inprog} in progress, {n_todo} todo)"
    )


def _build_task_reminder_body(tasks: tuple[Task, ...]) -> str:
    actionable = [t for t in tasks if t.status in ("pending", "in_progress")]
    lines = [
        "The task tools haven't been used recently. If you're working on "
        "tasks that would benefit from tracking progress, consider using "
        "TaskCreate to add new tasks and TaskUpdate to update task status "
        "(set to in_progress when starting, completed when done). Also "
        "consider cleaning up the task list if it has become stale. Only "
        "use these if relevant to the current work. This is just a gentle "
        "reminder - ignore if not applicable. Make sure that you NEVER "
        "mention this reminder to the user.",
    ]
    if actionable:
        lines.append("")
        lines.append(TASK_REMINDER_MARKER)
        lines.append("")
        for task in actionable:
            lines.append(f"#{task.id}. [{task.status}] {task.subject}")
    return "\n".join(lines)


# ---- misc helpers --------------------------------------------------------


def supports_openai_tools(slot: Slot) -> bool:
    """Whether this slot accepts OpenAI Chat Completions function tools."""
    model = slot.model.lower()
    provider = slot.provider.lower()
    if provider == "ihep" and model.startswith("anthropic/"):
        return False
    return True


def _compact_error(exc: Exception, *, limit: int = 600) -> str:
    text = str(exc).replace("\n", " ").strip() or exc.__class__.__name__
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def _extract_tool_calls(assistant_msg: dict[str, Any]) -> list[dict[str, str]]:
    raw_tool_calls = assistant_msg.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return []
    tool_calls: list[dict[str, str]] = []
    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, dict):
            continue
        function = raw_tool_call.get("function")
        if not isinstance(function, dict):
            continue
        tool_calls.append(
            {
                "id": str(raw_tool_call.get("id") or ""),
                "name": str(function.get("name") or ""),
                "arguments": str(function.get("arguments") or ""),
            }
        )
    return tool_calls


def _record_from_openai(msg: dict[str, Any]) -> MessageRecord:
    """Build a MessageRecord that preserves extra OpenAI fields.

    Mirrors :func:`neutrix.store.openai_to_record` but inlined here so
    CM doesn't reach into another module just for the conversion.
    """
    role = msg.get("role") or "system"
    if role not in ("user", "assistant", "system", "tool"):
        role = "system"
    content = msg.get("content")
    if content is not None and not isinstance(content, str):
        content = str(content)
    tool_call_id_raw = msg.get("tool_call_id")
    tool_call_id = str(tool_call_id_raw) if tool_call_id_raw is not None else None
    known = {"role", "content", "tool_call_id"}
    extra = {k: v for k, v in msg.items() if k not in known}
    return MessageRecord(
        role=role,  # type: ignore[arg-type]
        content=content,
        tool_call_id=tool_call_id,
        extra=extra or None,
    )


def messages_from_records(records: Iterable[MessageRecord]) -> list[dict[str, Any]]:
    """Build an OpenAI-format messages list from typed records."""
    out: list[dict[str, Any]] = []
    for record in records:
        msg: dict[str, Any] = {"role": record.role, "content": record.content}
        if record.tool_call_id is not None:
            msg["tool_call_id"] = record.tool_call_id
        if record.extra:
            msg.update(record.extra)
        out.append(msg)
    return out
