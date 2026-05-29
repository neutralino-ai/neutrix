"""Canonical in-memory record of a chat session.

ChatStore is the single owner of:
- the past message list,
- the queue of user inputs typed while the assistant is busy,
- the in-progress assistant stream text,
- the list of pending tool calls.

Any renderer (terminal, future web, tests) reads from this store and
awaits :py:meth:`ChatStore.changes` to know when to refresh. This module
imports nothing from ``tui``, ``terminal_chat``, ``agent_loop``, ``llm``,
``tools``, or ``onboard`` — the dependency arrow points outward only.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal

from loguru import logger

Role = Literal["user", "assistant", "system", "tool"]
TaskStatus = Literal["pending", "in_progress", "completed"]


@dataclass(frozen=True)
class Task:
    """One LLM-tracked work item.

    Mirrors the minimal shape of Claude Code's todo entries: a stable
    id, a short subject, an optional description, and a tri-state status.
    Ids are monotonic positive integers serialized as strings so they
    survive transcript save/load and stay short enough to mention in
    conversation.
    """

    id: str
    subject: str
    description: str = ""
    status: TaskStatus = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class MessageRecord:
    """One settled turn in the conversation.

    ``content`` may be ``None`` to mirror the OpenAI Chat Completions
    convention where an assistant message that only carries tool calls
    has ``content = None``. Renderers that don't speak that convention
    should treat ``None`` as an empty string.

    ``extra`` preserves provider-specific fields (``tool_calls``,
    ``name``, etc.) that the typed record cannot otherwise carry, so
    save → load round-trips through :mod:`neutrix.transcript` stay
    lossless. By convention callers do not mutate ``extra`` after
    constructing the record.
    """

    role: Role
    content: str | None
    ts: datetime = field(default_factory=datetime.now)
    tool_name: str | None = None
    tool_call_id: str | None = None
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class QueuedUserMessage:
    """A user input typed while the assistant was busy."""

    text: str
    queued_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class PendingToolCall:
    """A tool call the LLM has requested but whose result is not back yet."""

    name: str
    arguments: str
    started_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class ToolRecord:
    """A completed tool result, indexed for `/tool N` expansion (v0.10.3).

    Pure data — no rendering logic (that lives in the view). Moved here from
    `terminal_chat.py` so the folded-result tray is canonical store state, not
    a view-private list: the store is the single state holder all actors see.
    """

    index: int
    name: str
    arguments: str
    result: str


class ChatStore:
    """Mutable owner of chat state with an async change-notification API.

    Mutations are synchronous; observers consume change notifications
    through :py:meth:`changes`, an async iterator that yields once per
    batch of mutations since the last yield. Multiple consecutive
    mutations between yields coalesce — the consumer re-reads the full
    store on each yield, so per-mutation granularity is unnecessary.
    """

    def __init__(self) -> None:
        self._messages: list[MessageRecord] = []
        self._queued: list[QueuedUserMessage] = []
        self._pending_assistant_text: str | None = None
        self._pending_tool_calls: list[PendingToolCall] = []
        self._tasks: list[Task] = []
        self._next_task_id: int = 1
        self._llm_active: bool = False
        self._folded_tool_results: list[ToolRecord] = []
        self._subscribers: set[asyncio.Event] = set()

    # --------------------------------------------------------------- reads

    @property
    def messages(self) -> tuple[MessageRecord, ...]:
        return tuple(self._messages)

    @property
    def queued_user_messages(self) -> tuple[QueuedUserMessage, ...]:
        return tuple(self._queued)

    @property
    def pending_assistant_text(self) -> str | None:
        return self._pending_assistant_text

    @property
    def pending_tool_calls(self) -> tuple[PendingToolCall, ...]:
        return tuple(self._pending_tool_calls)

    @property
    def tasks(self) -> tuple[Task, ...]:
        return tuple(self._tasks)

    @property
    def llm_active(self) -> bool:
        return self._llm_active

    @property
    def folded_tool_results(self) -> tuple[ToolRecord, ...]:
        return tuple(self._folded_tool_results)

    # -------------------------------------------------------------- writes

    def append_message(self, msg: MessageRecord) -> None:
        self._messages.append(msg)
        self._notify()

    def enqueue_user(self, text: str) -> QueuedUserMessage:
        item = QueuedUserMessage(text=text)
        self._queued.append(item)
        self._notify()
        return item

    def dequeue_user(self) -> QueuedUserMessage | None:
        if not self._queued:
            return None
        item = self._queued.pop(0)
        self._notify()
        return item

    def start_assistant_stream(self) -> None:
        self._pending_assistant_text = ""
        self._notify()

    def extend_assistant_stream(self, delta: str) -> None:
        if self._pending_assistant_text is None:
            self._pending_assistant_text = ""
        self._pending_assistant_text += delta
        self._notify()

    def finish_assistant_stream(self) -> MessageRecord | None:
        """Append the streamed text as a message and clear pending text.

        Returns the appended record, or ``None`` if no stream was active.
        Even when ``None`` is returned the change is notified, because
        the cleared pending text is observable state.
        """
        text = self._pending_assistant_text
        self._pending_assistant_text = None
        if text is None:
            self._notify()
            return None
        record = MessageRecord(role="assistant", content=text)
        self._messages.append(record)
        self._notify()
        return record

    def add_folded_tool_result(
        self, name: str, arguments: str, result: str
    ) -> ToolRecord:
        """Append a completed tool result to the folded tray (v0.10.3).

        The store assigns the 1-based ``index`` (used by ``/tool N``), so the
        view holds no counter of its own. Returns the created record.
        """
        record = ToolRecord(
            index=len(self._folded_tool_results) + 1,
            name=name,
            arguments=arguments,
            result=result,
        )
        self._folded_tool_results.append(record)
        self._notify()
        return record

    def add_pending_tool_call(self, name: str, arguments: str) -> PendingToolCall:
        call = PendingToolCall(name=name, arguments=arguments)
        self._pending_tool_calls.append(call)
        self._notify()
        return call

    def remove_pending_tool_call(self, name: str) -> PendingToolCall | None:
        """Remove and return the first pending call matching ``name``.

        Falls back to the first pending call regardless of name if no
        exact match exists, mirroring the behavior the terminal renderer
        used before the store existed. Returns ``None`` if no pending
        calls.
        """
        for index, call in enumerate(self._pending_tool_calls):
            if call.name == name:
                removed = self._pending_tool_calls.pop(index)
                self._notify()
                return removed
        if self._pending_tool_calls:
            removed = self._pending_tool_calls.pop(0)
            self._notify()
            return removed
        return None

    def clear_pending_tool_calls(self) -> None:
        if not self._pending_tool_calls:
            return
        self._pending_tool_calls.clear()
        self._notify()

    def add_task(self, subject: str, description: str = "") -> Task:
        """Append a new ``pending`` task and return it."""
        task = Task(
            id=str(self._next_task_id),
            subject=subject,
            description=description,
        )
        self._next_task_id += 1
        self._tasks.append(task)
        self._notify()
        return task

    def update_task(
        self,
        task_id: str,
        *,
        status: TaskStatus | None = None,
        subject: str | None = None,
        description: str | None = None,
    ) -> Task | None:
        """Update one or more fields of an existing task.

        Returns the new record, or ``None`` if the id is unknown. The
        ``updated_at`` field is refreshed whenever any field changes.
        """
        for index, existing in enumerate(self._tasks):
            if existing.id != task_id:
                continue
            changes: dict[str, Any] = {}
            if status is not None and status != existing.status:
                changes["status"] = status
            if subject is not None and subject != existing.subject:
                changes["subject"] = subject
            if description is not None and description != existing.description:
                changes["description"] = description
            if not changes:
                return existing
            changes["updated_at"] = datetime.now()
            new_task = replace(existing, **changes)
            self._tasks[index] = new_task
            self._notify()
            return new_task
        return None

    def remove_task(self, task_id: str) -> Task | None:
        """Remove the matching task; return the removed record, or
        ``None`` if the id is unknown.
        """
        for index, existing in enumerate(self._tasks):
            if existing.id == task_id:
                removed = self._tasks.pop(index)
                self._notify()
                return removed
        return None

    def replace_tasks(self, tasks: Iterable[Task]) -> None:
        """Replace the current task list and reset the id counter.

        Used by :func:`neutrix.transcript.load` to seed tasks from disk.
        The next-id counter resumes from ``max(loaded_ids) + 1`` so newly
        added tasks never collide with previously deleted ones.
        """
        self._tasks = list(tasks)
        max_id = 0
        for task in self._tasks:
            try:
                value = int(task.id)
            except (TypeError, ValueError):
                continue
            if value > max_id:
                max_id = value
        self._next_task_id = max_id + 1
        self._notify()

    def reset(self, system_prompt: str | None = None) -> None:
        """Drop all state. Optionally re-seed with a system prompt."""
        self._messages.clear()
        self._queued.clear()
        self._pending_assistant_text = None
        self._pending_tool_calls.clear()
        self._tasks.clear()
        self._next_task_id = 1
        self._llm_active = False
        self._folded_tool_results.clear()
        if system_prompt is not None:
            self._messages.append(
                MessageRecord(role="system", content=system_prompt)
            )
        self._notify()

    # --------------------------------------------------------------- reducer

    def apply(self, event: Any) -> None:
        """Mutate the store from an :class:`AgentEvent`.

        The single entry point used by renderers that consume an
        ``Agent.stream_reply`` event stream. Only the kinds listed
        below have an effect; other kinds are no-ops.

        Accepts ``event: Any`` (read reflectively) so this module does
        not import :class:`neutrix.agent_loop.AgentEvent` — that would
        be a circular import, since ``agent_loop`` imports
        :class:`ChatStore`.
        """
        kind = getattr(event, "kind", None)
        data = getattr(event, "data", None)
        if kind == "llm_request_start":
            self._llm_active = True
            self._notify()
        elif kind == "llm_request_end":
            self._llm_active = False
            self._notify()
        elif kind == "tool_call":
            name = str((data or {}).get("name") or "")
            arguments = str((data or {}).get("arguments") or "")
            self.add_pending_tool_call(name, arguments)
        elif kind == "tool_result":
            name = str((data or {}).get("name") or "")
            self.remove_pending_tool_call(name)
        # token / assistant / done / error: no store change.

    # -------------------------------------------------------- observation

    async def changes(self) -> AsyncIterator[None]:
        """Yield once per batch of mutations since the last yield.

        Each call to ``changes()`` creates an independent subscription.
        Consecutive mutations between yields coalesce into a single
        wake-up — the consumer is expected to re-read the store on
        every yield, so no information is lost.
        """
        event = asyncio.Event()
        self._subscribers.add(event)
        try:
            while True:
                await event.wait()
                event.clear()
                yield None
        finally:
            self._subscribers.discard(event)

    # ---------------------------------------------------------- internal

    def _notify(self) -> None:
        for event in self._subscribers:
            try:
                event.set()
            except Exception:
                logger.exception("ChatStore subscriber notification failed")


# -------------------------------------------------------- OpenAI bridge

# These conversion helpers live in store.py because they bridge the
# typed store to the OpenAI-format messages list that :class:`Agent`
# and the LLM client still own as of v0.7.0. When agent_loop refactors
# into a pure controller (v0.8.0), the OpenAI bridge can move into the
# controller and store.py can shed any awareness of that format.

_OPENAI_KNOWN_KEYS = {"role", "content", "tool_call_id"}


def openai_to_record(raw: dict[str, Any]) -> MessageRecord:
    """Build a :class:`MessageRecord` from an OpenAI-format message dict.

    Anything beyond ``role``, ``content``, and ``tool_call_id`` is kept
    in ``extra`` so that round-tripping back through
    :func:`record_to_openai` is lossless.
    """
    role_raw = raw.get("role", "system")
    role: Role = (
        role_raw if role_raw in ("user", "assistant", "system", "tool") else "system"
    )
    content = raw.get("content")
    if content is not None and not isinstance(content, str):
        content = str(content)
    tool_call_id_raw = raw.get("tool_call_id")
    tool_call_id = str(tool_call_id_raw) if tool_call_id_raw is not None else None
    extra = {k: v for k, v in raw.items() if k not in _OPENAI_KNOWN_KEYS}
    return MessageRecord(
        role=role,
        content=content,
        tool_call_id=tool_call_id,
        extra=extra or None,
    )


def record_to_openai(record: MessageRecord) -> dict[str, Any]:
    """Render a :class:`MessageRecord` as an OpenAI Chat Completions message."""
    out: dict[str, Any] = {"role": record.role, "content": record.content}
    if record.tool_call_id is not None:
        out["tool_call_id"] = record.tool_call_id
    if record.extra:
        out.update(record.extra)
    return out
