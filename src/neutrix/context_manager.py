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
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol

from loguru import logger

if TYPE_CHECKING:
    from neutrix.prompts import AskUserPort

from neutrix.compaction import (
    CompactionOutcome,
    compact_messages,
    compact_to_token_budget,
    should_compact,
    smart_compact,
    truncate_large_tool_results,
)
from neutrix.config import Slot
from neutrix.executor import Executor, ToolEvent
from neutrix.llm import (
    INTERRUPTED_BY_USER_MARKER,
    LLMEvent,
    LLMResponse,
)
from neutrix.store import ChatStore, CompactionEvent, MessageRecord, Task
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

# v1.4.9: the no-progress watchdog polls at min(llm_timeout_s, this) so a tiny
# test timeout (0.1s) still fires inside its wait window while a 300s prod
# timeout only wakes every few seconds.
_WATCHDOG_POLL_CEILING_S = 2.0

# v0.10.4 Smart Advisor: a judged suggestion is injected as a pseudo-user turn
# wrapped in these tags, so it renders distinctly and is excluded from Up-arrow
# recall — the same treatment as the task reminder / compact markers.
ADVISOR_TAG_OPEN = "<advisor>"
ADVISOR_TAG_CLOSE = "</advisor>"

CancelReason = Literal["user", "timeout"]


class State(str, Enum):
    """Conversation-loop state. See module docstring for transitions."""

    IDLE = "IDLE"
    AWAITING_LLM = "AWAITING_LLM"
    AWAITING_EXECUTOR = "AWAITING_EXECUTOR"
    CANCELLING = "CANCELLING"


# ---- events ----------------------------------------------------------------


@dataclass(frozen=True)
class LLMRoundBundle:
    """Frozen snapshot of the channels sent to the LLM in a round (v0.10.2).

    The single source of truth for the visibility-parity invariant
    (``.claude/rules/visibility-parity.md``): the renderer must surface every
    populated channel here. Built on demand by
    :py:meth:`ContextManager.round_bundle`; consumed by the invariant test.
    """

    system: str
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...] | None


@dataclass(frozen=True)
class UserMessageEvent:
    """User typed a non-slash message — append and drive a turn."""

    text: str


@dataclass(frozen=True)
class CancelEvent:
    """User pressed Esc / Ctrl+C-while-busy, or the LLM watchdog fired.

    Handled synchronously by :py:meth:`ContextManager.handle_event` —
    the UI can also call :py:meth:`ContextManager.cancel` directly
    from a sync key binding. ``reason`` distinguishes the user-initiated
    cancel (v0.9.3 ``[interrupted by user]`` marker) from the
    timeout-watchdog cancel (v0.9.5 ``[LLM timeout after Ns]``
    assistant marker).
    """

    reason: CancelReason = "user"


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
    # v0.10.0 subagent support. ``tool_names`` (None = all builtins) scopes
    # which tool schemas ``_call_llm`` advertises — a subagent omits ``Agent``
    # so recursion is structurally impossible. ``max_rounds`` (None =
    # unbounded) caps the ``_drive`` LLM-round loop so an unattended subagent
    # cannot run away. Both default to the pre-v0.10.0 behavior for the main
    # chat, which passes neither.
    tool_names: frozenset[str] | None = None
    max_rounds: int | None = None
    # v1.4.8 interactive port (CC's `canUseTool` role). Injected by the UI
    # (TerminalChat); the CM is the ONLY layer that holds it, so the Executor
    # stays a pure event leaf. None everywhere non-interactive (tests, piped
    # stdin, inside a subagent) → AskUserQuestion degrades; permission is denied
    # directly in the Executor, never via this port.
    ask_user: AskUserPort | None = None
    state: State = field(default=State.IDLE, init=False)
    last_progress_at: float | None = field(default=None, init=False)
    # v1.5.0: wall-clock start of the current busy phase (this LLM round / this
    # tool dispatch), for the status-bar elapsed field. Distinct from
    # ``last_progress_at`` (which is bumped per token); ``None`` while IDLE.
    phase_started_at: float | None = field(default=None, init=False)
    cancel_reason: CancelReason = field(default="user", init=False)
    # v0.10.1 streaming / v1.4.7 live render: the in-progress assistant text
    # for the current round lives in ``store.pending_assistant_text`` (the
    # single state holder — v0.10.3). It drives the live preview AND is what
    # _do_cancel commits on a mid-stream cancel (keep-partial). Started/extended
    # in _call_llm; cleared in the same beat as the committed append.
    _drive_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _llm_timeout_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

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
        # background thread need it. It also carries the slot so the
        # dispatch layer can forward it to tools that declare it (v0.10.0
        # ``Agent`` builds a subagent LLM from the parent slot).
        self.executor.store = self.store
        self.executor.slot = self.slot

    # ------------------------------------------------------------------ queries

    def is_busy(self) -> bool:
        return self.state != State.IDLE

    def supports_tools(self) -> bool:
        return supports_openai_tools(self.slot)

    def effective_tools_enabled(self) -> bool:
        return self.use_tools and self.supports_tools()

    def round_bundle(self) -> LLMRoundBundle:
        """Snapshot the channels the LLM would receive now (v0.10.2 parity).

        ``system`` is the leading system message's content (the prompt);
        ``messages`` is the full payload; ``tools`` is the schema list when
        tools are effectively enabled, else ``None`` — exactly what
        :py:meth:`_call_llm` sends. The single source of truth the renderer
        and the visibility-parity invariant test both refer to.
        """
        system = ""
        if self.messages and self.messages[0].get("role") == "system":
            system = str(self.messages[0].get("content") or "")
        tools = (
            tuple(get_schemas(self.tool_names)) if self.effective_tools_enabled() else None
        )
        return LLMRoundBundle(
            system=system,
            messages=tuple(self.messages),
            tools=tools,
        )

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
            if self.state in (State.IDLE, State.CANCELLING):
                return
            self.cancel_reason = event.reason
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

    def cancel(self, *, reason: CancelReason = "user") -> bool:
        """Sync convenience for :class:`CancelEvent`. Returns True iff
        something was actually in flight (state was non-IDLE).

        ``reason="user"`` (the default) replays the v0.9.3 cancel-as-steer
        behavior — an ``[interrupted by user]`` user marker is appended
        to ``messages`` so the next LLM call sees the steer.
        ``reason="timeout"`` is invoked by the LLM watchdog: no user
        marker, and the drive loop appends a fresh
        ``[LLM timeout after Ns]`` assistant message on unwind.
        """
        if self.state in (State.IDLE, State.CANCELLING):
            return False
        self.cancel_reason = reason
        self._do_cancel()
        return True

    # ------------------------------------------------------------------ slots

    def switch(self, slot: Slot) -> None:
        self.slot = slot
        self.llm.switch(slot)

    # ---------------------------------------------------------------- compact

    async def compact(self) -> CompactionOutcome:
        """Mechanically drop the oldest ~50 % of history (v0.9.6).

        Direct method (not an event) so the caller gets the dropped
        counts back, mirroring :py:meth:`cancel`. Compacts BOTH
        ``messages`` (the LLM payload) and ``store`` (render/save
        source) and PRESERVES tasks — compaction trims context, not the
        live work list. ``_cancel_and_wait`` is a no-op when IDLE (the
        ``/compact`` command refuses while busy) but keeps the method
        safe for a future event-driven caller.

        Returns a :class:`~neutrix.compaction.CompactionOutcome`;
        ``did_compact=False`` leaves history untouched (conversation too
        short to drop a tool-round-safe slice).
        """
        await self._cancel_and_wait()
        # v0.10.5: prefer summary-based compaction (one LLM call on the active
        # slot, run while IDLE here); fall back to the mechanical drop if the
        # summary is empty/fails, so /compact always does something useful.
        new_messages, outcome = await smart_compact(
            self.messages,
            llm=self.llm,
            model=self.slot.model,
            max_context_tokens=self.slot.max_context_tokens,
        )
        kind = "summary"
        if not outcome.did_compact:
            new_messages, outcome = compact_messages(self.messages)
            kind = "mechanical"
        if not outcome.did_compact:
            return outcome
        self._apply_compaction(new_messages, outcome, kind=kind)
        return outcome

    def _apply_compaction(
        self, new_messages: list[dict[str, Any]], outcome: CompactionOutcome, *, kind: str
    ) -> None:
        """Swap in the compacted messages, rebuild the store, record the event.

        Mirrors :py:meth:`compact`'s store rebuild and PRESERVES tasks. The
        single place the compacted result lands in ``messages``/store.
        """
        tasks = self.store.tasks
        self.messages = new_messages
        self.store.reset()
        for msg in new_messages:
            self.store.append_message(_record_from_openai(msg))
        if tasks:
            self.store.replace_tasks(tasks)
        self.store.add_compaction_event(
            CompactionEvent(
                turns_compacted=outcome.turns_dropped,
                original_tokens=outcome.approx_tokens_dropped,
                summary_tokens=0,
                kind=kind,
            )
        )

    async def _maybe_auto_compact(self) -> None:
        """Threshold-triggered summary compaction (v0.10.5), once per turn.

        Called at the top of :py:meth:`_drive` before any LLM call (IDLE w.r.t.
        the LLM, so it can safely reuse ``self.llm``). No-op when the slot has no
        ``max_context_tokens`` or the payload is under threshold.
        """
        if not should_compact(
            self.messages, max_context_tokens=self.slot.max_context_tokens
        ):
            return
        new_messages, outcome = await smart_compact(
            self.messages,
            llm=self.llm,
            model=self.slot.model,
            max_context_tokens=self.slot.max_context_tokens,
        )
        if outcome.did_compact:
            self._apply_compaction(new_messages, outcome, kind="summary")

    def _recover_from_prompt_too_long(self) -> bool:
        """React to a provider prompt-too-long error (v0.10.5 hardening #2/#3).

        First truncate oversized ``role:tool`` bodies (the single-huge-message
        case), then drop oldest turns under 80% of the window. Returns True iff
        anything was trimmed (so retrying the round is worthwhile).
        """
        budget = int((self.slot.max_context_tokens or 0) * 0.8)
        truncated, n_truncated = truncate_large_tool_results(self.messages, cap=8000)
        new_messages, outcome = compact_to_token_budget(truncated, budget=budget)
        if outcome.did_compact:
            self._apply_compaction(new_messages, outcome, kind="budget")
            return True
        if n_truncated:
            self._apply_compaction(
                truncated, CompactionOutcome(True, 0, 0), kind="truncate"
            )
            return True
        return False

    # ----------------------------------------------------------------- rewind

    async def rewind_to(self, message_index: int) -> int:
        """Drop ``self.messages[message_index:]`` and rebuild the store.

        Direct method (mirrors :py:meth:`compact`): trims BOTH ``messages``
        (the LLM payload) and ``store`` (render/save source), PRESERVES
        tasks, and returns the number of messages dropped. ``message_index``
        is snapped to a tool-round boundary so a rewind that would bisect a
        round drops the whole round — the kept head never ends on an
        ``assistant`` with unanswered ``tool_calls`` or a dangling
        ``role:tool`` result. Returns ``0`` (history untouched) when the
        snapped index drops nothing.

        Destructive (v0.9.7 split #1, Follow CC): the dropped suffix is
        gone, not retained — multi-branch history is a non-goal.
        """
        await self._cancel_and_wait()
        index = self._safe_rewind_index(message_index)
        dropped = len(self.messages) - index
        if dropped <= 0:
            return 0
        tasks = self.store.tasks
        self.messages = self.messages[:index]
        self.store.reset()
        for msg in self.messages:
            self.store.append_message(_record_from_openai(msg))
        if tasks:
            self.store.replace_tasks(tasks)
        return dropped

    def _safe_rewind_index(self, message_index: int) -> int:
        """Clamp ``message_index`` to ``[system-prefix, len]`` then snap it
        backward so the kept head ``messages[:index]`` ends on a complete
        turn — never on an ``assistant`` with pending ``tool_calls`` or a
        ``role:tool`` result whose round was cut. The round-safety mirror of
        :func:`neutrix.compaction.compact_messages`, applied at the head end.
        """
        prefix = 0
        for msg in self.messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                prefix += 1
            else:
                break
        index = max(prefix, min(message_index, len(self.messages)))
        while index > prefix and _is_unfinished_tail(self.messages[index - 1]):
            index -= 1
        return index

    # ---------------------------------------------------------------- advisor

    def inject_advisor_message(self, text: str) -> None:
        """Inject a v0.10.4 Advisor suggestion as a pseudo-user turn.

        The Advisor is a third actor but must NOT mutate ``messages``/store
        itself (only CM does) — it routes its judged suggestion through here.
        Wrapped in ``<advisor>`` tags so it renders distinctly and is excluded
        from Up-arrow recall. Appends to ``messages`` and the store in lockstep
        (mirrors :py:meth:`_append_user_message`); the next LLM round sees it as
        input. Call only while IDLE (the turn-end caller guarantees it).
        """
        content = f"{ADVISOR_TAG_OPEN}{text}{ADVISOR_TAG_CLOSE}"
        self.messages.append({"role": "user", "content": content})
        self.store.append_message(MessageRecord(role="user", content=content))

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
        # v0.10.1 keep-partial-on-cancel (split #2): commit whatever streamed
        # text arrived this round as a partial assistant turn BEFORE the
        # marker (ordering landmine #3), so cancel-as-steer carries the prior
        # assistant intent. Runs for both user and timeout reasons. Orphan
        # tool_use (if any) is repaired by the pairing layer at next send.
        partial = self.store.pending_assistant_text
        if partial:
            self._append_assistant_message({"role": "assistant", "content": partial})
            self.store.clear_pending_assistant_text()
        # User-initiated cancel: append the steer marker. Timeout
        # cancels skip the user marker — the drive loop's
        # ``_finalize_cancel`` will append the
        # ``[LLM timeout after Ns]`` assistant message instead.
        # Orphan tool_use in the latest assistant message is
        # intentionally NOT repaired here — the pairing layer in
        # llm.py handles it on the next API send.
        if self.cancel_reason == "user":
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
        rounds = 0
        ptl_retried = False
        try:
            # v0.10.5: threshold-triggered summary compaction before the first
            # LLM call, once per turn (no-op unless the slot has a window and
            # the payload is over threshold).
            await self._maybe_auto_compact()
            while True:
                self.state = State.AWAITING_LLM
                self.last_progress_at = time.monotonic()
                self.phase_started_at = self.last_progress_at
                self._llm_timeout_task = asyncio.create_task(
                    self._llm_timeout_watchdog()
                )
                self._maybe_inject_system_reminder()
                try:
                    try:
                        assistant_msg = await self._call_llm()
                    except asyncio.CancelledError:
                        self._consume_cancel()
                        self._finalize_cancel()
                        return
                    except Exception as exc:
                        err = _compact_error(exc)
                        self.store.clear_pending_assistant_text()
                        # v0.10.5 hardening #2/#3: a prompt-too-long error is
                        # recoverable — truncate oversized tool bodies + drop
                        # oldest under budget, then retry the round once.
                        if (
                            not ptl_retried
                            and self.slot.max_context_tokens
                            and _is_prompt_too_long(exc)
                            and self._recover_from_prompt_too_long()
                        ):
                            ptl_retried = True
                            logger.warning("prompt too long; compacted and retrying")
                            continue
                        logger.warning("LLM call failed: {}", err)
                        # Discard any partial stream text on a hard error
                        # (v0.10.1 split #5): the [LLM error] message is the
                        # outcome — one assistant message, not partial+error.
                        self._append_assistant_message(
                            {
                                "role": "assistant",
                                "content": f"{LLM_ERROR_PREFIX}{err}]",
                            }
                        )
                        return
                finally:
                    self._cancel_watchdog()
                    self.last_progress_at = None

                if self.state == State.CANCELLING:
                    return

                self._append_assistant_message(assistant_msg)
                # Clear the live-preview pending text in the SAME synchronous
                # beat as the committed append (no await between) so the render
                # watcher never sees both the record and a stale preview.
                self.store.clear_pending_assistant_text()
                rounds += 1

                tool_calls = _extract_tool_calls(assistant_msg)
                if not tool_calls:
                    return

                # Runaway guard (v0.10.0): an unattended subagent caps its
                # LLM rounds. On the cap we stop BEFORE dispatching this
                # round's tools, leaving the last assistant message with
                # unanswered tool_calls — run_subagent reads that as
                # "hit the limit before finishing". None = unbounded (the
                # main chat), so behavior there is unchanged.
                if self.max_rounds is not None and rounds >= self.max_rounds:
                    logger.warning(
                        "drive loop hit max_rounds={} — stopping", self.max_rounds
                    )
                    return

                self.state = State.AWAITING_EXECUTOR
                self.phase_started_at = time.monotonic()
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
            self.phase_started_at = None

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

    async def _llm_timeout_watchdog(self) -> None:
        """No-progress watchdog (v1.4.9): cancel the round only when NO token
        has arrived for ``slot.llm_timeout_s`` — not after a fixed wall-clock
        budget from round start.

        ``last_progress_at`` is set at round start and bumped on every streamed
        token (:py:meth:`_call_llm`), so a steadily-streaming response — even a
        slow multi-minute one from a model like deepseek-v4-pro — is NEVER
        cancelled, while a genuinely hung connection (dead proxy, the
        CLOSE-WAIT case) is still caught after one quiet ``llm_timeout_s``.

        Spawned on every :attr:`State.AWAITING_LLM` entry; cancelled by
        :py:meth:`_cancel_watchdog` on exit. Polls at a sub-interval derived
        from the timeout so a tiny test timeout still fires inside its window;
        the per-iteration state guard retires it the instant the round ends.
        """
        timeout_s = self.slot.llm_timeout_s
        poll = min(timeout_s, _WATCHDOG_POLL_CEILING_S)
        while True:
            try:
                await asyncio.sleep(poll)
            except asyncio.CancelledError:
                return
            if self.state != State.AWAITING_LLM:
                return
            last = self.last_progress_at
            if last is None:  # round is wrapping up; _cancel_watchdog will retire us
                continue
            idle = time.monotonic() - last
            if idle > timeout_s:
                logger.error("LLM made no progress for {:.0f}s — cancelling", idle)
                self.cancel(reason="timeout")
                return

    def _cancel_watchdog(self) -> None:
        task = self._llm_timeout_task
        self._llm_timeout_task = None
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()

    def _finalize_cancel(self) -> None:
        """Per-reason cleanup after the drive task absorbs a cancel.

        For ``reason="user"`` the v0.9.3 ``[interrupted by user]`` marker
        was already appended in :py:meth:`_do_cancel`; nothing more to
        do here. For ``reason="timeout"`` no marker was appended (the
        user did not interrupt), so the drive loop appends a fresh
        ``[LLM timeout after Ns]`` assistant message that the next
        transcript render picks up. ``cancel_reason`` resets to the
        default so a later turn does not inherit the prior reason.
        """
        if self.cancel_reason == "timeout":
            elapsed = int(self.slot.llm_timeout_s)
            self._append_assistant_message(
                {
                    "role": "assistant",
                    "content": f"[LLM timeout after {elapsed}s]",
                }
            )
        self.cancel_reason = "user"

    async def _call_llm(self) -> dict[str, Any]:
        """Make one LLM round; return the assistant message dict.

        Streaming (v0.10.1): ``"token"`` events accumulate into
        ``self._streaming_partial`` so a cancel mid-round can keep the bytes
        that arrived; the terminal ``"assistant"`` event carries the assembled
        message that the happy path appends (unchanged).
        """
        tools = get_schemas(self.tool_names) if self.effective_tools_enabled() else None
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": None}
        self.store.start_assistant_stream()
        async for event in self.llm.stream_response(
            model=self.slot.model,
            messages=self.messages,
            tools=tools,
        ):
            if not isinstance(event, LLMEvent):  # pragma: no cover - defensive
                continue
            if event.kind == "token":
                # v1.4.9: every streamed token resets the no-progress clock so
                # the watchdog measures INACTIVITY, not total round time, and
                # the UI stall hint stops false-firing on a slow-but-live model.
                self.last_progress_at = time.monotonic()
                if isinstance(event.data, str):
                    self.store.extend_assistant_stream(event.data)
            elif event.kind == "assistant":
                payload = event.data
                if isinstance(payload, LLMResponse):
                    assistant_msg = payload.message
                elif isinstance(payload, dict):
                    assistant_msg = payload
        return assistant_msg

    async def _dispatch_tools(self, tool_calls: list[dict[str, str]]) -> None:
        """Drive the executor's event stream and apply each event to messages.

        The executor is a bidirectional async generator (v1.4.8): it yields
        ``tool_started`` / ``tool_finished`` like before, and — for the
        interactive AskUserQuestion tool (permission is denied in the Executor) — a
        ``needs_user_input`` event. The CM is the ONLY layer that holds the
        ``ask_user`` port: it resolves the prompt (via the UI) and feeds the
        :class:`~neutrix.prompts.Answer` back in with ``gen.asend(answer)``
        (``None`` when there is no interactive consumer). The Executor never
        calls the UI; it only yields and receives on its own channel, so
        UI→CM→Executor layering holds. Sequential — results append in order.
        """
        gen = self.executor.dispatch_all(tool_calls)
        send: Any = None
        try:
            while True:
                try:
                    event = await gen.asend(send)
                except StopAsyncIteration:
                    return
                send = None
                if not isinstance(event, ToolEvent):  # pragma: no cover - defensive
                    continue
                data = event.data
                if event.kind == "needs_user_input":
                    # Relay to the human (CM owns the port); the answer (or None
                    # when non-interactive) is sent back on the next asend.
                    spec = data.get("spec")
                    send = await self.ask_user(spec) if self.ask_user is not None else None
                    continue
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
        finally:
            # Close the suspended generator (e.g. cancel mid-prompt left it
            # parked at a `needs_user_input` yield) so it unwinds cleanly.
            # Best-effort: a re-delivered CancelledError (BaseException) still
            # propagates; only mundane cleanup errors are swallowed.
            try:
                await gen.aclose()
            except Exception:
                pass

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


def is_advisor_message(content: Any) -> bool:
    """Whether ``content`` is an injected v0.10.4 ``<advisor>`` suggestion."""
    return isinstance(content, str) and content.startswith(ADVISOR_TAG_OPEN)


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


def _is_prompt_too_long(exc: Exception) -> bool:
    """Heuristic: does this provider error mean the payload exceeded the window?

    String-matches the common phrasings across OpenAI-compatible gateways (no
    structured error code is portable). Used to trigger the v0.10.5 reactive
    compact-and-retry.
    """
    msg = str(exc).lower()
    needles = (
        "context length",
        "context_length",
        "maximum context",
        "context window",
        "too long",
        "prompt is too long",
        "reduce the length",
        "string too long",
        "413",
    )
    return any(n in msg for n in needles)


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


def _is_unfinished_tail(msg: Any) -> bool:
    """Whether keeping a head that ENDS on ``msg`` would leave a tool round
    open — an ``assistant`` whose ``tool_calls`` results were dropped, or a
    ``role:tool`` result still awaiting the assistant's follow-up. Used by
    :py:meth:`ContextManager._safe_rewind_index` to snap a rewind cut back
    to a clean turn boundary.
    """
    if not isinstance(msg, dict):
        return False
    role = msg.get("role")
    if role == "tool":
        return True
    if role == "assistant":
        tool_calls = msg.get("tool_calls")
        return isinstance(tool_calls, list) and len(tool_calls) > 0
    return False


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
