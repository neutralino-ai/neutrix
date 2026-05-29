"""Subagent framework (v0.10.0) — a fresh-context worker behind the ``Agent`` tool.

The structural answer to context-explode: spawn a worker with its own
:class:`~neutrix.context_manager.ContextManager` (own message list, own
:class:`~neutrix.store.ChatStore`, own :class:`~neutrix.executor.Executor`,
a scoped tool allowlist), run the LLM/tool loop to completion, and return only
the final assistant text. The parent's context grows by one ``tool_result``
instead of by the whole sub-task.

Why reuse ``ContextManager`` rather than write a loop (split #2): the round
loop, tool dispatch, cancel-and-wait, and tool-round-boundary handling already
live there and are tested. ``run_subagent`` only adds the runaway cap
(``max_turns`` → CM ``max_rounds``), the tool scoping (``tool_names``), and a
cancel bridge.

Cancellation (split #7, corrected): the ``Agent`` tool runs in the executor's
worker thread (``asyncio.to_thread``) on its *own* event loop, so an
``asyncio.Event`` can't cross loops — the token is a :class:`threading.Event`
the parent ``Executor`` sets on cancel. The watcher's only job is to stop the
subagent's token/subprocess burn; the ``[cancelled by user]`` the parent LLM
sees comes from the existing v0.9.3 cancel path, not from this return value.
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any

from loguru import logger

from neutrix.config import Slot
from neutrix.context_manager import ChatLLM, ContextManager, UserMessageEvent
from neutrix.executor import Executor
from neutrix.store import ChatStore

SUBAGENT_MAX_TURNS = 25
SUBAGENT_MAX_RESULT_CHARS = 100_000

SUBAGENT_SYSTEM_PROMPT = (
    "You are a focused sub-agent dispatched by a controller to complete one "
    "self-contained task and report back. You have tools; use them to do the "
    "work yourself.\n\n"
    "CRITICAL: your final message IS the return value — it is handed verbatim "
    "to the controller as the result of the task, not shown to a human. So:\n"
    "- Return the answer or the gathered data directly. No 'Sure, I'll…' "
    "preamble, no sign-off.\n"
    "- You cannot ask questions back — there is no user to answer. If the task "
    "is ambiguous, make the most reasonable assumption, state it briefly, and "
    "proceed.\n"
    "- When the task is done, stop calling tools and write the final result.\n"
    "Be thorough but concise; the controller only sees what you write last."
)

_TRUNCATION_SUFFIX = "\n…[subagent output truncated]"


@dataclass(frozen=True)
class SubagentResult:
    """Outcome of one :func:`run_subagent` call.

    ``final_text`` is what the ``Agent`` tool returns to the parent LLM
    (already capped). ``turn_count`` / ``cancelled`` / ``error`` are
    out-of-band metadata; on the Esc path the whole result is discarded with
    the abandoned ``to_thread`` future, so ``cancelled`` is meaningful only on
    a non-Esc cancel (e.g. a test firing the event directly).
    """

    final_text: str
    turn_count: int
    cancelled: bool = False
    error: str | None = None


async def run_subagent(
    *,
    user_prompt: str,
    slot: Slot,
    llm: ChatLLM,
    tool_names: frozenset[str] | None,
    max_turns: int = SUBAGENT_MAX_TURNS,
    cancel_event: threading.Event | None = None,
    system_prompt: str = SUBAGENT_SYSTEM_PROMPT,
) -> SubagentResult:
    """Run a fresh-context subagent to completion and return its final text.

    Builds a fresh :class:`ContextManager` (own store/executor, scoped tools,
    ``max_rounds=max_turns``), submits ``user_prompt``, drives the existing
    round loop until it settles, and extracts the last assistant text. A
    cancel-watcher task (if ``cancel_event`` is given) polls the event and
    calls :py:meth:`ContextManager.cancel` to stop the burn.
    """
    store = ChatStore()
    executor = Executor()
    cm = ContextManager(
        slot=slot,
        llm=llm,
        executor=executor,
        store=store,
        system_prompt=system_prompt,
        use_tools=True,
        tool_names=tool_names,
        max_rounds=max_turns,
    )

    watcher: asyncio.Task[None] | None = None
    if cancel_event is not None:
        watcher = asyncio.create_task(_watch_cancel(cancel_event, cm))

    error: str | None = None
    try:
        await cm.handle_event(UserMessageEvent(user_prompt))
    except Exception as exc:  # pragma: no cover - CM swallows most internally
        error = _compact(exc)
        logger.warning("run_subagent caught: {}", error)
    finally:
        if watcher is not None:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass

    cancelled = cancel_event is not None and cancel_event.is_set()
    final_text = _extract_final_text(cm.messages, max_turns=max_turns)
    if len(final_text) > SUBAGENT_MAX_RESULT_CHARS:
        final_text = final_text[:SUBAGENT_MAX_RESULT_CHARS] + _TRUNCATION_SUFFIX
    return SubagentResult(
        final_text=final_text,
        turn_count=_count_assistant_turns(cm.messages),
        cancelled=cancelled,
        error=error,
    )


async def _watch_cancel(cancel_event: threading.Event, cm: ContextManager) -> None:
    """Poll the cross-loop cancel token; cancel the subagent CM when set."""
    try:
        while not cancel_event.is_set():
            await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        return
    cm.cancel()


def _extract_final_text(messages: list[dict[str, Any]], *, max_turns: int) -> str:
    """Last assistant text, or a turn-limit note if the tail is unfinished."""
    last_assistant: dict[str, Any] | None = None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            last_assistant = msg
            break
    if last_assistant is None:
        return ""
    content = last_assistant.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if last_assistant.get("tool_calls"):
        return (
            f"[subagent reached the {max_turns}-turn limit before producing "
            "a final answer]"
        )
    return content if isinstance(content, str) else ""


def _count_assistant_turns(messages: list[dict[str, Any]]) -> int:
    return sum(
        1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant"
    )


def _compact(exc: Exception, *, limit: int = 300) -> str:
    text = str(exc).replace("\n", " ").strip() or exc.__class__.__name__
    return text if len(text) <= limit else f"{text[:limit].rstrip()}..."
