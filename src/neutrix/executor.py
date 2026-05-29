"""Tool dispatch + cancellable Popen pool — no message ownership.

v0.9.3 narrows the Executor's responsibilities to two things:

1. **Subprocess pool.** Tools like :func:`neutrix.tools._run_shell`
   register a :class:`subprocess.Popen` here before they block on
   ``communicate()``. :py:meth:`Executor.cancel` tree-kills the whole
   pool so the parked I/O calls return promptly.
2. **Tool dispatch.** :py:meth:`Executor.dispatch_all` is an async
   generator that yields :class:`ToolEvent` instances —
   ``tool_started`` before each tool runs, ``tool_finished`` after.
   The ContextManager iterates these and decides what to do.

The v0.9.2 Executor owned the per-turn rollback snapshot and mutated
``agent.messages``. v0.9.3 removes that: history is the
ContextManager's job; the Executor only does tool I/O.
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from neutrix.permissions import (
    USER_DENIED,
    PermissionPolicy,
    apply_always_rule,
    block_reason,
    decide,
    permission_question,
    verdict_from_answer,
)
from neutrix.prompts import (
    ASK_NOT_AVAILABLE,
    format_answers_result,
    parse_question_spec,
)
from neutrix.store import ChatStore
from neutrix.tools import dispatch

if TYPE_CHECKING:
    from neutrix.config import Slot


@dataclass(frozen=True)
class ToolEvent:
    """Event yielded by :py:meth:`Executor.dispatch_all`.

    Kinds:
      ``tool_started``  — data={"tool_call_id", "tool_name", "args"}
      ``tool_finished`` — data={"tool_call_id", "tool_name", "content", "ok"}
    """

    kind: str
    data: dict[str, Any]


@dataclass
class Executor:
    """Cancellable Popen pool + tool dispatch loop.

    ``store`` is forwarded to dispatch so Task* tools can mutate the
    canonical store. Set by :class:`ContextManager` at wiring time.
    """

    store: ChatStore | None = None
    # Set by ContextManager at wiring time; forwarded to tools that declare a
    # ``slot`` kwarg (v0.10.0 ``Agent`` builds its subagent LLM from it).
    slot: Slot | None = None
    # v1.1.0 read-before-edit: resolved paths the LLM has Read this session.
    # Edit/Write check membership; per-session (a subagent gets a fresh
    # Executor → fresh read-state, correctly isolated).
    read_paths: set[str] = field(default_factory=set, repr=False)
    # v1.4.0 permissions. Default mode "auto" allows normal ops but blocks
    # clearly-destructive Bash (user-directed default); "allow-all" disables
    # all checks. An empty policy + auto = "runs everything except dangerous
    # shell commands".
    policy: PermissionPolicy = field(default_factory=PermissionPolicy)
    permission_mode: str = "auto"
    _pool: list[subprocess.Popen] = field(default_factory=list, repr=False)
    _cancel_events: list[threading.Event] = field(default_factory=list, repr=False)
    _cancel_requested: bool = field(default=False, repr=False)

    def mark_read(self, resolved_path: str) -> None:
        """Record that ``Read`` has seen this path (v1.1.0 read-before-edit)."""
        self.read_paths.add(resolved_path)

    def register_cancellable(self, proc: subprocess.Popen) -> None:
        self._pool.append(proc)

    def unregister_cancellable(self, proc: subprocess.Popen) -> None:
        try:
            self._pool.remove(proc)
        except ValueError:
            pass

    def register_cancel_event(self, event: threading.Event) -> None:
        """Register a cross-loop cancel token (v0.10.0 subagent bridge).

        The ``Agent`` tool runs its subagent on a worker-thread event loop;
        an :class:`asyncio.Event` can't cross loops, so the subagent watcher
        polls this :class:`threading.Event`, which :py:meth:`cancel` sets.
        """
        self._cancel_events.append(event)

    def unregister_cancel_event(self, event: threading.Event) -> None:
        try:
            self._cancel_events.remove(event)
        except ValueError:
            pass

    def cancel(self) -> None:
        """Tree-kill every still-registered Popen; flag cancel.

        Idempotent — safe to call when no turn is in flight. The
        ``_cancel_requested`` flag lets :func:`neutrix.tools._run_shell`
        return the canonical ``"[cancelled by user]"`` placeholder when
        its Popen exits negative under our kill, instead of leaking a
        terminated-by-signal exit-code line. The ContextManager
        unrelatedly cancels the drive task so the
        ``asyncio.to_thread`` parked on a pure-compute tool is
        abandoned (pure-compute tool cancellation stays a non-goal —
        the daemon thread runs to completion and its result is dropped).
        """
        self._cancel_requested = True
        for proc in list(self._pool):
            _tree_kill(proc)
        self._pool.clear()
        # Trip every registered subagent cancel token so its watcher (on
        # another event loop) wakes and unwinds the subagent.
        for event in list(self._cancel_events):
            event.set()

    async def dispatch_all(
        self,
        tool_calls: list[dict[str, str]],
    ) -> AsyncIterator[ToolEvent]:
        """Dispatch each tool sequentially; yield events as they happen.

        Each tool call yields one ``tool_started`` then one
        ``tool_finished``. Dispatch runs via :func:`asyncio.to_thread`
        so the event loop stays responsive for the cancel broadcast.
        Sequential rather than concurrent — v0.9.3 keeps the v0.9.2
        ordering since the LLMs we drive rarely emit multiple
        tool_calls in one round and concurrent dispatch is an unused
        optimization.

        Each ``tool_call`` is a dict ``{"id", "name", "arguments"}``.
        The ``arguments`` field is the JSON string the LLM produced.
        """
        self._cancel_requested = False
        for tc in tool_calls:
            tcid = str(tc.get("id") or "")
            name = str(tc.get("name") or "")
            args = str(tc.get("arguments") or "")
            yield ToolEvent(
                "tool_started",
                {"tool_call_id": tcid, "tool_name": name, "args": args},
            )
            # v1.4.0/v1.4.8 permission gate before any side effect. `ask` is a
            # real yes/always/no prompt when the consumer can answer (a
            # `needs_user_input` event the CM drives via `.asend()`), else
            # block-with-notice. The Executor builds the request and interprets
            # the reply but NEVER touches the UI — it only yields/receives on
            # its channel, keeping UI→CM→Executor layering intact.
            verdict = decide(name, args, mode=self.permission_mode, policy=self.policy)
            if verdict == "deny":
                yield _finished(tcid, name, block_reason(name, "deny"), ok=False)
                continue
            if verdict == "ask":
                answer = yield _needs_input(permission_question(name, args, verdict))
                if answer is None:  # no interactive consumer → v1.4.0 block-notice
                    yield _finished(
                        tcid, name, block_reason(name, "ask", self.permission_mode),
                        ok=False,
                    )
                    continue
                decision = verdict_from_answer(answer)
                if decision == "no":
                    yield _finished(tcid, name, USER_DENIED, ok=False)
                    continue
                if decision == "always":
                    self.policy = apply_always_rule(self.policy, name, args)
                # yes / always → fall through and run the tool

            # v1.4.8: AskUserQuestion is interactive — round-trip through the
            # `needs_user_input` channel, never to_thread. A ``None`` reply means
            # no interactive consumer (inside a subagent / headless).
            if name == "AskUserQuestion":
                try:
                    spec = parse_question_spec(args)
                except ValueError as exc:
                    yield _finished(tcid, name, f"ERROR: {exc}", ok=False)
                    continue
                answer = yield _needs_input(spec)
                if answer is None:
                    yield _finished(tcid, name, ASK_NOT_AVAILABLE, ok=False)
                    continue
                yield _finished(tcid, name, format_answers_result(answer), ok=True)
                continue

            try:
                result = await asyncio.to_thread(
                    dispatch, name, args, store=self.store, executor=self, slot=self.slot
                )
            except Exception as exc:  # pragma: no cover - dispatch itself catches
                logger.exception("dispatch raised for {}", name)
                result = f"ERROR: tool crashed: {exc}"
            ok = not (isinstance(result, str) and result.startswith("ERROR:"))
            yield _finished(tcid, name, result, ok=ok)


def _finished(tcid: str, name: str, content: str, *, ok: bool) -> ToolEvent:
    """Build a ``tool_finished`` event (keeps dispatch_all branches terse)."""
    return ToolEvent(
        "tool_finished",
        {"tool_call_id": tcid, "tool_name": name, "content": content, "ok": ok},
    )


def _needs_input(spec: Any) -> ToolEvent:
    """Build a ``needs_user_input`` request event (v1.4.8).

    Yielded by :py:meth:`Executor.dispatch_all`; the consumer (ContextManager)
    drives it with ``gen.asend(answer)`` — passing an
    :class:`~neutrix.prompts.Answer` if it has an interactive port, or ``None``
    if not. The Executor never holds the port, so it never calls the UI.
    """
    return ToolEvent("needs_user_input", {"spec": spec})


def _tree_kill(proc: subprocess.Popen, grace_seconds: float = 0.2) -> None:
    """Send SIGTERM to the process group, then SIGKILL after a grace.

    Python analog of Claude Code's ``tree-kill`` for ``run_shell``'s
    ``start_new_session=True`` children. POSIX-only; on Windows the
    Popen pool is empty (we never register anything there).
    """
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception as exc:
        logger.debug("tree_kill SIGKILL fallthrough swallowed: {}", exc)
