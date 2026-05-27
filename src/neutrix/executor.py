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
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from neutrix.store import ChatStore
from neutrix.tools import dispatch


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
    _pool: list[subprocess.Popen] = field(default_factory=list, repr=False)
    _cancel_requested: bool = field(default=False, repr=False)

    def register_cancellable(self, proc: subprocess.Popen) -> None:
        self._pool.append(proc)

    def unregister_cancellable(self, proc: subprocess.Popen) -> None:
        try:
            self._pool.remove(proc)
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
            try:
                result = await asyncio.to_thread(
                    dispatch, name, args, store=self.store, executor=self
                )
            except Exception as exc:  # pragma: no cover - dispatch itself catches
                logger.exception("dispatch raised for {}", name)
                result = f"ERROR: tool crashed: {exc}"
            ok = not (isinstance(result, str) and result.startswith("ERROR:"))
            yield ToolEvent(
                "tool_finished",
                {
                    "tool_call_id": tcid,
                    "tool_name": name,
                    "content": result,
                    "ok": ok,
                },
            )


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
