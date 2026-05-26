"""Owns the in-flight turn: cancellable-process pool + rollback snapshot.

The :class:`Executor` wraps an :class:`~neutrix.agent_loop.Agent` for one
turn at a time. The agent is the stateless message router; the executor
holds the per-turn state the controller needs to broadcast cancel against
— specifically the pre-turn ``len(agent.messages)`` snapshot (used to
roll history back so a cancelled turn doesn't leave the conversation in
a 400-able shape) and the pool of ``subprocess.Popen`` handles tools
have registered for tree-kill on cancel.
"""
from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from loguru import logger

from neutrix.agent_loop import Agent, AgentEvent


@dataclass
class Executor:
    """Per-turn state holder + cancel entry-point.

    Tools register a cancellable :class:`subprocess.Popen` with
    :py:meth:`register_cancellable` before they block on
    ``communicate()``; they unregister in a ``finally`` clause. On
    cancel, the executor tree-kills every still-registered process,
    rolls ``agent.messages`` back to the pre-turn length, and sets
    :py:attr:`_cancel_requested` so the tool wrapper can render the
    canonical ``[cancelled by user]`` placeholder when its Popen exits
    negative.
    """

    agent: Agent
    _pool: list[subprocess.Popen] = field(default_factory=list, repr=False)
    _turn_snapshot_len: int | None = field(default=None, repr=False)
    _cancel_requested: bool = field(default=False, repr=False)

    def register_cancellable(self, proc: subprocess.Popen) -> None:
        self._pool.append(proc)

    def unregister_cancellable(self, proc: subprocess.Popen) -> None:
        try:
            self._pool.remove(proc)
        except ValueError:
            pass

    def cancel(self) -> None:
        """Tree-kill the pool, roll history back, flag cancel.

        Idempotent — a no-op when no turn is in flight (the snapshot
        slot is the in-flight indicator). Synchronous so the controller
        can call this from any task, including the one currently
        awaiting on a tool registered with the pool.
        """
        if self._turn_snapshot_len is None:
            return
        self._cancel_requested = True
        for proc in list(self._pool):
            _tree_kill(proc)
        self._pool.clear()
        self.agent.rollback_to(self._turn_snapshot_len)

    async def stream_turn(
        self,
        user_text: str,
    ) -> AsyncIterator[AgentEvent]:
        """Forward ``agent.stream_reply`` with pre/post-turn bookkeeping.

        Sets the rollback snapshot at the top of the turn and clears
        it (plus the pool, defensively) on unwind. The cancel flag is
        reset for each fresh turn so a previously cancelled turn does
        not poison the next one. When the cancel flag is set during
        the turn (via :py:meth:`cancel`), the rollback is RE-applied
        on unwind so any post-cancel message synthesis (the agent's
        run-to-completion epilogue, executed if the LLM iterator
        merely runs out instead of raising :class:`CancelledError`)
        is undone — that's what makes the idle-state contract
        ``agent.messages == pre`` hold whether or not the outer task
        is also cancelled.
        """
        snapshot_len = len(self.agent.messages)
        self._turn_snapshot_len = snapshot_len
        self._cancel_requested = False
        try:
            async for event in self.agent.stream_reply(
                user_text, executor=self
            ):
                yield event
        finally:
            # PEP 525 safe — no yield in finally. Pure assignment +
            # in-place clears.
            was_cancelled = self._cancel_requested
            self._turn_snapshot_len = None
            self._cancel_requested = False
            self._pool.clear()
            if was_cancelled:
                self.agent.rollback_to(snapshot_len)


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
