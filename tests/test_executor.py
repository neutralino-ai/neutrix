"""Tests for the minimal-surface Executor + tree-kill helper."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import pytest

from neutrix.executor import Executor, ToolEvent
from neutrix.tools import _run_shell

# ---- dispatch_all event surface -------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_all_emits_started_then_finished_in_order(monkeypatch):
    """A vanilla tool dispatch yields tool_started then tool_finished."""
    monkeypatch.setattr(
        "neutrix.executor.dispatch",
        lambda name, arguments, **_: f"ran {name} with {arguments}",
    )
    executor = Executor()
    events: list[ToolEvent] = []
    async for event in executor.dispatch_all(
        [{"id": "c1", "name": "echo", "arguments": "{}"}]
    ):
        events.append(event)

    assert [e.kind for e in events] == ["tool_started", "tool_finished"]
    assert events[0].data == {
        "tool_call_id": "c1",
        "tool_name": "echo",
        "args": "{}",
    }
    assert events[1].data["content"] == "ran echo with {}"
    assert events[1].data["ok"] is True


@pytest.mark.asyncio
async def test_dispatch_all_multiple_tools_sequential(monkeypatch):
    """Two tool_calls → two pairs of events in dispatch order."""
    monkeypatch.setattr(
        "neutrix.executor.dispatch",
        lambda name, arguments, **_: f"{name}_done",
    )
    executor = Executor()
    events = [
        e
        async for e in executor.dispatch_all(
            [
                {"id": "c1", "name": "first", "arguments": "{}"},
                {"id": "c2", "name": "second", "arguments": "{}"},
            ]
        )
    ]
    kinds = [e.kind for e in events]
    assert kinds == [
        "tool_started",
        "tool_finished",
        "tool_started",
        "tool_finished",
    ]
    # tool_call_id flows through both events for each pair.
    assert events[0].data["tool_call_id"] == "c1"
    assert events[1].data["tool_call_id"] == "c1"
    assert events[2].data["tool_call_id"] == "c2"
    assert events[3].data["tool_call_id"] == "c2"


@pytest.mark.asyncio
async def test_dispatch_all_marks_error_tools_as_not_ok(monkeypatch):
    """A tool that returns ``"ERROR: ..."`` flags ``ok=False``."""
    monkeypatch.setattr(
        "neutrix.executor.dispatch",
        lambda name, arguments, **_: "ERROR: nope",
    )
    executor = Executor()
    events = [
        e
        async for e in executor.dispatch_all(
            [{"id": "c1", "name": "bad", "arguments": "{}"}]
        )
    ]
    assert events[-1].data["ok"] is False
    assert events[-1].data["content"] == "ERROR: nope"


# ---- cancel() ----------------------------------------------------------


def test_cancel_idle_is_noop():
    """No Popen in pool → cancel is a clean no-op."""
    executor = Executor()
    executor.cancel()
    assert executor._pool == []


def test_cancel_does_not_touch_any_message_state():
    """v0.9.2's executor mutated agent.messages on cancel; v0.9.3
    must NOT — the Executor has no messages awareness at all."""
    executor = Executor()
    # The dataclass has no agent / messages field; only pool + flag.
    fields = {f.name for f in executor.__dataclass_fields__.values()}
    assert "agent" not in fields
    assert "_turn_snapshot_len" not in fields


# ---- run_shell + tree-kill regression ---------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only: killpg")
def test_run_shell_registers_popen_and_tree_kill_unblocks_communicate():
    """Launch ``sleep 30`` via ``_run_shell`` on a thread; cancel;
    assert the process dies fast and the pool is empty."""
    executor = Executor()

    result: list[str] = []

    def runner() -> None:
        result.append(_run_shell("sleep 30", timeout=60, executor=executor))

    t = threading.Thread(target=runner, daemon=True)
    t.start()
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
    assert elapsed < 1.5
    assert executor._pool == []
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert result, "runner thread produced no result"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only: killpg")
def test_run_shell_returns_cancelled_marker_when_tree_killed():
    """A cancelled run_shell returns ``[cancelled by user]`` content
    instead of leaking the terminated-by-signal exit-code line.
    The pairing layer in llm.py also synthesizes the same marker as a
    fallback when the tool_finished event never lands."""
    executor = Executor()

    result: list[str] = []

    def runner() -> None:
        result.append(_run_shell("sleep 30", timeout=60, executor=executor))

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    for _ in range(50):
        if executor._pool:
            break
        time.sleep(0.02)
    executor.cancel()
    t.join(timeout=2.0)

    assert result == ["[cancelled by user]"]


# ---- v1.6.2 Bash timeout: deadlock-free kill, clamp, partial output -------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only: killpg")
def test_run_shell_timeout_tree_kills_grandchild_and_keeps_partial_output():
    """v1.6.2 regression for the timeout-path deadlock: a command that times
    out AND leaves a backgrounded grandchild holding the stdout pipe must NOT
    block on the post-kill drain (the old ``proc.kill()`` + bare
    ``communicate()`` waited for the orphaned grandchild's pipe EOF, ~forever).
    The whole process *group* is tree-killed, the bounded drain returns fast,
    and the partial stdout printed before the hang is preserved."""
    result: list[str] = []

    def runner() -> None:
        # ``echo started`` → partial stdout; ``sleep 30 &`` → a backgrounded
        # grandchild that inherits the pipe; foreground ``sleep 30`` keeps the
        # shell alive past the 1 s timeout.
        result.append(_run_shell("echo started; sleep 30 & sleep 30", timeout=1))

    t = threading.Thread(target=runner, daemon=True)
    start = time.monotonic()
    t.start()
    t.join(timeout=8.0)
    elapsed = time.monotonic() - start

    assert not t.is_alive(), f"_run_shell deadlocked on timeout ({elapsed:.1f}s)"
    assert elapsed < 7.0, f"timeout path too slow ({elapsed:.1f}s)"
    assert result, "runner produced no result"
    assert result[0].startswith("ERROR: command timed out after 1s")
    assert "started" in result[0]  # partial stdout preserved


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only: killpg")
def test_run_shell_timeout_clamps_supplied_value_to_max(monkeypatch):
    """A timeout above the cap is clamped to ``_BASH_MAX_TIMEOUT_S`` and the
    clamp is noted in the result so the LLM learns the ceiling."""
    monkeypatch.setattr("neutrix.tools._BASH_MAX_TIMEOUT_S", 1)
    out = _run_shell("sleep 30", timeout=999)
    assert "ERROR: command timed out after 1s" in out
    assert "requested timeout 999s clamped to max 1s" in out


def test_run_shell_default_timeout_runs_fast_command():
    """Omitting ``timeout`` resolves to the (long) default; a fast command
    still returns its exit_code + stdout unchanged."""
    out = _run_shell("echo hi")
    assert "exit_code: 0" in out
    assert "hi" in out


def test_env_int_parses_and_bounds_invariant(monkeypatch):
    """``_env_int`` returns a positive override or the fallback; the max cap is
    held >= the default so an env override can't invert the bound."""
    from neutrix import tools

    monkeypatch.delenv("NEUTRIX_BASH_DEFAULT_TIMEOUT_S", raising=False)
    assert tools._env_int("NEUTRIX_BASH_DEFAULT_TIMEOUT_S", 120) == 120
    monkeypatch.setenv("NEUTRIX_BASH_DEFAULT_TIMEOUT_S", "45")
    assert tools._env_int("NEUTRIX_BASH_DEFAULT_TIMEOUT_S", 120) == 45
    monkeypatch.setenv("NEUTRIX_BASH_DEFAULT_TIMEOUT_S", "0")  # non-positive → fallback
    assert tools._env_int("NEUTRIX_BASH_DEFAULT_TIMEOUT_S", 120) == 120
    monkeypatch.setenv("NEUTRIX_BASH_DEFAULT_TIMEOUT_S", "nope")  # invalid → fallback
    assert tools._env_int("NEUTRIX_BASH_DEFAULT_TIMEOUT_S", 120) == 120
    assert tools._BASH_MAX_TIMEOUT_S >= tools._BASH_DEFAULT_TIMEOUT_S


def test_register_and_unregister_cancellable_round_trip():
    """Defensive: the executor pool is a plain list; unregister of a
    proc not currently registered is a no-op."""
    executor = Executor()

    class StubProc:
        pid = 123

        def poll(self):
            return 0

    proc = StubProc()
    executor.register_cancellable(proc)  # type: ignore[arg-type]
    assert executor._pool == [proc]
    executor.unregister_cancellable(proc)  # type: ignore[arg-type]
    assert executor._pool == []
    # Second unregister doesn't raise.
    executor.unregister_cancellable(proc)  # type: ignore[arg-type]


def test_tree_kill_noop_on_finished_process():
    """A Popen that's already finished doesn't get killed again."""
    from neutrix.executor import _tree_kill

    proc = subprocess.Popen(["true"])
    proc.wait()
    # _tree_kill should be a clean no-op (proc.poll() returns 0).
    _tree_kill(proc)
