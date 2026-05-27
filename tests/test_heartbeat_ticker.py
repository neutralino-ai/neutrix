"""Heartbeat ticker loop tests (v0.9.4).

Uses a deterministic ``asyncio.sleep`` override so the assertions
about tick counts and wake-up timing stay stable. The production
default applies ±10% jitter (see ``jittered_sleep``); the jitter
itself is unit-tested in :func:`test_jitter_stays_within_bounds`.
"""
from __future__ import annotations

import asyncio
import random
import time

import pytest

from neutrix.context_manager import State
from neutrix.store import ChatStore, MessageRecord
from neutrix.terminal_chat import (
    HEARTBEAT_JITTER_RATIO,
    HEARTBEAT_TICK_MS,
    heartbeat_loop,
    jittered_sleep,
)


async def _cancel(task: asyncio.Task[None]) -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_ticker_increments_while_busy() -> None:
    """While CM is busy, on_tick fires roughly every sleep_seconds."""
    store = ChatStore()
    state: list[State] = [State.AWAITING_LLM]
    timestamps: list[float] = []

    def on_tick() -> None:
        timestamps.append(time.monotonic())

    task = asyncio.create_task(
        heartbeat_loop(
            lambda: state[0],
            store,
            on_tick,
            sleep_seconds=0.1,
            sleep_fn=asyncio.sleep,
        )
    )
    try:
        await asyncio.sleep(0.4)
        assert len(timestamps) >= 3
        assert timestamps == sorted(timestamps)
    finally:
        await _cancel(task)


@pytest.mark.asyncio
async def test_ticker_idle_when_state_idle() -> None:
    """While CM is IDLE, on_tick never fires."""
    store = ChatStore()
    state: list[State] = [State.IDLE]
    timestamps: list[float] = []

    task = asyncio.create_task(
        heartbeat_loop(
            lambda: state[0],
            store,
            lambda: timestamps.append(time.monotonic()),
            sleep_seconds=0.1,
            sleep_fn=asyncio.sleep,
        )
    )
    try:
        await asyncio.sleep(0.4)
        assert timestamps == []
    finally:
        await _cancel(task)


@pytest.mark.asyncio
async def test_ticker_wakes_on_store_change() -> None:
    """IDLE → AWAITING_LLM via store mutation wakes the ticker within 200 ms."""
    store = ChatStore()
    state: list[State] = [State.IDLE]
    timestamps: list[float] = []

    task = asyncio.create_task(
        heartbeat_loop(
            lambda: state[0],
            store,
            lambda: timestamps.append(time.monotonic()),
            sleep_seconds=0.1,
            sleep_fn=asyncio.sleep,
        )
    )
    try:
        # Give the loop a moment to subscribe to store.changes().
        await asyncio.sleep(0.05)

        start = time.monotonic()
        state[0] = State.AWAITING_LLM
        store.append_message(MessageRecord(role="user", content="hi"))

        for _ in range(25):
            if timestamps:
                break
            await asyncio.sleep(0.01)

        assert timestamps, "ticker did not wake within 250 ms of store mutation"
        assert timestamps[0] - start < 0.2
    finally:
        await _cancel(task)


@pytest.mark.asyncio
async def test_ticker_clean_shutdown_on_cancel() -> None:
    """Cancelling the ticker task exits within one tick without raising
    anything other than CancelledError."""
    store = ChatStore()
    state: list[State] = [State.AWAITING_LLM]
    task = asyncio.create_task(
        heartbeat_loop(
            lambda: state[0],
            store,
            lambda: None,
            sleep_seconds=0.1,
            sleep_fn=asyncio.sleep,
        )
    )
    await asyncio.sleep(0.05)

    start = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.5)
    elapsed = time.monotonic() - start
    assert elapsed < 0.15


@pytest.mark.asyncio
async def test_jitter_stays_within_bounds(monkeypatch) -> None:
    """``jittered_sleep`` applies a multiplier in [1-r, 1+r]; mean ~ 1.0."""
    rng = random.Random(0xC0FFEE)
    nominal = 0.1
    multipliers: list[float] = []

    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        multipliers.append(seconds / nominal)
        await real_sleep(0)

    monkeypatch.setattr("neutrix.terminal_chat.asyncio.sleep", fake_sleep)

    for _ in range(200):
        await jittered_sleep(nominal, jitter_ratio=HEARTBEAT_JITTER_RATIO, rng=rng)

    assert len(multipliers) == 200
    for m in multipliers:
        assert 1.0 - HEARTBEAT_JITTER_RATIO - 1e-9 <= m <= 1.0 + HEARTBEAT_JITTER_RATIO + 1e-9
    mean = sum(multipliers) / len(multipliers)
    assert abs(mean - 1.0) < 0.02


@pytest.mark.asyncio
async def test_jitter_zero_is_exact(monkeypatch) -> None:
    """``jitter_ratio=0`` disables jitter entirely (used by deterministic tests)."""
    observed: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        observed.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr("neutrix.terminal_chat.asyncio.sleep", fake_sleep)

    await jittered_sleep(0.123, jitter_ratio=0.0)
    assert observed == [0.123]


def test_default_tick_period_is_calm_breathing() -> None:
    """Sanity guard: the default cycle is in the human resting-calm band.

    Locks the Phase-2 reopen decision against accidental regression
    back to a fast cadence. HEARTBEAT_TICK_MS * HEARTBEAT_CYCLE_FRAMES
    must fall inside [2.0, 5.0] seconds — the 12-30 BPM range that
    qualifies as resting calm breathing.
    """
    from neutrix.terminal_chat import HEARTBEAT_CYCLE_FRAMES

    cycle_seconds = HEARTBEAT_TICK_MS / 1000 * HEARTBEAT_CYCLE_FRAMES
    assert 2.0 <= cycle_seconds <= 5.0, (
        f"cycle period {cycle_seconds:.2f}s outside the 2-5s calm-breathing band"
    )
