"""Heartbeat ticker loop tests (v0.9.4 → v0.9.8).

Uses a deterministic ``asyncio.sleep`` override so the assertions about
tick counts and wake-up timing stay stable. v0.9.8 replaces the jittered
brightness cadence with a strict on/off blink and adds the
``on_enter_busy`` reset hook (so a turn opens on a visible dot).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from neutrix.context_manager import State
from neutrix.store import ChatStore, MessageRecord
from neutrix.terminal_chat import (
    HEARTBEAT_BLINK_INTERVAL_MS,
    heartbeat_loop,
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
    """Cancelling the ticker exits within one tick, raising only CancelledError."""
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


# ---- v0.9.8 on_enter_busy reset (first-frame-visible) ---------------------


@pytest.mark.asyncio
async def test_on_enter_busy_fires_before_first_tick() -> None:
    """on_enter_busy is called once on IDLE→busy, ahead of the first tick."""
    store = ChatStore()
    state: list[State] = [State.AWAITING_LLM]  # already busy at start
    events: list[str] = []

    task = asyncio.create_task(
        heartbeat_loop(
            lambda: state[0],
            store,
            lambda: events.append("tick"),
            sleep_seconds=0.05,
            sleep_fn=asyncio.sleep,
            on_enter_busy=lambda: events.append("enter"),
        )
    )
    try:
        await asyncio.sleep(0.16)
        assert events, "no events fired"
        assert events[0] == "enter"  # reset happens before any tick
        assert "tick" in events[1:]
        assert events.count("enter") == 1  # one continuous busy phase
    finally:
        await _cancel(task)


@pytest.mark.asyncio
async def test_on_enter_busy_resets_counter_so_turn_opens_visible() -> None:
    """Models the app wiring: enter resets the blink counter to 0 (even →
    visible dot), then ticks advance it. Guarantees no blank-dot turn start.
    """
    store = ChatStore()
    state: list[State] = [State.AWAITING_LLM]
    tick = [5]  # leftover odd-ish value from a previous turn
    seen_after_enter: list[int] = []

    def on_enter_busy() -> None:
        tick[0] = 0
        seen_after_enter.append(tick[0])

    def on_tick() -> None:
        tick[0] += 1

    task = asyncio.create_task(
        heartbeat_loop(
            lambda: state[0],
            store,
            on_tick,
            sleep_seconds=0.05,
            sleep_fn=asyncio.sleep,
            on_enter_busy=on_enter_busy,
        )
    )
    try:
        await asyncio.sleep(0.16)
        assert seen_after_enter == [0]  # reset to 0 exactly once, at busy entry
        assert tick[0] >= 1  # then ticks advanced
    finally:
        await _cancel(task)


def test_blink_interval_is_calm() -> None:
    """Guard against regressing to a fast / 120 Hz cadence: one full on+off
    wink cycle (two toggles) must sit in a calm 0.5-4 s band.
    """
    cycle_seconds = HEARTBEAT_BLINK_INTERVAL_MS / 1000 * 2
    assert 0.5 <= cycle_seconds <= 4.0, (
        f"blink cycle {cycle_seconds:.2f}s outside the calm 0.5-4 s band"
    )
