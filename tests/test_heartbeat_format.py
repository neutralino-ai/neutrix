"""Pure-function tests for the heartbeat renderer (v0.9.4 → v0.9.8).

v0.9.8 replaces the brightness-fade glyph with an on/off *presence wink*
(split #1): the dot is ``HEARTBEAT_GLYPH`` on even ``tick`` and a
same-width blank on odd ``tick``. There is no gradient, so the renderer
emits a single style — ``HEARTBEAT_GLYPH_STYLE`` normally,
``HEARTBEAT_STALLED_GLYPH_STYLE`` (red) when stalled. The label logic and
the v0.9.5 stall hint are unchanged.

Cases derived from ``docs/PRDs/v0.9.8-liveness-motion.md`` § Acceptance.
"""

from __future__ import annotations

import time

from neutrix.context_manager import State
from neutrix.store import ChatStore, PendingToolCall
from neutrix.terminal_chat import (
    HEARTBEAT_GLYPH,
    HEARTBEAT_GLYPH_STYLE,
    HEARTBEAT_LABEL_STYLE,
    HEARTBEAT_STALL_FLOOR_S,
    HEARTBEAT_STALLED_GLYPH_STYLE,
    format_heartbeat,
    stall_threshold_for,
)


def _glyph_fragment(state: State, store: ChatStore, tick: int) -> tuple[str, str]:
    return format_heartbeat(state, store, tick)[0]


def _label_fragment(state: State, store: ChatStore, tick: int) -> tuple[str, str]:
    return format_heartbeat(state, store, tick)[1]


# ---- labels ---------------------------------------------------------------


def test_idle_returns_empty() -> None:
    assert format_heartbeat(State.IDLE, ChatStore(), 0) == []


def test_awaiting_llm_label() -> None:
    fragments = format_heartbeat(State.AWAITING_LLM, ChatStore(), 0)
    text = "".join(t for _s, t in fragments)
    assert "LLM" in text
    glyph_style, glyph_text = fragments[0]
    assert HEARTBEAT_GLYPH in glyph_text
    assert glyph_style == HEARTBEAT_GLYPH_STYLE


def test_awaiting_executor_label_uses_tool_head() -> None:
    store = ChatStore()
    store.add_pending_tool_call("run_shell", '{"command":"sleep 5"}')
    store.add_pending_tool_call("read_file", '{"path":"x"}')
    assert isinstance(store.pending_tool_calls[0], PendingToolCall)

    fragments = format_heartbeat(State.AWAITING_EXECUTOR, store, 0)
    text = "".join(t for _s, t in fragments)
    assert "tool: run_shell" in text
    assert "read_file" not in text
    assert fragments[0][0] == HEARTBEAT_GLYPH_STYLE  # tool dot is white, not red


def test_awaiting_executor_with_no_pending_falls_back() -> None:
    """Defensive: tiny window between state transition and add_pending_tool_call."""
    fragments = format_heartbeat(State.AWAITING_EXECUTOR, ChatStore(), 0)
    text = "".join(t for _s, t in fragments)
    assert "tool" in text


def test_cancelling_label() -> None:
    fragments = format_heartbeat(State.CANCELLING, ChatStore(), 0)
    text = "".join(t for _s, t in fragments)
    assert "cancelling" in text
    assert fragments[0][0] == HEARTBEAT_GLYPH_STYLE  # not stalled → white


# ---- presence wink (v0.9.8 split #1) --------------------------------------


def test_visible_on_even_tick() -> None:
    style, text = _glyph_fragment(State.AWAITING_LLM, ChatStore(), 0)
    assert text == f"{HEARTBEAT_GLYPH} "
    assert style == HEARTBEAT_GLYPH_STYLE


def test_blank_on_odd_tick() -> None:
    _style, text = _glyph_fragment(State.AWAITING_LLM, ChatStore(), 1)
    assert text.strip() == ""  # dot winked off


def test_glyph_field_is_constant_width() -> None:
    """Visible and blank fields are the same width so the label never shifts."""
    on = _glyph_fragment(State.AWAITING_LLM, ChatStore(), 0)[1]
    off = _glyph_fragment(State.AWAITING_LLM, ChatStore(), 1)[1]
    assert len(on) == len(off) == 2


def test_wink_alternates_by_tick_parity() -> None:
    state = State.AWAITING_LLM
    store = ChatStore()
    for tick in range(6):
        text = _glyph_fragment(state, store, tick)[1]
        if tick % 2 == 0:
            assert HEARTBEAT_GLYPH in text
        else:
            assert text.strip() == ""


def test_label_style_is_always_bright_regardless_of_wink() -> None:
    """The label fragment style does NOT depend on tick — only the dot winks."""
    state = State.AWAITING_LLM
    store = ChatStore()
    styles = {_label_fragment(state, store, t)[0] for t in range(6)}
    assert styles == {HEARTBEAT_LABEL_STYLE}


# ---- v0.9.5 stall hint (v0.9.8 colour swap) -------------------------------


def test_stall_hint_off_when_last_progress_at_is_none() -> None:
    """v0.9.4 callers (no kwarg) see the normal white dot."""
    fragments = format_heartbeat(State.AWAITING_LLM, ChatStore(), 0)
    assert fragments[0][0] == HEARTBEAT_GLYPH_STYLE
    assert "stalled" not in fragments[1][1]


def test_stall_hint_off_below_threshold() -> None:
    fresh = time.monotonic() - 0.1
    fragments = format_heartbeat(State.AWAITING_LLM, ChatStore(), 0, last_progress_at=fresh)
    assert fragments[0][0] == HEARTBEAT_GLYPH_STYLE
    assert "stalled" not in fragments[1][1]


def test_stall_hint_on_above_threshold() -> None:
    stale = time.monotonic() - (HEARTBEAT_STALL_FLOOR_S + 5.0)
    fragments = format_heartbeat(State.AWAITING_LLM, ChatStore(), 0, last_progress_at=stale)
    assert fragments[0][0] == HEARTBEAT_STALLED_GLYPH_STYLE
    assert "LLM (stalled)" in fragments[1][1]


def test_stalled_dot_still_winks() -> None:
    """Stalled changes colour, not the wink: even tick = red dot, odd = blank."""
    stale = time.monotonic() - (HEARTBEAT_STALL_FLOOR_S + 5.0)
    on = format_heartbeat(State.AWAITING_LLM, ChatStore(), 0, last_progress_at=stale)
    off = format_heartbeat(State.AWAITING_LLM, ChatStore(), 1, last_progress_at=stale)
    assert on[0][0] == HEARTBEAT_STALLED_GLYPH_STYLE
    assert HEARTBEAT_GLYPH in on[0][1]
    assert off[0][1].strip() == ""
    assert off[0][0] == HEARTBEAT_STALLED_GLYPH_STYLE  # colour persists across the wink


def test_stall_hint_threshold_is_customizable() -> None:
    """Tests can drive the threshold without sleeping 5 s."""
    moment_ago = time.monotonic() - 0.2
    stalled = format_heartbeat(
        State.AWAITING_LLM,
        ChatStore(),
        0,
        last_progress_at=moment_ago,
        stall_threshold_s=0.1,
    )
    fresh = format_heartbeat(
        State.AWAITING_LLM,
        ChatStore(),
        0,
        last_progress_at=moment_ago,
        stall_threshold_s=1.0,
    )
    assert stalled[0][0] == HEARTBEAT_STALLED_GLYPH_STYLE
    assert "stalled" in stalled[1][1]
    assert fresh[0][0] == HEARTBEAT_GLYPH_STYLE
    assert "stalled" not in fresh[1][1]


def test_stall_threshold_derives_from_timeout_with_floor() -> None:
    """Single-knob: stall scales with the slot timeout, floored."""
    assert stall_threshold_for(300.0) == 50.0
    assert stall_threshold_for(600.0) == 100.0
    assert stall_threshold_for(12.0) == HEARTBEAT_STALL_FLOOR_S
    assert stall_threshold_for(0.1) == HEARTBEAT_STALL_FLOOR_S


def test_stall_hint_only_during_awaiting_llm() -> None:
    """Stall semantics apply only to AWAITING_LLM — tool runs are exempt."""
    stale = time.monotonic() - (HEARTBEAT_STALL_FLOOR_S + 5.0)
    store = ChatStore()
    store.add_pending_tool_call("run_shell", '{"command":"sleep 30"}')
    fragments = format_heartbeat(State.AWAITING_EXECUTOR, store, 0, last_progress_at=stale)
    assert fragments[0][0] == HEARTBEAT_GLYPH_STYLE
    assert "stalled" not in fragments[1][1]
