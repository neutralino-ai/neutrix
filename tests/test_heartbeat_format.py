"""Pure-function tests for the heartbeat renderer (v0.9.4).

Cases derived from ``docs/PRDs/v0.9.4-heartbeat.md`` § Acceptance.

After the Phase-2 reopen, the brightness cycle is a 40-frame
truecolor gradient (``HEARTBEAT_CYCLE_FRAMES = 40``, smooth
raised-cosine breath, ~4 s period at the default tick). Tests
assert (a) the cycle is genuinely smooth (≥20 distinct hex
colors), (b) each value is a valid ``fg:#xxxxxx`` style, and (c)
sampling at quarter-cycle points returns 4 distinct values.
"""
from __future__ import annotations

import re

from neutrix.context_manager import State
from neutrix.store import ChatStore, PendingToolCall
from neutrix.terminal_chat import (
    HEARTBEAT_BRIGHTNESS_CYCLE,
    HEARTBEAT_CYCLE_FRAMES,
    HEARTBEAT_GLYPH,
    HEARTBEAT_LABEL_STYLE,
    format_heartbeat,
)

HEX_STYLE_RE = re.compile(r"^fg:#[0-9a-f]{6}$")


def _glyph_fragment(state: State, store: ChatStore, tick: int) -> tuple[str, str]:
    fragments = format_heartbeat(state, store, tick)
    return fragments[0]


def _label_fragment(state: State, store: ChatStore, tick: int) -> tuple[str, str]:
    fragments = format_heartbeat(state, store, tick)
    return fragments[1]


def test_idle_returns_empty() -> None:
    assert format_heartbeat(State.IDLE, ChatStore(), 0) == []


def test_awaiting_llm_label() -> None:
    fragments = format_heartbeat(State.AWAITING_LLM, ChatStore(), 0)
    text = "".join(t for _s, t in fragments)
    assert "LLM" in text
    glyph_style, glyph_text = fragments[0]
    assert HEARTBEAT_GLYPH in glyph_text
    assert glyph_style in HEARTBEAT_BRIGHTNESS_CYCLE


def test_awaiting_executor_label_uses_tool_head() -> None:
    store = ChatStore()
    store.add_pending_tool_call("run_shell", '{"command":"sleep 5"}')
    store.add_pending_tool_call("read_file", '{"path":"x"}')
    assert isinstance(store.pending_tool_calls[0], PendingToolCall)

    fragments = format_heartbeat(State.AWAITING_EXECUTOR, store, 0)
    text = "".join(t for _s, t in fragments)
    assert "tool: run_shell" in text
    assert "read_file" not in text


def test_awaiting_executor_with_no_pending_falls_back() -> None:
    """Defensive: tiny window between state transition and add_pending_tool_call."""
    fragments = format_heartbeat(State.AWAITING_EXECUTOR, ChatStore(), 0)
    text = "".join(t for _s, t in fragments)
    assert "tool" in text


def test_cancelling_label() -> None:
    fragments = format_heartbeat(State.CANCELLING, ChatStore(), 0)
    text = "".join(t for _s, t in fragments)
    assert "cancelling" in text


def test_brightness_cycle_is_truecolor_gradient() -> None:
    """Every value is fg:#rrggbb; cycle has ≥20 distinct shades."""
    assert len(HEARTBEAT_BRIGHTNESS_CYCLE) == HEARTBEAT_CYCLE_FRAMES
    for style in HEARTBEAT_BRIGHTNESS_CYCLE:
        assert HEX_STYLE_RE.match(style), f"not a hex style: {style!r}"
    assert len(set(HEARTBEAT_BRIGHTNESS_CYCLE)) >= 20


def test_brightness_cycles_by_tick() -> None:
    """Four ticks across one half-cycle ascent produce four distinct styles.

    Sampling only the ascending half (frames 0..N/2) avoids the
    mirror-symmetry of the breath curve, where pairs of frames
    equidistant from the midpoint have identical brightness.
    """
    state = State.AWAITING_LLM
    store = ChatStore()
    # Four evenly spaced ticks in the trough→peak ascent.
    ascent = (
        0,
        HEARTBEAT_CYCLE_FRAMES // 8,
        HEARTBEAT_CYCLE_FRAMES // 4,
        3 * HEARTBEAT_CYCLE_FRAMES // 8,
    )
    fragments = [format_heartbeat(state, store, t) for t in ascent]
    glyph_styles = [f[0][0] for f in fragments]
    glyph_texts = [f[0][1] for f in fragments]
    assert len(set(glyph_styles)) == 4, f"expected 4 distinct styles, got {glyph_styles}"
    for style in glyph_styles:
        assert style in HEARTBEAT_BRIGHTNESS_CYCLE
    assert len(set(glyph_texts)) == 1
    assert HEARTBEAT_GLYPH in glyph_texts[0]


def test_brightness_cycles_modulo_cycle_frames() -> None:
    """tick = K and tick = K + HEARTBEAT_CYCLE_FRAMES yield identical style."""
    state = State.AWAITING_LLM
    store = ChatStore()
    for k in (0, 7, 13, HEARTBEAT_CYCLE_FRAMES - 1):
        assert (
            _glyph_fragment(state, store, k)[0]
            == _glyph_fragment(state, store, k + HEARTBEAT_CYCLE_FRAMES)[0]
        )
        assert (
            _glyph_fragment(state, store, k)[0]
            == _glyph_fragment(state, store, k + 3 * HEARTBEAT_CYCLE_FRAMES)[0]
        )


def test_label_style_is_always_bright() -> None:
    """The label fragment style does NOT depend on tick — only the glyph fades."""
    state = State.AWAITING_LLM
    store = ChatStore()
    styles = {
        _label_fragment(state, store, t)[0]
        for t in range(2 * HEARTBEAT_CYCLE_FRAMES)
    }
    assert styles == {HEARTBEAT_LABEL_STYLE}


def test_breathing_curve_is_symmetric_peak_at_midpoint() -> None:
    """frame 0 ≈ frame N ≈ trough; frame N/2 ≈ peak."""
    # Inspect the cycle as numeric grayscale (peak gray > trough gray).
    def _gray(style: str) -> int:
        # Style is "fg:#rrggbb"; r=g=b in this design.
        return int(style[4:6], 16)

    trough_frame = _gray(HEARTBEAT_BRIGHTNESS_CYCLE[0])
    peak_frame = _gray(HEARTBEAT_BRIGHTNESS_CYCLE[HEARTBEAT_CYCLE_FRAMES // 2])
    assert peak_frame > trough_frame
    # Symmetry: frame k and frame N-k should be approximately equal.
    for k in range(1, HEARTBEAT_CYCLE_FRAMES // 2):
        mirror = HEARTBEAT_CYCLE_FRAMES - k
        assert abs(
            _gray(HEARTBEAT_BRIGHTNESS_CYCLE[k])
            - _gray(HEARTBEAT_BRIGHTNESS_CYCLE[mirror])
        ) <= 1  # round-off tolerance
