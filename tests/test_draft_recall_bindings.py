"""Wiring tests for the v0.9.7 recall key bindings.

``tests/test_rewind.py`` exercises the ``RecallState`` state machine in
isolation. This file checks the *binding layer* in
:func:`neutrix.terminal_chat.build_draft_key_bindings` — which key maps to
which handler, the filter conditions that gate Up/Down, and the recall-aware
Esc fallthrough — without spinning up a live prompt_toolkit application.

The handlers are anonymous (``def _``); we retrieve each ``Binding`` by its
key value and invoke ``.handler``/``.filter`` directly. Filters call
``get_app()``, so they are evaluated inside a ``set_app`` context with a
stub app exposing ``current_buffer``.
"""

from __future__ import annotations

import pytest

prompt_toolkit = pytest.importorskip("prompt_toolkit")

from prompt_toolkit.application.current import set_app  # noqa: E402
from prompt_toolkit.buffer import Buffer  # noqa: E402

from neutrix.terminal_chat import RecallState, build_draft_key_bindings  # noqa: E402


class _StubApp:
    """Minimal stand-in for the running Application (filters read current_buffer)."""

    def __init__(self, buffer: Buffer) -> None:
        self.current_buffer = buffer
        self.exited: tuple[str | None, BaseException | None] | None = None

    def exit(self, result: str | None = None, exception: BaseException | None = None) -> None:
        self.exited = (result, exception)

    def invalidate(self) -> None:  # pragma: no cover - unused by these paths
        pass


class _Event:
    """Stub key-press event: carries the active buffer and app."""

    def __init__(self, buffer: Buffer) -> None:
        self.current_buffer = buffer
        self.app = _StubApp(buffer)


def _binding(kb, key_value: str):
    """Return the single binding whose only key has ``.value == key_value``."""
    matches = [
        b for b in kb.bindings if len(b.keys) == 1 and getattr(b.keys[0], "value", None) == key_value
    ]
    assert len(matches) == 1, f"expected exactly one {key_value!r} binding, got {len(matches)}"
    return matches[0]


def _filter_with_buffer(binding, buffer: Buffer) -> bool:
    """Evaluate a binding's filter as prompt_toolkit would, with ``buffer`` active."""
    with set_app(_StubApp(buffer)):
        return bool(binding.filter())


def _kb(recall_state: RecallState, turns: list[str], cancel_hook=None):
    return build_draft_key_bindings(
        cancel_hook=cancel_hook,
        recall_provider=lambda: turns,
        recall_state=recall_state,
    )


# ---- key presence -----------------------------------------------------------


def test_recall_bindings_present_only_with_provider() -> None:
    """Up/Down recall bindings appear iff recall_provider + recall_state given."""
    with_recall = _kb(RecallState(), ["a"])
    keys_with = {b.keys[0].value for b in with_recall.bindings if len(b.keys) == 1}
    assert {"up", "down"} <= keys_with

    without = build_draft_key_bindings()
    keys_without = {b.keys[0].value for b in without.bindings if len(b.keys) == 1}
    assert "up" not in keys_without and "down" not in keys_without


# ---- Up filter: starts recall only on an empty buffer -----------------------


def test_up_filter_true_on_empty_buffer() -> None:
    binding = _binding(_kb(RecallState(), ["a", "b"]), "up")
    assert _filter_with_buffer(binding, Buffer()) is True


def test_up_filter_false_with_text_and_recall_inactive() -> None:
    """Non-empty buffer + not recalling → Up does ordinary multi-line cursor-up."""
    binding = _binding(_kb(RecallState(), ["a", "b"]), "up")
    buf = Buffer()
    buf.text = "half-typed"
    assert _filter_with_buffer(binding, buf) is False


def test_up_filter_true_with_text_once_recall_active() -> None:
    """Once recalling, Up keeps walking even though the buffer is non-empty."""
    rs = RecallState()
    rs.up(["a", "b"])  # activate
    binding = _binding(_kb(rs, ["a", "b"]), "up")
    buf = Buffer()
    buf.text = "b"  # recalled text is in the buffer
    assert _filter_with_buffer(binding, buf) is True


# ---- Down filter: only while recalling --------------------------------------


def test_down_filter_only_when_recall_active() -> None:
    rs = RecallState()
    binding = _binding(_kb(rs, ["a", "b"]), "down")
    assert _filter_with_buffer(binding, Buffer()) is False
    rs.up(["a", "b"])
    assert _filter_with_buffer(binding, Buffer()) is True


# ---- handlers walk turns into the buffer ------------------------------------


def test_up_handler_fills_buffer_newest_first() -> None:
    rs = RecallState()
    binding = _binding(_kb(rs, ["old", "mid", "new"]), "up")
    ev = _Event(Buffer())
    binding.handler(ev)
    assert ev.current_buffer.text == "new"
    assert ev.current_buffer.cursor_position == len("new")
    binding.handler(ev)
    assert ev.current_buffer.text == "mid"


def test_down_handler_walks_forward_then_clears() -> None:
    rs = RecallState()
    turns = ["old", "mid", "new"]
    up = _binding(_kb(rs, turns), "up")
    down = _binding(_kb(rs, turns), "down")
    ev = _Event(Buffer())
    up.handler(ev)  # new
    up.handler(ev)  # mid
    down.handler(ev)
    assert ev.current_buffer.text == "new"
    down.handler(ev)
    assert ev.current_buffer.text == ""  # back to the fresh draft
    assert not rs.active


# ---- Esc: exits recall when active, else falls through to cancel ------------


def test_escape_exits_recall_and_clears_without_cancelling() -> None:
    calls: list[bool] = []
    rs = RecallState()
    rs.up(["a", "b"])  # recall active
    esc = _binding(_kb(rs, ["a", "b"], cancel_hook=lambda: calls.append(True) or True), "escape")
    buf = Buffer()
    buf.text = "b"
    esc.handler(_Event(buf))
    assert buf.text == ""
    assert not rs.active
    assert calls == []  # cancel NOT invoked while recall was active


def test_escape_falls_through_to_cancel_when_not_recalling() -> None:
    calls: list[bool] = []
    rs = RecallState()  # inactive
    esc = _binding(_kb(rs, ["a", "b"], cancel_hook=lambda: calls.append(True) or True), "escape")
    esc.handler(_Event(Buffer()))
    assert calls == [True]  # cancel fired


# ---- Enter resets recall before submitting ----------------------------------


def test_enter_resets_recall_and_submits() -> None:
    rs = RecallState()
    rs.up(["a", "b"])  # recall active
    enter = _binding(_kb(rs, ["a", "b"]), "c-m")
    buf = Buffer()
    buf.text = "b"
    ev = _Event(buf)
    enter.handler(ev)
    assert not rs.active  # recall cleared on submit
    assert ev.app.exited == ("b", None)
