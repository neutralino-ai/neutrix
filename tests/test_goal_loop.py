"""v1.6.0 — native /goal autonomous loop.

Termination paths are the priority: an autonomous loop's failure mode is
non-termination, so the sentinel / cap / Esc / typed-message exits are tested
hardest.
"""
from __future__ import annotations

import asyncio

import pytest
from test_terminal_chat import FakeLLM, _assistant_text, _make_chat, _make_ctx

from neutrix.context_manager import (
    GOAL_DONE_SENTINEL,
    GOAL_REMINDER_MARKER,
    TASK_REMINDER_TAG_OPEN,
    is_goal_reminder,
)
from neutrix.terminal_chat import _GOAL_KICK, GOAL_MAX_STEPS


def _working_rounds(n: int) -> list:
    """n LLM rounds of plain assistant text that never emit the sentinel."""
    return [[_assistant_text(f"working step {i}")] for i in range(n)]


# ---- is_goal_reminder + continue_goal (the CM mechanism) ------------------


def test_is_goal_reminder_discriminates():
    good = f"{TASK_REMINDER_TAG_OPEN}\n{GOAL_REMINDER_MARKER} do X\n</system-reminder>"
    assert is_goal_reminder(good)
    assert not is_goal_reminder("<system-reminder>\nHere are the existing tasks:\n…")
    assert not is_goal_reminder("plain text")
    assert not is_goal_reminder(None)


@pytest.mark.asyncio
async def test_continue_goal_injects_one_reminder_and_drives(tmp_path):
    ctx = _make_ctx(FakeLLM([[_assistant_text("did a thing")]]))
    before = len(ctx.messages)
    await ctx.continue_goal("optimize X")
    new = ctx.messages[before:]
    reminders = [m for m in new if m.get("role") == "user"]
    assert len(reminders) == 1
    assert is_goal_reminder(reminders[0]["content"])
    assert "optimize X" in reminders[0]["content"]
    assert any(m.get("role") == "assistant" and m.get("content") == "did a thing" for m in new)


# ---- termination paths ----------------------------------------------------


@pytest.mark.asyncio
async def test_sentinel_stops_the_loop(tmp_path):
    rounds = [
        [_assistant_text("step a")],
        [_assistant_text("step b")],
        [_assistant_text(f"all done\n{GOAL_DONE_SENTINEL}")],
    ]
    ctx = _make_ctx(FakeLLM(rounds))
    chat, _o, _p = _make_chat(ctx, tmp_path, [])
    chat._goal = "do the thing"
    chat._goal_step = 0
    await chat._run_goal_continuations()
    assert chat._goal is None          # cleared on completion
    assert ctx.llm.rounds == []        # all 3 continuations ran, then the sentinel stopped it


@pytest.mark.asyncio
async def test_cap_stops_the_loop_gracefully(tmp_path):
    # never emits the sentinel → the cap is the hard termination guarantee.
    ctx = _make_ctx(FakeLLM(_working_rounds(GOAL_MAX_STEPS + 5)))
    chat, _o, _p = _make_chat(ctx, tmp_path, [])
    chat._goal = "endless"
    chat._goal_step = 0
    await chat._run_goal_continuations()
    assert chat._goal is None                 # cleared on cap (not a crash)
    assert len(ctx.llm.rounds) == 5           # exactly GOAL_MAX_STEPS continuations ran


@pytest.mark.asyncio
async def test_typed_message_releases_the_goal(tmp_path):
    ctx = _make_ctx(FakeLLM(_working_rounds(10)))
    chat, _o, _p = _make_chat(ctx, tmp_path, [])
    chat._goal = "x"
    chat._goal_step = 0
    chat._goal_interrupt = True   # the input loop sets this on a typed message
    await chat._run_goal_continuations()
    assert chat._goal is None
    assert len(ctx.llm.rounds) == 10   # released immediately — no continuation ran


def test_esc_clears_the_goal(tmp_path):
    ctx = _make_ctx(FakeLLM([]))
    chat, _o, _p = _make_chat(ctx, tmp_path, [])
    chat._goal = "x"
    chat._goal_step = 4
    fired = chat.try_cancel_current_stream()   # the Esc keybinding's cancel_hook
    assert chat._goal is None and chat._goal_step == 0
    assert fired is True         # had_goal → True even with nothing in flight


@pytest.mark.asyncio
async def test_goal_cleared_mid_turn_stops_loop(tmp_path, monkeypatch):
    # mimic Esc during a continuation: the goal is cleared while the turn runs.
    ctx = _make_ctx(FakeLLM([]))
    chat, _o, _p = _make_chat(ctx, tmp_path, [])
    chat._goal = "x"
    chat._goal_step = 0
    calls = {"n": 0}

    async def fake_continue(goal):
        calls["n"] += 1
        chat._goal = None  # Esc cleared it mid-turn

    monkeypatch.setattr(chat.ctx, "continue_goal", fake_continue)
    await chat._run_goal_continuations()
    assert calls["n"] == 1   # one continuation, then saw _goal is None and returned


# ---- sentinel false-positive guard ----------------------------------------


def test_goal_completed_requires_final_line(tmp_path):
    ctx = _make_ctx(FakeLLM([]))
    chat, _o, _p = _make_chat(ctx, tmp_path, [])
    ctx.messages.append({"role": "assistant", "content": f"{GOAL_DONE_SENTINEL}\nstill working"})
    assert chat._goal_completed() is False  # sentinel not the final line
    ctx.messages.append({"role": "assistant", "content": "I'll end with the token when done"})
    assert chat._goal_completed() is False  # echoed, not the final line
    ctx.messages.append({"role": "assistant", "content": f"finished\n{GOAL_DONE_SENTINEL}\n"})
    assert chat._goal_completed() is True   # final non-empty line is the sentinel


def test_goal_completed_ignores_tool_results(tmp_path):
    ctx = _make_ctx(FakeLLM([]))
    chat, _o, _p = _make_chat(ctx, tmp_path, [])
    ctx.messages.append({"role": "assistant", "content": "working"})
    ctx.messages.append({"role": "tool", "tool_call_id": "c1", "content": GOAL_DONE_SENTINEL})
    # the last ASSISTANT message decides — the tool result is ignored.
    assert chat._goal_completed() is False


# ---- command surface ------------------------------------------------------


@pytest.mark.asyncio
async def test_cmd_goal_set_show_clear(tmp_path):
    ctx = _make_ctx(FakeLLM([]))
    chat, _o, _p = _make_chat(ctx, tmp_path, [])
    chat._input_queue = asyncio.Queue()
    await chat._cmd_goal(["optimize", "the", "fit"])
    assert chat._goal == "optimize the fit" and chat._goal_step == 0
    assert chat._input_queue.get_nowait() is _GOAL_KICK   # worker kicked
    await chat._cmd_goal([])           # show — no state change
    assert chat._goal == "optimize the fit"
    await chat._cmd_goal(["clear"])
    assert chat._goal is None
