"""Tests for the v0.10.4 Smart Advisor (trigger policy + run_once + wiring)."""
from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from neutrix.advisor import Advisor, AdvisorOutcome
from neutrix.config import Config, Slot
from neutrix.context_manager import (
    ADVISOR_TAG_OPEN,
    ContextManager,
    is_advisor_message,
)
from neutrix.executor import Executor
from neutrix.llm import LLMEvent, LLMResponse
from neutrix.store import ChatStore, Task
from neutrix.terminal_chat import (
    TerminalChat,
    recallable_user_turns,
    user_turn_indices,
)


def _slot() -> Slot:
    return Slot(
        name="fast",
        provider="test",
        model="test-model",
        base_url="https://example.test/v1",
        api_key="sk-test",
    )


def _config(tmp_path: Path) -> Config:
    return Config(
        providers={"test": {"base_url": "https://example.test/v1", "api_key": "sk-test"}},
        slots={"fast": {"provider": "test", "model": "test-model"}},
        path=tmp_path / "config.yaml",
    )


class _OneShotLLM:
    """Yields a single assistant message (content and/or tool_calls)."""

    def __init__(self, *, content: str | None = None, tool_calls: list[dict] | None = None) -> None:
        self.message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls is not None:
            self.message["tool_calls"] = tool_calls
        self.calls: list[dict[str, Any]] = []

    def switch(self, slot: Slot) -> None:  # pragma: no cover
        pass

    def stop(self) -> None:  # pragma: no cover
        pass

    async def stream_response(self, *, model, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        yield LLMEvent("assistant", LLMResponse(self.message, "stop"))


def _task_call(name: str, args: str = "{}", call_id: str = "c1") -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


# ---- trigger policy -------------------------------------------------------


def test_periodic_trigger_fires_at_cadence() -> None:
    adv = Advisor(cadence_turns=3)
    assert not adv.should_run()
    adv.note_turn()
    adv.note_turn()
    assert not adv.should_run()
    adv.note_turn()
    assert adv.should_run()


def test_run_lock_blocks_reentry() -> None:
    adv = Advisor(cadence_turns=1)
    adv.note_turn()
    adv._running = True
    assert not adv.should_run()


def test_session_cap_stops_auto_trigger() -> None:
    adv = Advisor(cadence_turns=1, max_runs=2)
    adv._runs = 2
    adv.note_turn()
    assert not adv.should_run()


def test_disabled_advisor_never_runs() -> None:
    adv = Advisor(enabled=False, cadence_turns=1)
    adv.note_turn()
    assert not adv.should_run()


# ---- run_once -------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_once_extracts_task_calls() -> None:
    adv = Advisor()
    llm = _OneShotLLM(tool_calls=[_task_call("TaskUpdate", '{"taskId":"3","status":"completed"}')])
    outcome = await adv.run_once(tasks=(), recent_turns=[], llm=llm, model="m")
    assert outcome.task_calls == (("TaskUpdate", '{"taskId":"3","status":"completed"}'),)
    assert outcome.suggestion is None
    assert adv.runs == 1


@pytest.mark.asyncio
async def test_run_once_extracts_suggestion() -> None:
    adv = Advisor()
    llm = _OneShotLLM(content="You finished task #1; focus on #2 next.")
    outcome = await adv.run_once(tasks=(), recent_turns=[], llm=llm, model="m")
    assert outcome.suggestion == "You finished task #1; focus on #2 next."
    assert outcome.task_calls == ()


@pytest.mark.asyncio
async def test_run_once_filters_non_task_tools() -> None:
    adv = Advisor()
    llm = _OneShotLLM(tool_calls=[_task_call("run_shell", '{"command":"rm -rf /"}')])
    outcome = await adv.run_once(tasks=(), recent_turns=[], llm=llm, model="m")
    assert outcome.task_calls == ()  # non-Task tool dropped (guardrail)


@pytest.mark.asyncio
async def test_run_once_is_reentrant_safe() -> None:
    """A concurrent run_once (e.g. /advise during a periodic run) is dropped.

    Guards the live hole: _worker_loop clears _busy before awaiting the advisor,
    so a forced /advise during the periodic run's network await isn't busy-
    guarded and forced skips should_run — run_once must self-guard.
    """
    adv = Advisor()
    adv._running = True  # a run is "in flight"
    llm = _OneShotLLM(content="should not be called")
    outcome = await adv.run_once(tasks=(), recent_turns=[], llm=llm, model="m")
    assert outcome == AdvisorOutcome(task_calls=(), suggestion=None)
    assert llm.calls == []  # no LLM call made while a run was in flight


@pytest.mark.asyncio
async def test_run_once_advertises_only_task_tools() -> None:
    adv = Advisor()
    llm = _OneShotLLM(content="ok")
    await adv.run_once(tasks=(Task(id="1", subject="x", status="pending"),), recent_turns=[], llm=llm, model="m")
    names = {t["function"]["name"] for t in llm.calls[0]["tools"]}
    assert names == {"TaskCreate", "TaskUpdate", "TaskList"}


# ---- CM injection + markers -----------------------------------------------


def _ctx() -> ContextManager:
    class _Stub:
        def switch(self, slot): ...  # pragma: no cover
        def stop(self): ...  # pragma: no cover
    return ContextManager(
        slot=_slot(),
        llm=_Stub(),
        executor=Executor(),
        store=ChatStore(),
        system_prompt="sp",
        messages=[{"role": "system", "content": "sp"}],
    )


def test_inject_advisor_message_wraps_and_syncs_store() -> None:
    ctx = _ctx()
    ctx.inject_advisor_message("consider deleting task #5")
    last = ctx.messages[-1]
    assert last["role"] == "user"
    assert last["content"] == "<advisor>consider deleting task #5</advisor>"
    assert is_advisor_message(last["content"])
    assert ctx.store.messages[-1].content == last["content"]


def test_advisor_message_excluded_from_recall() -> None:
    msgs = [
        {"role": "system", "content": "sp"},
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": f"{ADVISOR_TAG_OPEN}advice</advisor>"},
    ]
    assert user_turn_indices(msgs) == [1]
    assert recallable_user_turns(msgs) == ["real question"]


# ---- rendering + orchestration --------------------------------------------


def _chat(ctx: ContextManager, tmp_path: Path) -> tuple[TerminalChat, StringIO]:
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=100)
    return (
        TerminalChat(ctx, config=_config(tmp_path), render_markdown=False, console=console),
        output,
    )


@pytest.mark.asyncio
async def test_advisor_message_renders_as_notice(tmp_path: Path) -> None:
    ctx = _ctx()
    ctx.inject_advisor_message("focus on the failing test")
    chat, output = _chat(ctx, tmp_path)
    await chat._render_initial_transcript()
    text = output.getvalue()
    assert "↳ advisor: focus on the failing test" in text
    assert "<advisor>" not in text  # not dumped as a raw user turn


@pytest.mark.asyncio
async def test_apply_outcome_mutates_store_and_injects(tmp_path: Path) -> None:
    ctx = _ctx()
    ctx.store.add_task("do the thing")  # task #1
    chat, output = _chat(ctx, tmp_path)
    outcome = AdvisorOutcome(
        task_calls=(("TaskUpdate", '{"taskId":"1","status":"completed"}'),),
        suggestion="nice work; nothing left",
    )
    await chat._apply_advisor_outcome(outcome)
    # Task channel: the store mutation landed + an audit notice printed.
    assert ctx.store.tasks[0].status == "completed"
    assert "↳ advisor:" in output.getvalue()
    # Suggestion channel: injected through CM as an <advisor> turn.
    assert is_advisor_message(ctx.messages[-1]["content"])


@pytest.mark.asyncio
async def test_maybe_run_advisor_skips_after_cancelled_turn(tmp_path: Path) -> None:
    """No advisor run when the last message is the interrupt marker."""
    ctx = _ctx()
    ctx.messages.append({"role": "user", "content": "[interrupted by user]"})
    chat, _ = _chat(ctx, tmp_path)
    chat.advisor = Advisor(cadence_turns=1)
    chat.advisor.note_turn()
    # Stub the llm so a stray run would be detectable (it must NOT be called).
    called = []
    chat._advisor_llm = _OneShotLLM(content="should not run")

    async def _spy_run(**kwargs):
        called.append(True)
        return AdvisorOutcome((), None)

    chat.advisor.run_once = _spy_run  # type: ignore[assignment]
    await chat._maybe_run_advisor()
    assert called == []  # skipped — last turn was cancelled
