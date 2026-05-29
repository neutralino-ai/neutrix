"""Tests for the v0.10.0 subagent framework (run_subagent + Agent tool).

The cancel test fires the ``threading.Event`` directly (the isolated unit),
NOT an Esc integration — per the corrected cancel data flow, on a real Esc the
parent drive-task cancel abandons the ``to_thread`` future and the
``[cancelled by user]`` marker reaches the parent through v0.9.3's pairing
layer, not through this return value.
"""
from __future__ import annotations

import threading
from typing import Any

import pytest

from neutrix.config import Slot
from neutrix.llm import LLMEvent, LLMResponse
from neutrix.subagent import (
    SUBAGENT_MAX_RESULT_CHARS,
    SubagentResult,
    run_subagent,
)


def _slot() -> Slot:
    return Slot(
        name="fast",
        provider="test",
        model="test-model",
        base_url="https://example.test/v1",
        api_key="sk-test",
    )


class _ScriptedLLM:
    """Yields pre-canned rounds; one ``rounds`` entry consumed per LLM call."""

    def __init__(self, rounds: list[list[LLMEvent]]) -> None:
        self.rounds = list(rounds)
        self.calls: list[dict[str, Any]] = []

    def switch(self, slot: Slot) -> None:  # pragma: no cover - unused
        pass

    def stop(self) -> None:  # pragma: no cover - unused
        pass

    async def stream_response(self, *, model, messages, tools=None):
        self.calls.append({"tools": tools})
        batch = self.rounds.pop(0) if self.rounds else [_text("")]
        for event in batch:
            yield event


class _LoopingLLM:
    """Always calls a tool — used to exercise the turn cap."""

    def __init__(self) -> None:
        self.calls = 0

    def switch(self, slot: Slot) -> None:  # pragma: no cover
        pass

    def stop(self) -> None:  # pragma: no cover
        pass

    async def stream_response(self, *, model, messages, tools=None):
        self.calls += 1
        yield _tool("list_dir", "{}", call_id=f"c{self.calls}")


def _text(text: str) -> LLMEvent:
    return LLMEvent(
        "assistant",
        LLMResponse(message={"role": "assistant", "content": text}, finish_reason="stop"),
    )


def _tool(name: str, args: str, call_id: str = "c1") -> LLMEvent:
    return LLMEvent(
        "assistant",
        LLMResponse(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}
                ],
            },
            finish_reason="tool_calls",
        ),
    )


@pytest.mark.asyncio
async def test_subagent_runs_to_completion() -> None:
    llm = _ScriptedLLM([[_text("the answer")]])
    result = await run_subagent(
        user_prompt="do it",
        slot=_slot(),
        llm=llm,
        tool_names=frozenset({"read_file"}),
    )
    assert isinstance(result, SubagentResult)
    assert result.final_text == "the answer"
    assert result.turn_count == 1
    assert result.cancelled is False
    assert result.error is None


@pytest.mark.asyncio
async def test_subagent_dispatches_tools(monkeypatch) -> None:
    """Subagent calls a tool, gets the result, then finalizes."""
    monkeypatch.setattr(
        "neutrix.executor.dispatch",
        lambda name, arguments, **_: f"ran {name}",
    )
    llm = _ScriptedLLM([[_tool("list_dir", "{}")], [_text("summarized")]])
    result = await run_subagent(
        user_prompt="explore",
        slot=_slot(),
        llm=llm,
        tool_names=frozenset({"list_dir"}),
    )
    assert result.final_text == "summarized"
    assert result.turn_count == 2
    # The subagent saw only its scoped tool set.
    names = {t["function"]["name"] for t in llm.calls[0]["tools"]}
    assert names == {"list_dir"}


@pytest.mark.asyncio
async def test_subagent_hits_turn_cap(monkeypatch) -> None:
    monkeypatch.setattr(
        "neutrix.executor.dispatch",
        lambda name, arguments, **_: f"ran {name}",
    )
    llm = _LoopingLLM()
    result = await run_subagent(
        user_prompt="loop forever",
        slot=_slot(),
        llm=llm,
        tool_names=frozenset({"list_dir"}),
        max_turns=3,
    )
    assert llm.calls == 3  # capped at 3 LLM rounds
    assert "3-turn limit" in result.final_text


@pytest.mark.asyncio
async def test_subagent_cancel_sets_flag() -> None:
    """Firing the cancel event mid-run unwinds the subagent (isolated unit)."""

    class _SuspendLLM:
        def switch(self, slot): ...  # pragma: no cover
        def stop(self): self.stopped = True
        async def stream_response(self, *, model, messages, tools=None):
            import asyncio

            await asyncio.sleep(10)  # parked until cancelled
            yield _text("never")  # pragma: no cover

    event = threading.Event()
    llm = _SuspendLLM()

    import asyncio

    async def _fire_soon() -> None:
        await asyncio.sleep(0.1)
        event.set()

    fire = asyncio.create_task(_fire_soon())
    result = await run_subagent(
        user_prompt="hang",
        slot=_slot(),
        llm=llm,
        tool_names=frozenset({"read_file"}),
        cancel_event=event,
    )
    await fire
    assert result.cancelled is True


@pytest.mark.asyncio
async def test_subagent_result_truncated() -> None:
    big = "x" * (SUBAGENT_MAX_RESULT_CHARS + 500)
    llm = _ScriptedLLM([[_text(big)]])
    result = await run_subagent(
        user_prompt="emit a lot",
        slot=_slot(),
        llm=llm,
        tool_names=frozenset(),
    )
    assert len(result.final_text) <= SUBAGENT_MAX_RESULT_CHARS + 64
    assert result.final_text.endswith("truncated]")


# ---- Agent tool dispatch path ---------------------------------------------


def test_agent_tool_blocks_recursion() -> None:
    """The contextvar backstop makes a nested Agent call return an ERROR."""
    from neutrix.tools import _agent, _inside_subagent

    token = _inside_subagent.set(True)
    try:
        out = _agent("desc", "prompt", slot=_slot())
    finally:
        _inside_subagent.reset(token)
    assert out.startswith("ERROR:")
    assert "single-level" in out


def test_agent_tool_rejects_unknown_subagent_type() -> None:
    from neutrix.tools import _agent

    out = _agent("desc", "prompt", subagent_type="code-reviewer", slot=_slot())
    assert out.startswith("ERROR:")
    assert "general-purpose" in out


def test_agent_tool_returns_subagent_output(monkeypatch) -> None:
    """Full path through the Agent tool func with a stubbed run_subagent."""
    import neutrix.subagent as subagent_mod

    async def _fake_run(**kwargs):
        return SubagentResult(final_text="delegated result", turn_count=2)

    monkeypatch.setattr(subagent_mod, "run_subagent", _fake_run)
    # Stub the LLM build so no real client is constructed.
    monkeypatch.setattr("neutrix.llm.OpenAIChatLLM", lambda slot: object())

    from neutrix.tools import _agent

    out = _agent("read files", "summarize foo.py", slot=_slot())
    assert out == "delegated result"


def test_subagent_tool_names_excludes_agent() -> None:
    from neutrix.tools import BUILTIN_TOOLS, subagent_tool_names

    names = subagent_tool_names()
    assert "Agent" not in names
    assert "Agent" in BUILTIN_TOOLS  # it IS a builtin, just not for subagents
    assert "read_file" in names


def test_agent_tool_full_path_with_stub_llm(monkeypatch) -> None:
    """End-to-end: _agent -> asyncio.run -> REAL run_subagent, only LLM stubbed.

    Exercises the sync-tool-in-worker-thread driving an async subagent on its
    own loop (the architecture's load-bearing seam), without a network call.
    """
    scripted = _ScriptedLLM([[_text("worker done")]])
    monkeypatch.setattr("neutrix.llm.OpenAIChatLLM", lambda slot: scripted)

    from neutrix.tools import _agent

    out = _agent("tiny task", "answer 2+2", slot=_slot())
    assert out == "worker done"
    # The subagent's LLM was advertised a tool set without Agent.
    advertised = {t["function"]["name"] for t in scripted.calls[0]["tools"]}
    assert "Agent" not in advertised
    assert "read_file" in advertised
