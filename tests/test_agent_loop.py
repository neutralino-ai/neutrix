"""Tests for the model/tool continuation loop."""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from neutrix.agent_loop import Agent
from neutrix.config import Slot
from neutrix.llm import LLMEvent, LLMResponse


def _slot() -> Slot:
    return Slot(
        name="fast",
        provider="test",
        model="test-model",
        base_url="https://example.test/v1",
        api_key="sk-test",
    )


def _ihep_claude_slot() -> Slot:
    return Slot(
        name="strong",
        provider="ihep",
        model="anthropic/claude-opus-4-7",
        base_url="https://aiapi.ihep.ac.cn/apiv2/",
        api_key="sk-test",
    )


class FakeLLM:
    def __init__(self, rounds: list[list[LLMEvent]]) -> None:
        self.rounds = rounds
        self.calls: list[dict[str, Any]] = []
        self.switched_to: Slot | None = None

    def switch(self, slot: Slot) -> None:
        self.switched_to = slot

    async def stream_response(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ):
        self.calls.append(
            {
                "model": model,
                "messages": [dict(message) for message in messages],
                "tools": tools,
            }
        )
        for event in self.rounds.pop(0):
            yield event


@pytest.mark.asyncio
async def test_agent_loop_appends_user_and_assistant_once():
    llm = FakeLLM(
        [
            [
                LLMEvent("token", "hello"),
                LLMEvent(
                    "assistant",
                    LLMResponse(
                        {"role": "assistant", "content": "hello"},
                        finish_reason="stop",
                    ),
                ),
            ]
        ]
    )
    agent = Agent(slot=_slot(), use_tools=False, llm=llm)

    events = [event async for event in agent.stream_reply("hi")]

    assert [event.kind for event in events] == ["token", "done"]
    assert [message["role"] for message in agent.messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert agent.messages[1]["content"] == "hi"
    assert agent.messages[2]["content"] == "hello"
    assert llm.calls[0]["messages"][-1] == {"role": "user", "content": "hi"}
    assert llm.calls[0]["tools"] is None


@pytest.mark.asyncio
async def test_agent_loop_emits_final_assistant_when_no_tokens_streamed():
    llm = FakeLLM(
        [
            [
                LLMEvent(
                    "assistant",
                    LLMResponse(
                        {"role": "assistant", "content": "final only"},
                        finish_reason="stop",
                    ),
                ),
            ]
        ]
    )
    agent = Agent(slot=_slot(), use_tools=False, llm=llm)

    events = [event async for event in agent.stream_reply("hi")]

    assert [(event.kind, event.data) for event in events] == [
        ("assistant", "final only"),
        ("done", None),
    ]
    assert agent.messages[-1] == {"role": "assistant", "content": "final only"}


@pytest.mark.asyncio
async def test_agent_loop_omits_openai_tools_for_ihep_anthropic_models(monkeypatch):
    monkeypatch.setattr(
        "neutrix.agent_loop.get_schemas",
        lambda: [{"type": "function", "function": {"name": "echo"}}],
    )
    llm = FakeLLM(
        [
            [
                LLMEvent(
                    "assistant",
                    LLMResponse(
                        {"role": "assistant", "content": "hello"},
                        finish_reason="stop",
                    ),
                ),
            ]
        ]
    )
    agent = Agent(slot=_ihep_claude_slot(), use_tools=True, llm=llm)

    events = [event async for event in agent.stream_reply("hi")]

    assert [(event.kind, event.data) for event in events] == [
        ("assistant", "hello"),
        ("done", None),
    ]
    assert llm.calls[0]["tools"] is None
    assert not agent.effective_tools_enabled()


@pytest.mark.asyncio
async def test_agent_loop_runs_tools_then_samples_follow_up(monkeypatch):
    monkeypatch.setattr(
        "neutrix.agent_loop.get_schemas",
        lambda: [{"type": "function", "function": {"name": "echo"}}],
    )
    monkeypatch.setattr(
        "neutrix.agent_loop.dispatch",
        lambda name, arguments: f"{name}:{arguments}",
    )
    llm = FakeLLM(
        [
            [
                LLMEvent(
                    "assistant",
                    LLMResponse(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "echo",
                                        "arguments": '{"x": 1}',
                                    },
                                }
                            ],
                        },
                        finish_reason="tool_calls",
                    ),
                )
            ],
            [
                LLMEvent("token", "done"),
                LLMEvent(
                    "assistant",
                    LLMResponse(
                        {"role": "assistant", "content": "done"},
                        finish_reason="stop",
                    ),
                ),
            ],
        ]
    )
    agent = Agent(slot=_slot(), use_tools=True, llm=llm)

    events = [event async for event in agent.stream_reply("hi")]

    assert [event.kind for event in events] == [
        "tool_call",
        "tool_result",
        "token",
        "done",
    ]
    assert [message["role"] for message in agent.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert agent.messages[3]["tool_call_id"] == "call_1"
    assert agent.messages[3]["content"] == 'echo:{"x": 1}'
    assert len(llm.calls) == 2
    assert llm.calls[1]["messages"][-1]["role"] == "tool"


@pytest.mark.asyncio
async def test_agent_loop_tool_dispatch_does_not_block_event_loop(monkeypatch):
    monkeypatch.setattr(
        "neutrix.agent_loop.get_schemas",
        lambda: [{"type": "function", "function": {"name": "slow"}}],
    )

    def slow_dispatch(name: str, arguments: str) -> str:
        time.sleep(0.2)
        return f"{name}:{arguments}"

    monkeypatch.setattr("neutrix.agent_loop.dispatch", slow_dispatch)
    llm = FakeLLM(
        [
            [
                LLMEvent(
                    "assistant",
                    LLMResponse(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "slow",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        finish_reason="tool_calls",
                    ),
                )
            ],
            [
                LLMEvent(
                    "assistant",
                    LLMResponse(
                        {"role": "assistant", "content": "done"},
                        finish_reason="stop",
                    ),
                ),
            ],
        ]
    )
    agent = Agent(slot=_slot(), use_tools=True, llm=llm)

    async def collect_events():
        return [event async for event in agent.stream_reply("hi")]

    task = asyncio.create_task(collect_events())
    start = time.perf_counter()
    await asyncio.sleep(0.05)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.15
    events = await asyncio.wait_for(task, timeout=1)
    assert [event.kind for event in events] == [
        "tool_call",
        "tool_result",
        "assistant",
        "done",
    ]
