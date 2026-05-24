"""Tests for the OpenAI-compatible final-response adapter."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from neutrix.config import Slot
from neutrix.llm import LLMResponse, OpenAIChatLLM


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


def _completion(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ]
    )


def _tool_call(
    *,
    tool_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=tool_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeCompletions:
    def __init__(self, completion: Any) -> None:
        self.completion = completion
        self.kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.completion


@pytest.mark.asyncio
async def test_openai_chat_llm_emits_final_assistant_message_without_tokens():
    completions = FakeCompletions(
        _completion(
            content="hello world",
            tool_calls=[
                _tool_call(
                    tool_id="call_1",
                    name="echo",
                    arguments='{"x": 1}',
                )
            ],
            finish_reason="tool_calls",
        )
    )
    llm = OpenAIChatLLM(_slot())
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    events = [
        event
        async for event in llm.stream_response(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "echo"}}],
        )
    ]

    assert [event.kind for event in events] == ["assistant"]
    response = events[-1].data
    assert isinstance(response, LLMResponse)
    assert response.finish_reason == "tool_calls"
    assert response.message["content"] == "hello world"
    assert response.message["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "echo", "arguments": '{"x": 1}'},
        }
    ]
    assert completions.kwargs is not None
    assert completions.kwargs["stream"] is False
    assert completions.kwargs["tools"] == [
        {"type": "function", "function": {"name": "echo"}}
    ]


@pytest.mark.asyncio
async def test_openai_chat_llm_handles_dict_completion_payloads():
    completions = FakeCompletions(
        {
            "choices": [
                {
                    "message": {
                        "content": "dict payload",
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "arguments": '{"ok": true}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    llm = OpenAIChatLLM(_slot())
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    events = [
        event
        async for event in llm.stream_response(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )
    ]

    response = events[-1].data
    assert response.message == {
        "role": "assistant",
        "content": "dict payload",
        "tool_calls": [
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "echo", "arguments": '{"ok": true}'},
            }
        ],
    }


@pytest.mark.asyncio
async def test_ihep_anthropic_request_sends_system_prompt_via_sdk_extra_body():
    completions = FakeCompletions(_completion(content="ok", finish_reason="stop"))
    llm = OpenAIChatLLM(_ihep_claude_slot())
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    events = [
        event
        async for event in llm.stream_response(
            model="anthropic/claude-opus-4-7",
            messages=[
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Say ok."},
            ],
        )
    ]

    assert [event.kind for event in events] == ["assistant"]
    assert completions.kwargs is not None
    assert completions.kwargs["stream"] is False
    assert completions.kwargs["messages"] == [{"role": "user", "content": "Say ok."}]
    assert completions.kwargs["extra_body"] == {"system": "Be brief."}


@pytest.mark.asyncio
async def test_ihep_anthropic_request_still_uses_openai_sdk_completion_create():
    completions = FakeCompletions(_completion(content="ok", finish_reason="stop"))
    llm = OpenAIChatLLM(_ihep_claude_slot())
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    events = [
        event
        async for event in llm.stream_response(
            model="anthropic/claude-opus-4-7",
            messages=[{"role": "user", "content": "Say ok."}],
        )
    ]

    assert [event.kind for event in events] == ["assistant"]
    assert completions.kwargs is not None
    assert completions.kwargs["messages"] == [{"role": "user", "content": "Say ok."}]
    assert "extra_body" not in completions.kwargs
