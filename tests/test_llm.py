"""Tests for the OpenAI-compatible streaming adapter."""
from __future__ import annotations

from collections.abc import AsyncIterator
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


def _chunk(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ]
    )


def _tool_delta(
    *,
    index: int = 0,
    tool_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        id=tool_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeCompletions:
    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self.chunks = chunks
        self.kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> AsyncIterator[SimpleNamespace]:
        self.kwargs = kwargs
        return self._stream()

    async def _stream(self) -> AsyncIterator[SimpleNamespace]:
        for chunk in self.chunks:
            yield chunk


@pytest.mark.asyncio
async def test_openai_chat_llm_streams_tokens_and_final_assistant_message():
    completions = FakeCompletions(
        [
            _chunk(content="hello "),
            _chunk(
                tool_calls=[
                    _tool_delta(tool_id="call_1", name="echo", arguments='{"x"')
                ]
            ),
            _chunk(content="world"),
            _chunk(
                tool_calls=[_tool_delta(arguments=": 1}")],
                finish_reason="tool_calls",
            ),
        ]
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

    assert [event.kind for event in events] == ["token", "token", "assistant"]
    assert [event.data for event in events[:2]] == ["hello ", "world"]
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
    assert completions.kwargs["stream"] is True
    assert completions.kwargs["tools"] == [
        {"type": "function", "function": {"name": "echo"}}
    ]
