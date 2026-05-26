"""Tests for the OpenAI-compatible streaming adapter."""
from __future__ import annotations

import asyncio
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


def _chunk(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    delta_kwargs: dict[str, Any] = {"content": content}
    if tool_calls is not None:
        delta_kwargs["tool_calls"] = tool_calls
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(**delta_kwargs),
                finish_reason=finish_reason,
            )
        ]
    )


def _tool_call_delta(
    *,
    index: int = 0,
    tool_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    fn_kwargs: dict[str, Any] = {}
    if name is not None:
        fn_kwargs["name"] = name
    if arguments is not None:
        fn_kwargs["arguments"] = arguments
    return SimpleNamespace(
        index=index,
        id=tool_id,
        type="function" if name is not None else None,
        function=SimpleNamespace(**fn_kwargs) if fn_kwargs else None,
    )


class FakeAsyncStream:
    """Minimal stand-in for the OpenAI SDK's AsyncStream.

    Iterates over a pre-built chunk list; tracks ``close()`` calls so
    cancellation tests can assert the SDK was told to tear down.
    """

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = list(chunks)
        self.close_called = False
        self._closed = False

    def __aiter__(self) -> FakeAsyncStream:
        return self

    async def __anext__(self) -> Any:
        if self._closed or not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    def close(self) -> None:
        self.close_called = True
        self._closed = True


class FakeCompletions:
    """Pretends to be ``client.chat.completions``."""

    def __init__(self, stream: FakeAsyncStream) -> None:
        self.stream = stream
        self.kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> FakeAsyncStream:
        self.kwargs = kwargs
        return self.stream


class SuspendingStream:
    """Stream that blocks indefinitely until ``close()`` is called."""

    def __init__(self) -> None:
        self._closed = asyncio.Event()
        self.close_called = False

    def __aiter__(self) -> SuspendingStream:
        return self

    async def __anext__(self) -> Any:
        await self._closed.wait()
        raise StopAsyncIteration

    def close(self) -> None:
        self.close_called = True
        self._closed.set()


@pytest.mark.asyncio
async def test_streaming_yields_token_events_then_assistant():
    stream = FakeAsyncStream(
        [
            _chunk(content="he"),
            _chunk(content="llo"),
            _chunk(finish_reason="stop"),
        ]
    )
    completions = FakeCompletions(stream)
    llm = OpenAIChatLLM(_slot())
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    events = [
        event
        async for event in llm.stream_response(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )
    ]

    assert [event.kind for event in events] == ["token", "token", "assistant"]
    assert [event.data for event in events[:2]] == ["he", "llo"]
    response = events[-1].data
    assert isinstance(response, LLMResponse)
    assert response.finish_reason == "stop"
    assert response.message == {"role": "assistant", "content": "hello"}
    assert completions.kwargs is not None
    assert completions.kwargs["stream"] is True


@pytest.mark.asyncio
async def test_streaming_accumulates_tool_calls_across_deltas():
    stream = FakeAsyncStream(
        [
            _chunk(
                tool_calls=[
                    _tool_call_delta(
                        tool_id="call_1",
                        name="echo",
                        arguments='{"x":',
                    )
                ],
            ),
            _chunk(
                tool_calls=[_tool_call_delta(arguments=" 1}")],
            ),
            _chunk(finish_reason="tool_calls"),
        ]
    )
    completions = FakeCompletions(stream)
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

    # Tool-only stream: no content tokens, just the final assistant event.
    assert [event.kind for event in events] == ["assistant"]
    response = events[-1].data
    assert response.finish_reason == "tool_calls"
    assert response.message["content"] is None
    assert response.message["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "echo", "arguments": '{"x": 1}'},
        }
    ]


@pytest.mark.asyncio
async def test_streaming_handles_dict_chunk_payloads():
    """Mirrors what some providers emit through the OpenAI gateway —
    dict payloads instead of SimpleNamespace-style attrs."""
    stream = FakeAsyncStream(
        [
            {"choices": [{"delta": {"content": "dict "}}]},
            {"choices": [{"delta": {"content": "payload"}}]},
            {"choices": [{"finish_reason": "stop", "delta": {}}]},
        ]
    )
    completions = FakeCompletions(stream)
    llm = OpenAIChatLLM(_slot())
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    events = [
        event
        async for event in llm.stream_response(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )
    ]
    assert [event.kind for event in events] == ["token", "token", "assistant"]
    response = events[-1].data
    assert response.message["content"] == "dict payload"


@pytest.mark.asyncio
async def test_ihep_anthropic_request_sends_system_prompt_via_sdk_extra_body():
    stream = FakeAsyncStream(
        [
            _chunk(content="ok"),
            _chunk(finish_reason="stop"),
        ]
    )
    completions = FakeCompletions(stream)
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

    assert [event.kind for event in events] == ["token", "assistant"]
    assert completions.kwargs is not None
    assert completions.kwargs["stream"] is True
    assert completions.kwargs["messages"] == [{"role": "user", "content": "Say ok."}]
    assert completions.kwargs["extra_body"] == {"system": "Be brief."}


@pytest.mark.asyncio
async def test_ihep_anthropic_request_still_uses_openai_sdk_completion_create():
    stream = FakeAsyncStream(
        [
            _chunk(content="ok"),
            _chunk(finish_reason="stop"),
        ]
    )
    completions = FakeCompletions(stream)
    llm = OpenAIChatLLM(_ihep_claude_slot())
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    events = [
        event
        async for event in llm.stream_response(
            model="anthropic/claude-opus-4-7",
            messages=[{"role": "user", "content": "Say ok."}],
        )
    ]

    assert [event.kind for event in events] == ["token", "assistant"]
    assert completions.kwargs is not None
    assert completions.kwargs["messages"] == [{"role": "user", "content": "Say ok."}]
    assert "extra_body" not in completions.kwargs


# ---- v0.9.2: LLM.stop() -----------------------------------------------------


def test_stop_on_idle_llm_is_noop():
    """``stop()`` with no active stream must not raise."""
    llm = OpenAIChatLLM(_slot())
    # No active stream ever attached — must be a clean no-op.
    llm.stop()
    assert llm._active_stream is None


@pytest.mark.asyncio
async def test_stop_closes_active_stream_and_unblocks_iterator():
    """The v0.9.2 cancel-broadcast contract: ``llm.stop()`` calls
    ``close()`` on the active stream so the iterator's next
    ``__anext__`` exits cleanly within a short budget."""
    stream = SuspendingStream()
    completions = FakeCompletions(stream)
    llm = OpenAIChatLLM(_slot())
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    started = asyncio.Event()

    async def drive() -> list[Any]:
        events: list[Any] = []
        async for event in llm.stream_response(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        ):
            started.set()
            events.append(event)
        return events

    task = asyncio.create_task(drive())
    # Wait until the stream is attached on the LLM. Polling kept tight
    # (~10 ms ticks) so the test cost stays small.
    for _ in range(50):
        await asyncio.sleep(0.01)
        if llm._active_stream is not None:
            break
    assert llm._active_stream is stream

    llm.stop()
    assert stream.close_called is True
    events = await asyncio.wait_for(task, timeout=0.5)
    # No tokens were yielded (the stream blocked); the trailing
    # assistant event still fires once the iterator unwinds, with an
    # empty assistant message.
    assert [event.kind for event in events] == ["assistant"]
    response = events[-1].data
    assert response.message == {"role": "assistant", "content": None}
