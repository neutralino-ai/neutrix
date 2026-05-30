"""Tests for the streaming OpenAI-compatible adapter (v0.10.1) + pairing layer."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from neutrix.config import Slot
from neutrix.llm import (
    CANCELLED_TOOL_RESULT,
    EMPTY_ASSISTANT_PLACEHOLDER,
    INTERRUPTED_BY_USER_MARKER,
    MISSING_TOOL_RESULT,
    LLMResponse,
    OpenAIChatLLM,
    _ensure_sdk_compliant,
    _ensure_tool_result_pairing,
    _repair_empty_assistants,
)


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


def _content_chunk(text: str | None, finish_reason: str | None = None) -> SimpleNamespace:
    """One streaming chunk carrying a content delta."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )
        ]
    )


def _tool_chunk(
    *,
    index: int = 0,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    """One streaming chunk carrying a tool_call delta (index-keyed)."""
    fn = SimpleNamespace(name=name, arguments=arguments)
    tc = SimpleNamespace(index=index, id=call_id, type="function", function=fn)
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=[tc]),
                finish_reason=finish_reason,
            )
        ]
    )


class FakeStream:
    """Async-iterable stand-in for openai's ``AsyncStream``; supports close()."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = list(chunks)
        self.closed = False

    def __aiter__(self) -> FakeStream:
        return self

    async def __anext__(self) -> Any:
        if self.closed or not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    def close(self) -> None:
        self.closed = True


class FakeCompletions:
    """Pretends to be ``client.chat.completions`` with streaming create."""

    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks
        self.kwargs: dict[str, Any] | None = None
        self.stream: FakeStream | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        self.stream = FakeStream(self.chunks)
        return self.stream


class HangingStream:
    """Async stream that parks on __anext__ until close() is called."""

    def __init__(self) -> None:
        self.closed = asyncio.Event()

    def __aiter__(self) -> HangingStream:
        return self

    async def __anext__(self) -> Any:
        await self.closed.wait()
        raise StopAsyncIteration

    def close(self) -> None:
        self.closed.set()


class HangingCompletions:
    """create() returns a stream that hangs until closed (tests stop())."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None
        self.entered = asyncio.Event()
        self.stream = HangingStream()

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        self.entered.set()
        return self.stream


# ---- pairing function (pure) -----------------------------------------------


def _tool_call(call_id: str, name: str = "echo", arguments: str = "{}") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_pairing_dedup_keeps_first_tool_result_only():
    """Two role:tool messages with the same tool_call_id → drop the second."""
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("c1")],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "first"},
        {"role": "tool", "tool_call_id": "c1", "content": "second"},
    ]
    out = _ensure_tool_result_pairing(messages)
    tool_messages = [m for m in out if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "first"


def test_pairing_synthesizes_missing_when_no_cancel_context():
    """Assistant with orphan tool_call, no [interrupted by user] marker
    → synthesize [tool result missing] immediately after."""
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("c1")],
        },
        {"role": "user", "content": "follow-up"},
    ]
    out = _ensure_tool_result_pairing(messages)
    assert out[2] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": MISSING_TOOL_RESULT,
    }
    # The follow-up user message is now at index 3, shifted by one.
    assert out[3] == {"role": "user", "content": "follow-up"}


def test_pairing_synthesizes_cancelled_when_interrupt_marker_present():
    """Assistant with orphan tool_call, [interrupted by user] marker AFTER
    → synthesize [cancelled by user] immediately after the orphan."""
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("c1")],
        },
        {"role": "user", "content": INTERRUPTED_BY_USER_MARKER},
        {"role": "user", "content": "instead, just ls"},
    ]
    out = _ensure_tool_result_pairing(messages)
    # Synthetic inserted at index 2 (immediately after the assistant).
    assert out[2] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": CANCELLED_TOOL_RESULT,
    }
    assert out[3]["content"] == INTERRUPTED_BY_USER_MARKER
    assert out[4]["content"] == "instead, just ls"


def test_pairing_synthesize_only_orphans_when_partial_pair():
    """Two tool_calls in one assistant message, one paired, one orphan →
    synthesize ONLY the orphan; the existing tool result stays put."""
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("c1"), _tool_call("c2")],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "first result"},
    ]
    out = _ensure_tool_result_pairing(messages)
    # The synthetic for c2 is inserted immediately after the assistant,
    # i.e. BEFORE the c1 tool result.
    assert out[2] == {
        "role": "tool",
        "tool_call_id": "c2",
        "content": MISSING_TOOL_RESULT,
    }
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "first result"}


def test_pairing_is_pure_on_input():
    """Input list identity preserved; output is a new list."""
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("c1")],
        },
    ]
    snapshot = [dict(m) for m in messages]
    out = _ensure_tool_result_pairing(messages)
    assert messages == snapshot  # unchanged
    assert out is not messages


def test_pairing_no_op_when_all_paired():
    """All tool_calls already have a matching tool_result → no synthesis."""
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("c1")],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ]
    out = _ensure_tool_result_pairing(messages)
    # Same shape; only difference is dict-copy identity.
    assert [m.get("role") for m in out] == ["user", "assistant", "tool"]
    assert len(out) == 3


def test_pairing_no_op_when_no_assistant_with_tool_calls():
    """A vanilla chat with no tool_calls anywhere → straight passthrough."""
    messages = [
        {"role": "system", "content": "sp"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = _ensure_tool_result_pairing(messages)
    assert [m["role"] for m in out] == ["system", "user", "assistant"]


def test_pairing_scans_only_latest_assistant_with_tool_calls():
    """Earlier assistant messages with tool_calls were paired in prior
    rounds; the function checks only the LATEST. An orphan in an
    earlier message is not re-synthesized."""
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("c1")],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "first round"},
        {"role": "user", "content": "next"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("c2")],
        },
    ]
    out = _ensure_tool_result_pairing(messages)
    # The synthetic targets c2 (orphan in latest), not c1.
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    # c1 still there, plus new synthetic for c2.
    assert any(m["tool_call_id"] == "c1" and m["content"] == "first round" for m in tool_msgs)
    assert any(m["tool_call_id"] == "c2" and m["content"] == MISSING_TOOL_RESULT for m in tool_msgs)


# ---- streaming (v0.10.1) --------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_yields_token_deltas_then_assistant():
    completions = FakeCompletions(
        [
            _content_chunk("hel"),
            _content_chunk("lo"),
            _content_chunk(None, finish_reason="stop"),
        ]
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

    assert [event.kind for event in events] == ["token", "token", "assistant"]
    assert [e.data for e in events if e.kind == "token"] == ["hel", "lo"]
    response = events[-1].data
    assert isinstance(response, LLMResponse)
    assert response.finish_reason == "stop"
    assert response.message == {"role": "assistant", "content": "hello"}
    assert completions.kwargs is not None
    assert completions.kwargs["stream"] is True


@pytest.mark.asyncio
async def test_streaming_rebuilds_tool_calls_index_keyed():
    """tool_call fragments across deltas reassemble into one call."""
    completions = FakeCompletions(
        [
            _tool_chunk(index=0, call_id="call_1", name="echo", arguments=""),
            _tool_chunk(index=0, arguments='{"x":'),
            _tool_chunk(index=0, arguments=" 1}"),
            _content_chunk(None, finish_reason="tool_calls"),
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

    assert [e.kind for e in events] == ["assistant"]  # no content tokens
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
async def test_streaming_handles_dict_chunks():
    """Some gateways stream dict chunks instead of SDK objects."""
    completions = FakeCompletions(
        [
            {"choices": [{"delta": {"content": "dict "}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "reply"}, "finish_reason": "stop"}]},
        ]
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
    assert [e.kind for e in events] == ["token", "token", "assistant"]
    assert events[-1].data.message["content"] == "dict reply"


@pytest.mark.asyncio
async def test_pairing_runs_before_outbound_request():
    """The outgoing payload is repaired before the SDK sees it."""
    completions = FakeCompletions([_content_chunk("ok", finish_reason="stop")])
    llm = OpenAIChatLLM(_slot())
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [_tool_call("c1")]},
        {"role": "user", "content": INTERRUPTED_BY_USER_MARKER},
    ]
    _ = [e async for e in llm.stream_response(model="m", messages=messages)]
    assert completions.kwargs is not None
    outbound = completions.kwargs["messages"]
    tool_msgs = [m for m in outbound if m.get("role") == "tool"]
    assert tool_msgs == [
        {"role": "tool", "tool_call_id": "c1", "content": CANCELLED_TOOL_RESULT}
    ]


@pytest.mark.asyncio
async def test_ihep_anthropic_request_sends_system_prompt_via_extra_body():
    completions = FakeCompletions([_content_chunk("ok", finish_reason="stop")])
    llm = OpenAIChatLLM(_ihep_claude_slot())
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    _ = [
        e
        async for e in llm.stream_response(
            model="anthropic/claude-opus-4-7",
            messages=[
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Say ok."},
            ],
        )
    ]

    assert completions.kwargs is not None
    assert completions.kwargs["stream"] is True
    assert completions.kwargs["messages"] == [{"role": "user", "content": "Say ok."}]
    assert completions.kwargs["extra_body"] == {"system": "Be brief."}


# ---- stop() ---------------------------------------------------------------


def test_stop_on_idle_llm_is_noop():
    llm = OpenAIChatLLM(_slot())
    llm.stop()
    assert llm._active_stream is None


@pytest.mark.asyncio
async def test_stop_closes_active_stream():
    """stop() closes the active stream so the iterator exits cleanly."""
    completions = HangingCompletions()
    llm = OpenAIChatLLM(_slot())
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    async def drive() -> None:
        async for _ in llm.stream_response(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        ):
            pass

    task = asyncio.create_task(drive())
    await completions.entered.wait()
    assert llm._active_stream is not None

    llm.stop()
    # Closing the stream makes the hung __anext__ raise StopAsyncIteration,
    # so the generator finishes normally (no CancelledError needed here).
    await asyncio.wait_for(task, timeout=0.5)
    assert llm._active_stream is None


# ---- v1.6.1 Bug #1: Anthropic Messages SSE inbound parse -------------------


def _anthropic_chunk(event: dict[str, Any]) -> SimpleNamespace:
    """A chunk as the AsyncOpenAI SDK yields it for an Anthropic SSE event:
    no usable ``choices``; the raw event in ``model_extra``."""
    return SimpleNamespace(choices=None, model_extra=event)


@pytest.mark.asyncio
async def test_anthropic_sse_text_streaming():
    """anthropic/* text arrives as content_block_delta/text_delta off model_extra."""
    completions = FakeCompletions(
        [
            _anthropic_chunk({"type": "message_start", "message": {"usage": {"input_tokens": 5}}}),
            _anthropic_chunk(
                {"type": "content_block_start", "index": 0,
                 "content_block": {"type": "text", "text": ""}}
            ),
            _anthropic_chunk({"type": "ping"}),
            _anthropic_chunk(
                {"type": "content_block_delta", "index": 0,
                 "delta": {"type": "text_delta", "text": "hel"}}
            ),
            _anthropic_chunk(
                {"type": "content_block_delta", "index": 0,
                 "delta": {"type": "text_delta", "text": "lo"}}
            ),
            _anthropic_chunk({"type": "content_block_stop", "index": 0}),
            _anthropic_chunk(
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                 "usage": {"output_tokens": 2}}
            ),
            _anthropic_chunk({"type": "message_stop"}),
        ]
    )
    llm = OpenAIChatLLM(_ihep_claude_slot())
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    events = [
        e
        async for e in llm.stream_response(
            model="anthropic/claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
        )
    ]
    assert [e.kind for e in events] == ["token", "token", "assistant"]
    assert [e.data for e in events if e.kind == "token"] == ["hel", "lo"]
    resp = events[-1].data
    assert resp.message == {"role": "assistant", "content": "hello"}
    assert resp.finish_reason == "stop"


@pytest.mark.asyncio
async def test_anthropic_sse_tool_use_streaming():
    """tool_use block (content_block_start + input_json_delta) → one tool_call."""
    completions = FakeCompletions(
        [
            _anthropic_chunk({"type": "message_start", "message": {}}),
            _anthropic_chunk(
                {"type": "content_block_start", "index": 0,
                 "content_block": {"type": "text", "text": ""}}
            ),
            _anthropic_chunk(
                {"type": "content_block_delta", "index": 0,
                 "delta": {"type": "text_delta", "text": "Let me check."}}
            ),
            _anthropic_chunk({"type": "content_block_stop", "index": 0}),
            _anthropic_chunk(
                {"type": "content_block_start", "index": 1,
                 "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather"}}
            ),
            _anthropic_chunk(
                {"type": "content_block_delta", "index": 1,
                 "delta": {"type": "input_json_delta", "partial_json": '{"city":'}}
            ),
            _anthropic_chunk(
                {"type": "content_block_delta", "index": 1,
                 "delta": {"type": "input_json_delta", "partial_json": ' "Paris"}'}}
            ),
            _anthropic_chunk({"type": "content_block_stop", "index": 1}),
            _anthropic_chunk({"type": "message_delta", "delta": {"stop_reason": "tool_use"}}),
            _anthropic_chunk({"type": "message_stop"}),
        ]
    )
    llm = OpenAIChatLLM(_ihep_claude_slot())
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    events = [
        e
        async for e in llm.stream_response(
            model="anthropic/claude-opus-4-7",
            messages=[{"role": "user", "content": "weather in Paris?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
    ]
    resp = events[-1].data
    assert resp.finish_reason == "tool_calls"
    assert resp.message["content"] == "Let me check."
    assert resp.message["tool_calls"] == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }
    ]


def test_anthropic_event_detection_is_typed():
    """A chunk is Anthropic only when model_extra carries a known event type —
    an OpenAI usage-only chunk (empty choices, no anthropic type) is NOT."""
    llm = OpenAIChatLLM(_slot())
    openai_usage = SimpleNamespace(choices=[], model_extra={"usage": {"prompt_tokens": 1}})
    assert llm._anthropic_event(openai_usage) is None
    anthropic = SimpleNamespace(
        choices=None, model_extra={"type": "content_block_delta", "delta": {}}
    )
    assert llm._anthropic_event(anthropic) is not None


# ---- v1.6.1 Bug #2: empty-assistant repair ---------------------------------


def test_repair_empty_assistant_null_no_tools_gets_placeholder():
    out = _repair_empty_assistants(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": None}]
    )
    assert out[1]["content"] == EMPTY_ASSISTANT_PLACEHOLDER


def test_repair_keeps_null_content_with_tool_calls():
    """A null-content assistant WITH tool_calls is a normal text-free tool call."""
    msg = {"role": "assistant", "content": None, "tool_calls": [_tool_call("c1")]}
    out = _repair_empty_assistants([msg])
    assert out[0]["content"] is None
    assert out[0]["tool_calls"] == [_tool_call("c1")]


def test_repair_empty_string_and_whitespace_get_placeholder():
    out = _repair_empty_assistants(
        [{"role": "assistant", "content": ""}, {"role": "assistant", "content": "   "}]
    )
    assert out[0]["content"] == EMPTY_ASSISTANT_PLACEHOLDER
    assert out[1]["content"] == EMPTY_ASSISTANT_PLACEHOLDER


def test_repair_keeps_real_content_and_non_assistants():
    out = _repair_empty_assistants(
        [
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": ""},
            {"role": "tool", "tool_call_id": "c1", "content": ""},
        ]
    )
    assert out[0]["content"] == "hello"
    assert out[1]["content"] == ""  # a user message is never repaired
    assert out[2]["content"] == ""  # a tool message is never repaired


def test_repair_empty_assistants_is_pure():
    msgs = [{"role": "assistant", "content": None}]
    snapshot = [dict(m) for m in msgs]
    out = _repair_empty_assistants(msgs)
    assert msgs == snapshot  # input unchanged
    assert out is not msgs


@pytest.mark.asyncio
async def test_repair_empty_assistants_runs_before_outbound_request():
    """An empty assistant turn is repaired in the outgoing payload."""
    completions = FakeCompletions([_content_chunk("ok", finish_reason="stop")])
    llm = OpenAIChatLLM(_slot())
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None},  # the poison
        {"role": "user", "content": "go"},
    ]
    _ = [e async for e in llm.stream_response(model="m", messages=messages)]
    outbound = completions.kwargs["messages"]
    assistants = [m for m in outbound if m.get("role") == "assistant"]
    assert assistants[0]["content"] == EMPTY_ASSISTANT_PLACEHOLDER


def test_ensure_sdk_compliant_composes_pairing_and_empty_repair():
    """The umbrella runs BOTH compliance steps: orphan tool_calls get a
    synthetic result AND a separate empty assistant turn gets the placeholder.
    A null-content assistant WITH tool_calls is left for pairing (not repaired)."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [_tool_call("c1")]},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": None},  # genuine empty turn
        {"role": "user", "content": "go"},
    ]
    out = _ensure_sdk_compliant(messages)
    # 1. orphan tool_call c1 got a synthetic result (pairing).
    assert any(
        m.get("role") == "tool"
        and m.get("tool_call_id") == "c1"
        and m["content"] == MISSING_TOOL_RESULT
        for m in out
    )
    # 2. the tool-call assistant keeps content=None (a text-free tool call).
    tc_assistant = next(m for m in out if m.get("tool_calls"))
    assert tc_assistant["content"] is None
    # 3. the genuine empty assistant turn got the placeholder (empty repair).
    assert any(
        m.get("role") == "assistant"
        and not m.get("tool_calls")
        and m.get("content") == EMPTY_ASSISTANT_PLACEHOLDER
        for m in out
    )
    # purity
    assert out is not messages
