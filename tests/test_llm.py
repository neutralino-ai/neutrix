"""Tests for the streaming OpenAI-compatible adapter (v0.10.1) + pairing layer."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import BadRequestError

from neutrix.config import Slot
from neutrix.llm import (
    CANCELLED_TOOL_RESULT,
    EMPTY_ASSISTANT_PLACEHOLDER,
    INTERRUPTED_BY_USER_MARKER,
    MISSING_TOOL_RESULT,
    LLMResponse,
    OpenAIChatLLM,
    Usage,
    _ensure_sdk_compliant,
    _ensure_tool_result_pairing,
    _repair_empty_assistants,
    _usage_from_anthropic,
    _usage_from_openai,
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


# ---- v1.7.0 usage / cost capture -------------------------------------------


def _usage_chunk(usage: dict[str, Any]) -> SimpleNamespace:
    """An OpenAI final chunk: empty ``choices``, a ``usage`` payload."""
    return SimpleNamespace(choices=[], usage=usage)


def _anthropic_event_chunk(event: dict[str, Any]) -> SimpleNamespace:
    """An anthropic-gateway chunk: no OpenAI choice, event in ``model_extra``."""
    return SimpleNamespace(choices=None, model_extra=event)


class _ProbingCompletions:
    """First create() (with include_usage) raises 400; the retry (without) wins."""

    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if "stream_options" in kwargs:
            req = httpx.Request("POST", "https://example.test/v1/chat/completions")
            raise BadRequestError(
                "include_usage unsupported",
                response=httpx.Response(400, request=req),
                body=None,
            )
        return FakeStream(list(self.chunks))


class _Always400Completions:
    """create() 400s on every call — a genuine error unrelated to include_usage."""

    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        req = httpx.Request("POST", "https://example.test/v1/chat/completions")
        raise BadRequestError(
            "unknown model", response=httpx.Response(400, request=req), body=None
        )


def test_usage_normalization_cache_accounting_asymmetry():
    """The silent-corruption guard: OpenAI's prompt_tokens INCLUDES cached (fresh
    input subtracts it); Anthropic reports cache SEPARATELY (no subtraction)."""
    oa = _usage_from_openai(
        {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 40},
        }
    )
    assert (oa.input, oa.cache_read, oa.output) == (60, 40, 10)
    assert oa.raw["prompt_tokens"] == 100  # raw kept as source of truth
    an = _usage_from_anthropic(
        {
            "input_tokens": 100,
            "output_tokens": 10,
            "cache_read_input_tokens": 40,
            "cache_creation_input_tokens": 5,
        }
    )
    assert (an.input, an.cache_read, an.cache_write, an.output) == (100, 40, 5, 10)
    assert (oa + Usage(input=1)).input == 61  # __add__ sums the four classes
    assert an.total == 155


def test_usage_from_openai_reads_deepseek_native_cache_fields():
    """v1.7.1: direct api.deepseek.com reports the hit/miss as ``prompt_cache_hit_tokens``
    / ``prompt_cache_miss_tokens`` (not the OpenAI-standard ``cached_tokens``,
    which it leaves 0). The normalizer reads the native fields."""
    u = _usage_from_openai(
        {
            "prompt_tokens": 1517,
            "completion_tokens": 35,
            "prompt_cache_hit_tokens": 1408,
            "prompt_cache_miss_tokens": 109,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
    )
    assert u.cache_read == 1408  # hit
    assert u.input == 109  # miss (native)
    assert u.output == 35
    assert u.cache_read + u.input == 1517  # hit + miss == prompt_tokens


def test_usage_from_openai_gateway_cache_read_fallback():
    """IHEP gateway warm call: the hit is in the standard ``cached_tokens`` AND a
    top-level ``cache_read_tokens``; miss = prompt - hit."""
    u = _usage_from_openai(
        {
            "prompt_tokens": 1517,
            "completion_tokens": 35,
            "prompt_tokens_details": {"cached_tokens": 1408},
            "cache_read_tokens": 1408,
        }
    )
    assert (u.cache_read, u.input, u.output) == (1408, 109, 35)


def test_usage_hit_miss_view():
    """v1.7.1 3-number display view: hit = cache_read; miss = input + cache_write."""
    u = Usage(input=100, output=50, cache_read=1408, cache_write=10)
    assert u.hit == 1408
    assert u.miss == 110  # 100 fresh input + 10 cache-write
    assert u.hit + u.miss == u.input + u.cache_read + u.cache_write


@pytest.mark.asyncio
async def test_openai_usage_captured_onto_llmresponse():
    llm = OpenAIChatLLM(_slot())
    chunks = [
        _content_chunk("hi", finish_reason="stop"),
        _usage_chunk(
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 30},
            }
        ),
    ]
    completions = FakeCompletions(chunks)
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    events = [
        e
        async for e in llm.stream_response(
            model="m", messages=[{"role": "user", "content": "x"}]
        )
    ]
    resp = next(e.data for e in events if e.kind == "assistant")
    assert resp.usage is not None
    assert (resp.usage.input, resp.usage.cache_read, resp.usage.output) == (70, 30, 20)
    assert completions.kwargs.get("stream_options") == {"include_usage": True}


@pytest.mark.asyncio
async def test_anthropic_usage_captured_and_no_include_usage_sent():
    llm = OpenAIChatLLM(_ihep_claude_slot())
    chunks = [
        _anthropic_event_chunk(
            {
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 200,
                        "cache_read_input_tokens": 50,
                        "cache_creation_input_tokens": 10,
                        "output_tokens": 1,
                    }
                },
            }
        ),
        _anthropic_event_chunk(
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}
        ),
        _anthropic_event_chunk(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"},
            }
        ),
        _anthropic_event_chunk(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 42},
            }
        ),
    ]
    completions = FakeCompletions(chunks)
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    events = [
        e
        async for e in llm.stream_response(
            model="anthropic/claude-opus-4-7",
            messages=[{"role": "user", "content": "x"}],
        )
    ]
    resp = next(e.data for e in events if e.kind == "assistant")
    assert resp.usage is not None
    # Anthropic input is NOT reduced by cache; output is the message_delta total.
    assert (resp.usage.input, resp.usage.cache_read, resp.usage.cache_write) == (
        200,
        50,
        10,
    )
    assert resp.usage.output == 42
    assert "stream_options" not in (completions.kwargs or {})


class _GatewayUsage:
    """Mimics the IHEP anthropic gateway's ``chunk.usage`` (verified live): an
    OpenAI ``CompletionUsage`` whose standard fields are ``null`` but whose
    ``model_extra`` (surfaced by ``model_dump``) carries the Anthropic counts."""

    def __init__(self, merged: dict[str, Any]) -> None:
        self._merged = merged

    def model_dump(self) -> dict[str, Any]:
        return dict(self._merged)


@pytest.mark.asyncio
async def test_anthropic_gateway_usage_on_chunk_usage_not_misread_as_openai():
    """Regression for the live-caught silent-corruption bug (v1.7.0 Acceptance
    #3): the IHEP anthropic gateway puts the Anthropic counts as ``model_extra``
    ON ``chunk.usage`` while its OpenAI fields are ``null``. Routing usage by
    presence sent that object to ``_usage_from_openai`` → all-zeros. The fix
    routes by PROTOCOL (the slot's gateway flag). This test would have failed
    before the fix (input/output == 0)."""
    llm = OpenAIChatLLM(_ihep_claude_slot())
    merged = {
        "completion_tokens": None,
        "prompt_tokens": None,
        "total_tokens": None,
        "completion_tokens_details": None,
        "prompt_tokens_details": None,
        "input_tokens": 200,
        "cache_read_input_tokens": 50,
        "cache_creation_input_tokens": 10,
        "output_tokens": 42,
    }
    chunks = [
        _anthropic_event_chunk(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"},
            }
        ),
        SimpleNamespace(choices=[], usage=_GatewayUsage(merged)),
    ]
    completions = FakeCompletions(chunks)
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    events = [
        e
        async for e in llm.stream_response(
            model="anthropic/claude-opus-4-7",
            messages=[{"role": "user", "content": "x"}],
        )
    ]
    resp = next(e.data for e in events if e.kind == "assistant")
    assert resp.usage is not None
    assert (resp.usage.input, resp.usage.cache_read, resp.usage.cache_write) == (200, 50, 10)
    assert resp.usage.output == 42
    # raw keeps the FULL provider payload (incl. the null OpenAI fields).
    assert resp.usage.raw["input_tokens"] == 200
    assert "stream_options" not in (completions.kwargs or {})


@pytest.mark.asyncio
async def test_include_usage_probe_retries_without_on_400_no_double_yield():
    llm = OpenAIChatLLM(_slot())
    completions = _ProbingCompletions([_content_chunk("hi", finish_reason="stop")])
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    events = [
        e
        async for e in llm.stream_response(
            model="m", messages=[{"role": "user", "content": "x"}]
        )
    ]
    tokens = [e.data for e in events if e.kind == "token"]
    assert tokens == ["hi"]  # exactly once — the retry did NOT double-yield
    assert len(completions.calls) == 2
    assert "stream_options" in completions.calls[0]
    assert "stream_options" not in completions.calls[1]
    assert llm._include_usage_supported is False  # cached → next turn skips probe


@pytest.mark.asyncio
async def test_genuine_400_propagates_after_include_usage_retry():
    llm = OpenAIChatLLM(_slot())
    completions = _Always400Completions()
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    with pytest.raises(BadRequestError):
        _ = [
            e
            async for e in llm.stream_response(
                model="m", messages=[{"role": "user", "content": "x"}]
            )
        ]
    assert completions.calls == 2  # probed, retried without, then the real error raised
