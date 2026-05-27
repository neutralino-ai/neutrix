"""Tests for the non-streaming OpenAI-compatible adapter + pairing layer."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from neutrix.config import Slot
from neutrix.llm import (
    CANCELLED_TOOL_RESULT,
    INTERRUPTED_BY_USER_MARKER,
    MISSING_TOOL_RESULT,
    LLMResponse,
    OpenAIChatLLM,
    _ensure_tool_result_pairing,
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


def _completion(
    *,
    content: str | None = "ok",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    """Build a non-streaming Chat Completions response object."""
    message_kwargs: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message_kwargs["tool_calls"] = [
            SimpleNamespace(
                id=tc["id"],
                type=tc.get("type", "function"),
                function=SimpleNamespace(**tc["function"]),
            )
            for tc in tool_calls
        ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(**message_kwargs),
                finish_reason=finish_reason,
            )
        ]
    )


class FakeCompletions:
    """Pretends to be ``client.chat.completions`` with non-streaming create."""

    def __init__(self, completion: Any) -> None:
        self.completion = completion
        self.kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.completion


class HangingCompletions:
    """Awaits forever — the test cancels it from outside."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None
        self.entered = asyncio.Event()

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        self.entered.set()
        await asyncio.Event().wait()
        # never returns
        return None  # pragma: no cover


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


# ---- streaming (now non-streaming) ----------------------------------------


@pytest.mark.asyncio
async def test_non_streaming_yields_one_assistant_event():
    completions = FakeCompletions(_completion(content="hello", finish_reason="stop"))
    llm = OpenAIChatLLM(_slot())
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    events = [
        event
        async for event in llm.stream_response(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )
    ]

    assert [event.kind for event in events] == ["assistant"]
    response = events[-1].data
    assert isinstance(response, LLMResponse)
    assert response.finish_reason == "stop"
    assert response.message == {"role": "assistant", "content": "hello"}
    assert completions.kwargs is not None
    assert completions.kwargs["stream"] is False


@pytest.mark.asyncio
async def test_non_streaming_returns_tool_calls():
    completions = FakeCompletions(
        _completion(
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"x": 1}'},
                }
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
async def test_non_streaming_handles_dict_message_payload():
    """Some gateways return dict payloads instead of SDK objects."""
    completions = FakeCompletions(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "dict reply"},
                    "finish_reason": "stop",
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
    assert [event.kind for event in events] == ["assistant"]
    response = events[-1].data
    assert response.message["content"] == "dict reply"


@pytest.mark.asyncio
async def test_pairing_runs_before_outbound_request():
    """The outgoing payload is repaired before the SDK sees it. An
    orphan tool_use in the input messages gets a synthetic tool_result
    in the outgoing request body."""
    completions = FakeCompletions(_completion(content="ok"))
    llm = OpenAIChatLLM(_slot())
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call("c1")],
        },
        {"role": "user", "content": INTERRUPTED_BY_USER_MARKER},
    ]
    _ = [
        e
        async for e in llm.stream_response(model="m", messages=messages)
    ]
    assert completions.kwargs is not None
    outbound = completions.kwargs["messages"]
    tool_msgs = [m for m in outbound if m.get("role") == "tool"]
    assert tool_msgs == [
        {"role": "tool", "tool_call_id": "c1", "content": CANCELLED_TOOL_RESULT}
    ]


@pytest.mark.asyncio
async def test_ihep_anthropic_request_sends_system_prompt_via_extra_body():
    completions = FakeCompletions(_completion(content="ok"))
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
    assert completions.kwargs["stream"] is False
    assert completions.kwargs["messages"] == [{"role": "user", "content": "Say ok."}]
    assert completions.kwargs["extra_body"] == {"system": "Be brief."}


# ---- stop() ---------------------------------------------------------------


def test_stop_on_idle_llm_is_noop():
    llm = OpenAIChatLLM(_slot())
    llm.stop()
    assert llm._active_task is None


@pytest.mark.asyncio
async def test_stop_cancels_awaiting_create_task():
    """stop() cancels the parked create() task so the awaiting
    stream_response raises CancelledError."""
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
    # The LLM should have an active task by now.
    for _ in range(50):
        await asyncio.sleep(0.01)
        if llm._active_task is not None:
            break
    assert llm._active_task is not None

    llm.stop()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.5)
    assert llm._active_task is None
