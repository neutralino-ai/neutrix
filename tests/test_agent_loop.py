"""Tests for the model/tool continuation loop."""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from neutrix.agent_loop import (
    TASK_REMINDER_MARKER,
    TASK_REMINDER_TAG_OPEN,
    Agent,
    assistant_turns_since_reminder,
    assistant_turns_since_task_management,
    build_task_reminder,
)
from neutrix.config import Slot
from neutrix.llm import LLMEvent, LLMResponse
from neutrix.store import ChatStore


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
        lambda name, arguments, **_: f"{name}:{arguments}",
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


# ---- v0.8.0 task reminder algorithm ----------------------------------------


def _assistant(content: str = "ok", **extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"role": "assistant", "content": content}
    base.update(extra)
    return base


def _user(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content}


def _task_call_msg(name: str = "TaskCreate") -> dict[str, Any]:
    return _assistant(
        content=None,
        tool_calls=[
            {
                "id": "c1",
                "type": "function",
                "function": {"name": name, "arguments": '{"subject": "x"}'},
            }
        ],
    )


def _reminder_msg(subject: str = "first") -> dict[str, Any]:
    body = (
        f"{TASK_REMINDER_TAG_OPEN}\n"
        "The task tools haven't been used recently. ...\n"
        "\n"
        f"{TASK_REMINDER_MARKER}\n"
        "\n"
        f"#1. [pending] {subject}\n"
        "</system-reminder>"
    )
    return _user(body)


def test_assistant_turns_since_task_management_no_prior_call():
    messages = [
        {"role": "system", "content": "sp"},
        _user("hi"),
        _assistant("a"),
        _user("again"),
        _assistant("b"),
    ]
    # Two assistant messages, none used the task tools.
    assert assistant_turns_since_task_management(messages) == 2


def test_assistant_turns_since_task_management_counts_after_call():
    messages = [
        _user("track this"),
        _task_call_msg("TaskCreate"),
        _user("ok"),
        _assistant("done"),
        _user("again"),
        _assistant("ok"),
    ]
    # Two assistant turns since the TaskCreate-bearing assistant turn.
    assert assistant_turns_since_task_management(messages) == 2


def test_assistant_turns_since_reminder_zero_when_none_exists():
    messages = [_user("hi"), _assistant("a"), _user("again"), _assistant("b")]
    # Two assistant turns, no prior reminder — count == total assistant turns.
    assert assistant_turns_since_reminder(messages) == 2


def test_assistant_turns_since_reminder_resets_after_reminder():
    messages = [_user("hi"), _assistant("a"), _reminder_msg(), _assistant("b")]
    # One assistant turn since the reminder.
    assert assistant_turns_since_reminder(messages) == 1


def test_build_task_reminder_skipped_when_no_actionable_tasks():
    store = ChatStore()
    store.add_task("done")
    store.update_task("1", status="completed")
    messages = [_user("hi")] + [_assistant("a")] * 15
    assert build_task_reminder(messages, store.tasks) is None


def test_build_task_reminder_skipped_below_thresholds():
    store = ChatStore()
    store.add_task("x")
    messages = [_user("hi")] + [_assistant("a")] * 5
    assert build_task_reminder(messages, store.tasks) is None


def test_build_task_reminder_emitted_when_both_thresholds_met():
    store = ChatStore()
    store.add_task("first")
    store.update_task("1", status="in_progress")
    store.add_task("second")  # pending
    store.add_task("ignored")
    store.update_task("3", status="completed")
    messages = [_user("hi")] + [_assistant("a")] * 10
    reminder = build_task_reminder(messages, store.tasks)
    assert reminder is not None
    assert reminder["role"] == "user"
    content = reminder["content"]
    assert content.startswith(TASK_REMINDER_TAG_OPEN)
    assert content.endswith("</system-reminder>")
    assert TASK_REMINDER_MARKER in content
    # Only actionable items listed.
    assert "#1. [in_progress] first" in content
    assert "#2. [pending] second" in content
    assert "ignored" not in content


def test_build_task_reminder_suppressed_after_recent_reminder():
    """A reminder emitted then 5 turns later: still in cooldown."""
    store = ChatStore()
    store.add_task("first")
    messages = (
        [_user("hi")]
        + [_assistant("a")] * 10
        + [_reminder_msg()]
        + [_user("k")]
        + [_assistant("a")] * 5
    )
    assert build_task_reminder(messages, store.tasks) is None


@pytest.mark.asyncio
async def test_stream_reply_injects_reminder_when_due(monkeypatch):
    """End-to-end: 10 assistant turns of fluff → next stream_reply appends
    exactly one reminder before the LLM call."""
    monkeypatch.setattr(
        "neutrix.agent_loop.get_schemas",
        lambda: [{"type": "function", "function": {"name": "echo"}}],
    )
    store = ChatStore()
    store.add_task("first")
    llm = FakeLLM(
        [
            [
                LLMEvent(
                    "assistant",
                    LLMResponse(
                        {"role": "assistant", "content": "ok"},
                        finish_reason="stop",
                    ),
                ),
            ]
        ]
    )
    agent = Agent(slot=_slot(), use_tools=True, llm=llm, store=store)
    # Pre-load enough assistant turns to satisfy both thresholds.
    agent.messages.extend([_user("hi")] + [_assistant("a")] * 10)

    events = [event async for event in agent.stream_reply("now what")]
    assert [event.kind for event in events] == ["assistant", "done"]

    # The LLM saw: user("now what"), then the reminder, before the assistant turn.
    sent = llm.calls[0]["messages"]
    user_indexes = [i for i, m in enumerate(sent) if m.get("role") == "user"]
    assert sent[user_indexes[-2]]["content"] == "now what"
    reminder = sent[user_indexes[-1]]
    assert reminder["content"].startswith(TASK_REMINDER_TAG_OPEN)
    assert TASK_REMINDER_MARKER in reminder["content"]

    # And only one reminder was appended for this stream_reply call.
    reminders = [
        m for m in agent.messages
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and TASK_REMINDER_MARKER in m["content"]
    ]
    assert len(reminders) == 1


@pytest.mark.asyncio
async def test_stream_reply_does_not_re_inject_during_tool_followup(monkeypatch):
    """The per-turn nudge fires once; subsequent tool-driven LLM rounds
    in the same stream_reply must not append a second reminder."""
    monkeypatch.setattr(
        "neutrix.agent_loop.get_schemas",
        lambda: [{"type": "function", "function": {"name": "echo"}}],
    )
    monkeypatch.setattr(
        "neutrix.agent_loop._dispatch_with_store",
        lambda name, arguments, store: "ok",
    )
    store = ChatStore()
    store.add_task("first")
    # Two rounds: tool call, then a final stop.
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
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "echo", "arguments": "{}"},
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
                )
            ],
        ]
    )
    agent = Agent(slot=_slot(), use_tools=True, llm=llm, store=store)
    agent.messages.extend([_user("hi")] + [_assistant("a")] * 10)

    [event async for event in agent.stream_reply("now what")]

    reminders = [
        m for m in agent.messages
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and TASK_REMINDER_MARKER in m["content"]
    ]
    assert len(reminders) == 1


@pytest.mark.asyncio
async def test_stream_reply_skips_reminder_when_no_store():
    """An Agent constructed without a store (legacy callers) must not crash
    and must not invent reminders."""
    llm = FakeLLM(
        [
            [
                LLMEvent(
                    "assistant",
                    LLMResponse(
                        {"role": "assistant", "content": "ok"},
                        finish_reason="stop",
                    ),
                )
            ]
        ]
    )
    agent = Agent(slot=_slot(), use_tools=False, llm=llm)
    [event async for event in agent.stream_reply("hi")]
    reminders = [
        m for m in agent.messages
        if isinstance(m.get("content"), str)
        and TASK_REMINDER_MARKER in m["content"]
    ]
    assert reminders == []


@pytest.mark.asyncio
async def test_task_tool_dispatch_receives_store(monkeypatch):
    """TaskCreate must mutate the agent's store via dispatch keyword."""
    monkeypatch.setattr(
        "neutrix.agent_loop.get_schemas",
        lambda: [{"type": "function", "function": {"name": "TaskCreate"}}],
    )
    store = ChatStore()
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
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": "TaskCreate",
                                        "arguments": '{"subject": "first"}',
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
                        {"role": "assistant", "content": "added"},
                        finish_reason="stop",
                    ),
                )
            ],
        ]
    )
    agent = Agent(slot=_slot(), use_tools=True, llm=llm, store=store)
    [event async for event in agent.stream_reply("track first")]
    assert [(t.id, t.subject) for t in store.tasks] == [("1", "first")]


@pytest.mark.asyncio
async def test_agent_loop_tool_dispatch_does_not_block_event_loop(monkeypatch):
    monkeypatch.setattr(
        "neutrix.agent_loop.get_schemas",
        lambda: [{"type": "function", "function": {"name": "slow"}}],
    )

    def slow_dispatch(name: str, arguments: str, **_: Any) -> str:
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
