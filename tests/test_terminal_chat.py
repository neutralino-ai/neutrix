"""Tests for the append-only terminal chat renderer."""
from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path
from queue import Queue
from typing import Any

import pytest
from rich.console import Console

from neutrix.agent_loop import AgentEvent
from neutrix.config import Config, Slot
from neutrix.terminal_chat import (
    TerminalChat,
    ToolRecord,
    approximate_token_count,
    delete_buffer_to_line_end,
    move_buffer_to_line_start,
    result_line_count,
)


def _slot(name: str = "strong") -> Slot:
    return Slot(
        name=name,
        provider="test",
        model=f"{name}-model",
        base_url="https://example.test/v1",
        api_key="sk-test",
    )


def _config(tmp_path: Path) -> Config:
    return Config(
        providers={"test": {"base_url": "https://example.test/v1", "api_key": "sk-test"}},
        slots={
            "fast": {"provider": "test", "model": "fast-model"},
            "strong": {"provider": "test", "model": "strong-model"},
        },
        path=tmp_path / "config.yaml",
    )


class ToolAgent:
    def __init__(self) -> None:
        self.slot = _slot()
        self.use_tools = True
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": "system prompt"}
        ]
        self.sent: list[str] = []

    def effective_tools_enabled(self) -> bool:
        return self.use_tools

    def switch(self, slot: Slot) -> None:
        self.slot = slot

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": "system prompt"}]

    async def stream_reply(self, user_text: str):
        self.sent.append(user_text)
        self.messages.append({"role": "user", "content": user_text})
        arguments = '{"path": "."}'
        result = "f a.txt\nf b.txt\n"
        yield AgentEvent("tool_call", {"name": "list_dir", "arguments": arguments})
        self.messages.append(
            {"role": "tool", "tool_call_id": "call_1", "content": result}
        )
        yield AgentEvent("tool_result", {"name": "list_dir", "result": result})
        self.messages.append({"role": "assistant", "content": "done"})
        yield AgentEvent("assistant", "done")
        yield AgentEvent("done")


class BlockingAgent(ToolAgent):
    def __init__(self) -> None:
        super().__init__()
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.releases: asyncio.Queue[None] = asyncio.Queue()

    async def stream_reply(self, user_text: str):
        self.sent.append(user_text)
        self.messages.append({"role": "user", "content": user_text})
        await self.started.put(user_text)
        await self.releases.get()
        self.messages.append({"role": "assistant", "content": f"reply {user_text}"})
        yield AgentEvent("assistant", f"reply {user_text}")
        yield AgentEvent("done")


def _chat(
    agent: ToolAgent,
    tmp_path: Path,
    inputs: list[str],
) -> tuple[TerminalChat, StringIO, list[str]]:
    output = StringIO()
    input_iter = iter(inputs)
    prompts: list[str] = []
    console = Console(
        file=output,
        force_terminal=False,
        color_system=None,
        width=100,
    )

    def input_func(prompt: str) -> str:
        prompts.append(prompt)
        return next(input_iter)

    chat = TerminalChat(
        agent,
        config=_config(tmp_path),
        render_markdown=False,
        input_func=input_func,
        console=console,
    )
    return chat, output, prompts


class FakeDocument:
    def __init__(self, text: str, cursor_position: int) -> None:
        self.text = text
        self.cursor_position = cursor_position

    @property
    def text_after_cursor(self) -> str:
        return self.text[self.cursor_position :]

    def get_start_of_line_position(self) -> int:
        line_start = self.text.rfind("\n", 0, self.cursor_position) + 1
        return line_start - self.cursor_position

    def get_end_of_line_position(self) -> int:
        line_end = self.text.find("\n", self.cursor_position)
        if line_end == -1:
            line_end = len(self.text)
        return line_end - self.cursor_position


class FakeBuffer:
    def __init__(self, text: str, cursor_position: int) -> None:
        self.text = text
        self.cursor_position = cursor_position

    @property
    def document(self) -> FakeDocument:
        return FakeDocument(self.text, self.cursor_position)

    def delete(self, count: int) -> None:
        self.text = (
            self.text[: self.cursor_position]
            + self.text[self.cursor_position + count :]
        )


def test_tool_result_summary_counts_lines_and_approx_tokens() -> None:
    assert result_line_count("") == 0
    assert result_line_count("one\ntwo\n") == 2
    assert approximate_token_count("one two\nthree") == 3

    record = ToolRecord(
        index=1,
        name="list_dir",
        arguments='{"path": "."}',
        result="one two\nthree",
    )

    assert record.summary == (
        '<- [tool 1] list_dir {"path": "."} | folded | 2 lines | ~3 tokens'
    )


def test_draft_key_helpers_match_readline_line_editing() -> None:
    buffer = FakeBuffer("abc\ndef", 6)
    move_buffer_to_line_start(buffer)
    assert buffer.cursor_position == 4

    buffer = FakeBuffer("abc\ndef", 5)
    delete_buffer_to_line_end(buffer)
    assert buffer.text == "abc\nd"

    buffer = FakeBuffer("abc\ndef", 3)
    delete_buffer_to_line_end(buffer)
    assert buffer.text == "abcdef"


def test_terminal_chat_folds_tool_results_and_expands_on_command(tmp_path: Path) -> None:
    agent = ToolAgent()
    chat, output, prompts = _chat(
        agent, tmp_path, ["please list", "/tool", "/tool 1", "/quit"]
    )

    chat.run()

    rendered = output.getvalue()
    assert prompts and all(prompt == "" for prompt in prompts)
    assert "you>" not in rendered
    assert "system:" not in rendered
    assert "user:" not in rendered
    assert "assistant:" not in rendered
    assert "assistant is responding" not in rendered
    assert "please list" in rendered
    assert "strong | test/strong-model | tools:on" not in rendered
    assert '-> list_dir {"path": "."}' in rendered
    assert (
        '<- [tool 1] list_dir {"path": "."} | folded | 2 lines | ~4 tokens'
        in rendered
    )
    assert "[tool 1] list_dir full result:" in rendered
    assert "f a.txt\nf b.txt" in rendered
    assert agent.messages[2]["content"] == "f a.txt\nf b.txt\n"


@pytest.mark.asyncio
async def test_terminal_chat_accepts_and_queues_input_while_agent_is_busy(
    tmp_path: Path,
) -> None:
    agent = BlockingAgent()
    output = StringIO()
    input_values: Queue[str] = Queue()
    prompts: list[str] = []

    def input_func(prompt: str) -> str:
        prompts.append(prompt)
        return input_values.get(timeout=2)

    chat = TerminalChat(
        agent,
        config=_config(tmp_path),
        render_markdown=False,
        input_func=input_func,
        console=Console(
            file=output,
            force_terminal=False,
            color_system=None,
            width=100,
        ),
    )
    task = asyncio.create_task(chat.run_async())

    input_values.put("first")
    assert await asyncio.wait_for(agent.started.get(), timeout=1) == "first"

    input_values.put("second")
    for _ in range(20):
        await asyncio.sleep(0.01)
        if "queued:1" in chat._status_text():
            break

    assert chat._busy
    assert "queued:1" in chat._status_text()

    await agent.releases.put(None)
    assert await asyncio.wait_for(agent.started.get(), timeout=1) == "second"
    await agent.releases.put(None)
    input_values.put("/quit")

    await asyncio.wait_for(task, timeout=2)

    assert prompts and all(prompt == "" for prompt in prompts)
    assert agent.sent == ["first", "second"]
    rendered = output.getvalue()
    assert "first" in rendered
    assert "second" in rendered
    assert "reply first" in rendered
    assert "reply second" in rendered


def test_terminal_chat_tool_toggle_updates_status(tmp_path: Path) -> None:
    agent = ToolAgent()
    chat, output, _prompts = _chat(agent, tmp_path, ["/tools off", "/quit"])

    chat.run()

    rendered = output.getvalue()
    assert "tool calling disabled" in rendered
    assert "tools:off" not in rendered
    assert "tools:off" in chat._status_text()
    assert agent.use_tools is False


def test_terminal_chat_folds_loaded_tool_messages_with_call_arguments(
    tmp_path: Path,
) -> None:
    agent = ToolAgent()
    agent.messages = [
        {"role": "system", "content": "system prompt"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "README.md"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "hello world\n"},
    ]
    chat, output, _prompts = _chat(agent, tmp_path, ["/quit"])

    chat.run()

    rendered = output.getvalue()
    assert '-> read_file {"path": "README.md"}' in rendered
    assert (
        '<- [tool 1] read_file {"path": "README.md"} | folded | 1 lines | ~2 tokens'
        in rendered
    )
