"""Headless tests for the main chat TUI."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from textual.widgets import Input, Static

from neutrix.agent import DEFAULT_SYSTEM_PROMPT, Agent, AgentEvent
from neutrix.config import Config, Slot
from neutrix.tui import Message, NeutrixApp


def _slot() -> Slot:
    return Slot(
        name="fast",
        provider="test",
        model="test-model",
        base_url="https://example.test/v1",
        api_key="sk-test",
    )


def _config(tmp_path: Path) -> Config:
    return Config(
        providers={"test": {"base_url": "https://example.test/v1", "api_key": "sk-test"}},
        slots={
            "fast": {"provider": "test", "model": "test-model"},
            "strong": {"provider": "test", "model": "test-model"},
        },
        path=tmp_path / "config.yaml",
    )


def _render_text(widget: Static) -> str:
    renderable = widget.render()
    if hasattr(renderable, "plain"):
        return renderable.plain
    return str(renderable)


def test_default_system_prompt_is_simple_chatbot_prompt():
    agent = Agent(slot=_slot(), use_tools=False)

    assert DEFAULT_SYSTEM_PROMPT == "You are a helpful assistant. Keep it simple."
    assert agent.messages == [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]


class StreamingAgent:
    def __init__(self) -> None:
        self.slot = _slot()
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.use_tools = False
        self.messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_reply(self, user_text: str):
        self.messages.append({"role": "user", "content": user_text})
        self.started.set()
        yield AgentEvent("token", "hello")
        await self.release.wait()
        yield AgentEvent("token", " world")
        self.messages.append({"role": "assistant", "content": "hello world"})


@pytest.mark.asyncio
async def test_main_chat_mount_shows_system_prompt_and_composer(tmp_path: Path):
    app = NeutrixApp(StreamingAgent(), config=_config(tmp_path))

    async with app.run_test() as pilot:
        await pilot.pause()

        system_prompt = app.query_one("#system-prompt", Static)
        input_box = app.query_one("#input", Input)

        assert DEFAULT_SYSTEM_PROMPT in _render_text(system_prompt)
        assert input_box.parent is app.query_one("#composer")
        assert not app.query_one("#chat").has_class("started")


@pytest.mark.asyncio
async def test_submit_streams_reply_and_shows_busy_indicator(tmp_path: Path):
    agent = StreamingAgent()
    app = NeutrixApp(agent, config=_config(tmp_path), render_markdown=False)

    async with app.run_test() as pilot:
        await pilot.pause()
        input_box = app.query_one("#input", Input)
        input_box.value = "hi"

        await pilot.press("enter")
        await asyncio.wait_for(agent.started.wait(), timeout=1.0)
        await pilot.pause()

        thinking = app.query_one("#thinking", Static)
        assert app.query_one("#chat").has_class("started")
        assert input_box.disabled
        assert thinking.has_class("active")
        assert "assistant is responding" in _render_text(thinking)

        agent.release.set()
        for _ in range(20):
            await pilot.pause()
            if not app._busy:
                break

        assert not app._busy
        assert not input_box.disabled
        assert not thinking.has_class("active")

        messages = list(app.query(Message))
        assert [message.role for message in messages] == ["user", "assistant"]
        assert messages[-1]._content == "hello world"
