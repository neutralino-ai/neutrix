"""Headless tests for the main chat TUI."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from textual.widgets import Input, Static

from neutrix.agent import DEFAULT_SYSTEM_PROMPT, Agent, AgentEvent
from neutrix.config import Config, Slot
from neutrix.onboard import KeyInput
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


def _onboard_config(tmp_path: Path) -> Config:
    return Config(
        providers={
            "ihep": {
                "base_url": "https://aiapi.ihep.ac.cn/apiv2/",
                "api_key": "",
            }
        },
        slots={
            "fast": {"provider": "ihep", "model": "anthropic/claude-haiku-4-5"},
            "strong": {"provider": "ihep", "model": "anthropic/claude-opus-4-7"},
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


class SpyAgent(StreamingAgent):
    def __init__(self) -> None:
        super().__init__()
        self.streamed_texts: list[str] = []

    async def stream_reply(self, user_text: str):
        self.streamed_texts.append(user_text)
        if False:
            yield AgentEvent("token", "")


@pytest.mark.asyncio
async def test_main_chat_mount_shows_system_prompt_and_composer(tmp_path: Path):
    app = NeutrixApp(StreamingAgent(), config=_config(tmp_path))

    async with app.run_test() as pilot:
        await pilot.pause()

        input_box = app.query_one("#input", Input)
        composer = app.query_one("#composer")
        messages = list(app.query(Message))

        assert [message.role for message in messages] == ["system"]
        assert DEFAULT_SYSTEM_PROMPT in messages[0]._content
        assert input_box.parent is composer
        assert composer.parent is app.query_one("#blocks")
        assert composer.has_class("draft")
        assert "User draft" in _render_text(composer.query_one(".block-label", Static))


@pytest.mark.asyncio
async def test_slash_command_feedback_stays_outside_model_blocks(tmp_path: Path):
    agent = StreamingAgent()
    app = NeutrixApp(agent, config=_config(tmp_path), render_markdown=False)

    async with app.run_test() as pilot:
        await pilot.pause()
        input_box = app.query_one("#input", Input)
        input_box.value = "/help"

        await pilot.press("enter")
        await pilot.pause()

        messages = list(app.query(Message))
        assert [message.role for message in messages] == ["system"]
        assert all("/help" not in message._content for message in messages)
        assert all("Commands:" not in message._content for message in messages)
        assert "Commands:" in _render_text(app.query_one("#notice", Static))
        assert agent.messages == [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
        ]


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
        assert [message.role for message in messages] == [
            "system",
            "user",
            "assistant",
        ]
        assert messages[1]._content == "hi"
        assert messages[-1]._content == "hello world"


@pytest.mark.asyncio
async def test_onboard_key_submit_does_not_become_chat_message(tmp_path: Path):
    agent = SpyAgent()
    app = NeutrixApp(agent, config=_onboard_config(tmp_path), render_markdown=False)
    secret = "sk-test-secret-do-not-echo"

    async with app.run_test() as pilot:
        await pilot.pause()
        input_box = app.query_one("#input", Input)
        input_box.value = "/onboard"
        await pilot.press("enter")
        await pilot.pause()

        key_input = app.screen.query_one("#key-ihep", KeyInput)
        key_input.focus()
        await pilot.pause()
        key_input.value = secret
        await pilot.press("enter")
        await pilot.pause()

        assert key_input._committed_value == secret
        assert all("/onboard" not in message._content for message in app.query(Message))
        assert all(secret not in message._content for message in app.query(Message))
        assert all(secret not in str(message) for message in agent.messages)
        assert agent.streamed_texts == []
