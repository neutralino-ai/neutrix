"""Async streaming chat agent with OpenAI-style tool calling."""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from neutrix.config import Slot
from neutrix.tools import dispatch, get_schemas

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Keep it simple."


@dataclass
class AgentEvent:
    """Stream event yielded by Agent.stream_reply."""

    kind: str  # "token" | "tool_call" | "tool_result" | "done" | "error" | "needs_tool"
    data: Any = None


@dataclass
class Agent:
    slot: Slot
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    use_tools: bool = True
    messages: list[dict[str, Any]] = field(default_factory=list)
    _client: AsyncOpenAI | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.messages:
            self.messages = [{"role": "system", "content": self.system_prompt}]
        self._rebuild_client()

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def switch(self, slot: Slot) -> None:
        self.slot = slot
        self._rebuild_client()

    def _rebuild_client(self) -> None:
        self._client = AsyncOpenAI(
            base_url=self.slot.base_url,
            api_key=self.slot.api_key,
        )

    async def stream_reply(self, user_text: str) -> AsyncIterator[AgentEvent]:
        """Append a user turn, stream assistant tokens, dispatch tool calls, repeat
        until the model stops with a non-tool finish_reason.
        """
        self.messages.append({"role": "user", "content": user_text})

        while True:
            try:
                async for ev in self._one_round():
                    yield ev
                    if ev.kind == "needs_tool":
                        break
                else:
                    yield AgentEvent("done")
                    return
                continue
            except Exception as e:
                logger.exception("agent round failed")
                yield AgentEvent("error", str(e))
                return

    async def _one_round(self) -> AsyncIterator[AgentEvent]:
        assert self._client is not None

        kwargs: dict[str, Any] = {
            "model": self.slot.model,
            "messages": self.messages,
            "stream": True,
        }
        if self.use_tools:
            kwargs["tools"] = get_schemas()

        content_parts: list[str] = []
        tool_calls: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None

        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                content_parts.append(delta.content)
                yield AgentEvent("token", delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    slot = tool_calls.setdefault(
                        idx, {"id": "", "name": "", "arguments": ""}
                    )
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function.arguments:
                            slot["arguments"] += tc.function.arguments

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts) or None,
        }
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls.values()
            ]
        self.messages.append(assistant_msg)

        if finish_reason == "tool_calls" or tool_calls:
            for tc in tool_calls.values():
                yield AgentEvent(
                    "tool_call",
                    {"name": tc["name"], "arguments": tc["arguments"]},
                )
                result = dispatch(tc["name"], tc["arguments"])
                yield AgentEvent(
                    "tool_result",
                    {"name": tc["name"], "result": result},
                )
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    }
                )
            yield AgentEvent("needs_tool")
            return
