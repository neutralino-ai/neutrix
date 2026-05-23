"""OpenAI-compatible streaming LLM client.

This layer performs one model request and translates provider chunks into
small events. It does not mutate conversation history and does not run tools.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from neutrix.config import Slot


@dataclass(frozen=True)
class LLMResponse:
    message: dict[str, Any]
    finish_reason: str | None


@dataclass(frozen=True)
class LLMEvent:
    kind: str  # "token" | "assistant"
    data: Any = None


class OpenAIChatLLM:
    """One-request OpenAI Chat Completions streaming adapter."""

    def __init__(self, slot: Slot) -> None:
        self.slot = slot
        self._client = self._build_client(slot)

    def switch(self, slot: Slot) -> None:
        self.slot = slot
        self._client = self._build_client(slot)

    def _build_client(self, slot: Slot) -> AsyncOpenAI:
        return AsyncOpenAI(base_url=slot.base_url, api_key=slot.api_key)

    async def stream_response(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

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
                yield LLMEvent("token", delta.content)

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    idx = tool_call.index
                    pending = tool_calls.setdefault(
                        idx, {"id": "", "name": "", "arguments": ""}
                    )
                    if tool_call.id:
                        pending["id"] = tool_call.id
                    if tool_call.function:
                        if tool_call.function.name:
                            pending["name"] = tool_call.function.name
                        if tool_call.function.arguments:
                            pending["arguments"] += tool_call.function.arguments

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts) or None,
        }
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                    },
                }
                for tool_call in tool_calls.values()
            ]

        yield LLMEvent(
            "assistant",
            LLMResponse(message=assistant_msg, finish_reason=finish_reason),
        )
