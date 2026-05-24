"""OpenAI-compatible final-response LLM client.

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
    kind: str  # "assistant"; token events are reserved for future streaming
    data: Any = None


class OpenAIChatLLM:
    """One-request OpenAI Chat Completions adapter."""

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
        outbound_messages, system_text = self._outbound_prompt(messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": outbound_messages,
            "stream": False,
        }
        if system_text:
            kwargs["extra_body"] = {"system": system_text}
        if tools:
            kwargs["tools"] = tools

        completion = await self._client.chat.completions.create(**kwargs)
        choice = self._first_choice(completion)

        finish_reason = self._read(choice, "finish_reason") if choice else None
        message = self._read(choice, "message", {}) if choice else {}
        content = self._read(message, "content")
        tool_calls = self._tool_calls(self._read(message, "tool_calls"))

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content or None}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls

        yield LLMEvent(
            "assistant",
            LLMResponse(message=assistant_msg, finish_reason=finish_reason),
        )

    def _first_choice(self, completion: Any) -> Any | None:
        choices = self._read(completion, "choices", []) or []
        return choices[0] if choices else None

    def _read(self, value: Any, key: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(key, default)
        return getattr(value, key, default)

    def _tool_calls(self, raw_tool_calls: Any) -> list[dict[str, Any]]:
        if not raw_tool_calls:
            return []

        tool_calls: list[dict[str, Any]] = []
        for raw_tool_call in raw_tool_calls:
            function = self._read(raw_tool_call, "function")
            if function is None:
                continue
            tool_calls.append(
                {
                    "id": str(self._read(raw_tool_call, "id", "") or ""),
                    "type": str(self._read(raw_tool_call, "type", "function") or "function"),
                    "function": {
                        "name": str(self._read(function, "name", "") or ""),
                        "arguments": str(self._read(function, "arguments", "") or ""),
                    },
                }
            )
        return tool_calls

    def _outbound_prompt(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str | None]:
        if not self._uses_anthropic_messages_gateway():
            return messages, None

        system_parts: list[str] = []
        outbound: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content") or ""
            if role == "system":
                system_parts.append(str(content))
            elif role in {"user", "assistant"}:
                outbound.append({"role": role, "content": content})
            elif role == "tool":
                outbound.append({"role": "user", "content": f"Tool result:\n{content}"})

        system_text = "\n\n".join(part for part in system_parts if part) or None
        return outbound, system_text

    def _uses_anthropic_messages_gateway(self) -> bool:
        return (
            self.slot.provider.lower() == "ihep"
            and self.slot.model.lower().startswith("anthropic/")
        )
