"""OpenAI-compatible streaming LLM client.

This layer performs one model request and translates provider chunks into
small events. It does not mutate conversation history and does not run tools.

``OpenAIChatLLM`` uses ``stream=True`` so token deltas surface as
``LLMEvent("token", str)`` events as they arrive, and so
:py:meth:`OpenAIChatLLM.stop` can close the underlying HTTP stream and
return control to the caller without waiting for the rest of the response.
The cancellation hook is what v0.9.2's Esc / Ctrl+C broadcast hangs on.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from neutrix.config import Slot


@dataclass(frozen=True)
class LLMResponse:
    message: dict[str, Any]
    finish_reason: str | None


@dataclass(frozen=True)
class LLMEvent:
    kind: str  # "token" (str delta) | "assistant" (LLMResponse)
    data: Any = None


class OpenAIChatLLM:
    """Streaming OpenAI Chat Completions adapter."""

    def __init__(self, slot: Slot) -> None:
        self.slot = slot
        self._client = self._build_client(slot)
        self._active_stream: Any = None

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
            "stream": True,
        }
        if system_text:
            kwargs["extra_body"] = {"system": system_text}
        if tools:
            kwargs["tools"] = tools

        stream = await self._client.chat.completions.create(**kwargs)
        self._active_stream = stream

        content_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        try:
            async for chunk in stream:
                choice = self._first_choice(chunk)
                if choice is None:
                    continue
                reason = self._read(choice, "finish_reason")
                if reason is not None:
                    finish_reason = reason
                delta = self._read(choice, "delta", {}) or {}
                content_delta = self._read(delta, "content")
                if isinstance(content_delta, str) and content_delta:
                    content_parts.append(content_delta)
                    yield LLMEvent("token", content_delta)
                tc_delta = self._read(delta, "tool_calls")
                if tc_delta:
                    self._accumulate_tool_calls(tool_calls_by_index, tc_delta)
        finally:
            # PEP 525 safe — no yield in finally. Pure assignment.
            self._active_stream = None

        content = "".join(content_parts) if content_parts else None
        tool_calls = self._finalize_tool_calls(tool_calls_by_index)

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls

        yield LLMEvent(
            "assistant",
            LLMResponse(message=assistant_msg, finish_reason=finish_reason),
        )

    def stop(self) -> None:
        """Abort the currently-iterating stream, if any.

        Closes the SDK's ``AsyncStream``, which tears down the HTTP
        connection so the iterator's next ``__anext__`` raises
        :class:`StopAsyncIteration` (or yields nothing further) and the
        wrapping ``async for`` exits. Synchronous so the controller can
        broadcast cancel from any task — including the one currently
        iterating the stream.

        Idempotent: a no-op when ``_active_stream`` is ``None``.
        Best-effort: exceptions from ``stream.close()`` are swallowed
        so the cancel broadcast can never raise.
        """
        stream = self._active_stream
        if stream is None:
            return
        try:
            close = getattr(stream, "close", None)
            if close is None:
                return
            result = close()
            # The SDK's AsyncStream.close may return a coroutine; we
            # can't await synchronously, but the side-effect of issuing
            # the close call is enough to unblock the iterator.
            if hasattr(result, "__await__") or hasattr(result, "close"):
                # Coroutine returned but we can't await it here — log
                # and let GC pick it up. The HTTP connection is already
                # being torn down by the SDK at this point.
                try:
                    result.close()  # type: ignore[union-attr]
                except Exception:
                    pass
        except Exception as exc:
            logger.debug("OpenAIChatLLM.stop swallowed: {}", exc)

    def _first_choice(self, completion: Any) -> Any | None:
        choices = self._read(completion, "choices", []) or []
        return choices[0] if choices else None

    def _read(self, value: Any, key: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(key, default)
        return getattr(value, key, default)

    def _accumulate_tool_calls(
        self,
        accumulator: dict[int, dict[str, Any]],
        deltas: Any,
    ) -> None:
        """Fold streaming tool_call deltas onto the index-keyed accumulator.

        OpenAI's streaming tool_calls arrive as a sequence of deltas
        keyed by ``index``. The first delta typically carries
        ``id`` + ``function.name``; subsequent deltas append to
        ``function.arguments``. We rebuild the final tool_calls list
        from this accumulator at end-of-stream.
        """
        for raw in deltas:
            index = self._read(raw, "index", 0) or 0
            slot = accumulator.setdefault(
                index,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            call_id = self._read(raw, "id")
            if call_id:
                slot["id"] = str(call_id)
            call_type = self._read(raw, "type")
            if call_type:
                slot["type"] = str(call_type)
            function = self._read(raw, "function")
            if function is None:
                continue
            name = self._read(function, "name")
            if name:
                slot["function"]["name"] = str(name)
            arguments = self._read(function, "arguments")
            if arguments:
                slot["function"]["arguments"] += str(arguments)

    def _finalize_tool_calls(
        self,
        accumulator: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not accumulator:
            return []
        return [accumulator[index] for index in sorted(accumulator)]

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
