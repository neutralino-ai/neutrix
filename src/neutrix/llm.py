"""OpenAI-compatible LLM client (non-streaming for v0.9.3).

v0.9.3 reverts the v0.9.2 ``stream=True`` switch. ``stream_response`` makes
one non-streaming Chat Completions call and yields exactly one
``LLMEvent("assistant", LLMResponse(...))``. Streaming re-enables in a later
PRD with the CC-aligned "keep partial text" semantic.

Two responsibilities live here:

1. **Outgoing-payload validity.** :func:`_ensure_tool_result_pairing` is a
   pure transform on the outgoing message list — dedup ``role:tool``
   messages by ``tool_call_id`` (first wins) and synthesize a
   ``role:tool`` placeholder for any orphan ``tool_use`` in the latest
   assistant message. Synthetic content is conditional on whether
   ``"[interrupted by user]"`` appears after the orphan, so a cancelled
   tool sees ``"[cancelled by user]"`` and a defensive missing-result
   sees ``"[tool result missing]"``. Pure transform on a copy — does
   NOT mutate ``messages`` (the ContextManager-as-sole-mutator rule).

2. **Cancellation.** :py:meth:`OpenAIChatLLM.stop` cancels the awaiting
   ``create`` task so the wrapping ``await`` raises ``CancelledError``
   and the caller unwinds. Idempotent.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from neutrix.config import Slot

INTERRUPTED_BY_USER_MARKER = "[interrupted by user]"
CANCELLED_TOOL_RESULT = "[cancelled by user]"
MISSING_TOOL_RESULT = "[tool result missing]"


@dataclass(frozen=True)
class LLMResponse:
    message: dict[str, Any]
    finish_reason: str | None


@dataclass(frozen=True)
class LLMEvent:
    kind: str  # "assistant" (LLMResponse)
    data: Any = None


def _ensure_tool_result_pairing(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a new list with tool_call/tool_result pairing enforced.

    Pure function — does not mutate ``messages``. Two passes:

    1. **Dedup:** for each ``tool_call_id`` referenced by a ``role:tool``
       message, keep only the FIRST occurrence; drop later duplicates.
       Pessimistic safety net for transcripts that have somehow grown
       redundant tool replies.
    2. **Synthesize:** for each ``tool_call.id`` in the LATEST assistant
       message with ``tool_calls`` that has no matching ``role:tool``
       message *anywhere* after it, insert a synthetic
       ``{"role": "tool", "tool_call_id": id, "content": ...}``
       *immediately after* the orphan assistant message. Synthetic
       content is:
         - :data:`CANCELLED_TOOL_RESULT` (``"[cancelled by user]"``) if any
           ``role:user`` message with content
           :data:`INTERRUPTED_BY_USER_MARKER` appears AFTER the orphan
           assistant message in the list — that's the cancel signature.
         - :data:`MISSING_TOOL_RESULT` (``"[tool result missing]"``)
           otherwise — defensive placeholder.

    Multi-tool-call assistant messages with partial pairing are
    handled: only the unpaired ``tool_call.id``s get synthetic results;
    the existing ``role:tool`` messages stay in place.

    Only the LATEST assistant-with-tool_calls can have orphans —
    earlier ones were paired during prior rounds. We scan the latest
    only.
    """
    # Pass 1 — dedup tool messages by tool_call_id.
    deduped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for msg in messages:
        if not isinstance(msg, dict):
            deduped.append(msg)
            continue
        if msg.get("role") == "tool":
            tcid = str(msg.get("tool_call_id") or "")
            if tcid and tcid in seen_ids:
                # Drop duplicate.
                continue
            if tcid:
                seen_ids.add(tcid)
        deduped.append(dict(msg))

    # Find the latest assistant-with-tool_calls index in the deduped list.
    latest_assistant_idx: int | None = None
    latest_tool_calls: list[dict[str, Any]] = []
    for i in range(len(deduped) - 1, -1, -1):
        msg = deduped[i]
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        tcs = msg.get("tool_calls")
        if isinstance(tcs, list) and tcs:
            latest_assistant_idx = i
            latest_tool_calls = [tc for tc in tcs if isinstance(tc, dict)]
            break

    if latest_assistant_idx is None:
        return deduped

    # Pass 2 — find orphan tool_call ids in the latest assistant message.
    after = deduped[latest_assistant_idx + 1 :]
    paired_ids: set[str] = set()
    cancel_marker_after = False
    for msg in after:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool":
            tcid = str(msg.get("tool_call_id") or "")
            if tcid:
                paired_ids.add(tcid)
        elif msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str) and content == INTERRUPTED_BY_USER_MARKER:
                cancel_marker_after = True

    synthetic_content = (
        CANCELLED_TOOL_RESULT if cancel_marker_after else MISSING_TOOL_RESULT
    )

    synthetics: list[dict[str, Any]] = []
    for tc in latest_tool_calls:
        tcid = str(tc.get("id") or "")
        if not tcid or tcid in paired_ids:
            continue
        synthetics.append(
            {
                "role": "tool",
                "tool_call_id": tcid,
                "content": synthetic_content,
            }
        )

    if not synthetics:
        return deduped

    # Insert synthetic tool messages immediately after the latest
    # assistant-with-tool_calls message.
    return (
        deduped[: latest_assistant_idx + 1]
        + synthetics
        + deduped[latest_assistant_idx + 1 :]
    )


class OpenAIChatLLM:
    """Non-streaming OpenAI Chat Completions adapter.

    One API call per :py:meth:`stream_response`; one
    ``LLMEvent("assistant", LLMResponse(...))`` emitted. ``stop()``
    cancels the awaiting ``create`` task so a parked
    ``stream_response`` returns via ``CancelledError``.
    """

    def __init__(self, slot: Slot) -> None:
        self.slot = slot
        self._client = self._build_client(slot)
        self._active_task: asyncio.Task[Any] | None = None

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
        paired = _ensure_tool_result_pairing(messages)
        outbound_messages, system_text = self._outbound_prompt(paired)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": outbound_messages,
            "stream": False,
        }
        if system_text:
            kwargs["extra_body"] = {"system": system_text}
        if tools:
            kwargs["tools"] = tools

        task = asyncio.ensure_future(self._client.chat.completions.create(**kwargs))
        self._active_task = task
        try:
            completion = await task
        finally:
            # PEP 525 safe — pure assignment, no yield in finally.
            self._active_task = None

        choice = self._first_choice(completion)
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": None}
        finish_reason: str | None = None
        if choice is not None:
            finish_reason = self._read(choice, "finish_reason")
            raw_message = self._read(choice, "message")
            if raw_message is not None:
                assistant_msg = self._coerce_message(raw_message)

        yield LLMEvent(
            "assistant",
            LLMResponse(message=assistant_msg, finish_reason=finish_reason),
        )

    def stop(self) -> None:
        """Cancel the awaiting ``create`` task, if any.

        Calls ``Task.cancel()`` so the parked ``await`` in
        :py:meth:`stream_response` raises :class:`asyncio.CancelledError`
        and the caller unwinds. Synchronous so the cancel broadcast can
        run from any task. Idempotent — a no-op when no request is in
        flight. Best-effort: exceptions are swallowed so the cancel
        broadcast never raises.
        """
        task = self._active_task
        if task is None:
            return
        if task.done():
            return
        try:
            task.cancel()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("OpenAIChatLLM.stop swallowed: {}", exc)

    def _first_choice(self, completion: Any) -> Any | None:
        choices = self._read(completion, "choices", []) or []
        return choices[0] if choices else None

    def _read(self, value: Any, key: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(key, default)
        return getattr(value, key, default)

    def _coerce_message(self, raw: Any) -> dict[str, Any]:
        """Normalize the SDK's message object to a plain dict.

        Handles both dict payloads (some gateways) and SDK
        :class:`openai.types.chat.ChatCompletionMessage` instances.
        Preserves ``role``, ``content``, ``tool_calls``.
        """
        role = self._read(raw, "role", "assistant") or "assistant"
        content = self._read(raw, "content")
        if content is not None and not isinstance(content, str):
            content = str(content)
        message: dict[str, Any] = {"role": str(role), "content": content}
        tool_calls = self._read(raw, "tool_calls")
        if tool_calls:
            message["tool_calls"] = [self._coerce_tool_call(tc) for tc in tool_calls]
        return message

    def _coerce_tool_call(self, raw: Any) -> dict[str, Any]:
        function = self._read(raw, "function")
        fn_name = self._read(function, "name", "") if function is not None else ""
        fn_args = self._read(function, "arguments", "") if function is not None else ""
        return {
            "id": str(self._read(raw, "id") or ""),
            "type": str(self._read(raw, "type") or "function"),
            "function": {
                "name": str(fn_name or ""),
                "arguments": str(fn_args or ""),
            },
        }

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
