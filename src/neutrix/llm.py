"""OpenAI-compatible streaming LLM client (v0.10.1).

v0.10.1 restores the ``stream=True`` path (removed in v0.9.3) merged with
v0.9.3's pairing layer. ``stream_response`` yields one ``LLMEvent("token", str)``
per content delta as it arrives, accumulates ``tool_calls`` index-keyed across
deltas, and yields one terminal ``LLMEvent("assistant", LLMResponse(...))`` with
the assembled message. Cancel keeps whatever text arrived — the
``ContextManager`` stashes it and commits it on cancel (PRD v0.10.1).

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

2. **Cancellation.** :py:meth:`OpenAIChatLLM.stop` closes the active
   ``AsyncStream`` (v0.9.2 eager teardown) so the iterator exits; the
   ``ContextManager`` drive-task-cancel additionally raises
   ``CancelledError`` into the iteration. Idempotent, best-effort.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger
from openai import AsyncOpenAI

from neutrix.config import Slot

INTERRUPTED_BY_USER_MARKER = "[interrupted by user]"
CANCELLED_TOOL_RESULT = "[cancelled by user]"
MISSING_TOOL_RESULT = "[tool result missing]"
# v1.6.1: placeholder for an empty assistant turn (null/blank content AND no
# tool_calls) on outbound. Such a turn — left by a dropped or no-text response
# (e.g. the pre-v1.6.1 anthropic-SSE bug) — makes strict backends
# (``openai/gpt-5.5``, the IHEP anthropic gateway) return an EMPTY reply with no
# error; ``deepseek/*`` tolerates it. Repaired to a placeholder rather than
# dropped, to preserve user↔assistant alternation (a drop creates consecutive
# user turns that strict backends reject).
EMPTY_ASSISTANT_PLACEHOLDER = "[no response]"

# v1.6.1: Anthropic Messages SSE — stop_reason → OpenAI finish_reason, and the
# event ``type``s we parse off ``chunk.model_extra`` (see
# :py:meth:`OpenAIChatLLM._anthropic_event`).
_ANTHROPIC_STOP_REASON = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "stop",
}
_ANTHROPIC_EVENT_TYPES = frozenset(
    {
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
        "ping",
    }
)


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


def _repair_empty_assistants(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a new list where every *empty assistant turn* gets a placeholder.

    An assistant message whose content is null/blank/whitespace AND which has
    **no** ``tool_calls`` is an empty turn — a dropped or no-text response.
    Sent verbatim it makes strict backends (``openai/gpt-5.5``, the IHEP
    anthropic gateway) return an EMPTY reply with no error (``deepseek/*``
    tolerates it), and a single such turn anywhere in history poisons the whole
    request. We replace the content with
    :data:`EMPTY_ASSISTANT_PLACEHOLDER` so the turn is API-valid everywhere
    while keeping role alternation intact (a drop would leave consecutive user
    turns that strict backends also reject).

    A null-content assistant *with* ``tool_calls`` is the normal text-free
    tool-call shape — left untouched. Pure transform on a copy — does NOT
    mutate ``messages`` (the ContextManager-as-sole-mutator rule). Runs on
    outbound so it heals histories already carrying such turns, not just
    prevents new ones.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if (
            isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and not msg.get("tool_calls")
        ):
            content = msg.get("content")
            if not (isinstance(content, str) and content.strip()):
                patched = dict(msg)
                patched["content"] = EMPTY_ASSISTANT_PLACEHOLDER
                out.append(patched)
                continue
        out.append(msg)
    return out


def _ensure_sdk_compliant(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The outbound payload-compliance layer — every transform that makes the
    message list valid for the API lives here, in one named seam. Pure
    (operates on copies; does not mutate ``messages``).

    Composes, in order:

    1. :func:`_ensure_tool_result_pairing` — dedup duplicate tool results and
       synthesize a result for any orphan ``tool_call``.
    2. :func:`_repair_empty_assistants` — replace an empty assistant turn
       (null/blank content AND no ``tool_calls``) with a placeholder so a
       strict backend doesn't choke on the empty block.

    Runs **before** :py:meth:`OpenAIChatLLM._outbound_prompt` so both steps see
    the canonical OpenAI-shaped message (the empty-block repair can tell a
    genuine empty turn from a text-free tool call by its ``tool_calls`` — a
    distinction the anthropic-gateway translation erases). New API-validity
    repairs belong here, beside the tool-pairing fix — the single seam the
    streaming send routes through.
    """
    out = _ensure_tool_result_pairing(messages)
    out = _repair_empty_assistants(out)
    return out


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
        self._active_stream: Any = None

    def switch(self, slot: Slot) -> None:
        self.slot = slot
        self._client = self._build_client(slot)

    def _build_client(self, slot: Slot) -> AsyncOpenAI:
        # v1.4.9: explicit transport timeout so a dead/half-closed connection
        # (e.g. a local proxy that dropped its upstream — observed as a
        # CLOSE-WAIT socket) raises ReadTimeout instead of hanging the turn
        # forever. ``read`` is the max gap between streamed chunks, so it
        # doubles as a transport-level no-progress cap: no bytes for
        # ``llm_timeout_s`` → APITimeoutError → a visible [LLM error], never an
        # infinite hang. ``connect`` stays short so an unreachable proxy fails
        # fast. This is the belt to the ContextManager watchdog's suspenders.
        return AsyncOpenAI(
            base_url=slot.base_url,
            api_key=slot.api_key,
            timeout=httpx.Timeout(slot.llm_timeout_s, connect=10.0),
            max_retries=2,
        )

    async def stream_response(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        compliant = _ensure_sdk_compliant(messages)
        outbound_messages, system_text = self._outbound_prompt(compliant)
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
                    # v1.6.1: the IHEP gateway streams Anthropic Messages SSE for
                    # anthropic/* models — there is no OpenAI ``choices``; the raw
                    # event survives in ``chunk.model_extra``. Fold its text /
                    # tool_use / stop_reason onto the SAME accumulators.
                    event = self._anthropic_event(chunk)
                    if event is not None:
                        text, a_reason = self._apply_anthropic_event(
                            event, content_parts, tool_calls_by_index
                        )
                        if a_reason is not None:
                            finish_reason = a_reason
                        if text:
                            yield LLMEvent("token", text)
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
            # PEP 525 safe — pure assignment, no yield in finally.
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
        """Close the active ``AsyncStream`` so the iterator exits.

        v0.10.1 mechanism: tearing the stream down makes the wrapping
        ``async for`` in :py:meth:`stream_response` stop, returning control
        to the caller. Synchronous so the cancel broadcast can run from any
        task. Idempotent — a no-op when no stream is in flight. Best-effort:
        exceptions are swallowed so the cancel broadcast never raises. The
        SDK's ``AsyncStream.close`` may return a coroutine that we can't await
        here; issuing the call is enough to begin teardown, and the
        ContextManager's drive-task-cancel unwinds the iteration regardless.
        """
        stream = self._active_stream
        if stream is None:
            return
        try:
            close = getattr(stream, "close", None)
            if close is None:
                return
            result = close()
            if hasattr(result, "close"):
                try:
                    result.close()
                except Exception:  # pragma: no cover - defensive
                    pass
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("OpenAIChatLLM.stop swallowed: {}", exc)

    def _accumulate_tool_calls(
        self,
        accumulator: dict[int, dict[str, Any]],
        deltas: Any,
    ) -> None:
        """Fold streaming tool_call deltas onto the index-keyed accumulator.

        OpenAI streaming tool_calls arrive keyed by ``index``: the first delta
        carries ``id`` + ``function.name``; later deltas append to
        ``function.arguments``. Rebuilt into the final list at end-of-stream.
        """
        for raw in deltas:
            index = self._read(raw, "index", 0) or 0
            slot = accumulator.setdefault(
                index,
                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
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

    def _anthropic_event(self, chunk: Any) -> dict[str, Any] | None:
        """Return the Anthropic Messages SSE event carried on ``chunk``, or None.

        The ``AsyncOpenAI`` SDK cannot map Anthropic events to ``choices`` (so
        :py:meth:`_first_choice` is None) but stashes the raw event dict in
        ``chunk.model_extra`` (pydantic extra-allow). Some gateways stream plain
        dict chunks instead, so we also accept a bare dict. A chunk counts as
        Anthropic only when that extra carries a known Anthropic event ``type``
        — so an OpenAI usage-only final chunk (empty ``choices``, no Anthropic
        ``type``) is never misrouted here.
        """
        extra = getattr(chunk, "model_extra", None)
        if extra is None and isinstance(chunk, dict):
            extra = chunk
        if not isinstance(extra, dict):
            return None
        etype = extra.get("type")
        if isinstance(etype, str) and etype in _ANTHROPIC_EVENT_TYPES:
            return extra
        return None

    def _apply_anthropic_event(
        self,
        event: dict[str, Any],
        content_parts: list[str],
        tool_calls_by_index: dict[int, dict[str, Any]],
    ) -> tuple[str | None, str | None]:
        """Fold one Anthropic Messages event onto the streaming accumulators.

        Returns ``(text_to_yield, finish_reason)``: ``text_to_yield`` is a text
        delta to surface as a ``token`` event (or None); ``finish_reason`` is
        the mapped OpenAI reason once ``message_delta`` carries a stop_reason
        (or None). Accumulates into the SAME ``content_parts`` /
        ``tool_calls_by_index`` as the OpenAI branch, so
        :py:meth:`_finalize_tool_calls` and the terminal assembly are
        unchanged. A block's ``delta.type`` (``text_delta`` vs
        ``input_json_delta``) is authoritative; ``thinking_delta`` and other
        block kinds are ignored (thinking display is a non-goal).
        """
        etype = event.get("type")
        if etype == "content_block_start":
            index = int(self._read(event, "index", 0) or 0)
            block = self._read(event, "content_block", {}) or {}
            if self._read(block, "type") == "tool_use":
                tool_calls_by_index[index] = {
                    "id": str(self._read(block, "id", "") or ""),
                    "type": "function",
                    "function": {
                        "name": str(self._read(block, "name", "") or ""),
                        "arguments": "",
                    },
                }
            return None, None
        if etype == "content_block_delta":
            index = int(self._read(event, "index", 0) or 0)
            delta = self._read(event, "delta", {}) or {}
            dtype = self._read(delta, "type")
            if dtype == "text_delta":
                text = self._read(delta, "text")
                if isinstance(text, str) and text:
                    content_parts.append(text)
                    return text, None
            elif dtype == "input_json_delta":
                partial = self._read(delta, "partial_json")
                call = tool_calls_by_index.get(index)
                if call is not None and isinstance(partial, str) and partial:
                    call["function"]["arguments"] += partial
            return None, None
        if etype == "message_delta":
            delta = self._read(event, "delta", {}) or {}
            stop = self._read(delta, "stop_reason")
            if isinstance(stop, str) and stop:
                return None, _ANTHROPIC_STOP_REASON.get(stop, "stop")
        return None, None

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
