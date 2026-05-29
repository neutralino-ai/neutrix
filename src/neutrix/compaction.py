"""Mechanical ``/compact`` — drop the oldest ~50 % of messages, no LLM.

v0.9.6 stopgap until v0.10.5 ships smart (summary-based) compaction.
The cut is round-boundary-safe: after computing a naive drop count it
snaps the cut *forward* past any ``role:tool`` message, so the kept
tail never begins on an orphan ``tool_result`` — which
:func:`neutrix.llm._ensure_tool_result_pairing` does NOT repair (it
only synthesizes results for orphan ``tool_use``). See
``docs/PRDs/v0.9.6-emergency-compact.md`` and
``docs/splits/v0.9.6-emergency-compact.html``.

v0.10.5 reuses :func:`compact_messages`'s cut computation and swaps the
placeholder-insertion step for a summarizer call against the dropped
slice.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from loguru import logger

from neutrix.llm import LLMEvent, LLMResponse

COMPACT_MARKER_OPEN = "<system-compact>"
COMPACT_MARKER_CLOSE = "</system-compact>"
# v0.10.5: a summary-based compaction replaces the cut segment with this.
SUMMARY_MARKER_OPEN = "<system-summary>"
SUMMARY_MARKER_CLOSE = "</system-summary>"

COMPACTION_SYSTEM_PROMPT = (
    "You compress a conversation between a user and a coding assistant into a "
    "compact summary that lets the assistant continue without the original "
    "turns. Preserve: decisions made, open work items / next steps, key facts "
    "and file paths, and any tool results the assistant will still need. Drop "
    "chit-chat and superseded detail. Write 1-2 short paragraphs (aim ~200 "
    "tokens), in plain prose, third person. Output only the summary."
)

_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class CompactionOutcome:
    """Result of a compaction attempt.

    ``did_compact`` is ``False`` when the conversation was too short to
    drop anything; in that case the caller receives the message list
    unchanged.
    """

    did_compact: bool
    turns_dropped: int
    approx_tokens_dropped: int


def is_compact_marker(content: Any) -> bool:
    """Whether ``content`` is a ``<system-compact>`` placeholder body.

    Unused by the v0.9.6 renderer — the live ``/compact`` path
    suppresses re-print and the marker renders raw on a later ``/load``.
    Provided for v0.10.2 visibility-parity to pick up, mirroring
    :func:`neutrix.context_manager.is_task_reminder`.
    """
    return isinstance(content, str) and content.startswith(COMPACT_MARKER_OPEN)


def compact_messages(
    messages: list[dict[str, Any]],
    *,
    keep_ratio: float = 0.5,
) -> tuple[list[dict[str, Any]], CompactionOutcome]:
    """Drop the oldest ``1 - keep_ratio`` of the non-system body.

    Pure — never mutates ``messages``. Steps:

    1. Preserve the leading ``role:system`` prefix verbatim (normally
       one message; robust to zero or several).
    2. Drop ``floor(len(body) * (1 - keep_ratio))`` of the oldest body
       messages. If that is zero, no-op (``did_compact=False``).
    3. Snap the cut *forward* past any ``role:tool`` message so the
       first kept message is never an orphan ``tool_result``.
    4. Insert one ``<system-compact>`` ``role:user`` placeholder between
       the system prefix and the kept tail.

    Returns ``(new_messages, outcome)``. When nothing can be dropped the
    original list is returned (as a fresh copy) with
    ``did_compact=False``.
    """
    prefix_len = _system_prefix_len(messages)
    body = messages[prefix_len:]
    drop = math.floor(len(body) * (1.0 - keep_ratio))
    if drop <= 0:
        return list(messages), CompactionOutcome(False, 0, 0)

    cut = drop
    while cut < len(body) and _is_tool_message(body[cut]):
        cut += 1
    if cut >= len(body):
        # Forward-snap consumed the whole body — no safe tail to keep.
        return list(messages), CompactionOutcome(False, 0, 0)

    dropped = body[:cut]
    kept = body[cut:]
    new_messages = [*messages[:prefix_len], _make_marker(len(dropped)), *kept]
    outcome = CompactionOutcome(
        did_compact=True,
        turns_dropped=len(dropped),
        approx_tokens_dropped=_approx_tokens(dropped),
    )
    return new_messages, outcome


def _system_prefix_len(messages: list[dict[str, Any]]) -> int:
    count = 0
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            count += 1
        else:
            break
    return count


def _is_tool_message(msg: Any) -> bool:
    return isinstance(msg, dict) and msg.get("role") == "tool"


def _make_marker(n_dropped: int) -> dict[str, Any]:
    return {
        "role": "user",
        "content": (
            f"{COMPACT_MARKER_OPEN}{n_dropped} earlier turns removed by "
            f"/compact (no summary){COMPACT_MARKER_CLOSE}"
        ),
    }


def _approx_tokens(messages: list[dict[str, Any]]) -> int:
    """Whitespace word-count token estimate over message content.

    Same heuristic as :func:`neutrix.terminal_chat.approximate_token_count`;
    the ``~`` in the user-facing notice signals it is approximate, so a
    real tokenizer dependency is unwarranted for a mechanical stopgap.
    """
    total = 0
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, str):
            total += len(_WORD_RE.findall(content))
    return total


# ===== v0.10.5: smart (summary) compaction + the >1M hardening ============
# CompactionEvent lives in neutrix.store (pure data, next to Task/ToolRecord)
# so the leaf store module doesn't import this LLM-dependent module.


def is_summary_marker(content: Any) -> bool:
    """Whether ``content`` is a v0.10.5 ``<system-summary>`` body."""
    return isinstance(content, str) and content.startswith(SUMMARY_MARKER_OPEN)


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Cheap whole-payload token estimate: ``len(json) // 4`` (PRD heuristic).

    Counts the full serialized payload (roles, tool_calls, tool_call_ids), not
    just text, so it tracks what the provider actually weighs. Approximate by
    design — a real tokenizer dependency isn't worth it for a budget gate.
    """
    try:
        serialized = json.dumps(messages, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        serialized = str(messages)
    return len(serialized) // 4


def should_compact(
    messages: list[dict[str, Any]],
    *,
    max_context_tokens: int | None,
    threshold: float = 0.75,
) -> bool:
    """True when the payload exceeds ``threshold`` of the slot's window.

    ``max_context_tokens=None`` ⇒ the window is unknown ⇒ auto-compaction is
    disabled (the user can still compact manually).
    """
    if not max_context_tokens or max_context_tokens <= 0:
        return False
    return estimate_tokens(messages) > threshold * max_context_tokens


def _msg_tokens(msg: dict[str, Any]) -> int:
    try:
        return len(json.dumps(msg, default=str)) // 4
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return len(str(msg)) // 4


def _snap_forward_past_tools(body: list[dict[str, Any]], cut: int) -> int:
    """Move ``cut`` forward past ``role:tool`` messages so ``body[cut:]`` never
    begins on an orphan tool result (the pairing layer can't repair that)."""
    while cut < len(body) and _is_tool_message(body[cut]):
        cut += 1
    return cut


def compact_to_token_budget(
    messages: list[dict[str, Any]],
    *,
    budget: int,
) -> tuple[list[dict[str, Any]], CompactionOutcome]:
    """Hardening #1: drop oldest body messages (round-safe) until under ``budget``.

    The under-the-limit guarantee message-count halving can't give: keeps
    dropping the oldest until ``estimate_tokens < budget`` (or only the system
    prefix + a clean tail remain). Pure. Inserts a ``<system-compact>`` marker
    for the dropped turns so the cut is visible (like v0.9.6).
    """
    prefix_len = _system_prefix_len(messages)
    body = list(messages[prefix_len:])
    if estimate_tokens(messages) <= budget or not body:
        return list(messages), CompactionOutcome(False, 0, 0)
    drop = 0
    while drop < len(body):
        cut = _snap_forward_past_tools(body, drop + 1)
        if cut >= len(body):
            break
        candidate = [*messages[:prefix_len], _make_marker(cut), *body[cut:]]
        drop = cut
        if estimate_tokens(candidate) <= budget:
            return candidate, CompactionOutcome(True, cut, _approx_tokens(body[:cut]))
    # Could not get under budget without dropping everything; drop all body.
    return list(messages), CompactionOutcome(False, 0, 0)


def truncate_large_tool_results(
    messages: list[dict[str, Any]],
    *,
    cap: int = 8000,
) -> tuple[list[dict[str, Any]], int]:
    """Hardening #3: truncate within an oversized ``role:tool`` body.

    Message-dropping can't fix a single huge *recent* message; this caps each
    ``role:tool`` content to ``cap`` chars with a marker suffix. Pure; returns
    ``(new_messages, n_truncated)``.
    """
    out: list[dict[str, Any]] = []
    n = 0
    for msg in messages:
        if (
            isinstance(msg, dict)
            and msg.get("role") == "tool"
            and isinstance(msg.get("content"), str)
            and len(msg["content"]) > cap
        ):
            body = msg["content"][:cap]
            new = dict(msg)
            new["content"] = f"{body}\n…[truncated {len(msg['content']) - cap} chars]"
            out.append(new)
            n += 1
        else:
            out.append(msg)
    return out, n


async def summarize_segment(
    segment: list[dict[str, Any]],
    *,
    llm: Any,
    model: str,
) -> str:
    """Summarize a conversation segment via one LLM call; '' on empty/failure.

    The segment is rendered to a single text blob and sent as one user message
    (so the summarize call's own payload has no tool-pairing constraints).
    """
    convo = _render_segment_text(segment)
    request = [
        {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
        {"role": "user", "content": convo},
    ]
    assistant_msg: dict[str, Any] = {"role": "assistant", "content": None}
    async for event in llm.stream_response(model=model, messages=request, tools=None):
        if not isinstance(event, LLMEvent):  # pragma: no cover - defensive
            continue
        if event.kind == "assistant":
            payload = event.data
            if isinstance(payload, LLMResponse):
                assistant_msg = payload.message
            elif isinstance(payload, dict):
                assistant_msg = payload
    content = assistant_msg.get("content")
    return content.strip() if isinstance(content, str) else ""


def _render_segment_text(segment: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for msg in segment:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "?"))
        content = msg.get("content")
        text = content.strip() if isinstance(content, str) else ""
        if msg.get("tool_calls"):
            names = [
                str(tc.get("function", {}).get("name", "?"))
                for tc in msg["tool_calls"]
                if isinstance(tc, dict)
            ]
            text = (text + f" [calls: {', '.join(names)}]").strip()
        lines.append(f"{role}: {text}" if text else f"{role}: (empty)")
    return "\n".join(lines)


async def smart_compact(
    messages: list[dict[str, Any]],
    *,
    llm: Any,
    model: str,
    max_context_tokens: int | None,
    keep_ratio: float = 0.5,
) -> tuple[list[dict[str, Any]], CompactionOutcome]:
    """Summary-based compaction (v0.10.5). Pure w.r.t. ``messages``.

    1. If already over ``max_context_tokens``, run a token-budget cut first so
       the summarize call is sendable (hardening #1 as the pre-summary primitive).
    2. Cut by token share: keep the most recent ``keep_ratio`` of the budget
       verbatim; the older segment (round-safe boundary) is summarized.
    3. Replace the segment with one ``<system-summary>`` ``role:user`` message.

    On an empty segment or a failed/empty summary, returns the original list
    with ``did_compact=False`` (caller surfaces a notice; no mutation).
    """
    working = list(messages)
    if max_context_tokens and estimate_tokens(working) > max_context_tokens:
        working, _ = compact_to_token_budget(
            working, budget=int(max_context_tokens * 0.8)
        )

    prefix_len = _system_prefix_len(working)
    body = working[prefix_len:]
    if not body:
        return list(messages), CompactionOutcome(False, 0, 0)

    budget_basis = max_context_tokens or estimate_tokens(working)
    keep_target = max(1, int(budget_basis * keep_ratio))
    kept_tokens = 0
    cut = 0
    for i in range(len(body) - 1, -1, -1):
        kept_tokens += _msg_tokens(body[i])
        if kept_tokens >= keep_target:
            cut = i
            break
    cut = _snap_forward_past_tools(body, cut)
    segment = body[:cut]
    kept = body[cut:]
    if not segment:
        return list(messages), CompactionOutcome(False, 0, 0)

    try:
        summary = await summarize_segment(
            [*working[:prefix_len], *segment], llm=llm, model=model
        )
    except Exception as exc:
        logger.warning("smart_compact summarize failed: {}", exc)
        return list(messages), CompactionOutcome(False, 0, 0)
    if not summary:
        return list(messages), CompactionOutcome(False, 0, 0)

    marker = {
        "role": "user",
        "content": f"{SUMMARY_MARKER_OPEN}{summary}{SUMMARY_MARKER_CLOSE}",
    }
    new_messages = [*working[:prefix_len], marker, *kept]
    return new_messages, CompactionOutcome(
        did_compact=True,
        turns_dropped=len(segment),
        approx_tokens_dropped=_approx_tokens(segment),
    )
