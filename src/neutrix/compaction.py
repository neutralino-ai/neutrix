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

import math
import re
from dataclasses import dataclass
from typing import Any

COMPACT_MARKER_OPEN = "<system-compact>"
COMPACT_MARKER_CLOSE = "</system-compact>"

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
