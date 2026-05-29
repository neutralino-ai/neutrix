"""Append-only terminal chat renderer wired to the ContextManager.

v0.9.3 removes the v0.9.2 controller + agent_loop pair. The UI now:

- emits :class:`~neutrix.context_manager.UserMessageEvent` /
  :class:`CancelEvent` / :class:`ClearEvent` / etc. to the
  :class:`~neutrix.context_manager.ContextManager`;
- renders by subscribing to :py:meth:`ChatStore.changes`, walking new
  records as they arrive.

The view never mutates :class:`ChatStore` or the CM's ``messages``
list directly — that's the CM's job. Cancel is a sync method on the
CM (the key binding can't await), kept honest by the
:class:`CancelEvent` async surface for non-keyboard callers.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
import time
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, TextIO

from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from neutrix import transcript
from neutrix.advisor import Advisor
from neutrix.compaction import (
    SUMMARY_MARKER_CLOSE,
    SUMMARY_MARKER_OPEN,
    is_compact_marker,
    is_summary_marker,
)
from neutrix.config import SLOT_NAMES, Config, ConfigError, Slot, load_config
from neutrix.context_files import expand_at_mentions
from neutrix.context_manager import (
    ADVISOR_TAG_CLOSE,
    ADVISOR_TAG_OPEN,
    ClearEvent,
    ContextManager,
    ReplaceHistoryEvent,
    SlotSwitchEvent,
    State,
    UserMessageEvent,
    format_reminder_notice,
    is_advisor_message,
    is_task_reminder,
)
from neutrix.skills import discover_skills, render_skill, skills_signature
from neutrix.store import ChatStore, MessageRecord, Task, ToolRecord
from neutrix.tools import BUILTIN_TOOLS, dispatch, get_schemas

QUEUED_PREFIX = "› "  # noqa: RUF001  -- U+203A is the chosen UI glyph

MAX_PANEL_ROWS = 5

HEARTBEAT_GLYPH = "●"
# v0.9.8: the liveness pulse animates the dot's *presence* (an on/off
# wink), not its brightness. A 256-color terminal (the user's
# tmux-256color) exposes only ~22 distinct grays — the xterm 24-step gray
# ramp #080808..#eeeeee — so a brightness fade bands no matter how it is
# paced or how fast it refreshes: the palette, not the clock, is the
# ceiling (which is why the v0.9.5 jump to 120 Hz didn't help). A 2-state
# presence toggle has nothing to interpolate, so it stays smooth on every
# terminal. Follows Claude Code's tool-use loader
# (components/ToolUseLoader.tsx + hooks/useBlink.ts, BLINK_INTERVAL_MS=600).
HEARTBEAT_BLINK_INTERVAL_MS = 600  # one on/off toggle per tick → 1.2 s cycle
HEARTBEAT_GLYPH_STYLE = "fg:ansiwhite bold"
# Stalled glyph (v0.9.5 intent, v0.9.8 mechanism): the dot keeps winking
# but turns red — a discrete colour swap, no gradient — mapping to CC's
# `error` colour for a call that has waited too long.
HEARTBEAT_STALLED_GLYPH_STYLE = "fg:ansired bold"
# Single-knob stall threshold (v0.9.5 post-gate revision): the stall
# hint is derived from the slot's hard timeout rather than carrying a
# separate magic number, so raising llm_timeout_s for a slow model
# also pushes the hint out and stops it flickering on healthy slow
# calls. Floored so a tiny per-slot timeout still leaves an
# early-warning window.
HEARTBEAT_STALL_FRACTION = 1 / 6
HEARTBEAT_STALL_FLOOR_S = 10.0
HEARTBEAT_LABEL_STYLE = "fg:ansiwhite bold"


def stall_threshold_for(llm_timeout_s: float) -> float:
    """Derive the stall-hint threshold from the hard timeout.

    ``max(HEARTBEAT_STALL_FLOOR_S, llm_timeout_s * HEARTBEAT_STALL_FRACTION)``.
    At the 300 s default this is ~50 s; raise the per-slot timeout and
    the hint moves out with it. The floor keeps a usable window when a
    slot sets an aggressively short timeout.
    """
    return max(HEARTBEAT_STALL_FLOOR_S, llm_timeout_s * HEARTBEAT_STALL_FRACTION)


_TASK_PANEL_ICON = {
    "in_progress": "◼",
    "pending": "◻",
    "completed": "✓",
}

_TASK_PANEL_STYLE = {
    "in_progress": "fg:ansicyan bold",
    "pending": "",
    "completed": "fg:ansigreen",
}

_TASK_PANEL_ORDER = {"in_progress": 0, "pending": 1, "completed": 2}

WORD_RE = re.compile(r"\S+")
SYSTEM_STYLE = "dim yellow"
USER_STYLE = "cyan"
ASSISTANT_STYLE = "white"
# v0.10.2 visibility parity: fold the system prompt only when it's long enough
# to dominate session start; short prompts (incl. the default) stay inline.
SYSTEM_FOLD_THRESHOLD = 200


def result_line_count(result: str) -> int:
    """Count displayed result lines without inventing a line for empty output."""
    if not result:
        return 0
    return len(result.splitlines())


def approximate_token_count(result: str) -> int:
    """Cheap, deterministic token estimate based on non-whitespace chunks."""
    return len(WORD_RE.findall(result))


def format_token_count(n: int) -> str:
    """Render an approximate token count for the /compact notice."""
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


STREAM_PREVIEW_LINES = 8


def _stream_preview(pending: str | None) -> str:
    """Bounded tail of the in-progress assistant text for the live region (v1.4.7).

    Shows the last ``STREAM_PREVIEW_LINES`` lines (the full text is committed to
    scrollback on finish); ``…`` marks earlier lines elided.
    """
    if not pending:
        return ""
    lines = pending.splitlines() or [pending]
    tail = lines[-STREAM_PREVIEW_LINES:]
    prefix = "… " if len(lines) > STREAM_PREVIEW_LINES else ""
    return prefix + "\n".join(tail)


def compact_inline(value: object, *, limit: int = 160) -> str:
    """Collapse a value to one terminal line with a bounded length."""
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _task_sort_key(task: Task) -> tuple[int, int]:
    try:
        id_int = int(task.id)
    except (TypeError, ValueError):
        id_int = 0
    return (_TASK_PANEL_ORDER.get(task.status, 99), id_int)


def format_task_panel(tasks: tuple[Task, ...]) -> list[tuple[str, str]]:
    """Render the always-on task panel as prompt_toolkit fragments."""
    if not tasks:
        return []
    ordered = sorted(tasks, key=_task_sort_key)
    visible = ordered[:MAX_PANEL_ROWS]
    truncated = ordered[MAX_PANEL_ROWS:]

    fragments: list[tuple[str, str]] = []
    for task in visible:
        icon = _TASK_PANEL_ICON.get(task.status, "?")
        style = _TASK_PANEL_STYLE.get(task.status, "")
        fragments.append((style, f"  {icon} #{task.id} [{task.status}] {task.subject}\n"))

    if truncated:
        n_inprog = sum(1 for t in truncated if t.status == "in_progress")
        n_pending = sum(1 for t in truncated if t.status == "pending")
        n_done = sum(1 for t in truncated if t.status == "completed")
        parts: list[str] = []
        if n_inprog:
            parts.append(f"{n_inprog} in progress")
        if n_pending:
            parts.append(f"{n_pending} pending")
        if n_done:
            parts.append(f"{n_done} done")
        if parts:
            fragments.append(("fg:ansibrightblack", f"  … +{', '.join(parts)}\n"))
    return fragments


async def heartbeat_loop(
    state_supplier: Callable[[], State],
    store: ChatStore,
    on_tick: Callable[[], None],
    *,
    sleep_seconds: float = HEARTBEAT_BLINK_INTERVAL_MS / 1000,
    sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    on_enter_busy: Callable[[], None] | None = None,
) -> None:
    """Drive the heartbeat: wink while busy, wait on store changes when idle.

    While ``state_supplier()`` is busy (anything except
    :attr:`State.IDLE`), awaits ``sleep_fn(sleep_seconds)`` and then
    calls ``on_tick`` — one blink toggle per tick. When the state is
    :attr:`State.IDLE`, blocks on the next :py:meth:`ChatStore.changes`
    yield — CM state transitions always accompany a store mutation, so the
    next busy phase wakes the loop. Cleanly cancellable.

    ``on_enter_busy`` (if given) is called once on each IDLE→busy
    transition, before the first tick, so the caller can reset its blink
    phase and guarantee a turn opens on a *visible* dot rather than a
    blank one. The default ``sleep_fn`` is :func:`asyncio.sleep` (strict
    period — a presence wink desyncs badly under jitter). Tests inject a
    fast deterministic ``sleep_fn`` / ``on_enter_busy`` to stabilize timing.
    """
    if sleep_fn is None:
        sleep_fn = asyncio.sleep
    changes = store.changes()
    try:
        while True:
            while state_supplier() == State.IDLE:
                await changes.__anext__()
            if on_enter_busy is not None:
                on_enter_busy()
            while state_supplier() != State.IDLE:
                await sleep_fn(sleep_seconds)
                on_tick()
    finally:
        await changes.aclose()


def format_heartbeat(
    state: State,
    store: ChatStore,
    tick: int,
    *,
    last_progress_at: float | None = None,
    stall_threshold_s: float = HEARTBEAT_STALL_FLOOR_S,
) -> list[tuple[str, str]]:
    """Render the liveness pulse above the input as prompt_toolkit fragments.

    Returns ``[]`` when ``state == IDLE``. Otherwise returns two
    fragments: the liveness glyph and a static label naming the current
    phase. The glyph *winks* on/off by visibility — ``HEARTBEAT_GLYPH``
    on even ``tick``, a same-width blank on odd ``tick`` (v0.9.8 split #1:
    a presence toggle, not a brightness fade, so nothing quantizes on a
    256-color terminal). The dot is bright white normally.

    When ``state == AWAITING_LLM`` and ``last_progress_at`` is set and
    more than ``stall_threshold_s`` seconds ago, the glyph winks red
    (``HEARTBEAT_STALLED_GLYPH_STYLE``) and the label becomes
    ``"LLM (stalled)"`` — v0.9.5 stall hint, v0.9.8 colour swap. The hint
    is UI-only; the hard timeout is enforced by
    :class:`~neutrix.context_manager.ContextManager`'s watchdog.
    """
    if state == State.IDLE:
        return []
    is_stalled = (
        state == State.AWAITING_LLM
        and last_progress_at is not None
        and (time.monotonic() - last_progress_at) > stall_threshold_s
    )
    if state == State.AWAITING_LLM:
        label = "LLM (stalled)" if is_stalled else "LLM"
    elif state == State.AWAITING_EXECUTOR:
        pending = store.pending_tool_calls
        label = f"tool: {pending[0].name}" if pending else "tool"
    elif state == State.CANCELLING:
        label = "cancelling…"
    else:  # pragma: no cover - defensive for future states
        label = state.value
    visible = tick % 2 == 0
    glyph_style = HEARTBEAT_STALLED_GLYPH_STYLE if is_stalled else HEARTBEAT_GLYPH_STYLE
    # On/off wink: a same-width blank when off, so the label never shifts.
    glyph = f"{HEARTBEAT_GLYPH} " if visible else "  "
    return [
        (glyph_style, glyph),
        (HEARTBEAT_LABEL_STYLE, f"{label}\n"),
    ]


TOOL_KEYWORD_WIDTH = len("tool_result")


# v0.10.3: ``ToolRecord`` is now pure data in :mod:`neutrix.store`; the
# summary-rendering logic stays here in the view (it depends on the view's
# ``compact_inline``/``result_line_count``/``approximate_token_count`` helpers).


def tool_record_keyword(record: ToolRecord) -> str:
    # v0.10.2 split #5: a subagent (Agent tool) result reads as "subagent",
    # reusing the whole tool-result fold/expand path.
    return "subagent" if record.name == "Agent" else "tool_result"


def tool_record_summary_body(record: ToolRecord) -> str:
    args = compact_inline(record.arguments or "{}")
    lines = result_line_count(record.result)
    approx_tokens = approximate_token_count(record.result)
    return (
        f" [tool {record.index}] {record.name} {args} | folded | "
        f"{lines} lines | ~{approx_tokens} tokens"
    )


def tool_record_summary(record: ToolRecord) -> str:
    return f"<- {tool_record_keyword(record).ljust(TOOL_KEYWORD_WIDTH)}{tool_record_summary_body(record)}"


def tool_record_summary_parts(record: ToolRecord) -> tuple[str, str, str]:
    return (
        "<- ",
        tool_record_keyword(record).ljust(TOOL_KEYWORD_WIDTH),
        tool_record_summary_body(record),
    )


InputFunc = Callable[[str], str]


class DraftReader:
    """Bottom draft editor — same shape as v0.9.2."""

    def __init__(
        self,
        *,
        message_supplier: Callable[[], object] = lambda: "",
        cancel_hook: Callable[[], bool] | None = None,
        recall_provider: Callable[[], list[str]] | None = None,
    ) -> None:
        self._message_supplier = message_supplier
        self.quit_state = QuitArmingState()
        self.cancel_hook = cancel_hook
        self.recall_provider = recall_provider
        self.recall_state = RecallState()
        self._session = self._build_session()

    def read(self) -> str:
        if self._session is None:
            return input("")
        self.recall_state.reset()
        return self._session.prompt(handle_sigint=False)

    async def read_async(self) -> str:
        if self._session is None:
            return await asyncio.to_thread(input, "")
        self.recall_state.reset()
        return await self._session.prompt_async(handle_sigint=False)

    @property
    def uses_prompt_toolkit(self) -> bool:
        return self._session is not None

    def _build_session(self):
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.enums import EditingMode
            from prompt_toolkit.formatted_text import FormattedText
            from prompt_toolkit.history import InMemoryHistory
        except ImportError:
            return None

        placeholder = FormattedText(
            [("fg:ansibrightblack", "Message the assistant  (/help for commands)")]
        )
        return PromptSession(
            self._message_supplier,
            multiline=True,
            editing_mode=EditingMode.EMACS,
            erase_when_done=True,
            placeholder=placeholder,
            prompt_continuation="",
            history=InMemoryHistory(),
            key_bindings=build_draft_key_bindings(
                self.quit_state,
                cancel_hook=self.cancel_hook,
                recall_provider=self.recall_provider,
                recall_state=self.recall_state,
            ),
        )


@dataclass
class QuitArmingState:
    """Pure-timer arming tracker for the Ctrl+C / Ctrl+D double-press."""

    QUIT_WINDOW_S: ClassVar[float] = 1.0
    armed_at: dict[str, float] = field(default_factory=dict)
    last_armed_key: str | None = None

    def within_window(self, key: str | None = None) -> bool:
        if key is None:
            return self.hint_text() is not None
        armed_at = self.armed_at.get(key, -math.inf)
        return time.monotonic() - armed_at < self.QUIT_WINDOW_S

    def arm(self, key: str) -> None:
        self.armed_at[key] = time.monotonic()
        self.last_armed_key = key

    def hint_text(self) -> str | None:
        if self.last_armed_key is None:
            return None
        if not self.within_window(self.last_armed_key):
            return None
        if self.last_armed_key == "c-c":
            return "press Ctrl+C again to exit"
        if self.last_armed_key == "c-d":
            return "press Ctrl+D again to exit"
        return None


def apply_enter_or_continuation(buffer) -> bool:
    """Bash- / Claude-style trailing-backslash line continuation."""
    if buffer.cursor_position != len(buffer.text):
        return False
    if not buffer.text.endswith("\\"):
        return False
    buffer.delete_before_cursor(count=1)
    buffer.newline()
    return True


@dataclass
class RecallState:
    """Cursor for Up/Down recall of prior user turns (v0.9.7).

    Decoupled from rewind (split #2): recall only fills the input buffer;
    rewinding history is the explicit ``/rewind`` command. ``cursor`` is 0
    for a fresh draft, ``k`` for the k-th most-recent prior turn. Pure /
    UI-free so it can be unit-tested without a prompt_toolkit app.
    """

    cursor: int = 0

    @property
    def active(self) -> bool:
        return self.cursor > 0

    def reset(self) -> None:
        self.cursor = 0

    def up(self, turns: list[str]) -> str | None:
        """Walk one turn further back; return the text to show, or ``None``
        when there is no history."""
        if not turns:
            return None
        self.cursor = min(self.cursor + 1, len(turns))
        return turns[len(turns) - self.cursor]

    def down(self, turns: list[str]) -> str | None:
        """Walk one turn forward; at the front, return to the fresh draft
        (``""``). Returns the text to show, or ``None`` when no history."""
        if not turns:
            return None
        if self.cursor <= 1:
            self.cursor = 0
            return ""
        self.cursor -= 1
        return turns[len(turns) - self.cursor]


def _is_real_user_prompt(content: object) -> bool:
    """A ``role:user`` message the user actually typed — excludes injected
    markers (task reminders, ``/compact`` placeholders) that share the role.
    """
    if not isinstance(content, str) or not content.strip():
        return False
    if (
        is_task_reminder(content)
        or is_compact_marker(content)
        or is_advisor_message(content)
        or is_summary_marker(content)
    ):
        return False
    return True


def user_turn_indices(messages: list[dict[str, Any]]) -> list[int]:
    """Message indices of real user turns, oldest first."""
    return [
        i
        for i, m in enumerate(messages)
        if isinstance(m, dict)
        and m.get("role") == "user"
        and _is_real_user_prompt(m.get("content"))
    ]


def recallable_user_turns(messages: list[dict[str, Any]]) -> list[str]:
    """Real user-turn texts, oldest first — the Up/Down recall source."""
    return [str(messages[i]["content"]) for i in user_turn_indices(messages)]


def build_draft_key_bindings(
    quit_state: QuitArmingState | None = None,
    *,
    cancel_hook: Callable[[], bool] | None = None,
    recall_provider: Callable[[], list[str]] | None = None,
    recall_state: RecallState | None = None,
):
    """Build explicit editor bindings for terminal draft input.

    ``cancel_hook`` is invoked on the first Ctrl+C / Esc while something is
    in flight. It returns ``True`` iff the cancel actually fired. The hook
    is :py:meth:`ContextManager.cancel` (sync).

    When ``recall_provider`` + ``recall_state`` are given (v0.9.7),
    ``Up``/``Down`` walk prior user turns into the buffer (decoupled from
    rewind — recall only edits the draft). ``Up`` starts recall only on an
    empty buffer (so multi-line cursor-up still works mid-draft); ``Esc``
    exits recall when active, else falls through to cancel.
    """
    try:
        from prompt_toolkit.application.current import get_app
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.key_binding import KeyBindings
    except ImportError:
        return None

    if quit_state is None:
        quit_state = QuitArmingState()

    def _try_cancel() -> bool:
        if cancel_hook is None:
            return False
        try:
            return bool(cancel_hook())
        except Exception as exc:
            logger.warning("cancel_hook raised: {}", exc)
            return False

    bindings = KeyBindings()

    def _arm_or_exit(
        event,
        chord: str,
        exit_exception: type[BaseException],
    ) -> None:
        if quit_state.within_window(chord):
            event.app.exit(exception=exit_exception)
            return
        quit_state.arm(chord)
        event.app.invalidate()

        async def _fade_hint() -> None:
            await asyncio.sleep(QuitArmingState.QUIT_WINDOW_S)
            event.app.invalidate()

        event.app.create_background_task(_fade_hint())

    @bindings.add("c-c")
    def _(event) -> None:
        if _try_cancel():
            return
        _arm_or_exit(event, "c-c", KeyboardInterrupt)

    @bindings.add("escape", eager=True)
    def _(event) -> None:
        if recall_state is not None and recall_state.active:
            recall_state.reset()
            buf = event.current_buffer
            buf.text = ""
            buf.cursor_position = 0
            return
        _try_cancel()

    @Condition
    def _buffer_is_empty() -> bool:
        return not get_app().current_buffer.text

    @bindings.add("c-d", filter=_buffer_is_empty)
    def _(event) -> None:
        _arm_or_exit(event, "c-d", EOFError)

    @bindings.add("c-z")
    def _(event) -> None:
        event.app.suspend_to_background()

    @bindings.add("enter")
    def _(event) -> None:
        buf = event.current_buffer
        if apply_enter_or_continuation(buf):
            return
        if recall_state is not None:
            recall_state.reset()
        event.app.exit(result=buf.text)

    @bindings.add("c-j")
    def _(event) -> None:
        event.current_buffer.newline()

    @bindings.add("c-a")
    def _(event) -> None:
        move_buffer_to_line_start(event.current_buffer)

    @bindings.add("c-k")
    def _(event) -> None:
        delete_buffer_to_line_end(event.current_buffer)

    if recall_provider is not None and recall_state is not None:

        def _set_recalled(event, text: str | None) -> None:
            if text is None:
                return
            buf = event.current_buffer
            buf.text = text
            buf.cursor_position = len(text)

        # Up starts recall only on an empty buffer (so multi-line cursor-up
        # still works while editing); once recalling, Up/Down walk turns.
        @bindings.add(
            "up",
            filter=Condition(lambda: recall_state.active or not get_app().current_buffer.text),
        )
        def _(event) -> None:
            _set_recalled(event, recall_state.up(recall_provider()))

        @bindings.add("down", filter=Condition(lambda: recall_state.active))
        def _(event) -> None:
            _set_recalled(event, recall_state.down(recall_provider()))

    return bindings


def move_buffer_to_line_start(buffer) -> None:
    buffer.cursor_position += buffer.document.get_start_of_line_position()


def delete_buffer_to_line_end(buffer) -> None:
    delete_count = buffer.document.get_end_of_line_position()
    if delete_count:
        buffer.delete(count=delete_count)
    elif buffer.document.text_after_cursor.startswith("\n"):
        buffer.delete(count=1)


class TerminalView:
    """Prompt and transcript rendering for the terminal frontend."""

    def __init__(
        self,
        *,
        message_supplier: Callable[[], object] = lambda: "",
        render_markdown: bool = True,
        input_func: InputFunc | None = None,
        console: Console | None = None,
        cancel_hook: Callable[[], bool] | None = None,
        recall_provider: Callable[[], list[str]] | None = None,
    ) -> None:
        self.render_markdown = render_markdown
        self.input_func = input_func
        self.console = console or Console()
        self._prompt_running = False
        self._draft_reader = (
            None
            if input_func is not None
            else DraftReader(
                message_supplier=message_supplier,
                cancel_hook=cancel_hook,
                recall_provider=recall_provider,
            )
        )

    async def read_input(self) -> str:
        if self.input_func is not None:
            return await asyncio.to_thread(self.input_func, "")
        assert self._draft_reader is not None
        self._prompt_running = True
        try:
            return await self._draft_reader.read_async()
        finally:
            self._prompt_running = False

    def output_patch(self):
        if self.input_func is not None:
            return nullcontext()
        if self.console.file is not sys.stdout:
            return nullcontext()
        try:
            from prompt_toolkit.patch_stdout import patch_stdout
        except ImportError:
            return nullcontext()
        return patch_stdout(raw=True)

    async def _render(self, render: Callable[[], None]) -> None:
        if self._prompt_running and self.input_func is None and self.console.file is sys.stdout:
            try:
                from prompt_toolkit.application import run_in_terminal
            except ImportError:
                render()
                return
            await run_in_terminal(render, in_executor=False)
            return
        render()

    def print_notice_now(self, content: str, *, style: str = "dim") -> None:
        self.console.print(Text(content, style=style))

    async def print_notice(self, content: str, *, style: str = "dim") -> None:
        await self._render(lambda: self.print_notice_now(content, style=style))

    def print_system_now(self, content: str) -> None:
        self.console.print(Text(content, style=SYSTEM_STYLE))

    async def print_system(self, content: str) -> None:
        await self._render(lambda: self.print_system_now(content))

    def print_user_now(self, content: str) -> None:
        self.console.print(Text(content, style=USER_STYLE))

    async def print_user(self, content: str) -> None:
        await self._render(lambda: self.print_user_now(content))

    def print_assistant_now(self, content: str) -> None:
        if self.render_markdown:
            self.console.print(Markdown(content), style=ASSISTANT_STYLE)
        else:
            self.console.print(Text(content, style=ASSISTANT_STYLE))

    async def print_assistant(self, content: str) -> None:
        await self._render(lambda: self.print_assistant_now(content))

    def print_tool_use_now(self, name: str, arguments: str) -> None:
        keyword = "tool_use".ljust(TOOL_KEYWORD_WIDTH)
        args = compact_inline(arguments or "{}")
        text = Text.assemble(
            ("-> ", "dim"),
            (keyword, "bold cyan"),
            (f" {name} {args}", "dim"),
        )
        self.console.print(text)

    async def print_tool_use(self, name: str, arguments: str) -> None:
        await self._render(lambda: self.print_tool_use_now(name, arguments))

    def print_tool_result_now(self, record: ToolRecord) -> None:
        prefix, keyword, suffix = tool_record_summary_parts(record)
        text = Text.assemble(
            (prefix, "yellow"),
            (keyword, "bold bright_green"),
            (suffix, "yellow"),
        )
        self.console.print(text)

    async def print_tool_result(self, record: ToolRecord) -> None:
        await self._render(lambda: self.print_tool_result_now(record))

    def write_raw_now(self, text: str) -> None:
        file: TextIO = self.console.file
        file.write(text)
        file.flush()

    async def write_raw(self, text: str) -> None:
        await self._render(lambda: self.write_raw_now(text))

    def print_plain_now(self, content: str) -> None:
        self.console.print(content)

    async def print_plain(self, content: str) -> None:
        await self._render(lambda: self.print_plain_now(content))

    def print_text_now(self, text: Text) -> None:
        self.console.print(text)

    async def print_text(self, text: Text) -> None:
        await self._render(lambda: self.print_text_now(text))


class TerminalChat:
    """Normal terminal chat loop with append-only output.

    v0.9.3 wires this view to :class:`ContextManager`. The view emits
    events; the CM mutates state. The view never touches
    :class:`ChatStore` directly — it reads via
    :py:meth:`ChatStore.changes` and renders.
    """

    def __init__(
        self,
        ctx: ContextManager,
        *,
        config: Config,
        render_markdown: bool = True,
        input_func: InputFunc | None = None,
        console: Console | None = None,
    ) -> None:
        self.ctx = ctx
        self.config = config
        self.store = ctx.store
        self.view = TerminalView(
            message_supplier=self._above_input,
            render_markdown=render_markdown,
            input_func=input_func,
            console=console,
            cancel_hook=self.try_cancel_current_stream,
            recall_provider=lambda: recallable_user_turns(self.ctx.messages),
        )
        self._running = True
        self._busy = False
        self._input_queue: asyncio.Queue[str] | None = None
        # v0.10.3: the folded-tool-result tray lives in the store
        # (store.folded_tool_results) — no view-private list.
        # v0.10.2 visibility parity: full text of folded session channels,
        # stashed so /show can re-print them below (expand-by-append).
        self._system_full: str = ""
        self._tools_full: list[dict[str, Any]] = []
        self._summary_full: str = ""  # v0.10.5: last compaction summary, for /show summary
        # Per-render lookup so a ``role:tool`` record can find the
        # arguments string that the matching assistant ``tool_call``
        # carried. Populated by the renderer as it walks assistant
        # records with ``tool_calls``.
        self._tool_call_lookup: dict[str, tuple[str, str]] = {}
        # Index of the last rendered message. The render watcher walks
        # forward through ``store.messages`` from this point.
        self._rendered_message_count: int = 0
        # Heartbeat blink-phase counter: even tick → dot visible, odd →
        # blank. Reset to 0 on each IDLE→busy transition (see
        # _heartbeat_ticker) so a turn opens on a visible dot.
        self._heartbeat_tick: int = 0
        # v0.10.4 Smart Advisor: a third actor consulted at turn-end. Its LLM
        # is a fresh client on the fast slot (never the shared ctx.llm), built
        # lazily so a session that never triggers it makes no client.
        self.advisor = Advisor()
        self._advisor_llm: Any = None
        # v1.3.0: markdown skills/commands (.claude/skills + .claude/commands),
        # hot-reloaded by a background poll on the dir signature.
        self._skills = discover_skills(os.getcwd())
        self._skills_sig = skills_signature(os.getcwd())

    def run(self) -> None:
        """Run the blocking terminal chat loop."""
        asyncio.run(self.run_async())

    async def run_async(self) -> None:
        await self._render_initial_transcript()
        await self._render_tool_schemas_block()
        self._input_queue = asyncio.Queue()
        worker = asyncio.create_task(self._worker_loop())
        renderer = asyncio.create_task(self._render_watcher())
        heartbeat = asyncio.create_task(self._heartbeat_ticker())
        skills_poller = asyncio.create_task(self._skills_poller())
        try:
            with self.view.output_patch():
                await self._input_loop()
            if self._input_queue is not None:
                await self._input_queue.join()
        finally:
            worker.cancel()
            renderer.cancel()
            heartbeat.cancel()
            skills_poller.cancel()
            for task in (worker, renderer, heartbeat, skills_poller):
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            # Final flush — pick up any records the render_watcher
            # hadn't gotten to before its task was cancelled. The
            # renderer subscribes to ChatStore.changes() and processes
            # batches asynchronously; on shutdown there may still be
            # unrendered records the worker just appended.
            await self._render_new_records()

    async def _input_loop(self) -> None:
        while self._running:
            try:
                text = await self.view.read_input()
            except EOFError:
                await self.view.print_notice("quit")
                return
            except KeyboardInterrupt:
                await self.view.print_notice("\nquit")
                return

            text = text.strip()
            if not text:
                continue
            if text.startswith("/"):
                await self._run_command(text)
                continue
            assert self._input_queue is not None
            self.store.enqueue_user(text)
            await self._input_queue.put(text)

    async def _worker_loop(self) -> None:
        assert self._input_queue is not None
        while True:
            text = await self._input_queue.get()
            self.store.dequeue_user()
            # v1.2.0: inline @path file-mentions into the turn (raw text stayed
            # in the queue panel; the expanded text becomes the user message).
            text = expand_at_mentions(text, os.getcwd())
            self._busy = True
            self._invalidate_app()
            try:
                await self.ctx.handle_event(UserMessageEvent(text))
            except Exception as exc:
                logger.exception("terminal chat worker failed")
                await self.view.print_notice(str(exc), style="bold red")
            finally:
                self._busy = False
                self._invalidate_app()
                self._input_queue.task_done()
            # v0.10.4: turn-end Advisor hook — awaited, runs while IDLE between
            # turns (the user keeps typing into the queue meanwhile).
            self.advisor.note_turn()
            await self._maybe_run_advisor()

    def _tool_status(self) -> str:
        if not self.ctx.use_tools:
            return "off"
        if not self.ctx.effective_tools_enabled():
            return "unsupported"
        return "on"

    def _status_line(self) -> str:
        slot = self.ctx.slot
        parts = [
            slot.name,
            f"{slot.provider}/{slot.model}",
            f"tools:{self._tool_status()}",
            f"msgs:{len(self.ctx.messages)}",
        ]
        if self.ctx.executor.permission_mode == "allow-all":
            parts.append("allow-all")
        if self._busy:
            parts.append("working")
        return " | ".join(parts)

    def _above_input(self):
        """Content rendered directly above the input cursor."""
        heartbeat = format_heartbeat(
            self.ctx.state,
            self.store,
            self._heartbeat_tick,
            last_progress_at=self.ctx.last_progress_at,
            stall_threshold_s=stall_threshold_for(self.ctx.slot.llm_timeout_s),
        )
        tasks = self.store.tasks
        queued = self.store.queued_user_messages
        quit_hint = self._quit_hint_text()
        # v1.4.7 live streaming preview: a bounded tail of the in-progress
        # assistant text, shown ONLY in this live region (committed text goes to
        # scrollback via _render_record — strictly disjoint channels).
        preview = _stream_preview(self.store.pending_assistant_text)
        if not heartbeat and not tasks and not queued and quit_hint is None and not preview:
            return ""
        try:
            from prompt_toolkit.formatted_text import FormattedText
        except ImportError:
            FormattedText = None  # type: ignore[assignment]

        if FormattedText is None:
            lines: list[str] = []
            heartbeat_text = "".join(text for _style, text in heartbeat).rstrip("\n")
            if heartbeat_text:
                lines.append(heartbeat_text)
            if preview:
                lines.append(preview)
            for _style, text in format_task_panel(tasks):
                lines.append(text.rstrip("\n"))
            for q in queued:
                lines.append(f"{QUEUED_PREFIX}{q.text}")
            if quit_hint is not None:
                lines.append(quit_hint)
            return "\n".join(lines) + "\n" if lines else ""

        fragments: list[tuple[str, str]] = list(heartbeat)
        if preview:
            fragments.append(("fg:ansibrightblack italic", f"{preview}\n"))
        fragments.extend(format_task_panel(tasks))
        for q in queued:
            fragments.append(("fg:ansibrightblack", f"{QUEUED_PREFIX}{q.text}\n"))
        if quit_hint is not None:
            fragments.append(("fg:ansibrightblack", f"{quit_hint}\n"))
        if not fragments:
            return ""
        return FormattedText(fragments)

    def _quit_hint_text(self) -> str | None:
        reader = self.view._draft_reader
        return reader.quit_state.hint_text() if reader is not None else None

    def _invalidate_app(self) -> None:
        try:
            from prompt_toolkit.application.current import get_app_or_none
        except ImportError:
            return
        app = get_app_or_none()
        if app is not None:
            app.invalidate()

    async def _heartbeat_ticker(self) -> None:
        """Run the heartbeat liveness pulse for the lifetime of the chat.

        See :func:`heartbeat_loop` for the loop semantics. While CM is
        busy the tick counter is advanced once per blink interval and the
        prompt_toolkit app invalidated; idle phases consume no CPU. On each
        IDLE→busy transition the counter is reset to 0 so the turn opens on
        a visible dot rather than a blank one.
        """

        def on_tick() -> None:
            self._heartbeat_tick += 1
            self._invalidate_app()

        def on_enter_busy() -> None:
            self._heartbeat_tick = 0
            self._invalidate_app()

        await heartbeat_loop(
            state_supplier=lambda: self.ctx.state,
            store=self.store,
            on_tick=on_tick,
            on_enter_busy=on_enter_busy,
        )

    async def _skills_poller(self) -> None:
        """Hot-reload skills/commands when the .claude dirs change (v1.3.0).

        Polls the cheap dir signature every 2s (no file-watcher dependency —
        keeps neutrix lean) and re-discovers only on a real change.
        """
        while True:
            await asyncio.sleep(2.0)
            try:
                sig = skills_signature(os.getcwd())
            except OSError:  # pragma: no cover - defensive
                continue
            if sig != self._skills_sig:
                self._skills_sig = sig
                self._skills = discover_skills(os.getcwd())
                await self.view.print_notice(
                    f"↻ skills reloaded ({len(self._skills)} available)", style="dim"
                )

    async def _render_watcher(self) -> None:
        """Subscribe to store mutations; render new messages + redraw input.

        Walks new ``store.messages`` records as they arrive and prints
        each in the appropriate style. Also invalidates the prompt_toolkit
        app so the queue/task panel above the cursor refreshes.
        """
        async for _ in self.store.changes():
            await self._render_new_records()
            self._invalidate_app()

    async def _render_initial_transcript(self) -> None:
        """Render every record currently in the store, once at startup."""
        self._tool_call_lookup.clear()
        self._rendered_message_count = 0
        await self._render_new_records()

    async def _render_tool_schemas_block(self) -> None:
        """v0.10.2 visibility parity: render the tool-schemas channel folded.

        The schemas sent to the LLM (the ``tools=`` bundle) are otherwise the
        one input channel the user never sees. Session-static, so rendered once
        at start; the full list is reachable via ``/show tools``.
        """
        if not self.ctx.effective_tools_enabled():
            return
        schemas = get_schemas(self.ctx.tool_names)
        self._tools_full = schemas
        approx_bytes = sum(len(json.dumps(s)) for s in schemas)
        await self.view.print_notice(
            f"[tools]    {len(schemas)} schemas · folded · {approx_bytes} B  (/show tools)",
            style="dim",
        )

    async def _render_new_records(self) -> None:
        records = self.store.messages
        if len(records) < self._rendered_message_count:
            # v0.10.5: a CM-internal compaction (auto-threshold or
            # prompt-too-long recovery) shrank the store from inside _drive —
            # unlike /compact, /rewind, /clear, /load which realign the cursor
            # themselves (synchronously, before yielding). Without this, the
            # monotonic cursor would exceed len(records) and the transcript
            # would go silent. Realign and surface the summary; the kept tail is
            # already in scrollback above, so it is NOT re-printed.
            self._rendered_message_count = len(records)
            await self._render_compaction_shrink(records)
            return
        while self._rendered_message_count < len(records):
            record = records[self._rendered_message_count]
            await self._render_record(record)
            self._rendered_message_count += 1

    async def _render_compaction_shrink(self, records: tuple[MessageRecord, ...]) -> None:
        """Surface a CM-internal compaction: stash the summary + print a notice."""
        for rec in reversed(records):
            content = rec.content
            if is_summary_marker(content):
                body = str(content)[len(SUMMARY_MARKER_OPEN) :]
                if body.endswith(SUMMARY_MARKER_CLOSE):
                    body = body[: -len(SUMMARY_MARKER_CLOSE)]
                self._summary_full = body.strip()
                approx = approximate_token_count(body)
                await self.view.print_notice(
                    f"[summary]  context auto-compacted · ~{approx} tokens · folded"
                    "  (/show summary)",
                    style="dim",
                )
                return
            if is_compact_marker(content):
                break
        await self.view.print_notice(
            "context compacted to fit the window", style="dim"
        )

    async def _render_record(self, record: MessageRecord) -> None:
        role = record.role
        content = record.content
        if role == "user" and isinstance(content, str) and is_task_reminder(content):
            await self.view.print_notice(format_reminder_notice(self.store.tasks), style="dim")
            return
        if role == "user" and is_advisor_message(content):
            # v0.10.4: a judged Advisor suggestion — rendered expanded (it
            # carries advice, not state echo), distinct from the folded reminder.
            body = str(content)[len(ADVISOR_TAG_OPEN) :]
            if body.endswith(ADVISOR_TAG_CLOSE):
                body = body[: -len(ADVISOR_TAG_CLOSE)]
            await self.view.print_notice(f"↳ advisor: {body.strip()}", style="italic cyan")
            return
        if role == "user" and is_summary_marker(content):
            # v0.10.5: a compaction summary — folded one-liner (visibility
            # parity); the full summary expands via /show summary.
            body = str(content)[len(SUMMARY_MARKER_OPEN) :]
            if body.endswith(SUMMARY_MARKER_CLOSE):
                body = body[: -len(SUMMARY_MARKER_CLOSE)]
            self._summary_full = body.strip()
            approx = approximate_token_count(body)
            await self.view.print_notice(
                f"[summary]  conversation compacted · ~{approx} tokens · folded  (/show summary)",
                style="dim",
            )
            return
        if role == "system":
            if content:
                text = str(content)
                # v0.10.2: fold a long system prompt to a one-line summary
                # (full text reachable via /show system); short prompts stay
                # inline, preserving the default-prompt behavior.
                if len(text) > SYSTEM_FOLD_THRESHOLD:
                    self._system_full = text
                    await self.view.print_notice(
                        f"[system]   prompt · folded · {len(text)} B  (/show system)",
                        style=SYSTEM_STYLE,
                    )
                else:
                    await self.view.print_system(text)
            return
        if role == "user":
            if content:
                await self.view.print_user(str(content))
            return
        if role == "assistant":
            if content:
                await self.view.print_assistant(str(content))
            for call_id, name, arguments in self._tool_calls_from_record(record):
                if call_id:
                    self._tool_call_lookup[call_id] = (name, arguments)
                await self.view.print_tool_use(name, arguments)
            return
        if role == "tool":
            call_id = record.tool_call_id or ""
            name, arguments = self._tool_call_lookup.get(
                call_id, (record.tool_name or "tool", "{}")
            )
            await self._store_and_print_tool_result(name, arguments, str(content or ""))
            return

    def _tool_calls_from_record(self, record: MessageRecord) -> list[tuple[str, str, str]]:
        extra = record.extra or {}
        tool_calls = extra.get("tool_calls")
        if not isinstance(tool_calls, list):
            return []
        calls: list[tuple[str, str, str]] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            call_id = str(tool_call.get("id") or "")
            name = str(function.get("name") or "unknown")
            arguments = str(function.get("arguments") or "{}")
            calls.append((call_id, name, arguments))
        return calls

    def try_cancel_current_stream(self) -> bool:
        """Key-binding entry point. Returns True iff cancel fired.

        Delegates to :py:meth:`ContextManager.cancel`. The CM handles
        idempotency (a second cancel while already cancelling returns
        False). The renderer paints the ``[interrupted by user]`` user
        message as soon as the CM appends it to the store, so the
        affordance is visible without a separate dim notice.
        """
        return self.ctx.cancel()

    # --------------------------------------------------------------- advisor

    def _advisor_slot(self) -> Slot:
        """The slot the Advisor uses — the cheap `fast` slot if configured."""
        try:
            return self.config.slot("fast")
        except Exception:
            return self.ctx.slot

    def _get_advisor_llm(self):
        if self._advisor_llm is None:
            from neutrix.llm import OpenAIChatLLM

            self._advisor_llm = OpenAIChatLLM(self._advisor_slot())
        return self._advisor_llm

    def _recent_turn_pairs(self) -> list[dict[str, Any]]:
        """Last K real user/assistant turns for the Advisor's context."""
        out: list[dict[str, Any]] = []
        for m in self.ctx.messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            if role == "user" and not _is_real_user_prompt(m.get("content")):
                continue  # skip injected markers/advisor turns
            out.append({"role": role, "content": m.get("content")})
        return out[-(self.advisor.recent_turns * 2) :]

    async def _maybe_run_advisor(self, *, forced: bool = False) -> None:
        """Turn-end Advisor hook (v0.10.4). Awaited between turns, IDLE only."""
        if not forced:
            if not self.advisor.should_run():
                return
            # Only after a turn that completed with an assistant reply — skip
            # after a cancelled turn (last message is the interrupt marker).
            if not (self.ctx.messages and self.ctx.messages[-1].get("role") == "assistant"):
                return
        try:
            outcome = await self.advisor.run_once(
                tasks=self.store.tasks,
                recent_turns=self._recent_turn_pairs(),
                llm=self._get_advisor_llm(),
                model=self._advisor_slot().model,
            )
        except Exception as exc:
            logger.warning("advisor run failed: {}", exc)
            return
        await self._apply_advisor_outcome(outcome)

    async def _apply_advisor_outcome(self, outcome: Any) -> None:
        """Apply both Advisor channels: task mutations + injected suggestion."""
        for name, args in outcome.task_calls:
            result = dispatch(name, args, store=self.store)
            await self.view.print_notice(f"↳ advisor: {result}", style="dim")
        if outcome.suggestion:
            # Routed through CM (single mutator); the render watcher then prints
            # the <advisor> branch as an expanded notice.
            self.ctx.inject_advisor_message(outcome.suggestion)

    async def _store_and_print_tool_result(
        self, name: str, arguments: str, result: str
    ) -> ToolRecord:
        # v0.10.3: the folded-result tray is canonical store state, not a
        # view-private list. The store assigns the index.
        record = self.store.add_folded_tool_result(name, arguments, result)
        await self.view.print_tool_result(record)
        return record

    async def _run_command(self, line: str) -> None:
        parts = line[1:].strip().split()
        cmd, args = (parts[0].lower() if parts else ""), parts[1:]
        handler = getattr(self, f"_cmd_{cmd}", None)
        if handler is None and cmd in self._skills:
            # v1.3.0: a markdown skill/command — enqueue its body (with
            # $ARGUMENTS/$1.. substituted) as a user turn.
            if self._busy:
                await self.view.print_notice(
                    f"/{cmd} waits for the assistant to finish", style="yellow"
                )
                return
            prompt = render_skill(self._skills[cmd], args)
            await self.view.print_notice(f"↳ /{cmd}", style="dim")
            self.store.enqueue_user(prompt)
            assert self._input_queue is not None
            await self._input_queue.put(prompt)
            return
        if handler is None:
            await self.view.print_notice(f"unknown command: /{cmd}. Try /help.", style="bold red")
            return
        if self._busy and cmd in {
            "fast", "strong", "save", "load", "onboard", "compact", "rewind", "advise",
        }:
            await self.view.print_notice(
                f"/{cmd} waits for the assistant to finish", style="yellow"
            )
            return
        try:
            result = handler(args)
            if hasattr(result, "__await__"):
                await result
        except Exception as exc:
            logger.exception("command /{} failed", cmd)
            await self.view.print_notice(f"/{cmd} failed: {exc}", style="bold red")

    async def _cmd_help(self, args: list[str]) -> None:
        await self.view.print_plain(
            "\n".join(
                [
                    "Commands:",
                    "  /help               show this",
                    "  /init               survey the repo and write a CLAUDE.md",
                    "  /allow              toggle auto ↔ allow-all permissions",
                    "  /status             show slot, model, tool state, message count",
                    "  /tasks              list tracked tasks (read-only)",
                    "  /fast               switch to the fast slot",
                    "  /strong             switch to the strong slot",
                    "  /model              show current slot/provider/model",
                    "  /onboard            re-enter onboarding (manage keys / slots)",
                    "  /save [PATH]        save session (default: sessions/<ts>.json)",
                    "  /load PATH          load session",
                    "  /clear              start a fresh conversation",
                    "  /compact            drop the oldest ~50% of history (no summary)",
                    "  /rewind [N]         drop the last N user turns (default 1); Up/Down recalls",
                    "  /tools              list tools",
                    "  /tools on|off       enable/disable tool calling",
                    "  /tool [N]           list folded tool results or expand one",
                    "  /show system|tools|summary  expand a folded block",
                    "  /advise             ask the Advisor to review the task list now",
                    "  /quit               exit",
                ]
            )
        )

    async def _cmd_status(self, args: list[str]) -> None:
        await self.view.print_plain(self._status_line())

    async def _cmd_advise(self, args: list[str]) -> None:
        """On-demand Advisor run (v0.10.4) — bypasses the periodic trigger."""
        await self.view.print_notice("↳ advisor: reviewing…", style="dim")
        await self._maybe_run_advisor(forced=True)

    async def _cmd_allow(self, args: list[str]) -> None:
        """Toggle permission mode: `auto` (default) ↔ `allow-all` (v1.4.0)."""
        ex = self.ctx.executor
        if ex.permission_mode == "allow-all":
            ex.permission_mode = "auto"
            await self.view.print_notice(
                "permissions: auto — destructive shell commands are blocked", style="green"
            )
        else:
            ex.permission_mode = "allow-all"
            await self.view.print_notice(
                "permissions: allow-all — every tool runs, no safety checks", style="yellow"
            )

    async def _cmd_init(self, args: list[str]) -> None:
        """v1.2.0: drive the agent to survey the repo and write a CLAUDE.md."""
        if self._busy:
            await self.view.print_notice(
                "/init waits for the assistant to finish", style="yellow"
            )
            return
        prompt = (
            "Survey this repository and write a concise CLAUDE.md at the project "
            "root for future coding agents. Use Glob/Grep/Read/Bash to detect the "
            "build, test, and lint commands, the language/stack, key conventions, "
            "and any non-obvious gotchas. Keep it short — only what an agent would "
            "otherwise get wrong; no generic advice, no file-by-file tour. Then "
            "Write it to CLAUDE.md."
        )
        await self.view.print_notice("↳ /init: surveying the repo…", style="dim")
        self.store.enqueue_user(prompt)
        assert self._input_queue is not None
        await self._input_queue.put(prompt)

    async def _cmd_show(self, args: list[str]) -> None:
        """Expand-by-append a folded LLM-input channel (v0.10.2 parity)."""
        what = args[0].lower() if args else ""
        if what == "system":
            if self._system_full:
                await self.view.print_system(self._system_full)
            else:
                await self.view.print_notice(
                    "system prompt is shown inline (not folded)", style="dim"
                )
        elif what == "tools":
            if self._tools_full:
                lines = []
                for schema in self._tools_full:
                    fn = schema.get("function", {}) if isinstance(schema, dict) else {}
                    name = fn.get("name", "?")
                    desc = " ".join(str(fn.get("description", "")).split())[:88]
                    lines.append(f"  {name} — {desc}")
                await self.view.print_text(Text("\n".join(lines), style="dim"))
            else:
                await self.view.print_notice("no tool schemas to show", style="dim")
        elif what == "summary":
            if self._summary_full:
                await self.view.print_text(Text(self._summary_full, style="dim"))
            else:
                await self.view.print_notice("no compaction summary yet", style="dim")
        else:
            await self.view.print_notice("usage: /show system|tools|summary", style="bold red")

    async def _cmd_tasks(self, args: list[str]) -> None:
        tasks = self.store.tasks
        if not tasks:
            await self.view.print_notice("no tasks", style="dim")
            return
        lines = [f"#{t.id} [{t.status}] {t.subject}" for t in tasks]
        await self.view.print_text(Text("\n".join(lines)))

    async def _cmd_fast(self, args: list[str]) -> None:
        await self._switch_slot("fast")

    async def _cmd_strong(self, args: list[str]) -> None:
        await self._switch_slot("strong")

    async def _switch_slot(self, name: str) -> None:
        slot = self.config.slot(name)
        await self.ctx.handle_event(SlotSwitchEvent(slot=slot))
        await self.view.print_notice(
            f"switched to {name}: {slot.provider}/{slot.model}",
            style="green",
        )

    async def _cmd_model(self, args: list[str]) -> None:
        slot = self.ctx.slot
        await self.view.print_plain(
            "\n".join(
                [
                    f"current: [{slot.name}] {slot.provider}/{slot.model}",
                    f"slots available: {', '.join(SLOT_NAMES)}",
                    "edit ~/.config/neutrix/config.yaml to change slot bindings",
                ]
            )
        )

    async def _cmd_save(self, args: list[str]) -> None:
        if args:
            path = Path(args[0])
        else:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = Path("sessions") / f"{ts}.json"
        out = transcript.save(
            path,
            self.store,
            provider=self.ctx.slot.provider,
            model=self.ctx.slot.model,
        )
        await self.view.print_notice(f"saved -> {out}", style="green")

    async def _cmd_load(self, args: list[str]) -> None:
        if not args:
            await self.view.print_notice("usage: /load PATH", style="bold red")
            return
        loaded, metadata = transcript.load(args[0])
        raw_messages = list(metadata["raw_messages"])
        records = loaded.messages
        await self.ctx.handle_event(
            ReplaceHistoryEvent(
                raw_messages=raw_messages,
                records=records,
                tasks=loaded.tasks,
            )
        )
        # store.reset() (via ReplaceHistoryEvent) already wiped folded_tool_results.
        self._tool_call_lookup.clear()
        self._rendered_message_count = 0
        await self.view.print_notice(
            f"loaded {args[0]} ({len(raw_messages)} msgs, "
            f"{len(loaded.tasks)} tasks); current slot unchanged",
            style="green",
        )
        await self._render_new_records()

    async def _cmd_clear(self, args: list[str]) -> None:
        await self.ctx.handle_event(ClearEvent())
        # store.reset() (via ClearEvent) already wiped folded_tool_results.
        self._tool_call_lookup.clear()
        self._rendered_message_count = 0
        await self.view.print_notice("conversation cleared", style="green")
        await self._render_new_records()

    async def _cmd_compact(self, args: list[str]) -> None:
        outcome = await self.ctx.compact()
        if not outcome.did_compact:
            await self.view.print_notice("nothing to compact (conversation too short)", style="dim")
            return
        # Notice-only: the kept tail is already in scrollback, so suppress
        # re-printing it and re-align the render index to the compacted
        # store length (v0.9.6 split #9).
        self._rendered_message_count = len(self.store.messages)
        await self.view.print_notice(
            f"compacted {outcome.turns_dropped} turns "
            f"(~{format_token_count(outcome.approx_tokens_dropped)} tokens dropped)",
            style="dim",
        )

    async def _cmd_rewind(self, args: list[str]) -> None:
        n = 1
        if args:
            try:
                n = int(args[0])
            except ValueError:
                await self.view.print_notice(
                    "usage: /rewind [N]  (drop the last N user turns, default 1)",
                    style="bold red",
                )
                return
        if n < 1:
            await self.view.print_notice("usage: /rewind [N]  (N >= 1)", style="bold red")
            return
        turns = user_turn_indices(self.ctx.messages)
        if not turns:
            await self.view.print_notice("nothing to rewind", style="dim")
            return
        n = min(n, len(turns))
        dropped = await self.ctx.rewind_to(turns[-n])
        if dropped <= 0:
            await self.view.print_notice("nothing to rewind", style="dim")
            return
        # Notice-only (v0.9.7 split #7): the dropped turns stay in scrollback
        # above — an append-only renderer can't un-print them — so re-align
        # the render index to the rewound store and print a forward marker.
        self._rendered_message_count = len(self.store.messages)
        remaining = len(user_turn_indices(self.ctx.messages))
        await self.view.print_notice(
            f"↶ rewound {n} turn{'s' if n != 1 else ''} (back to turn {remaining})",
            style="dim",
        )

    async def _cmd_tools(self, args: list[str]) -> None:
        if args and args[0] in ("on", "off"):
            self.ctx.use_tools = args[0] == "on"
            await self.view.print_notice(
                f"tool calling {'enabled' if self.ctx.use_tools else 'disabled'}",
                style="green",
            )
            return

        lines = ["available tools:"]
        for tool in BUILTIN_TOOLS.values():
            lines.append(f"  - {tool.name}: {tool.description}")
        lines.append(f"status: {self._tool_status()}")
        await self.view.print_plain("\n".join(lines))

    async def _cmd_tool(self, args: list[str]) -> None:
        records = self.store.folded_tool_results
        if not records:
            await self.view.print_notice("no folded tool results", style="dim")
            return

        if not args:
            for record in records[-20:]:
                await self.view.print_tool_result(record)
            return

        try:
            index = int(args[0])
        except ValueError:
            await self.view.print_notice("usage: /tool N", style="bold red")
            return

        if index < 1 or index > len(records):
            await self.view.print_notice(f"unknown tool result: {index}", style="bold red")
            return

        record = records[index - 1]
        await self.view.print_text(
            Text(f"[tool {record.index}] {record.name} full result:", style="bold")
        )
        await self.view.print_plain(record.result)

    async def _cmd_onboard(self, args: list[str]) -> None:
        from neutrix.onboard import run_onboarding

        run_onboarding(self.config)
        try:
            self.config = load_config(self.config.path)
        except ConfigError as exc:
            await self.view.print_notice(f"config reload failed: {exc}", style="bold red")
            return
        await self.view.print_notice(
            "back from onboarding. Use /fast or /strong to switch to a newly-bound slot; "
            "current slot unchanged.",
            style="green",
        )

    def _cmd_quit(self, args: list[str]) -> None:
        self._running = False
