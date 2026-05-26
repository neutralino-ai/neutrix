"""Append-only terminal chat renderer.

The main chat uses ordinary terminal scrollback instead of a fullscreen app.
The agent still owns conversation state; this module only renders events and
handles slash commands.
"""
from __future__ import annotations

import asyncio
import math
import re
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import ClassVar, TextIO

from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from neutrix import transcript
from neutrix.agent_loop import (
    Agent,
    AgentEvent,
    format_reminder_notice,
    is_task_reminder,
)
from neutrix.config import SLOT_NAMES, Config, ConfigError, load_config
from neutrix.store import ChatStore, MessageRecord, Task, openai_to_record
from neutrix.tools import BUILTIN_TOOLS

QUEUED_PREFIX = "› "  # noqa: RUF001  -- U+203A is the chosen UI glyph

MAX_PANEL_ROWS = 5

_TASK_PANEL_ICON = {
    "in_progress": "◼",  # noqa: RUF001  -- U+25FC BLACK MEDIUM SQUARE
    "pending": "◻",  # noqa: RUF001  -- U+25FB WHITE MEDIUM SQUARE
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


def result_line_count(result: str) -> int:
    """Count displayed result lines without inventing a line for empty output."""
    if not result:
        return 0
    return len(result.splitlines())


def approximate_token_count(result: str) -> int:
    """Cheap, deterministic token estimate based on non-whitespace chunks."""
    return len(WORD_RE.findall(result))


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
    """Render the always-on task panel as prompt_toolkit fragments.

    Returns an empty list when ``tasks`` is empty so the input cursor
    sits at its natural position. Otherwise sorts by
    (in_progress → pending → completed, id ascending), caps the visible
    rows at :data:`MAX_PANEL_ROWS`, and appends a dim
    ``"  … +N in progress, N pending, N done\\n"`` overflow line if any
    tasks were truncated.

    Pure function — the panel content depends only on ``tasks``, which
    lets tests assert ordering and overflow without spinning up the
    full chat surface.
    """
    if not tasks:
        return []
    ordered = sorted(tasks, key=_task_sort_key)
    visible = ordered[:MAX_PANEL_ROWS]
    truncated = ordered[MAX_PANEL_ROWS:]

    fragments: list[tuple[str, str]] = []
    for task in visible:
        icon = _TASK_PANEL_ICON.get(task.status, "?")
        style = _TASK_PANEL_STYLE.get(task.status, "")
        fragments.append(
            (style, f"  {icon} #{task.id} [{task.status}] {task.subject}\n")
        )

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


# Width of the longest keyword, "tool_result". Keyword strings are
# right-padded to this width so the body name aligns vertically between
# the `-> tool_use` and `<- tool_result` lines (and across calls).
TOOL_KEYWORD_WIDTH = len("tool_result")


@dataclass(frozen=True)
class ToolRecord:
    index: int
    name: str
    arguments: str
    result: str

    def _summary_body(self) -> str:
        args = compact_inline(self.arguments or "{}")
        lines = result_line_count(self.result)
        approx_tokens = approximate_token_count(self.result)
        return (
            f" [tool {self.index}] {self.name} {args} | folded | "
            f"{lines} lines | ~{approx_tokens} tokens"
        )

    @property
    def summary(self) -> str:
        return f"<- {'tool_result'.ljust(TOOL_KEYWORD_WIDTH)}{self._summary_body()}"

    def summary_parts(self) -> tuple[str, str, str]:
        """Return (prefix, padded_keyword, suffix) for colored rendering."""
        return ("<- ", "tool_result".ljust(TOOL_KEYWORD_WIDTH), self._summary_body())


InputFunc = Callable[[str], str]


class DraftReader:
    """Bottom draft editor, using prompt_toolkit when it is installed.

    ``message_supplier`` populates the area shown directly above the
    input cursor (used for the queued-user-messages display). It is a
    callable re-evaluated on each render, so
    :py:meth:`prompt_toolkit.application.invalidate` refreshes it
    without a periodic timer.

    There is intentionally no bottom toolbar: in prompt_toolkit's
    append-only-with-bottom-input mode, every stdout write (every
    streamed token, every tool result) triggers a hide-restore cycle
    of the prompt + toolbar area, which made the toolbar visibly
    blink during the assistant's response. Status info is available
    on demand via the ``/status`` command.
    """

    def __init__(
        self,
        *,
        message_supplier: Callable[[], object] = lambda: "",
    ) -> None:
        self._message_supplier = message_supplier
        self.quit_state = QuitArmingState()
        self._session = self._build_session()

    def read(self) -> str:
        if self._session is None:
            return input("")
        return self._session.prompt(handle_sigint=False)

    async def read_async(self) -> str:
        if self._session is None:
            return await asyncio.to_thread(input, "")
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

        # No bottom_toolbar and no refresh_interval. The queued
        # messages render above the input via ``message_supplier``;
        # there is no other persistent UI region that would need
        # periodic refresh, and removing the toolbar eliminates the
        # blink that was visible whenever the assistant streamed
        # output (every stdout write hides-then-restores the prompt
        # area, which made the toolbar disappear for one frame).
        # ``handle_sigint=False`` is passed to each ``prompt``/
        # ``prompt_async`` call (it's a method arg, not a constructor
        # arg) so SIGINT reaches our explicit c-c binding instead of
        # being translated to KeyboardInterrupt before the binding
        # fires (the binding implements the Codex-style double-press
        # exit).
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
            key_bindings=build_draft_key_bindings(self.quit_state),
        )


@dataclass
class QuitArmingState:
    """Pure-timer arming tracker for the Ctrl+C / Ctrl+D double-press.

    Each chord (``"c-c"``, ``"c-d"``) carries its own independent
    ``armed_at`` timestamp in :data:`armed_at`. Cross-key presses
    never touch the other chord's clock; the displayed hint refreshes
    to name the most recently pressed chord, but each timer keeps
    running on its own schedule. Only the same chord can confirm —
    a second press of THIS chord, within THIS chord's own window.

    Earlier designs (shared-state confirm-either-key,
    re-arm-with-reset-timer) were rejected at successive Phase-3
    review gates — see ``docs/PRDs/v0.9.1-keyboard.md``.
    """

    QUIT_WINDOW_S: ClassVar[float] = 1.0
    armed_at: dict[str, float] = field(default_factory=dict)
    last_armed_key: str | None = None

    def within_window(self, key: str | None = None) -> bool:
        """Two-mode predicate.

        ``key is None`` (renderer): True iff the most-recently-armed
        chord is still within its own window — drives hint visibility.

        ``key`` set (binding): True iff that chord's own timer is
        still within :data:`QUIT_WINDOW_S`. Cross-key arms do not
        affect this — each chord's timer is independent.
        """
        if key is None:
            return self.hint_text() is not None
        armed_at = self.armed_at.get(key, -math.inf)
        return time.monotonic() - armed_at < self.QUIT_WINDOW_S

    def arm(self, key: str) -> None:
        self.armed_at[key] = time.monotonic()
        self.last_armed_key = key

    def hint_text(self) -> str | None:
        """Renderer-facing hint string. Names the *most recently
        pressed* chord, as long as its own window is still open.

        The "fallback to the OTHER chord if its window is still
        open" branch is unreachable: if the most-recent chord has
        expired, the earlier chord — armed strictly further in
        the past — must also have expired.
        """
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
    """Bash- / Claude-style trailing-backslash line continuation.

    If the buffer ends with ``\\`` and the cursor is at end-of-buffer,
    strip the backslash and insert a newline, returning ``True`` to
    signal "treat this Enter as a newline, not a submit." Otherwise
    returns ``False`` so the caller submits as normal.
    """
    if buffer.cursor_position != len(buffer.text):
        return False
    if not buffer.text.endswith("\\"):
        return False
    buffer.delete_before_cursor(count=1)
    buffer.newline()
    return True


def build_draft_key_bindings(quit_state: QuitArmingState | None = None):
    """Build explicit editor bindings for terminal draft input.

    ``quit_state`` carries the Ctrl+C arming timer; pass a fresh
    :class:`QuitArmingState` per :class:`DraftReader` so two sessions
    don't share state. ``None`` is accepted for unit tests that
    don't care about the quit dance.
    """
    try:
        from prompt_toolkit.application.current import get_app
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.key_binding import KeyBindings
    except ImportError:
        return None

    if quit_state is None:
        quit_state = QuitArmingState()

    bindings = KeyBindings()

    def _arm_or_exit(
        event,
        chord: str,
        exit_exception: type[BaseException],
    ) -> None:
        """Shared body for the c-c and c-d quit-confirm bindings.

        Independent per-chord arming: ``within_window(chord)`` is
        True only if THIS chord's own timer is still within
        :data:`QuitArmingState.QUIT_WINDOW_S`. A cross-key press
        falls through to arm only its own chord, leaving the other
        chord's clock untouched. So ``Ctrl+C → Ctrl+D → Ctrl+C``
        within the original 1 s of the first press exits via
        Ctrl+C: the intervening Ctrl+D never touched c-c's timer.
        """
        if quit_state.within_window(chord):
            event.app.exit(exception=exit_exception)
            return
        quit_state.arm(chord)
        event.app.invalidate()  # paint the hint now

        # Schedule a redraw QUIT_WINDOW_S later so the hint clears on
        # its own without needing the user to press another key. The
        # background task lives on the app's task group and is
        # cancelled automatically when prompt_async returns.
        async def _fade_hint() -> None:
            await asyncio.sleep(QuitArmingState.QUIT_WINDOW_S)
            event.app.invalidate()

        event.app.create_background_task(_fade_hint())

    @bindings.add("c-c")
    def _(event) -> None:
        _arm_or_exit(event, "c-c", KeyboardInterrupt)

    # Ctrl+D enters the quit dance ONLY when the buffer is empty;
    # otherwise it falls through to prompt_toolkit's default Emacs
    # binding (forward-delete-character). The Condition filter is what
    # makes that fall-through work — without it the binding would
    # always fire and the user could not forward-delete inside a
    # draft. Shares the QuitArmingState instance with c-c, but
    # per-chord arming means only the same key can confirm.
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
        event.app.exit(result=buf.text)

    # Newline keys. Codex's textarea.rs:347-355 inserts \n for Ctrl+J,
    # Ctrl+M, and Enter with any modifier. We can only bind what
    # prompt_toolkit's key parser recognizes — ``s-enter`` and
    # ``c-enter`` are not in its Keys enum (terminals usually send
    # the same byte for Enter regardless of Shift/Ctrl unless CSI-u
    # keyboard protocol is on, which prompt_toolkit doesn't yet
    # ship). So we cover ``c-j`` and ``escape, enter`` (Alt+Enter),
    # which work on every terminal.
    @bindings.add("c-j")
    def _(event) -> None:
        event.current_buffer.newline()

    @bindings.add("escape", "enter")
    def _(event) -> None:
        event.current_buffer.newline()

    @bindings.add("c-a")
    def _(event) -> None:
        move_buffer_to_line_start(event.current_buffer)

    @bindings.add("c-k")
    def _(event) -> None:
        delete_buffer_to_line_end(event.current_buffer)

    return bindings


def move_buffer_to_line_start(buffer) -> None:
    """Move a prompt_toolkit buffer cursor to the current logical line start."""
    buffer.cursor_position += buffer.document.get_start_of_line_position()


def delete_buffer_to_line_end(buffer) -> None:
    """Delete from cursor to the current logical line end."""
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
    ) -> None:
        self.render_markdown = render_markdown
        self.input_func = input_func
        self.console = console or Console()
        self._prompt_running = False
        self._draft_reader = (
            None
            if input_func is not None
            else DraftReader(message_supplier=message_supplier)
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
        prefix, keyword, suffix = record.summary_parts()
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
    """Normal terminal chat loop with append-only output."""

    def __init__(
        self,
        agent: Agent,
        *,
        config: Config,
        render_markdown: bool = True,
        input_func: InputFunc | None = None,
        console: Console | None = None,
    ) -> None:
        self.agent = agent
        self.config = config
        self.view = TerminalView(
            message_supplier=self._above_input,
            render_markdown=render_markdown,
            input_func=input_func,
            console=console,
        )
        self._running = True
        self._busy = False
        self._input_queue: asyncio.Queue[str] | None = None
        self._tool_records: list[ToolRecord] = []
        self._streaming_assistant = False
        self.store = ChatStore()
        self.agent.store = self.store
        self._seed_store_from_agent()

    def _seed_store_from_agent(self) -> None:
        """Mirror the agent's current message list into the store.

        Called at construction and after /load and /clear. The agent still
        owns the OpenAI-format list it feeds back to the LLM; the store
        is the typed view future renderers will read.
        """
        for raw in self.agent.messages:
            if isinstance(raw, dict):
                self.store.append_message(openai_to_record(raw))

    def run(self) -> None:
        """Run the blocking terminal chat loop."""
        asyncio.run(self.run_async())

    async def run_async(self) -> None:
        """Run prompt input and agent work concurrently."""
        await self._render_transcript()
        self._input_queue = asyncio.Queue()
        worker = asyncio.create_task(self._worker_loop())
        invalidator = asyncio.create_task(self._invalidation_watcher())
        try:
            with self.view.output_patch():
                await self._input_loop()
            if self._input_queue is not None:
                await self._input_queue.join()
        finally:
            worker.cancel()
            invalidator.cancel()
            for task in (worker, invalidator):
                try:
                    await task
                except asyncio.CancelledError:
                    pass

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
            self._busy = True
            self._invalidate_app()
            try:
                await self.view.print_user(text)
                self.store.append_message(
                    MessageRecord(role="user", content=text)
                )
                await self._send_message(text)
            finally:
                self._busy = False
                self._invalidate_app()
                self._input_queue.task_done()

    def _tool_status(self) -> str:
        if not self.agent.use_tools:
            return "off"
        enabled = getattr(self.agent, "effective_tools_enabled", None)
        if callable(enabled) and not enabled():
            return "unsupported"
        return "on"

    def _status_line(self) -> str:
        """Single-line status string used by the ``/status`` command."""
        slot = self.agent.slot
        parts = [
            slot.name,
            f"{slot.provider}/{slot.model}",
            f"tools:{self._tool_status()}",
            f"msgs:{len(self.agent.messages)}",
        ]
        if self._busy:
            parts.append("working")
        return " | ".join(parts)

    def _above_input(self):
        """Content rendered directly above the input cursor.

        Stacks (top → bottom):

        1. The always-visible task panel (when any tasks exist) — see
           :func:`format_task_panel` for ordering and overflow rules.
        2. Queued user messages — dim, prefixed with ``QUEUED_PREFIX``,
           one per line, drawn on the lines just above the cursor.
        3. The dim-gray quit-confirm hint, when the quit-shortcut
           is armed (v0.9.1 — see :class:`QuitArmingState`). The
           exact wording is chord-specific — ``press Ctrl+C again to
           exit`` after Ctrl+C, ``press Ctrl+D again to exit`` after
           Ctrl+D — so the user always sees the exact key they need
           to press to confirm. Shares the ``fg:ansibrightblack``
           dim style with queued messages — the affordance reads as
           part of the muted hierarchy, not a warning.

        Returns an empty string when there's nothing to show so the
        input cursor sits at its natural position. Returns
        ``FormattedText`` when prompt_toolkit is installed; otherwise a
        plain str fallback (used by tests that mock the input).
        """
        tasks = self.store.tasks
        queued = self.store.queued_user_messages
        quit_hint = self._quit_hint_text()
        if not tasks and not queued and quit_hint is None:
            return ""
        try:
            from prompt_toolkit.formatted_text import FormattedText
        except ImportError:
            FormattedText = None  # type: ignore[assignment]

        if FormattedText is None:
            lines: list[str] = []
            for _style, text in format_task_panel(tasks):
                lines.append(text.rstrip("\n"))
            for q in queued:
                lines.append(f"{QUEUED_PREFIX}{q.text}")
            if quit_hint is not None:
                lines.append(quit_hint)
            return "\n".join(lines) + "\n" if lines else ""

        fragments: list[tuple[str, str]] = list(format_task_panel(tasks))
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
        """Force the running prompt_toolkit app to redraw, if any."""
        try:
            from prompt_toolkit.application.current import get_app_or_none
        except ImportError:
            return
        app = get_app_or_none()
        if app is not None:
            app.invalidate()

    async def _invalidation_watcher(self) -> None:
        """Subscribe to store mutations and trigger one redraw per batch.

        Replaces the previous 0.5-s ``refresh_interval`` polling. The
        screen now only refreshes when something actually changed
        (queue, pending tool calls, new messages, streamed text), so
        there is no rhythmic flash between updates.
        """
        async for _ in self.store.changes():
            self._invalidate_app()

    async def _render_transcript(self) -> None:
        tool_call_lookup: dict[str, tuple[str, str]] = {}
        for message in self.agent.messages:
            role = str(message.get("role") or "")
            content = message.get("content")
            if role == "user" and is_task_reminder(content):
                # v0.8.0 reminder body — render the folded notice instead
                # of leaking the templated text as a plain user block.
                await self.view.print_notice(
                    format_reminder_notice(self.store.tasks), style="dim"
                )
                continue
            if role == "system" and content:
                await self.view.print_system(str(content))
            elif role == "user" and content:
                await self.view.print_user(str(content))
            elif role == "assistant":
                if content:
                    await self.view.print_assistant(str(content))
                for call_id, name, arguments in self._tool_calls_from_message(message):
                    if call_id:
                        tool_call_lookup[call_id] = (name, arguments)
                    await self.view.print_tool_use(name, arguments)
            elif role == "tool" and content is not None:
                call_id = str(message.get("tool_call_id") or "")
                name, arguments = tool_call_lookup.get(call_id, ("tool", "{}"))
                await self._store_and_print_tool_result(name, arguments, str(content))

    def _tool_calls_from_message(
        self, message: dict[str, object]
    ) -> list[tuple[str, str, str]]:
        tool_calls = message.get("tool_calls")
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

    async def _send_message(self, text: str) -> None:
        self.store.clear_pending_tool_calls()
        self._streaming_assistant = False
        try:
            pre_message_count = len(self.agent.messages)
            assistant_started = False
            # tool_arg_cache lives for the duration of one _send_message
            # call. store.apply() removes pending entries on tool_result
            # before the renderer sees the event, so the renderer can no
            # longer reach back into store.pending_tool_calls for the
            # original arguments string — we record them here on tool_call
            # and FIFO-pop on tool_result. Discarded at end of the call.
            tool_arg_cache: dict[str, list[str]] = {}
            async for event in self.agent.stream_reply(text):
                self.store.apply(event)
                assistant_started = await self._handle_event(
                    event, assistant_started, tool_arg_cache
                )
            if self._streaming_assistant:
                await self.view.write_raw("\n")
                self._streaming_assistant = False
            # The agent appended its assistant turn to its own list; mirror
            # the latest message into the store as well so future renderers
            # can read a complete typed transcript. Fold any v0.8.0 task
            # reminder that landed during this turn into a dim notice so
            # the user sees that the harness nudged the LLM.
            await self._render_new_reminders(pre_message_count)
            self._mirror_new_agent_messages()
        except Exception as exc:
            logger.exception("terminal chat worker failed")
            await self.view.print_notice(str(exc), style="bold red")

    async def _render_new_reminders(self, pre_count: int) -> None:
        for raw in self.agent.messages[pre_count:]:
            if not isinstance(raw, dict):
                continue
            if raw.get("role") != "user":
                continue
            if is_task_reminder(raw.get("content")):
                await self.view.print_notice(
                    format_reminder_notice(self.store.tasks), style="dim"
                )

    def _mirror_new_agent_messages(self) -> None:
        """Append any agent messages not yet in the store."""
        stored = len(self.store.messages)
        for raw in self.agent.messages[stored:]:
            if isinstance(raw, dict):
                self.store.append_message(openai_to_record(raw))

    async def _handle_event(
        self,
        event: AgentEvent,
        assistant_started: bool,
        tool_arg_cache: dict[str, list[str]],
    ) -> bool:
        if event.kind == "token":
            if not assistant_started:
                assistant_started = True
                self._streaming_assistant = True
            await self.view.write_raw(str(event.data))
            return assistant_started

        if self._streaming_assistant:
            await self.view.write_raw("\n")
            self._streaming_assistant = False

        if event.kind == "assistant":
            await self.view.print_assistant(str(event.data))
            return True

        if event.kind == "tool_call":
            name = str(event.data["name"])
            arguments = str(event.data["arguments"])
            tool_arg_cache.setdefault(name, []).append(arguments)
            await self.view.print_tool_use(name, arguments)
            return assistant_started

        if event.kind == "tool_result":
            name = str(event.data["name"])
            queued = tool_arg_cache.get(name) or []
            arguments = queued.pop(0) if queued else "{}"
            result = str(event.data["result"])
            await self._store_and_print_tool_result(name, arguments, result)
            return assistant_started

        if event.kind == "error":
            await self.view.print_notice(str(event.data), style="bold red")
            return assistant_started

        return assistant_started

    async def _store_and_print_tool_result(
        self, name: str, arguments: str, result: str
    ) -> ToolRecord:
        record = ToolRecord(
            index=len(self._tool_records) + 1,
            name=name,
            arguments=arguments,
            result=result,
        )
        self._tool_records.append(record)
        await self.view.print_tool_result(record)
        return record

    async def _run_command(self, line: str) -> None:
        parts = line[1:].strip().split()
        cmd, args = (parts[0].lower() if parts else ""), parts[1:]
        handler = getattr(self, f"_cmd_{cmd}", None)
        if handler is None:
            await self.view.print_notice(
                f"unknown command: /{cmd}. Try /help.", style="bold red"
            )
            return
        if self._busy and cmd in {"fast", "strong", "save", "load", "clear", "onboard"}:
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
                    "  /status             show slot, model, tool state, message count",
                    "  /tasks              list tracked tasks (read-only)",
                    "  /fast               switch to the fast slot",
                    "  /strong             switch to the strong slot",
                    "  /model              show current slot/provider/model",
                    "  /onboard            re-enter onboarding (manage keys / slots)",
                    "  /save [PATH]        save session (default: sessions/<ts>.json)",
                    "  /load PATH          load session",
                    "  /clear              start a fresh conversation",
                    "  /tools              list tools",
                    "  /tools on|off       enable/disable tool calling",
                    "  /tool [N]           list folded tool results or expand one",
                    "  /quit               exit",
                ]
            )
        )

    async def _cmd_status(self, args: list[str]) -> None:
        """Print the current slot, model, tool state, and message count.

        Replaces the persistent bottom toolbar: the toolbar was visibly
        blinking during streaming output because every stdout write
        triggers a hide-restore cycle of the prompt area. Status info
        is now available on demand here.
        """
        await self.view.print_plain(self._status_line())

    async def _cmd_tasks(self, args: list[str]) -> None:
        """Print the current task list (read directly from the store).

        The LLM writes tasks via the TaskCreate / TaskUpdate tools; this
        command is a read-only on-demand view of the same state.

        The lines render through ``print_text`` (not ``print_plain``) so
        the bracketed status tags ``[pending]`` etc. are not parsed as
        Rich BBCode markup and disappear.
        """
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
        self.agent.switch(slot)
        await self.view.print_notice(
            f"switched to {name}: {slot.provider}/{slot.model}",
            style="green",
        )

    async def _cmd_model(self, args: list[str]) -> None:
        slot = self.agent.slot
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
            provider=self.agent.slot.provider,
            model=self.agent.slot.model,
        )
        await self.view.print_notice(f"saved -> {out}", style="green")

    async def _cmd_load(self, args: list[str]) -> None:
        if not args:
            await self.view.print_notice("usage: /load PATH", style="bold red")
            return
        loaded, metadata = transcript.load(args[0])
        self.agent.messages = list(metadata["raw_messages"])
        self.store.reset()
        self._seed_store_from_agent()
        # Preserve tasks across /load — transcript.load() already populated
        # the throwaway store; copy them onto the live store so the next
        # /tasks (and the LLM via TaskList) sees them.
        if loaded.tasks:
            self.store.replace_tasks(loaded.tasks)
        self._tool_records.clear()
        await self.view.print_notice(
            f"loaded {args[0]} ({len(self.agent.messages)} msgs, "
            f"{len(self.store.tasks)} tasks); current slot unchanged",
            style="green",
        )
        await self._render_transcript()

    async def _cmd_clear(self, args: list[str]) -> None:
        self.agent.reset()
        self.store.reset()
        self._seed_store_from_agent()
        self._tool_records.clear()
        await self.view.print_notice("conversation cleared", style="green")
        await self._render_transcript()

    async def _cmd_tools(self, args: list[str]) -> None:
        if args and args[0] in ("on", "off"):
            self.agent.use_tools = args[0] == "on"
            await self.view.print_notice(
                f"tool calling {'enabled' if self.agent.use_tools else 'disabled'}",
                style="green",
            )
            return

        lines = ["available tools:"]
        for tool in BUILTIN_TOOLS.values():
            lines.append(f"  - {tool.name}: {tool.description}")
        lines.append(f"status: {self._tool_status()}")
        await self.view.print_plain("\n".join(lines))

    async def _cmd_tool(self, args: list[str]) -> None:
        if not self._tool_records:
            await self.view.print_notice("no folded tool results", style="dim")
            return

        if not args:
            for record in self._tool_records[-20:]:
                await self.view.print_tool_result(record)
            return

        try:
            index = int(args[0])
        except ValueError:
            await self.view.print_notice("usage: /tool N", style="bold red")
            return

        if index < 1 or index > len(self._tool_records):
            await self.view.print_notice(f"unknown tool result: {index}", style="bold red")
            return

        record = self._tool_records[index - 1]
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
