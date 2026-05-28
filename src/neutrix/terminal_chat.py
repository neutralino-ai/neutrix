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
import math
import random
import re
import sys
import time
from collections.abc import Awaitable, Callable
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
from neutrix.config import SLOT_NAMES, Config, ConfigError, load_config
from neutrix.context_manager import (
    ClearEvent,
    ContextManager,
    ReplaceHistoryEvent,
    SlotSwitchEvent,
    State,
    UserMessageEvent,
    format_reminder_notice,
    is_task_reminder,
)
from neutrix.store import ChatStore, MessageRecord, Task
from neutrix.tools import BUILTIN_TOOLS

QUEUED_PREFIX = "› "  # noqa: RUF001  -- U+203A is the chosen UI glyph

MAX_PANEL_ROWS = 5

HEARTBEAT_GLYPH = "●"
# Breathing cadence. v0.9.5 raises the refresh from 10 Hz (the v0.9.4
# 100 ms/tick) to 120 Hz so the fade reads as a continuous glow rather
# than ~10 visible brightness steps/s — 10 fps sits below the
# smooth-motion perception floor. The 4 s period (resting-calm
# ~15 BPM) is unchanged; the frame count scales with the refresh so
# one breath still spans exactly one period.
HEARTBEAT_BREATH_PERIOD_S = 4.0
HEARTBEAT_REFRESH_HZ = 120
HEARTBEAT_CYCLE_FRAMES = round(HEARTBEAT_REFRESH_HZ * HEARTBEAT_BREATH_PERIOD_S)  # 480
HEARTBEAT_TICK_MS = 1000 / HEARTBEAT_REFRESH_HZ  # ≈ 8.33 ms/frame
HEARTBEAT_JITTER_RATIO = 0.10
HEARTBEAT_TROUGH_RGB: tuple[int, int, int] = (60, 60, 60)
HEARTBEAT_PEAK_RGB: tuple[int, int, int] = (255, 255, 255)
# Stalled palette (v0.9.5 split #3): the breathing rhythm continues
# — only the gradient anchors swap. Low-brightness red to bright red
# so the dot keeps reading as "alive, but waiting too long."
HEARTBEAT_STALLED_TROUGH_RGB: tuple[int, int, int] = (60, 0, 0)
HEARTBEAT_STALLED_PEAK_RGB: tuple[int, int, int] = (255, 60, 60)
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


def _build_brightness_cycle(
    trough: tuple[int, int, int],
    peak: tuple[int, int, int],
) -> tuple[str, ...]:
    """Precompute HEARTBEAT_CYCLE_FRAMES hex-color style strings along
    a raised-cosine breathing curve from trough (frame 0) to peak
    (frame N/2) to trough (frame N).
    """
    cycle: list[str] = []
    for frame in range(HEARTBEAT_CYCLE_FRAMES):
        # Raised cosine in [0, 1]: 0 at frame 0 and N, 1 at frame N/2.
        progress = (1 - math.cos(2 * math.pi * frame / HEARTBEAT_CYCLE_FRAMES)) / 2
        r = round(trough[0] + (peak[0] - trough[0]) * progress)
        g = round(trough[1] + (peak[1] - trough[1]) * progress)
        b = round(trough[2] + (peak[2] - trough[2]) * progress)
        cycle.append(f"fg:#{r:02x}{g:02x}{b:02x}")
    return tuple(cycle)


HEARTBEAT_BRIGHTNESS_CYCLE: tuple[str, ...] = _build_brightness_cycle(
    HEARTBEAT_TROUGH_RGB, HEARTBEAT_PEAK_RGB
)
HEARTBEAT_STALLED_CYCLE: tuple[str, ...] = _build_brightness_cycle(
    HEARTBEAT_STALLED_TROUGH_RGB, HEARTBEAT_STALLED_PEAK_RGB
)


async def jittered_sleep(
    nominal_seconds: float,
    *,
    jitter_ratio: float = HEARTBEAT_JITTER_RATIO,
    rng: random.Random | None = None,
) -> None:
    """Sleep ``nominal_seconds`` with a uniform ±jitter_ratio multiplier.

    Default RNG is the :mod:`random` module-level singleton; tests
    can pass a seeded :class:`random.Random` for determinism. With
    ``jitter_ratio=0.0`` the sleep is exact (no randomness).
    """
    if jitter_ratio <= 0:
        await asyncio.sleep(nominal_seconds)
        return
    sampler = rng.uniform if rng is not None else random.uniform
    factor = sampler(1.0 - jitter_ratio, 1.0 + jitter_ratio)
    await asyncio.sleep(nominal_seconds * factor)

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


async def heartbeat_loop(
    state_supplier: Callable[[], State],
    store: ChatStore,
    on_tick: Callable[[], None],
    *,
    sleep_seconds: float = HEARTBEAT_TICK_MS / 1000,
    sleep_fn: Callable[[float], Awaitable[None]] | None = None,
) -> None:
    """Drive the heartbeat: tick while busy, wait on store changes when idle.

    While ``state_supplier()`` is busy (anything except
    :attr:`State.IDLE`), awaits ``sleep_fn(sleep_seconds)`` and then
    calls ``on_tick``. When the state is :attr:`State.IDLE`, blocks
    on the next :py:meth:`ChatStore.changes` yield — CM state
    transitions always accompany a store mutation, so the next busy
    phase wakes the loop. Cleanly cancellable.

    The default ``sleep_fn`` is :func:`jittered_sleep` which applies
    ±10% noise to each tick for an organic, less-mechanical breathing
    cadence. Tests pass a deterministic sleep (plain
    :func:`asyncio.sleep`) to stabilize timing assertions.
    """
    if sleep_fn is None:
        sleep_fn = jittered_sleep
    changes = store.changes()
    try:
        while True:
            while state_supplier() == State.IDLE:
                await changes.__anext__()
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
    fragments: the breathing glyph (``HEARTBEAT_GLYPH`` styled per the
    active cycle table) and a static label that names the current
    phase. The glyph fades smoothly along a truecolor gradient; the
    label stays bright (split #2 — Steve-Jobs-mode breathing dot,
    split #13 — truecolor smoothing).

    When ``state == AWAITING_LLM`` and ``last_progress_at`` is set and
    more than ``stall_threshold_s`` seconds ago, the palette swaps to
    ``HEARTBEAT_STALLED_CYCLE`` (red gradient) and the label becomes
    ``"LLM (stalled)"`` — v0.9.5 split #1 / #2 / #3. The stall hint is
    UI-only; the hard timeout is enforced by
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
    cycle = HEARTBEAT_STALLED_CYCLE if is_stalled else HEARTBEAT_BRIGHTNESS_CYCLE
    glyph_style = cycle[tick % HEARTBEAT_CYCLE_FRAMES]
    return [
        (glyph_style, f"{HEARTBEAT_GLYPH} "),
        (HEARTBEAT_LABEL_STYLE, f"{label}\n"),
    ]


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
        return ("<- ", "tool_result".ljust(TOOL_KEYWORD_WIDTH), self._summary_body())


InputFunc = Callable[[str], str]


class DraftReader:
    """Bottom draft editor — same shape as v0.9.2."""

    def __init__(
        self,
        *,
        message_supplier: Callable[[], object] = lambda: "",
        cancel_hook: Callable[[], bool] | None = None,
    ) -> None:
        self._message_supplier = message_supplier
        self.quit_state = QuitArmingState()
        self.cancel_hook = cancel_hook
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


def build_draft_key_bindings(
    quit_state: QuitArmingState | None = None,
    *,
    cancel_hook: Callable[[], bool] | None = None,
):
    """Build explicit editor bindings for terminal draft input.

    ``cancel_hook`` is invoked on Esc and on the first Ctrl+C while
    something is in flight. It returns ``True`` iff the cancel actually
    fired. The hook is :py:meth:`ContextManager.cancel` (sync).
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
        )
        self._running = True
        self._busy = False
        self._input_queue: asyncio.Queue[str] | None = None
        self._tool_records: list[ToolRecord] = []
        # Per-render lookup so a ``role:tool`` record can find the
        # arguments string that the matching assistant ``tool_call``
        # carried. Populated by the renderer as it walks assistant
        # records with ``tool_calls``.
        self._tool_call_lookup: dict[str, tuple[str, str]] = {}
        # Index of the last rendered message. The render watcher walks
        # forward through ``store.messages`` from this point.
        self._rendered_message_count: int = 0
        # Monotonic frame counter feeding the heartbeat brightness cycle.
        self._heartbeat_tick: int = 0

    def run(self) -> None:
        """Run the blocking terminal chat loop."""
        asyncio.run(self.run_async())

    async def run_async(self) -> None:
        await self._render_initial_transcript()
        self._input_queue = asyncio.Queue()
        worker = asyncio.create_task(self._worker_loop())
        renderer = asyncio.create_task(self._render_watcher())
        heartbeat = asyncio.create_task(self._heartbeat_ticker())
        try:
            with self.view.output_patch():
                await self._input_loop()
            if self._input_queue is not None:
                await self._input_queue.join()
        finally:
            worker.cancel()
            renderer.cancel()
            heartbeat.cancel()
            for task in (worker, renderer, heartbeat):
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
        if not heartbeat and not tasks and not queued and quit_hint is None:
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
            for _style, text in format_task_panel(tasks):
                lines.append(text.rstrip("\n"))
            for q in queued:
                lines.append(f"{QUEUED_PREFIX}{q.text}")
            if quit_hint is not None:
                lines.append(quit_hint)
            return "\n".join(lines) + "\n" if lines else ""

        fragments: list[tuple[str, str]] = list(heartbeat)
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

        See :func:`heartbeat_loop` for the loop semantics. The tick
        counter is incremented and the prompt_toolkit app invalidated
        once per frame while CM is busy; idle phases consume no CPU.
        """
        def on_tick() -> None:
            self._heartbeat_tick += 1
            self._invalidate_app()

        await heartbeat_loop(
            state_supplier=lambda: self.ctx.state,
            store=self.store,
            on_tick=on_tick,
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

    async def _render_new_records(self) -> None:
        records = self.store.messages
        while self._rendered_message_count < len(records):
            record = records[self._rendered_message_count]
            await self._render_record(record)
            self._rendered_message_count += 1

    async def _render_record(self, record: MessageRecord) -> None:
        role = record.role
        content = record.content
        if role == "user" and isinstance(content, str) and is_task_reminder(content):
            await self.view.print_notice(
                format_reminder_notice(self.store.tasks), style="dim"
            )
            return
        if role == "system":
            if content:
                await self.view.print_system(str(content))
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
            name, arguments = self._tool_call_lookup.get(call_id, (record.tool_name or "tool", "{}"))
            await self._store_and_print_tool_result(name, arguments, str(content or ""))
            return

    def _tool_calls_from_record(
        self, record: MessageRecord
    ) -> list[tuple[str, str, str]]:
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
        if self._busy and cmd in {"fast", "strong", "save", "load", "onboard", "compact"}:
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
                    "  /compact            drop the oldest ~50% of history (no summary)",
                    "  /tools              list tools",
                    "  /tools on|off       enable/disable tool calling",
                    "  /tool [N]           list folded tool results or expand one",
                    "  /quit               exit",
                ]
            )
        )

    async def _cmd_status(self, args: list[str]) -> None:
        await self.view.print_plain(self._status_line())

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
        self._tool_records.clear()
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
        self._tool_records.clear()
        self._tool_call_lookup.clear()
        self._rendered_message_count = 0
        await self.view.print_notice("conversation cleared", style="green")
        await self._render_new_records()

    async def _cmd_compact(self, args: list[str]) -> None:
        outcome = await self.ctx.compact()
        if not outcome.did_compact:
            await self.view.print_notice(
                "nothing to compact (conversation too short)", style="dim"
            )
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
