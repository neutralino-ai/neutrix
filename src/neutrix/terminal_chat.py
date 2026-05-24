"""Append-only terminal chat renderer.

The main chat uses ordinary terminal scrollback instead of a fullscreen app.
The agent still owns conversation state; this module only renders events and
handles slash commands.
"""
from __future__ import annotations

import asyncio
import re
import sys
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from neutrix import transcript
from neutrix.agent_loop import Agent, AgentEvent
from neutrix.config import SLOT_NAMES, Config, ConfigError, load_config
from neutrix.store import ChatStore, MessageRecord, openai_to_record
from neutrix.tools import BUILTIN_TOOLS

QUEUED_PREFIX = "› "  # noqa: RUF001  -- U+203A is the chosen UI glyph

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


@dataclass(frozen=True)
class ToolRecord:
    index: int
    name: str
    arguments: str
    result: str

    @property
    def summary(self) -> str:
        args = compact_inline(self.arguments or "{}")
        lines = result_line_count(self.result)
        approx_tokens = approximate_token_count(self.result)
        return (
            f"<- [tool {self.index}] {self.name} {args} | folded | "
            f"{lines} lines | ~{approx_tokens} tokens"
        )


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
        self._session = self._build_session()

    def read(self) -> str:
        if self._session is None:
            return input("")
        return self._session.prompt()

    async def read_async(self) -> str:
        if self._session is None:
            return await asyncio.to_thread(input, "")
        return await self._session.prompt_async()

    @property
    def uses_prompt_toolkit(self) -> bool:
        return self._session is not None

    def _build_session(self):
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.enums import EditingMode
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
        return PromptSession(
            self._message_supplier,
            multiline=True,
            editing_mode=EditingMode.EMACS,
            erase_when_done=True,
            placeholder="Message the assistant  (/help for commands)",
            prompt_continuation="",
            history=InMemoryHistory(),
            key_bindings=build_draft_key_bindings(),
        )


def build_draft_key_bindings():
    """Build explicit editor bindings for terminal draft input."""
    try:
        from prompt_toolkit.key_binding import KeyBindings
    except ImportError:
        return None

    bindings = KeyBindings()

    @bindings.add("enter")
    def _(event) -> None:
        event.app.exit(result=event.current_buffer.text)

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
            message_supplier=self._queued_message,
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

    def _queued_message(self):
        """Content rendered directly above the input cursor.

        Each queued user message is one dim-foreground line prefixed
        with ``QUEUED_PREFIX``; the input cursor lands on the line below
        the last queued item. Returns an empty string when no messages
        are queued, so the input cursor sits at its natural position.
        Returns FormattedText when prompt_toolkit is installed, plain
        str otherwise.
        """
        queued = self.store.queued_user_messages
        if not queued:
            return ""
        try:
            from prompt_toolkit.formatted_text import FormattedText
        except ImportError:
            return "\n".join(f"{QUEUED_PREFIX}{q.text}" for q in queued) + "\n"
        fragments: list[tuple[str, str]] = [
            ("fg:ansibrightblack", f"{QUEUED_PREFIX}{q.text}\n") for q in queued
        ]
        return FormattedText(fragments)

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
                    await self.view.print_notice(
                        f"-> {name} {compact_inline(arguments or '{}')}"
                    )
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
            assistant_started = False
            async for event in self.agent.stream_reply(text):
                assistant_started = await self._handle_event(event, assistant_started)
            if self._streaming_assistant:
                await self.view.write_raw("\n")
                self._streaming_assistant = False
            # The agent appended its assistant turn to its own list; mirror
            # the latest message into the store as well so future renderers
            # can read a complete typed transcript.
            self._mirror_new_agent_messages()
        except Exception as exc:
            logger.exception("terminal chat worker failed")
            await self.view.print_notice(str(exc), style="bold red")

    def _mirror_new_agent_messages(self) -> None:
        """Append any agent messages not yet in the store."""
        stored = len(self.store.messages)
        for raw in self.agent.messages[stored:]:
            if isinstance(raw, dict):
                self.store.append_message(openai_to_record(raw))

    async def _handle_event(self, event: AgentEvent, assistant_started: bool) -> bool:
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
            self.store.add_pending_tool_call(name, arguments)
            await self.view.print_notice(f"-> {name} {compact_inline(arguments or '{}')}")
            return assistant_started

        if event.kind == "tool_result":
            name = str(event.data["name"])
            arguments = self._pop_tool_arguments(name)
            result = str(event.data["result"])
            await self._store_and_print_tool_result(name, arguments, result)
            return assistant_started

        if event.kind == "error":
            await self.view.print_notice(str(event.data), style="bold red")
            return assistant_started

        return assistant_started

    def _pop_tool_arguments(self, name: str) -> str:
        call = self.store.remove_pending_tool_call(name)
        if call is None:
            return "{}"
        return call.arguments

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
        await self.view.print_notice(record.summary, style="yellow")
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
        _store, metadata = transcript.load(args[0])
        self.agent.messages = list(metadata["raw_messages"])
        self.store.reset()
        self._seed_store_from_agent()
        self._tool_records.clear()
        await self.view.print_notice(
            f"loaded {args[0]} ({len(self.agent.messages)} msgs); "
            f"current slot unchanged",
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
                await self.view.print_notice(record.summary, style="yellow")
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
