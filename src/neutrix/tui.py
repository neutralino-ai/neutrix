"""Textual TUI: model-visible block list + input + status bar.

Slash commands:
    /help                 show commands
    /fast                 switch to the fast slot
    /strong               switch to the strong slot
    /model                show current slot / provider / model
    /onboard              re-enter the onboarding TUI to manage keys/slots
    /save [PATH]          save session to JSON
    /load PATH            load session from JSON
    /clear                start a fresh conversation
    /tools                list / toggle tools (on|off)
    /quit                 exit
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import ClassVar

from loguru import logger
from rich.markdown import Markdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.timer import Timer
from textual.widgets import Footer, Header, Input, Static

from neutrix import __version__
from neutrix.agent_loop import Agent, AgentEvent
from neutrix.config import SLOT_NAMES, Config, ConfigError, load_config
from neutrix.session import dump as session_dump
from neutrix.session import load as session_load
from neutrix.tools import BUILTIN_TOOLS

ROLE_STYLE = {
    "user": "",
    "assistant": "",
    "tool": "bold yellow",
    "system": "bold yellow",
    "error": "bold red",
}

ROLE_LABEL = {
    "user": "User",
    "assistant": "LLM",
    "system": "System",
    "tool": "Tool",
    "error": "Error",
}

NOTICE_STYLE = {
    "info": "dim",
    "success": "green",
    "warning": "yellow",
    "error": "bold red",
}


class Message(Static):
    """Single message bubble; updated in-place during streaming."""

    def __init__(self, role: str, content: str = "", *, markdown: bool = False) -> None:
        super().__init__(classes=f"block message role-{role}")
        self.role = role
        self._content = content
        self._markdown = markdown
        self.border_title = ROLE_LABEL.get(role, role.title())
        self._refresh()

    def append(self, text: str) -> None:
        self._content += text
        self._refresh()

    def _refresh(self) -> None:
        style = ROLE_STYLE.get(self.role, "white")
        if self._markdown and self.role == "assistant":
            self.update(Markdown(self._content) if self._content else Text(""))
        else:
            self.update(Text(self._content, style=style if self.role == "error" else ""))


class NeutrixApp(App):
    CSS = """
    Screen {
        layout: vertical;
        background: $background;
    }

    #chat {
        height: 1fr;
        width: 1fr;
        padding: 1 0;
        align: left top;
    }

    #blocks {
        height: 1fr;
        width: 1fr;
        padding: 0 0 1 0;
    }

    .block {
        width: 1fr;
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
        border: solid $surface;
        background: $surface;
    }

    #thinking {
        display: none;
        height: 1;
        margin: 0 0 1 0;
        color: $warning;
    }
    #thinking.active {
        display: block;
    }
    #input {
        height: 3;
        border: none;
        background: $surface;
        padding: 0 1;
    }
    #input:focus {
        background: $boost;
    }
    #input:disabled {
        color: $text-muted;
    }
    #notice {
        min-height: 1;
        max-height: 8;
        margin: 0 2;
        color: $text-muted;
    }
    #status {
        height: 1;
        margin: 0 2;
        color: $text-muted;
    }

    .role-user {
        border: solid $boost;
        background: $surface;
    }
    .role-assistant {
        border: solid $primary;
        background: $panel;
        color: $text;
    }
    .role-system {
        border: solid $warning;
        background: $panel;
        color: $warning;
    }
    .role-tool {
        border: solid $warning;
    }
    .role-error {
        border: solid $error;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_log", "Notice", show=True),
    ]

    def __init__(
        self,
        agent: Agent,
        *,
        config: Config,
        render_markdown: bool = True,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.config = config
        self.render_markdown = render_markdown
        self._busy = False
        self._thinking_tick = 0
        self._thinking_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="chat"):
            with VerticalScroll(id="blocks"):
                with Vertical(id="composer", classes="block role-user draft") as composer:
                    composer.border_title = ROLE_LABEL["user"]
                    yield Static("", id="thinking")
                    yield Input(
                        placeholder="Message the assistant  (/help for commands)",
                        id="input",
                    )
        yield Static("", id="notice")
        yield Static(self._status_text(), id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"neutrix v{__version__}"
        self._refresh_subtitle()
        self._render_model_blocks()
        self.query_one("#input", Input).focus()

    # ----- UI helpers ---------------------------------------------------------

    def _status_text(self) -> str:
        tools = "on" if self.agent.use_tools else "off"
        s = self.agent.slot
        return (
            f" [{s.name}] {s.provider} · {s.model} · "
            f"tools:{tools} · msgs:{len(self.agent.messages)} "
        )

    def _refresh_status(self) -> None:
        self.query_one("#status", Static).update(self._status_text())
        self._refresh_subtitle()

    def _refresh_subtitle(self) -> None:
        s = self.agent.slot
        self.sub_title = f"{s.name} · {s.provider}/{s.model}"

    def _message_content(self, message: dict[str, object]) -> str:
        content = message.get("content")
        if content:
            return str(content)

        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            return ""

        lines: list[str] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name") or "unknown"
            arguments = function.get("arguments") or ""
            lines.append(f"tool call: {name}({arguments})")
        return "\n".join(lines)

    def _clear_model_blocks(self) -> None:
        for child in list(self.query(Message)):
            child.remove()

    def _render_model_blocks(self) -> None:
        self._clear_model_blocks()
        for message in self.agent.messages:
            role = str(message.get("role") or "")
            content = self._message_content(message)
            if content:
                self._post(role, str(content), markdown=role == "assistant")
        self.query_one("#blocks", VerticalScroll).scroll_end(animate=False)

    def _notice(self, content: str, *, severity: str = "info") -> None:
        style = NOTICE_STYLE.get(severity, NOTICE_STYLE["info"])
        self.query_one("#notice", Static).update(Text(content, style=style))

    def _post(self, role: str, content: str, *, markdown: bool = False) -> Message:
        msg = Message(role, content, markdown=markdown)
        blocks = self.query_one("#blocks", VerticalScroll)
        composer = self.query_one("#composer", Vertical)
        blocks.mount(msg, before=composer)
        blocks.scroll_end(animate=False)
        return msg

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        try:
            input_box = self.query_one("#input", Input)
            thinking = self.query_one("#thinking", Static)
        except NoMatches:
            return

        input_box.disabled = busy
        input_box.placeholder = (
            "Assistant is responding..."
            if busy
            else "Message the assistant  (/help for commands)"
        )

        if busy:
            self._thinking_tick = 0
            thinking.add_class("active")
            self._tick_thinking()
            if self._thinking_timer is None:
                self._thinking_timer = self.set_interval(0.4, self._tick_thinking)
            return

        if self._thinking_timer is not None:
            self._thinking_timer.stop()
            self._thinking_timer = None
        thinking.remove_class("active")
        thinking.update("")
        input_box.focus()

    def _tick_thinking(self) -> None:
        dots = "." * ((self._thinking_tick % 3) + 1)
        try:
            self.query_one("#thinking", Static).update(
                f"assistant is responding{dots}"
            )
        except NoMatches:
            return
        self._thinking_tick += 1

    def _is_chat_input(self, input_widget: Input) -> bool:
        try:
            composer = self.query_one("#composer", Vertical)
        except NoMatches:
            return False
        return input_widget.id == "input" and input_widget.parent is composer

    # ----- input handler ------------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if not self._is_chat_input(event.input):
            return
        event.stop()
        text = event.value.strip()
        if not text or self._busy:
            return
        event.input.value = ""
        if text.startswith("/"):
            await self._run_command(text)
            return
        self._post("user", text)
        self._set_busy(True)
        self.run_worker(self._send_to_model(text), exclusive=True)

    async def _send_to_model(self, text: str) -> None:
        if not self._busy:
            self._set_busy(True)
        assistant: Message | None = None
        try:
            async for ev in self.agent.stream_reply(text):
                assistant = self._handle_event(ev, assistant)
        except Exception as exc:
            logger.exception("model worker failed")
            self._post("error", str(exc))
            self._notice(str(exc), severity="error")
        finally:
            if assistant is not None and not assistant._content:
                assistant.remove()
            self._set_busy(False)
            self._refresh_status()

    def _handle_event(
        self, ev: AgentEvent, assistant: Message | None
    ) -> Message | None:
        if ev.kind == "token":
            if assistant is None:
                assistant = self._post(
                    "assistant", "", markdown=self.render_markdown
                )
            assistant.append(ev.data)
            self.query_one("#blocks", VerticalScroll).scroll_end(animate=False)
            return assistant
        elif ev.kind == "tool_call":
            self._post("tool", f"→ {ev.data['name']}({ev.data['arguments']})")
            return None
        elif ev.kind == "tool_result":
            result = ev.data["result"]
            preview = result if len(result) < 800 else result[:800] + "\n…(truncated)"
            self._post("tool", f"← {ev.data['name']}:\n{preview}")
            return None
        elif ev.kind == "error":
            content = str(ev.data)
            self._post("error", content)
            self._notice(content, severity="error")
        return assistant

    # ----- slash commands -----------------------------------------------------

    async def _run_command(self, line: str) -> None:
        parts = line[1:].strip().split()
        cmd, args = (parts[0].lower() if parts else ""), parts[1:]
        try:
            handler = getattr(self, f"_cmd_{cmd}", None)
            if handler is None:
                self._notice(
                    f"unknown command: /{cmd}. Try /help.", severity="error"
                )
                return
            await handler(args)
        except Exception as e:
            logger.exception("command /{} failed", cmd)
            self._notice(f"/{cmd} failed: {e}", severity="error")
        self._refresh_status()

    async def _cmd_help(self, args: list[str]) -> None:
        lines = [
            "Commands:",
            "  /help               show this",
            "  /fast               switch to the fast slot",
            "  /strong             switch to the strong slot",
            "  /model              show current slot/provider/model",
            "  /onboard            re-enter onboarding (manage keys / slots)",
            "  /save [PATH]        save session (default: sessions/<ts>.json)",
            "  /load PATH          load session",
            "  /clear              start fresh conversation",
            "  /tools              list tools",
            "  /tools on|off       enable/disable tool calling",
            "  /quit               exit",
        ]
        self._notice("\n".join(lines))

    async def _cmd_fast(self, args: list[str]) -> None:
        await self._switch_slot("fast")

    async def _cmd_strong(self, args: list[str]) -> None:
        await self._switch_slot("strong")

    async def _switch_slot(self, name: str) -> None:
        slot = self.config.slot(name)
        self.agent.switch(slot)
        self._notice(
            f"switched to {name}: {slot.provider}/{slot.model}",
            severity="success",
        )

    async def _cmd_model(self, args: list[str]) -> None:
        s = self.agent.slot
        lines = [
            f"current: [{s.name}] {s.provider}/{s.model}",
            f"slots available: {', '.join(SLOT_NAMES)}",
            "edit ~/.config/neutrix/config.yaml to change slot bindings",
        ]
        self._notice("\n".join(lines))

    async def _cmd_save(self, args: list[str]) -> None:
        if args:
            path = Path(args[0])
        else:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = Path("sessions") / f"{ts}.json"
        out = session_dump(
            path,
            provider=self.agent.slot.provider,
            model=self.agent.slot.model,
            messages=self.agent.messages,
        )
        self._notice(f"saved → {out}", severity="success")

    async def _cmd_load(self, args: list[str]) -> None:
        if not args:
            self._notice("usage: /load PATH", severity="error")
            return
        payload = session_load(args[0])
        self.agent.messages = payload["messages"]
        self._render_model_blocks()
        self._notice(
            f"loaded {args[0]} ({len(self.agent.messages)} msgs); "
            f"current slot unchanged",
            severity="success",
        )

    async def _cmd_clear(self, args: list[str]) -> None:
        self.agent.reset()
        self._render_model_blocks()
        self._notice("conversation cleared", severity="success")

    async def _cmd_tools(self, args: list[str]) -> None:
        if args and args[0] in ("on", "off"):
            self.agent.use_tools = args[0] == "on"
            self._notice(
                f"tool calling {'enabled' if self.agent.use_tools else 'disabled'}",
                severity="success",
            )
            return
        lines = ["available tools:"]
        for t in BUILTIN_TOOLS.values():
            lines.append(f"  • {t.name} — {t.description}")
        lines.append(f"status: {'on' if self.agent.use_tools else 'off'}")
        self._notice("\n".join(lines))

    async def _cmd_onboard(self, args: list[str]) -> None:
        from neutrix.onboard import OnboardScreen

        self.push_screen(
            OnboardScreen(self.config), callback=self._on_onboard_done
        )

    def _on_onboard_done(self, saved: bool | None) -> None:
        try:
            self.config = load_config(self.config.path)
        except ConfigError as e:
            self._notice(f"config reload failed: {e}", severity="error")
            return
        self._notice(
            "back from onboarding. Use /fast or /strong to switch to a "
            "newly-bound slot; current slot unchanged.",
            severity="success",
        )
        self._refresh_status()

    async def _cmd_quit(self, args: list[str]) -> None:
        self.exit()

    # ----- bindings -----------------------------------------------------------

    def action_clear_log(self) -> None:
        self._notice("")
