"""Textual TUI: scrollable chat log + input + status bar.

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
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Static

from neutrix import __version__
from neutrix.agent import Agent, AgentEvent
from neutrix.config import SLOT_NAMES, Config, ConfigError, load_config
from neutrix.session import dump as session_dump
from neutrix.session import load as session_load
from neutrix.tools import BUILTIN_TOOLS

ROLE_STYLE = {
    "user": "bold cyan",
    "assistant": "bold green",
    "tool": "bold yellow",
    "system": "dim",
    "error": "bold red",
}


class Message(Static):
    """Single message bubble; updated in-place during streaming."""

    def __init__(self, role: str, content: str = "", *, markdown: bool = False) -> None:
        super().__init__()
        self.role = role
        self._content = content
        self._markdown = markdown
        self._refresh()

    def append(self, text: str) -> None:
        self._content += text
        self._refresh()

    def _refresh(self) -> None:
        style = ROLE_STYLE.get(self.role, "white")
        prefix = Text(f"{self.role}: ", style=style)
        if self._markdown and self.role == "assistant":
            self.update(Markdown(self._content) if self._content else prefix)
        else:
            body = Text(self._content)
            self.update(prefix + body)


class NeutrixApp(App):
    CSS = """
    Screen { layout: vertical; }
    #log {
        height: 1fr;
        padding: 0 1;
        border: round $primary;
    }
    Message {
        margin: 0 0 1 0;
        padding: 0;
    }
    #input { dock: bottom; }
    #status { dock: bottom; height: 1; background: $boost; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_log", "Clear", show=True),
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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield VerticalScroll(id="log")
        yield Static(self._status_text(), id="status")
        yield Input(placeholder="Send a message  (/help for commands)", id="input")
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"neutrix v{__version__}"
        self._refresh_subtitle()
        self.query_one("#input", Input).focus()
        self._post(
            "system",
            f"connected to {self.agent.slot.name} slot — "
            f"{self.agent.slot.provider}/{self.agent.slot.model}. "
            f"Type /help for commands.",
        )

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

    def _post(self, role: str, content: str, *, markdown: bool = False) -> Message:
        msg = Message(role, content, markdown=markdown)
        log = self.query_one("#log", VerticalScroll)
        log.mount(msg)
        log.scroll_end(animate=False)
        return msg

    # ----- input handler ------------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text or self._busy:
            return
        if text.startswith("/"):
            await self._run_command(text)
            return
        self._post("user", text)
        self.run_worker(self._send_to_model(text), exclusive=True)

    async def _send_to_model(self, text: str) -> None:
        self._busy = True
        assistant = self._post("assistant", "", markdown=self.render_markdown)
        try:
            async for ev in self.agent.stream_reply(text):
                self._handle_event(ev, assistant)
                if ev.kind in ("tool_call", "tool_result"):
                    assistant = self._post(
                        "assistant", "", markdown=self.render_markdown
                    )
        finally:
            if not assistant._content:
                assistant.remove()
            self._busy = False
            self._refresh_status()

    def _handle_event(self, ev: AgentEvent, assistant: Message) -> None:
        if ev.kind == "token":
            assistant.append(ev.data)
            self.query_one("#log", VerticalScroll).scroll_end(animate=False)
        elif ev.kind == "tool_call":
            self._post("tool", f"→ {ev.data['name']}({ev.data['arguments']})")
        elif ev.kind == "tool_result":
            result = ev.data["result"]
            preview = result if len(result) < 800 else result[:800] + "\n…(truncated)"
            self._post("tool", f"← {ev.data['name']}:\n{preview}")
        elif ev.kind == "error":
            self._post("error", str(ev.data))

    # ----- slash commands -----------------------------------------------------

    async def _run_command(self, line: str) -> None:
        parts = line[1:].strip().split()
        cmd, args = (parts[0].lower() if parts else ""), parts[1:]
        try:
            handler = getattr(self, f"_cmd_{cmd}", None)
            if handler is None:
                self._post("error", f"unknown command: /{cmd}. Try /help.")
                return
            await handler(args)
        except Exception as e:
            logger.exception("command /{} failed", cmd)
            self._post("error", f"/{cmd} failed: {e}")
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
        self._post("system", "\n".join(lines))

    async def _cmd_fast(self, args: list[str]) -> None:
        await self._switch_slot("fast")

    async def _cmd_strong(self, args: list[str]) -> None:
        await self._switch_slot("strong")

    async def _switch_slot(self, name: str) -> None:
        slot = self.config.slot(name)
        self.agent.switch(slot)
        self._post("system", f"switched to {name}: {slot.provider}/{slot.model}")

    async def _cmd_model(self, args: list[str]) -> None:
        s = self.agent.slot
        lines = [
            f"current: [{s.name}] {s.provider}/{s.model}",
            f"slots available: {', '.join(SLOT_NAMES)}",
            "edit ~/.config/neutrix/config.yaml to change slot bindings",
        ]
        self._post("system", "\n".join(lines))

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
        self._post("system", f"saved → {out}")

    async def _cmd_load(self, args: list[str]) -> None:
        if not args:
            self._post("error", "usage: /load PATH")
            return
        payload = session_load(args[0])
        self.agent.messages = payload["messages"]
        self._post(
            "system",
            f"loaded {args[0]} ({len(self.agent.messages)} msgs); "
            f"current slot unchanged",
        )

    async def _cmd_clear(self, args: list[str]) -> None:
        self.agent.reset()
        log = self.query_one("#log", VerticalScroll)
        for child in list(log.children):
            child.remove()
        self._post("system", "conversation cleared")

    async def _cmd_tools(self, args: list[str]) -> None:
        if args and args[0] in ("on", "off"):
            self.agent.use_tools = args[0] == "on"
            self._post(
                "system",
                f"tool calling {'enabled' if self.agent.use_tools else 'disabled'}",
            )
            return
        lines = ["available tools:"]
        for t in BUILTIN_TOOLS.values():
            lines.append(f"  • {t.name} — {t.description}")
        lines.append(f"status: {'on' if self.agent.use_tools else 'off'}")
        self._post("system", "\n".join(lines))

    async def _cmd_onboard(self, args: list[str]) -> None:
        from neutrix.onboard import OnboardScreen

        self.push_screen(
            OnboardScreen(self.config), callback=self._on_onboard_done
        )

    def _on_onboard_done(self, saved: bool | None) -> None:
        try:
            self.config = load_config(self.config.path)
        except ConfigError as e:
            self._post("error", f"config reload failed: {e}")
            return
        self._post(
            "system",
            "back from onboarding. Use /fast or /strong to switch to a "
            "newly-bound slot; current slot unchanged.",
        )
        self._refresh_status()

    async def _cmd_quit(self, args: list[str]) -> None:
        self.exit()

    # ----- bindings -----------------------------------------------------------

    def action_clear_log(self) -> None:
        log = self.query_one("#log", VerticalScroll)
        for child in list(log.children):
            child.remove()
