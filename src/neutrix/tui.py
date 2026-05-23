"""Textual TUI: scrollable chat log + input + status bar.

Slash commands:
    /help                 show commands
    /model [P [M]]        switch provider / model
    /save [PATH]          save session to JSON
    /load PATH            load session from JSON
    /clear                start a fresh conversation
    /tools                toggle / list tools
    /quit                 exit

Anything else is sent to the model.
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
from neutrix.config import PROVIDERS, get_provider
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

    def set_content(self, text: str) -> None:
        self._content = text
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
        render_markdown: bool = True,
    ) -> None:
        super().__init__()
        self.agent = agent
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
        self.sub_title = f"{self.agent.provider.name} / {self.agent.model}"
        self.query_one("#input", Input).focus()
        self._post(
            "system",
            f"connected to {self.agent.provider.name} ({self.agent.model}). "
            f"Type /help for commands.",
        )

    # ----- UI helpers ---------------------------------------------------------

    def _status_text(self) -> str:
        tools = "on" if self.agent.use_tools else "off"
        return (
            f" {self.agent.provider.name} · {self.agent.model} · "
            f"tools:{tools} · msgs:{len(self.agent.messages)} "
        )

    def _refresh_status(self) -> None:
        self.query_one("#status", Static).update(self._status_text())
        self.sub_title = f"{self.agent.provider.name} / {self.agent.model}"

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
        # "done" and "needs_tool" are control-only

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
            "  /help                 show this",
            "  /model                show current",
            "  /model PROVIDER       switch provider (deepseek|glm|claude)",
            "  /model PROVIDER MODEL switch provider + model",
            "  /save [PATH]          save session (default: sessions/<ts>.json)",
            "  /load PATH            load session",
            "  /clear                start fresh conversation",
            "  /tools                list tools / toggle on|off",
            "  /tools on|off         enable/disable tool calling",
            "  /quit                 exit",
        ]
        self._post("system", "\n".join(lines))

    async def _cmd_model(self, args: list[str]) -> None:
        if not args:
            self._post(
                "system",
                f"current: {self.agent.provider.name} / {self.agent.model}\n"
                f"available providers: {', '.join(PROVIDERS)}",
            )
            return
        provider = get_provider(args[0])
        model = args[1] if len(args) > 1 else provider.default_model
        self.agent.switch(provider, model)
        self._post("system", f"switched to {provider.name} / {model}")

    async def _cmd_save(self, args: list[str]) -> None:
        if args:
            path = Path(args[0])
        else:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = Path("sessions") / f"{ts}.json"
        out = session_dump(
            path,
            provider=self.agent.provider.name,
            model=self.agent.model,
            messages=self.agent.messages,
        )
        self._post("system", f"saved → {out}")

    async def _cmd_load(self, args: list[str]) -> None:
        if not args:
            self._post("error", "usage: /load PATH")
            return
        payload = session_load(args[0])
        provider = get_provider(payload["provider"])
        self.agent.switch(provider, payload["model"])
        self.agent.messages = payload["messages"]
        self._post(
            "system",
            f"loaded {args[0]} ({len(self.agent.messages)} msgs, "
            f"{payload['provider']}/{payload['model']})",
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

    async def _cmd_quit(self, args: list[str]) -> None:
        self.exit()

    # ----- bindings -----------------------------------------------------------

    def action_clear_log(self) -> None:
        log = self.query_one("#log", VerticalScroll)
        for child in list(log.children):
            child.remove()
