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

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from loguru import logger
from rich.markdown import Markdown
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message as TextualMessage
from textual.timer import Timer
from textual.widgets import Static, TextArea

from neutrix import transcript
from neutrix.agent_loop import (
    Agent,
    AgentEvent,
    format_reminder_notice,
    is_task_reminder,
)
from neutrix.config import SLOT_NAMES, Config, ConfigError, load_config
from neutrix.store import ChatStore, openai_to_record
from neutrix.tools import BUILTIN_TOOLS

ROLE_STYLE = {
    "user": "",
    "assistant": "",
    "tool": "bold yellow",
    "system": "bold yellow",
    "error": "bold red",
    "reminder": "dim",
}

ROLE_LABEL = {
    "user": "",
    "assistant": "",
    "system": "",
    "tool": "Tool",
    "error": "Error",
    "reminder": "",
}

NOTICE_STYLE = {
    "info": "dim",
    "success": "green",
    "warning": "yellow",
    "error": "bold red",
}


class DraftInput(TextArea):
    """Multiline chat draft that keeps Enter as submit."""

    MAX_VISIBLE_LINES: ClassVar[int] = 6

    @dataclass
    class Submitted(TextualMessage):
        draft: DraftInput
        value: str

        @property
        def control(self) -> DraftInput:
            return self.draft

    def __init__(self, *, placeholder: str, id: str) -> None:
        super().__init__(
            "",
            id=id,
            placeholder=placeholder,
            compact=True,
            highlight_cursor_line=False,
            show_line_numbers=False,
            soft_wrap=True,
        )
        self._sync_height()

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, value: str) -> None:
        self.load_text(value)
        self._sync_height()

    def _sync_height(self) -> None:
        lines = max(1, self.text.count("\n") + 1)
        self.styles.height = min(lines, self.MAX_VISIBLE_LINES)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area is self:
            self._sync_height()

    async def _on_key(self, event: events.Key) -> None:
        if self.disabled or self.read_only:
            await super()._on_key(event)
            return

        if event.key in {"up", "down", "ctrl+up", "ctrl+down"}:
            event.stop()
            event.prevent_default()
            app = self.app
            if event.key == "up":
                app.action_focus_block(-1)
            elif event.key == "down":
                app.action_focus_block(1)
            elif event.key == "ctrl+up":
                app.action_focus_block_page(-1)
            else:
                app.action_focus_block_page(1)
            return

        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self, self.text))
            return

        if event.key in {"shift+enter", "alt+enter", "ctrl+j"}:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return

        await super()._on_key(event)


class Message(Static):
    """Single message bubble; updated in-place during streaming."""

    can_focus = True

    def __init__(self, role: str, content: str = "", *, markdown: bool = False) -> None:
        super().__init__(classes=f"block message role-{role}")
        self.role = role
        self._content = content
        self._markdown = markdown
        label = ROLE_LABEL.get(role, role.title())
        if label:
            self.border_title = label
        self._refresh()

    def append(self, text: str) -> None:
        self._content += text
        self._refresh()

    def _refresh(self) -> None:
        style = ROLE_STYLE.get(self.role, "white")
        if self._markdown and self.role == "assistant":
            self.update(Markdown(self._content) if self._content else Text(""))
        else:
            inline_style = style if self.role in {"error", "reminder"} else ""
            self.update(Text(self._content, style=inline_style))


class NeutrixApp(App):
    TITLE = ""
    SUB_TITLE = ""

    CSS = """
    Screen {
        layout: vertical;
        background: #10100e;
    }

    #chat {
        height: 1fr;
        width: 1fr;
        padding: 0;
        align: left top;
    }

    #blocks {
        height: 1fr;
        width: 1fr;
        padding: 0 0;
        border: none;
        background: #10100e;
    }

    #messages-status {
        height: 1;
        margin: 0 1;
        color: #e8e2d6;
    }

    .block {
        width: 1fr;
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
        border: none;
        background: #1e1c18;
        color: #e8e2d6;
    }
    .message:focus {
        background: #363129;
    }

    #thinking {
        display: none;
        height: 1;
        margin: 0;
        color: #d6a01d;
    }
    #thinking.active {
        display: block;
    }
    #input {
        height: 1;
        max-height: 6;
        border: none;
        padding: 0;
        background: #1e1c18;
        color: #e8e2d6;
    }
    #input:focus {
        background: #1e1c18;
    }
    #input:disabled {
        color: #9a9284;
        background: #1e1c18;
    }
    #input .text-area--cursor-line {
        background: #1e1c18;
    }
    #notice {
        min-height: 1;
        max-height: 8;
        margin: 0 1;
        color: #9a9284;
    }

    .role-user {
        background: #1e1c18;
        color: #e8e2d6;
    }
    .role-assistant {
        background: #2a2824;
        color: #e8e2d6;
    }
    .role-system {
        background: #2a2824;
        color: #d6a01d;
    }
    .role-tool {
        background: #2a2824;
        color: #d6a01d;
    }
    .role-error {
        background: #1e1c18;
        color: #c65a4a;
    }
    .role-reminder {
        background: #10100e;
        color: #9a9284;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("up", "focus_block(-1)", "Previous block", show=False, priority=True),
        Binding("down", "focus_block(1)", "Next block", show=False, priority=True),
        Binding(
            "ctrl+up",
            "focus_block_page(-1)",
            "Previous page",
            show=False,
            priority=True,
        ),
        Binding(
            "ctrl+down",
            "focus_block_page(1)",
            "Next page",
            show=False,
            priority=True,
        ),
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
        self.store = ChatStore()
        self.agent.store = self.store

    def compose(self) -> ComposeResult:
        with Vertical(id="chat"):
            with VerticalScroll(id="blocks"):
                with Vertical(id="composer", classes="block role-user draft"):
                    yield Static("", id="thinking")
                    yield DraftInput(
                        placeholder="Message the assistant  (/help for commands)",
                        id="input",
                    )
        yield Static("", id="notice")
        yield Static(self._status_text(), id="messages-status")

    def on_mount(self) -> None:
        self.title = ""
        self.sub_title = ""
        self._render_model_blocks()
        self.query_one("#input", DraftInput).focus()

    # ----- UI helpers ---------------------------------------------------------

    def _status_text(self) -> str:
        tools = self._tool_status()
        s = self.agent.slot
        return (
            f" {s.provider} · {s.model} · tools:{tools} · "
            f"msgs:{len(self.agent.messages)} "
        )

    def _tool_status(self) -> str:
        if not self.agent.use_tools:
            return "off"
        enabled = getattr(self.agent, "effective_tools_enabled", None)
        if callable(enabled) and not enabled():
            return "unsupported"
        return "on"

    def _refresh_status(self) -> None:
        self.query_one("#messages-status", Static).update(self._status_text())

    def _block_focus_targets(self) -> list[Message | DraftInput]:
        targets: list[Message | DraftInput] = list(self.query(Message))
        try:
            targets.append(self.query_one("#input", DraftInput))
        except NoMatches:
            pass
        return targets

    def _focused_block_index(self, targets: list[Message | DraftInput]) -> int:
        focused = self.focused
        if focused in targets:
            return targets.index(focused)
        return len(targets) - 1

    def action_focus_block(self, delta: int) -> None:
        targets = self._block_focus_targets()
        if not targets:
            return
        index = max(0, min(len(targets) - 1, self._focused_block_index(targets) + delta))
        target = targets[index]
        target.focus()
        target.scroll_visible(animate=False)

    def action_focus_block_page(self, delta: int) -> None:
        try:
            blocks = self.query_one("#blocks", VerticalScroll)
            page = max(1, blocks.size.height // 3)
        except NoMatches:
            page = 6
        self.action_focus_block(delta * page)

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
            if role == "user" and is_task_reminder(message.get("content")):
                # v0.8.0 reminder body — render the folded notice instead
                # of leaking the templated text as a user block.
                self._post("reminder", format_reminder_notice(self.store.tasks))
                continue
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
            input_box = self.query_one("#input", DraftInput)
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

    def _is_chat_input(self, input_widget: DraftInput) -> bool:
        try:
            composer = self.query_one("#composer", Vertical)
        except NoMatches:
            return False
        return input_widget.id == "input" and input_widget.parent is composer

    # ----- input handler ------------------------------------------------------

    async def on_draft_input_submitted(self, event: DraftInput.Submitted) -> None:
        if not self._is_chat_input(event.draft):
            return
        event.stop()
        text = event.value.strip()
        if not text or self._busy:
            return
        event.draft.value = ""
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
        if ev.kind in {"token", "assistant"}:
            if assistant is None:
                assistant = self._post(
                    "assistant", "", markdown=self.render_markdown
                )
            assistant.append(str(ev.data))
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
            "  /tasks              list tracked tasks (read-only)",
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
        # Mirror the agent's messages into a fresh store so it carries
        # the same shape as terminal_chat — the live tasks come from
        # self.store, which the agent has been mutating directly.
        snapshot = ChatStore()
        for raw in self.agent.messages:
            if isinstance(raw, dict):
                snapshot.append_message(openai_to_record(raw))
        if self.store.tasks:
            snapshot.replace_tasks(self.store.tasks)
        out = transcript.save(
            path,
            snapshot,
            provider=self.agent.slot.provider,
            model=self.agent.slot.model,
        )
        self._notice(f"saved → {out}", severity="success")

    async def _cmd_load(self, args: list[str]) -> None:
        if not args:
            self._notice("usage: /load PATH", severity="error")
            return
        loaded, metadata = transcript.load(args[0])
        self.agent.messages = list(metadata["raw_messages"])
        self.store.reset()
        if loaded.tasks:
            self.store.replace_tasks(loaded.tasks)
        self._render_model_blocks()
        self._notice(
            f"loaded {args[0]} ({len(self.agent.messages)} msgs, "
            f"{len(self.store.tasks)} tasks); current slot unchanged",
            severity="success",
        )

    async def _cmd_clear(self, args: list[str]) -> None:
        self.agent.reset()
        self.store.reset()
        self._render_model_blocks()
        self._notice("conversation cleared", severity="success")

    async def _cmd_tasks(self, args: list[str]) -> None:
        tasks = self.store.tasks
        if not tasks:
            self._notice("no tasks")
            return
        lines = [f"#{t.id} [{t.status}] {t.subject}" for t in tasks]
        self._notice("\n".join(lines))

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
        lines.append(f"status: {self._tool_status()}")
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
