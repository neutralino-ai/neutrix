"""First-run onboarding TUI.

Triggered from `cli.py` when neither the `fast` nor `strong` slot resolves
(both bound providers have empty api_key). Lets the user paste an api_key
inline, verify a model with one keystroke, and assign verified models to
the fast/strong slots — all without leaving the terminal.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import ClassVar

from loguru import logger
from openai import AsyncOpenAI
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.timer import Timer
from textual.widgets import Footer, Header, Input, Static

from neutrix import __version__
from neutrix.config import (
    PROVIDER_DEFAULT_MODELS,
    Config,
    save_config,
)


@dataclass
class _ProviderState:
    name: str
    base_url: str
    api_key: str
    models: list[str]
    statuses: dict[str, str] = field(default_factory=dict)  # model -> '?' | '✓' | '✗' | '…'


async def verify_model(base_url: str, api_key: str, model: str) -> bool:
    """Send a 1-token request to confirm (base_url, api_key, model) work."""
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    try:
        await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            ),
            timeout=8.0,
        )
        return True
    except Exception as e:
        logger.warning("verify {} failed: {}", model, e)
        return False


class ModelRow(Static):
    """Focusable row showing one model's status and slot assignments."""

    can_focus = True

    def __init__(self, provider: str, model: str) -> None:
        super().__init__(classes="model-row")
        self.provider = provider
        self.model = model
        self.status = "?"
        self.slot_tags: list[str] = []
        self._refresh()

    def _refresh(self) -> None:
        tags = f"  [b yellow]{'/'.join(self.slot_tags)}[/b yellow]" if self.slot_tags else ""
        self.update(f"    [{self.status}] {self.model}{tags}")


class ProviderSection(Vertical):
    """Header + base_url + api_key Input + list of ModelRow per provider."""

    def __init__(self, state: _ProviderState) -> None:
        super().__init__(classes="provider")
        self.state = state

    def compose(self) -> ComposeResult:
        yield Static(f"[b cyan]{self.state.name}[/b cyan]")
        yield Static(f"  base_url: [dim]{self.state.base_url}[/dim]")
        yield Input(
            value=self.state.api_key,
            placeholder="EMPTY  —  paste api_key here and press Enter",
            password=True,
            id=f"key-{self.state.name}",
            classes="api-key",
        )
        yield Static("  models:")
        for m in self.state.models:
            yield ModelRow(self.state.name, m)


class OnboardApp(App[bool]):
    """Returns True if the user saved+launched, False if they quit."""

    CSS = """
    Screen { layout: vertical; }
    #intro {
        padding: 0 1;
        color: $text;
    }
    #scroll {
        height: 1fr;
        padding: 1;
        border: round $primary;
    }
    ProviderSection {
        margin: 0 0 1 0;
        height: auto;
    }
    .api-key {
        margin: 0 0 0 2;
        width: 80;
    }
    .model-row {
        padding: 0;
    }
    ModelRow:focus {
        background: $accent 30%;
        color: $text;
    }
    #status { dock: bottom; height: 1; background: $boost; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("v", "verify", "Verify"),
        Binding("f", "set_fast", "Set fast"),
        Binding("g", "set_strong", "Set strong"),
        Binding("s", "save_and_launch", "Save & launch"),
        Binding("q", "quit_onboard", "Quit"),
        Binding("up", "focus_previous", "Up", show=False, priority=True),
        Binding("down", "focus_next", "Down", show=False, priority=True),
        Binding("ctrl+c", "confirm_quit", "Quit (Ctrl+C x2)", priority=True),
        Binding("escape", "cancel_quit", "Cancel quit", show=False, priority=True),
    ]

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.provider_state: dict[str, _ProviderState] = {
            name: _ProviderState(
                name=name,
                base_url=(prov or {}).get("base_url", ""),
                api_key=(prov or {}).get("api_key", ""),
                models=list(PROVIDER_DEFAULT_MODELS.get(name, [])),
            )
            for name, prov in config.providers.items()
        }
        self.fast_choice: dict[str, str] | None = None
        self.strong_choice: dict[str, str] | None = None
        self._quit_pending: bool = False
        self._quit_timer: Timer | None = None

    # ----- compose ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(
            "Welcome to neutrix. Paste at least one api_key (Enter to save),"
            " focus a model row and press [b]v[/b] to verify,"
            " then [b]f[/b]/[b]g[/b] to assign fast/strong,"
            " then [b]s[/b] to launch the chat.",
            id="intro",
        )
        with VerticalScroll(id="scroll"):
            for state in self.provider_state.values():
                yield ProviderSection(state)
        yield Static(self._status_text(), id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"neutrix v{__version__} · onboarding"
        self.sub_title = str(self.config.path)
        rows = list(self.query(ModelRow))
        if rows:
            rows[0].focus()

    # ----- status -------------------------------------------------------------

    def _status_text(self) -> str:
        fast = f"{self.fast_choice['provider']}/{self.fast_choice['model']}" if self.fast_choice else "—"
        strong = f"{self.strong_choice['provider']}/{self.strong_choice['model']}" if self.strong_choice else "—"
        return f" fast: {fast}  ·  strong: {strong} "

    def _refresh_status(self) -> None:
        self.query_one("#status", Static).update(self._status_text())

    # ----- inline api_key persistence ----------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        provider = event.input.id.removeprefix("key-") if event.input.id else ""
        if provider not in self.provider_state:
            return
        new_key = event.value.strip()
        self.provider_state[provider].api_key = new_key
        # reset verification statuses for this provider — the key changed
        self.provider_state[provider].statuses.clear()
        for row in self.query(ModelRow):
            if row.provider == provider:
                row.status = "?"
                row._refresh()
        self._persist_yaml(fast=self.fast_choice, strong=self.strong_choice)
        self.notify(
            f"saved {provider} api_key to {self.config.path}",
            severity="information",
        )

    def _persist_yaml(
        self,
        *,
        fast: dict[str, str] | None,
        strong: dict[str, str] | None,
    ) -> None:
        cfg = Config(
            providers={
                name: {"base_url": s.base_url, "api_key": s.api_key}
                for name, s in self.provider_state.items()
            },
            slots=self.config.slots,
            path=self.config.path,
        )
        save_config(
            cfg,
            fast=fast if fast is not None else cfg.slots.get("fast"),
            strong=strong if strong is not None else cfg.slots.get("strong"),
        )

    # ----- actions ------------------------------------------------------------

    def _focused_row(self) -> ModelRow | None:
        return self.focused if isinstance(self.focused, ModelRow) else None

    async def action_verify(self) -> None:
        row = self._focused_row()
        if row is None:
            self.notify("focus a model row first (Tab to navigate)", severity="warning")
            return
        state = self.provider_state[row.provider]
        if not state.api_key:
            self.notify(f"set {row.provider}'s api_key first", severity="warning")
            return
        row.status = "…"
        row._refresh()
        ok = await verify_model(state.base_url, state.api_key, row.model)
        row.status = "✓" if ok else "✗"
        state.statuses[row.model] = row.status
        row._refresh()

    def action_set_fast(self) -> None:
        self._assign("fast")

    def action_set_strong(self) -> None:
        self._assign("strong")

    def _assign(self, slot_name: str) -> None:
        row = self._focused_row()
        if row is None or row.status != "✓":
            self.notify("focus a verified (✓) model first", severity="warning")
            return
        choice = {"provider": row.provider, "model": row.model}
        if slot_name == "fast":
            self.fast_choice = choice
        else:
            self.strong_choice = choice
        self._refresh_slot_tags()
        self._refresh_status()

    def _refresh_slot_tags(self) -> None:
        for row in self.query(ModelRow):
            tags: list[str] = []
            if (
                self.fast_choice
                and self.fast_choice["provider"] == row.provider
                and self.fast_choice["model"] == row.model
            ):
                tags.append("fast")
            if (
                self.strong_choice
                and self.strong_choice["provider"] == row.provider
                and self.strong_choice["model"] == row.model
            ):
                tags.append("strong")
            row.slot_tags = tags
            row._refresh()

    def action_save_and_launch(self) -> None:
        if self.fast_choice is None and self.strong_choice is None:
            self.notify(
                "assign at least one slot first (f or g on a verified ✓ model)",
                severity="error",
            )
            return
        fast = self.fast_choice or self.strong_choice
        strong = self.strong_choice or self.fast_choice
        self._persist_yaml(fast=fast, strong=strong)
        self.exit(True)

    def action_quit_onboard(self) -> None:
        self.exit(False)

    # ----- two-tap Ctrl+C quit ------------------------------------------------

    def action_confirm_quit(self) -> None:
        if self._quit_pending:
            self.exit(False)
            return
        self._quit_pending = True
        self.notify(
            "press Ctrl+C again to quit, Esc to cancel",
            severity="warning",
            timeout=5,
        )
        if self._quit_timer is not None:
            self._quit_timer.stop()
        self._quit_timer = self.set_timer(5.0, self._reset_quit_pending)

    def action_cancel_quit(self) -> None:
        if self._quit_pending:
            self._reset_quit_pending()
            self.notify("quit cancelled")

    def _reset_quit_pending(self) -> None:
        self._quit_pending = False
        if self._quit_timer is not None:
            self._quit_timer.stop()
            self._quit_timer = None


def run_onboarding(config: Config) -> bool:
    """Launch the onboarding TUI. Returns True if user saved + launched."""
    app = OnboardApp(config)
    result = app.run()
    return bool(result)
