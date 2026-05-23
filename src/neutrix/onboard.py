"""Onboarding TUI — manage api keys, verify models, bind fast/strong slots.

Two entry points share the same `OnboardScreen`:

- **First-run** (`cli.py`): `run_onboarding(config)` boots `OnboardApp`,
  which pushes `OnboardScreen` as its initial screen. Used when neither
  the `fast` nor the `strong` slot resolves.
- **Mid-chat** (`tui.py`): `/onboard` slash command pushes
  `OnboardScreen` directly onto the chat App's screen stack.

`OnboardScreen.dismiss(True)` means "saved and ready to use the YAML";
`dismiss(False)` means the user backed out.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import ClassVar

from loguru import logger
from openai import AsyncOpenAI
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import Footer, Header, Input, Static

from neutrix import __version__
from neutrix.config import (
    PROVIDER_DEFAULT_MODELS,
    Config,
    save_config,
)

# ----- status labels --------------------------------------------------------

UNKNOWN = "unknown"
VERIFIED = "verified"
FAILED = "failed"
VERIFYING = "verifying"

_STATUS_ICON = {
    UNKNOWN: "[dim]○[/dim]",
    VERIFIED: "[b $success]✓[/b $success]",
    FAILED: "[b $error]✗[/b $error]",
    VERIFYING: "[b $warning]…[/b $warning]",
}

_STATUS_LABEL = {
    UNKNOWN: "[dim]unknown[/dim]",
    VERIFIED: "[$success]verified[/$success]",
    FAILED: "[$error]failed[/$error]",
    VERIFYING: "[$warning]verifying…[/$warning]",
}


@dataclass
class _ProviderState:
    name: str
    base_url: str
    api_key: str
    models: list[str]
    model_status: dict[str, str] = field(default_factory=dict)
    key_saved: bool = False


# ----- verification ---------------------------------------------------------


async def _check_one(client: AsyncOpenAI, model: str) -> bool:
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


async def verify_model(base_url: str, api_key: str, model: str) -> bool:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    return await _check_one(client, model)


async def verify_models(
    base_url: str, api_key: str, models: list[str]
) -> dict[str, bool]:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    results = await asyncio.gather(*(_check_one(client, m) for m in models))
    return dict(zip(models, results, strict=True))


# ----- shared focus nav helper ---------------------------------------------


def _navigate_focus(widget: Static | Input, event: events.Key) -> bool:
    """Translate up/down into screen focus navigation. Returns True if handled."""
    if event.key == "down":
        widget.screen.focus_next()
        event.stop()
        return True
    if event.key == "up":
        widget.screen.focus_previous()
        event.stop()
        return True
    return False


# ----- widgets --------------------------------------------------------------


class FocusScroll(VerticalScroll):
    """VerticalScroll that doesn't grab focus, so arrows never get trapped
    in scroll mode. Textual still auto-scrolls to keep the focused child
    visible, and mouse wheel / PgUp / PgDn still scroll."""

    can_focus = False


class ModelRow(Static):
    """Focusable row: status icon · model name · status label · slot tags."""

    can_focus = True

    def __init__(self, provider: str, model: str, status: str = UNKNOWN) -> None:
        super().__init__(classes="model-row")
        self.provider = provider
        self.model = model
        self.status = status
        self.slot_tags: list[str] = []
        self._refresh()

    def _refresh(self) -> None:
        icon = _STATUS_ICON[self.status]
        label = _STATUS_LABEL[self.status]
        tags = (
            "  " + " ".join(f"[b $accent]▸ {t}[/b $accent]" for t in self.slot_tags)
            if self.slot_tags
            else ""
        )
        # model name padded so labels line up roughly
        self.update(f" {icon}  {self.model:<42} {label}{tags}")

    def on_key(self, event: events.Key) -> None:
        _navigate_focus(self, event)


class VerifyAllRow(Static):
    """Focusable special row: press `v` to verify every model in the provider
    in parallel."""

    can_focus = True

    def __init__(self, provider: str) -> None:
        super().__init__(classes="model-row all-row")
        self.provider = provider
        self.update(
            " [b $accent]▶[/b $accent]  [b](all)[/b]                                  "
            "[dim]press v to verify all in parallel[/dim]"
        )

    def on_key(self, event: events.Key) -> None:
        _navigate_focus(self, event)


class KeyInput(Input):
    """Single-line Input with explicit edit semantics:

    - Focus clears the visible buffer (committed value preserved internally).
    - Tab / Up / Down without Enter restores the committed value on blur.
    - Enter with an empty buffer is treated as "no change" by the screen.
    - Enter with a non-empty buffer is treated as the new committed value.
    - After blur, the visible field always returns to the committed masked value.

    `_committed_value` is the source of truth for the saved key; the
    visible `value` is just the editing buffer. We commit the baseline
    inside `action_submit` *before* posting the Submitted message so
    any blur that races ahead of `on_input_submitted` reads an
    already-up-to-date `_committed_value` and doesn't revert.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._committed_value: str = self.value

    @property
    def _value(self) -> Text:
        """Render password content as the conventional `****` mask."""
        if self.password:
            return Text("*" * len(self.value), no_wrap=True, overflow="ignore", end="")
        return super()._value

    async def action_submit(self) -> None:  # type: ignore[override]
        buf = self.value.strip()
        if buf:
            # Promote typed buffer to committed baseline *before* the
            # Submitted message is posted, so the inevitable blur from
            # the screen calling focus_next() can't revert against a
            # stale baseline.
            self._committed_value = buf
            if self.value != buf:
                self.value = buf
        # Post the standard Submitted message (mirrors Input.action_submit).
        self.post_message(self.Submitted(self, self.value, None))

    def on_focus(self, event: events.Focus) -> None:
        # Fresh editing buffer; committed value restored on blur / empty Enter.
        if self.value:
            self.value = ""
        self.cursor_position = 0
        self.scroll_home(
            animate=False,
            force=True,
            immediate=True,
            y_axis=False,
        )

    def restore_committed_display(self, *, force: bool = False) -> None:
        """Show the committed key whenever this field is no longer editing."""
        if self.has_focus and not force:
            return
        if self.value != self._committed_value:
            self.value = self._committed_value
        self.cursor_position = len(self.value)
        self.scroll_home(
            animate=False,
            force=True,
            immediate=True,
            y_axis=False,
        )
        self.refresh()

    def on_blur(self, event: events.Blur) -> None:
        self.restore_committed_display(force=True)

    def on_key(self, event: events.Key) -> None:
        _navigate_focus(self, event)


class ProviderSection(Vertical):
    """One bordered card per provider: title, base_url, api_key row, models."""

    def __init__(self, state: _ProviderState) -> None:
        super().__init__(classes="provider-card")
        self.state = state
        self.border_title = state.name

    def compose(self) -> ComposeResult:
        yield Static(
            f"[dim]{self.state.base_url}[/dim]",
            classes="base-url",
        )
        with Horizontal(classes="key-row"):
            yield Static("[b]api key[/b]", classes="key-label")
            yield KeyInput(
                value=self.state.api_key,
                placeholder="EMPTY — paste api_key and press Enter",
                password=True,
                id=f"key-{self.state.name}",
                classes="api-key",
            )
            yield Static(
                "[$success]saved[/$success]" if self.state.api_key else "",
                id=f"saved-{self.state.name}",
                classes="saved-tag",
            )
        yield Static("[b]Models[/b]", classes="models-label")
        yield VerifyAllRow(self.state.name)
        for m in self.state.models:
            status = self.state.model_status.get(m, UNKNOWN)
            yield ModelRow(self.state.name, m, status=status)


# ----- screen ---------------------------------------------------------------


class OnboardScreen(Screen[bool]):
    """Onboarding screen. dismiss(True) = saved; dismiss(False) = cancelled."""

    CSS = """
    OnboardScreen { layout: vertical; background: $background; }

    #intro {
        padding: 1 2 0 2;
        color: $text-muted;
    }

    #scroll {
        height: 1fr;
        padding: 1 2;
    }

    ProviderSection {
        height: auto;
        margin: 0 0 1 0;
        padding: 1 2;
        border: round $primary;
        background: $boost;
    }
    ProviderSection > .base-url {
        margin: 0 0 1 0;
    }
    ProviderSection > .key-row {
        height: 1;
        margin: 0 0 1 0;
    }
    ProviderSection > .key-row > .key-label {
        width: 9;
        content-align: left middle;
    }
    ProviderSection > .key-row > .api-key {
        border: none;
        background: $surface;
        padding: 0 1;
        height: 1;
        min-height: 1;
        width: 1fr;
        margin: 0 1;
    }
    ProviderSection > .key-row > .api-key:focus {
        background: $accent 30%;
    }
    ProviderSection > .key-row > .saved-tag {
        width: auto;
        content-align: left middle;
    }
    ProviderSection > .models-label {
        margin: 0 0 0 0;
        color: $text-muted;
    }
    .model-row {
        padding: 0 0 0 1;
        height: 1;
    }
    .model-row:focus {
        background: $accent 25%;
    }

    #message {
        dock: bottom;
        height: 1;
        padding: 0 2;
        background: $boost;
    }
    #status {
        dock: bottom;
        height: 1;
        background: $boost;
        padding: 0 2;
        color: $text-muted;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("v", "verify", "Verify"),
        Binding("f", "set_fast", "Set fast"),
        Binding("g", "set_strong", "Set strong"),
        Binding("q", "quit_onboard", "Done"),
        Binding("ctrl+c", "confirm_quit", "Exit app (Ctrl+C x2)", priority=True),
        Binding("escape", "cancel_quit", "Cancel quit", show=False, priority=True),
    ]

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.provider_state: dict[str, _ProviderState] = {}
        for name, prov in config.providers.items():
            prov = prov or {}
            self.provider_state[name] = _ProviderState(
                name=name,
                base_url=prov.get("base_url", ""),
                api_key=prov.get("api_key", ""),
                models=list(PROVIDER_DEFAULT_MODELS.get(name, [])),
                model_status=dict(prov.get("model_status") or {}),
                key_saved=bool(prov.get("api_key")),
            )
        self.fast_choice: dict[str, str] | None = self._slot_choice("fast")
        self.strong_choice: dict[str, str] | None = self._slot_choice("strong")
        self._quit_pending: bool = False
        self._quit_timer: Timer | None = None
        self._msg_timer: Timer | None = None

    def _slot_choice(self, name: str) -> dict[str, str] | None:
        spec = self.config.slots.get(name) or {}
        p, m = spec.get("provider"), spec.get("model")
        if p and m:
            return {"provider": p, "model": m}
        return None

    # ----- compose ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(
            "[b]Onboarding.[/b]  Paste an api key, [b]Enter[/b] saves it."
            "  Focus a model row, [b]v[/b] verifies (or [b]v[/b] on [b](all)[/b] for parallel)."
            "  [b]f[/b]/[b]g[/b] assign fast/strong. Everything auto-saves; [b]q[/b] when done.",
            id="intro",
        )
        with FocusScroll(id="scroll"):
            for state in self.provider_state.values():
                yield ProviderSection(state)
        yield Static("", id="message")
        yield Static(self._status_text(), id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = str(self.config.path)
        self._refresh_slot_tags()
        self._refresh_status()
        first = self.query(ModelRow).first()
        if first is not None:
            first.focus()

    # ----- status -------------------------------------------------------------

    def _status_text(self) -> str:
        fast = (
            f"{self.fast_choice['provider']}/{self.fast_choice['model']}"
            if self.fast_choice
            else "—"
        )
        strong = (
            f"{self.strong_choice['provider']}/{self.strong_choice['model']}"
            if self.strong_choice
            else "—"
        )
        return f" fast: {fast}     strong: {strong} "

    def _refresh_status(self) -> None:
        self.query_one("#status", Static).update(self._status_text())

    def _set_saved_indicator(self, provider: str, on: bool) -> None:
        try:
            self.query_one(f"#saved-{provider}", Static).update(
                "[$success]saved[/$success]" if on else ""
            )
        except Exception:
            pass

    # ----- inline message bar -------------------------------------------------

    _MSG_COLORS: ClassVar[dict[str, str]] = {
        "info": "$text-muted",
        "warning": "$warning",
        "error": "$error",
        "success": "$success",
    }

    def _notify(self, text: str, severity: str = "info", *, persistent: bool = False) -> None:
        color = self._MSG_COLORS.get(severity, "$text-muted")
        try:
            bar = self.query_one("#message", Static)
        except Exception:
            return
        bar.update(f"[{color}]{text}[/{color}]")
        if self._msg_timer is not None:
            self._msg_timer.stop()
            self._msg_timer = None
        if not persistent:
            self._msg_timer = self.set_timer(4.0, self._clear_message)

    def _clear_message(self) -> None:
        try:
            self.query_one("#message", Static).update("")
        except Exception:
            pass
        self._msg_timer = None

    # Suppress Textual's floating toasts on this screen — route any caller
    # (framework, workers, future code) through the inline bar instead.
    def notify(  # type: ignore[override]
        self,
        message,
        *,
        title: str = "",
        severity: str = "information",
        timeout=None,
        markup: bool = True,
    ) -> None:
        sev_map = {
            "information": "info",
            "warning": "warning",
            "error": "error",
        }
        self._notify(str(message), sev_map.get(severity, "info"))

    # ----- inline api_key persistence ----------------------------------------

    def _restore_key_display_after_focus(self, input_widget: Input) -> None:
        if not isinstance(input_widget, KeyInput):
            return
        input_widget.restore_committed_display(force=True)
        self.call_after_refresh(input_widget.restore_committed_display, force=True)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        provider = event.input.id.removeprefix("key-") if event.input.id else ""
        if provider not in self.provider_state:
            return
        new_key = event.value.strip()

        # Empty Enter = "no change". Restore the visible buffer to the
        # committed value, advance focus, leave state and YAML alone.
        if not new_key:
            if isinstance(event.input, KeyInput):
                event.input.value = event.input._committed_value
            self.focus_next()
            self._restore_key_display_after_focus(event.input)
            return

        # The Input itself promoted _committed_value during action_submit
        # (so an early blur doesn't revert). We just ensure value matches.
        event.input.value = new_key

        state = self.provider_state[provider]
        # If key changed, drop stale verification statuses.
        if new_key != state.api_key:
            state.model_status.clear()
            for row in self.query(ModelRow):
                if row.provider == provider:
                    row.status = UNKNOWN
                    row._refresh()
        state.api_key = new_key
        state.key_saved = True
        self._set_saved_indicator(provider, True)
        self._persist_yaml(fast=self.fast_choice, strong=self.strong_choice)
        self._notify(f"saved {provider} api_key", "success")
        # Advance focus so the saved indicator is visible and the user
        # is unblocked. _committed_value is already current (set by
        # action_submit), so the outgoing blur on the Input won't revert.
        self.focus_next()
        self._restore_key_display_after_focus(event.input)

    def _persist_yaml(
        self,
        *,
        fast: dict[str, str] | None,
        strong: dict[str, str] | None,
    ) -> None:
        cfg = Config(
            providers={
                name: {
                    "base_url": s.base_url,
                    "api_key": s.api_key,
                    "model_status": dict(s.model_status),
                }
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

    def _focused_all(self) -> VerifyAllRow | None:
        return self.focused if isinstance(self.focused, VerifyAllRow) else None

    def action_verify(self) -> None:
        all_row = self._focused_all()
        if all_row is not None:
            self.run_worker(self._verify_all(all_row.provider), exclusive=False)
            return
        row = self._focused_row()
        if row is None:
            self._notify("focus a model row or the (all) row first", "warning")
            return
        state = self.provider_state[row.provider]
        if not state.api_key:
            self._notify(f"set {row.provider}'s api_key first", "warning")
            return
        self.run_worker(self._verify_one(row), exclusive=False)

    async def _verify_one(self, row: ModelRow) -> None:
        state = self.provider_state[row.provider]
        row.status = VERIFYING
        row._refresh()
        ok = await verify_model(state.base_url, state.api_key, row.model)
        row.status = VERIFIED if ok else FAILED
        state.model_status[row.model] = row.status
        row._refresh()
        self._persist_yaml(fast=self.fast_choice, strong=self.strong_choice)
        self._notify(
            f"{row.provider}/{row.model}: {row.status}",
            "success" if ok else "error",
        )
        self._maybe_auto_assign()

    async def _verify_all(self, provider: str) -> None:
        state = self.provider_state[provider]
        if not state.api_key:
            self._notify(f"set {provider}'s api_key first", "warning")
            return
        rows = [r for r in self.query(ModelRow) if r.provider == provider]
        if not rows:
            return
        for r in rows:
            r.status = VERIFYING
            r._refresh()
        self._notify(f"verifying {len(rows)} models for {provider}…", "info")
        results = await verify_models(
            state.base_url, state.api_key, [r.model for r in rows]
        )
        passed = 0
        for r in rows:
            ok = results[r.model]
            r.status = VERIFIED if ok else FAILED
            state.model_status[r.model] = r.status
            r._refresh()
            if ok:
                passed += 1
        self._persist_yaml(fast=self.fast_choice, strong=self.strong_choice)
        self._notify(
            f"{provider}: {passed}/{len(rows)} verified",
            "success" if passed == len(rows) else "warning",
        )
        self._maybe_auto_assign()

    def action_set_fast(self) -> None:
        self._assign("fast")

    def action_set_strong(self) -> None:
        self._assign("strong")

    def _assign(self, slot_name: str) -> None:
        row = self._focused_row()
        if row is None or row.status != VERIFIED:
            self._notify("focus a verified (✓) model first", "warning")
            return
        choice = {"provider": row.provider, "model": row.model}
        if slot_name == "fast":
            self.fast_choice = choice
        else:
            self.strong_choice = choice
        self._refresh_slot_tags()
        self._refresh_status()
        self._persist_yaml(fast=self.fast_choice, strong=self.strong_choice)
        self._notify(
            f"{slot_name} → {row.provider}/{row.model}", "success"
        )

    def _maybe_auto_assign(self) -> None:
        """If a slot is unset, bind it to the first verified model. Never
        overwrites a user choice."""
        verified = [
            {"provider": r.provider, "model": r.model}
            for r in self.query(ModelRow)
            if r.status == VERIFIED
        ]
        if not verified:
            return
        changed = False
        if self.fast_choice is None:
            self.fast_choice = verified[0]
            changed = True
        if self.strong_choice is None:
            self.strong_choice = next(
                (v for v in verified if v != self.fast_choice),
                self.fast_choice,
            )
            changed = True
        if changed:
            self._refresh_slot_tags()
            self._refresh_status()
            self._persist_yaml(fast=self.fast_choice, strong=self.strong_choice)
            self._notify(
                f"auto-assigned fast → {self.fast_choice['model']}, "
                f"strong → {self.strong_choice['model']}",
                "info",
            )

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

    def action_quit_onboard(self) -> None:
        # Everything is auto-saved; "done" just returns to whatever is below.
        self.dismiss(True)

    # ----- two-tap Ctrl+C quit ------------------------------------------------

    def action_confirm_quit(self) -> None:
        if self._quit_pending:
            # Hard exit — terminates the whole app, including the chat
            # when onboarding was reached via /onboard.
            self.app.exit()
            return
        self._quit_pending = True
        self._notify(
            "press Ctrl+C again to exit the app, Esc to cancel",
            "warning",
            persistent=True,
        )
        if self._quit_timer is not None:
            self._quit_timer.stop()
        self._quit_timer = self.set_timer(5.0, self._reset_quit_pending)

    def action_cancel_quit(self) -> None:
        if self._quit_pending:
            self._reset_quit_pending()
            self._notify("quit cancelled", "info")

    def _reset_quit_pending(self) -> None:
        self._quit_pending = False
        if self._quit_timer is not None:
            self._quit_timer.stop()
            self._quit_timer = None
        self._clear_message()


class OnboardApp(App[bool]):
    """Standalone wrapper used on first-run from `cli.py`."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config

    def on_mount(self) -> None:
        self.title = f"neutrix v{__version__} · onboarding"
        self.push_screen(OnboardScreen(self._config), callback=self._on_done)

    def _on_done(self, result: bool | None) -> None:
        self.exit(bool(result))


def run_onboarding(config: Config) -> bool:
    """Launch the onboarding TUI. Returns True if user saved + launched."""
    app = OnboardApp(config)
    result = app.run()
    return bool(result)
