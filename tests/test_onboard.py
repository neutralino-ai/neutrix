"""Headless tests for the onboarding TUI — covers every user-facing behavior."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from neutrix.config import bootstrap_config, load_config
from neutrix.onboard import (
    FAILED,
    UNKNOWN,
    VERIFIED,
    KeyInput,
    ModelRow,
    OnboardApp,
    OnboardScreen,
    VerifyAllRow,
)


# ----- fixtures --------------------------------------------------------------


@pytest.fixture
def cfg_path(tmp_path: Path) -> Path:
    """A bootstrapped config with empty api_keys (default template)."""
    p = tmp_path / "config.yaml"
    bootstrap_config(p)
    return p


@pytest.fixture
def cfg_with_key(tmp_path: Path) -> Path:
    """A bootstrapped config with ihep api_key pre-filled."""
    p = tmp_path / "config.yaml"
    bootstrap_config(p)
    p.write_text(p.read_text().replace('api_key: ""', "api_key: sk-test", 1))
    return p


@pytest.fixture
def cfg_unset_slots(tmp_path: Path) -> Path:
    """A config with ihep key set but both fast/strong empty {} — auto-assign target."""
    p = tmp_path / "config.yaml"
    p.write_text(
        """providers:
  ihep:
    base_url: https://aiapi.ihep.ac.cn/apiv2/
    api_key: sk-test
fast: {}
strong: {}
"""
    )
    return p


async def _ok(*a, **kw):
    return True


async def _ok_all(base_url, api_key, models):
    return dict.fromkeys(models, True)


async def _fail(*a, **kw):
    return False


# ----- compose ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_compose_has_one_section_per_provider(cfg_path: Path):
    cfg = load_config(cfg_path)
    app = OnboardApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause()
        scr = app.screen
        from neutrix.onboard import ProviderSection
        assert len(list(scr.query(ProviderSection))) == 3
        assert len(list(scr.query(VerifyAllRow))) == 3
        assert len(list(scr.query(KeyInput))) == 3
        # plus one ModelRow per model in PROVIDER_DEFAULT_MODELS
        from neutrix.config import PROVIDER_DEFAULT_MODELS
        expected = sum(len(v) for v in PROVIDER_DEFAULT_MODELS.values())
        assert len(list(scr.query(ModelRow))) == expected


@pytest.mark.asyncio
async def test_focus_chain_excludes_vertical_scroll(cfg_path: Path):
    """Issue: focus must never land on VerticalScroll (it eats arrows)."""
    cfg = load_config(cfg_path)
    app = OnboardApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause()
        kinds = {type(w).__name__ for w in app.screen.focus_chain}
        assert "VerticalScroll" not in kinds
        assert "FocusScroll" not in kinds
        # only our focusables
        assert kinds <= {"ModelRow", "KeyInput", "VerifyAllRow"}


# ----- api_key Input ---------------------------------------------------------


@pytest.mark.asyncio
async def test_api_key_enter_persists_and_advances_focus(cfg_path: Path):
    cfg = load_config(cfg_path)
    app = OnboardApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause()
        inp = app.screen.query_one("#key-ihep", KeyInput)
        inp.focus()
        await pilot.pause()
        inp.value = "sk-new"
        await pilot.press("enter")
        await pilot.pause()
        # value still on the Input
        assert inp.value == "sk-new"
        # committed baseline updated
        assert inp._committed_value == "sk-new"
        # focus advanced to the next focusable
        assert app.screen.focused is not inp
        # YAML written
        reloaded = load_config(cfg_path)
        assert reloaded.providers["ihep"]["api_key"] == "sk-new"


@pytest.mark.asyncio
async def test_api_key_blur_without_enter_reverts(cfg_path: Path):
    cfg = load_config(cfg_path)
    app = OnboardApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause()
        inp = app.screen.query_one("#key-ihep", KeyInput)
        inp.focus()
        await pilot.pause()
        original = inp._committed_value  # empty by default
        inp.value = "partial-typing"
        # Tab away without Enter
        await pilot.press("tab")
        await pilot.pause()
        assert inp.value == original  # reverted


@pytest.mark.asyncio
async def test_api_key_change_clears_model_status(cfg_with_key: Path):
    """When the user replaces an api_key, any prior verifications go stale."""
    cfg = load_config(cfg_with_key)
    # seed a verified status
    cfg.providers["ihep"]["model_status"] = {
        "anthropic/claude-haiku-4-5": "verified",
    }
    from neutrix.config import save_config
    save_config(cfg, path=cfg_with_key)

    cfg2 = load_config(cfg_with_key)
    app = OnboardApp(cfg2)
    async with app.run_test() as pilot:
        await pilot.pause()
        # confirm the row loaded as verified
        row = next(r for r in app.screen.query(ModelRow)
                   if r.model == "anthropic/claude-haiku-4-5")
        assert row.status == VERIFIED
        # replace the key
        inp = app.screen.query_one("#key-ihep", KeyInput)
        inp.focus()
        await pilot.pause()
        inp.value = "sk-different"
        await pilot.press("enter")
        await pilot.pause()
        # row reverted to UNKNOWN
        assert row.status == UNKNOWN
        # YAML model_status for ihep got cleared
        reloaded = load_config(cfg_with_key)
        ms = reloaded.providers["ihep"].get("model_status") or {}
        assert not ms


# ----- verify ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_persists_model_status(cfg_with_key: Path):
    cfg = load_config(cfg_with_key)
    with patch("neutrix.onboard.verify_model", side_effect=_ok):
        app = OnboardApp(cfg)
        async with app.run_test() as pilot:
            await pilot.pause()
            row = next(r for r in app.screen.query(ModelRow) if r.provider == "ihep")
            row.focus()
            await pilot.pause()
            await pilot.press("v")
            # let the background worker run
            await asyncio.sleep(0.1)
            await pilot.pause()
            assert row.status == VERIFIED
            reloaded = load_config(cfg_with_key)
            ms = reloaded.providers["ihep"].get("model_status") or {}
            assert ms.get(row.model) == "verified"


@pytest.mark.asyncio
async def test_verify_failed_marks_row_failed(cfg_with_key: Path):
    cfg = load_config(cfg_with_key)
    with patch("neutrix.onboard.verify_model", side_effect=_fail):
        app = OnboardApp(cfg)
        async with app.run_test() as pilot:
            await pilot.pause()
            row = next(r for r in app.screen.query(ModelRow) if r.provider == "ihep")
            row.focus()
            await pilot.pause()
            await pilot.press("v")
            await asyncio.sleep(0.1)
            await pilot.pause()
            assert row.status == FAILED


@pytest.mark.asyncio
async def test_verify_is_non_blocking(cfg_with_key: Path):
    """Focus must be movable while a verify is in flight."""
    cfg = load_config(cfg_with_key)

    async def slow(*a, **kw):
        await asyncio.sleep(0.5)
        return True

    with patch("neutrix.onboard.verify_model", side_effect=slow):
        app = OnboardApp(cfg)
        async with app.run_test() as pilot:
            await pilot.pause()
            row = next(r for r in app.screen.query(ModelRow) if r.provider == "ihep")
            row.focus()
            await pilot.pause()
            await pilot.press("v")
            await pilot.pause()  # worker started, sleeping
            before = app.screen.focused
            await pilot.press("down")
            await pilot.pause()
            assert app.screen.focused is not before  # focus moved while verifying


@pytest.mark.asyncio
async def test_verify_all_runs_in_parallel(cfg_with_key: Path):
    cfg = load_config(cfg_with_key)
    with patch("neutrix.onboard.verify_models", side_effect=_ok_all):
        app = OnboardApp(cfg)
        async with app.run_test() as pilot:
            await pilot.pause()
            all_row = next(r for r in app.screen.query(VerifyAllRow)
                           if r.provider == "ihep")
            all_row.focus()
            await pilot.pause()
            await pilot.press("v")
            await asyncio.sleep(0.1)
            await pilot.pause()
            ihep_rows = [r for r in app.screen.query(ModelRow) if r.provider == "ihep"]
            assert all(r.status == VERIFIED for r in ihep_rows)


@pytest.mark.asyncio
async def test_model_status_loaded_from_yaml(cfg_with_key: Path):
    """Reopening shows previously verified/failed statuses."""
    cfg = load_config(cfg_with_key)
    cfg.providers["ihep"]["model_status"] = {
        "anthropic/claude-haiku-4-5": "verified",
        "anthropic/claude-opus-4-7": "failed",
    }
    from neutrix.config import save_config
    save_config(cfg, path=cfg_with_key)

    cfg2 = load_config(cfg_with_key)
    app = OnboardApp(cfg2)
    async with app.run_test() as pilot:
        await pilot.pause()
        rows = {r.model: r for r in app.screen.query(ModelRow)
                if r.provider == "ihep"}
        assert rows["anthropic/claude-haiku-4-5"].status == VERIFIED
        assert rows["anthropic/claude-opus-4-7"].status == FAILED
        assert rows["anthropic/claude-sonnet-4-6"].status == UNKNOWN


# ----- slot assignment + auto-assign ----------------------------------------


@pytest.mark.asyncio
async def test_f_assigns_fast_and_persists(cfg_with_key: Path):
    cfg = load_config(cfg_with_key)
    with patch("neutrix.onboard.verify_model", side_effect=_ok):
        app = OnboardApp(cfg)
        async with app.run_test() as pilot:
            await pilot.pause()
            row = next(r for r in app.screen.query(ModelRow) if r.provider == "ihep")
            row.focus()
            await pilot.press("v")
            await asyncio.sleep(0.1)
            await pilot.pause()
            await pilot.press("f")
            await pilot.pause()
            assert "fast" in row.slot_tags
            reloaded = load_config(cfg_with_key)
            assert reloaded.slots["fast"]["model"] == row.model


@pytest.mark.asyncio
async def test_slot_tags_render_on_reopen(cfg_with_key: Path):
    """Issue 3: existing fast/strong in YAML should show as tags."""
    cfg = load_config(cfg_with_key)
    # cfg_with_key uses bootstrap defaults which already include fast/strong
    assert cfg.slots["fast"]["provider"] == "ihep"
    app = OnboardApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause()
        tagged = {r.model: r.slot_tags for r in app.screen.query(ModelRow) if r.slot_tags}
        assert tagged.get("anthropic/claude-haiku-4-5") == ["fast"]
        assert tagged.get("anthropic/claude-opus-4-7") == ["strong"]


@pytest.mark.asyncio
async def test_auto_assign_fast_and_strong_when_unset(cfg_unset_slots: Path):
    """Issue 4: verifying a model when slots are {} auto-binds both slots."""
    cfg = load_config(cfg_unset_slots)
    with patch("neutrix.onboard.verify_model", side_effect=_ok):
        app = OnboardApp(cfg)
        async with app.run_test() as pilot:
            await pilot.pause()
            row = next(r for r in app.screen.query(ModelRow) if r.provider == "ihep")
            row.focus()
            await pilot.press("v")
            await asyncio.sleep(0.1)
            await pilot.pause()
            reloaded = load_config(cfg_unset_slots)
            # Both slots got auto-assigned to the only verified model.
            assert reloaded.slots["fast"]["model"] == row.model
            assert reloaded.slots["strong"]["model"] == row.model


@pytest.mark.asyncio
async def test_auto_assign_never_overrides_existing(cfg_with_key: Path):
    """fast/strong already set in YAML should NOT change on first verify."""
    cfg = load_config(cfg_with_key)
    original_fast = cfg.slots["fast"]["model"]
    original_strong = cfg.slots["strong"]["model"]
    with patch("neutrix.onboard.verify_model", side_effect=_ok):
        app = OnboardApp(cfg)
        async with app.run_test() as pilot:
            await pilot.pause()
            # verify a model that isn't either of the bound slots
            row = next(
                r for r in app.screen.query(ModelRow)
                if r.provider == "ihep" and r.model != original_fast and r.model != original_strong
            )
            row.focus()
            await pilot.press("v")
            await asyncio.sleep(0.1)
            await pilot.pause()
            reloaded = load_config(cfg_with_key)
            assert reloaded.slots["fast"]["model"] == original_fast
            assert reloaded.slots["strong"]["model"] == original_strong


# ----- quit / exit ----------------------------------------------------------


@pytest.mark.asyncio
async def test_q_dismisses_with_true(cfg_with_key: Path):
    cfg = load_config(cfg_with_key)
    app = OnboardApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
    assert app.return_value is True


@pytest.mark.asyncio
async def test_ctrl_c_twice_hard_exits_app(cfg_with_key: Path):
    cfg = load_config(cfg_with_key)
    app = OnboardApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+c")
        await pilot.pause()
        scr = app.screen
        assert isinstance(scr, OnboardScreen) and scr._quit_pending
        await pilot.press("ctrl+c")
        await pilot.pause()
    # app exited (no specific result, but app.exit was called)
    assert app.is_running is False or app._exit


@pytest.mark.asyncio
async def test_esc_cancels_pending_ctrl_c(cfg_with_key: Path):
    cfg = load_config(cfg_with_key)
    app = OnboardApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause()
        scr = app.screen
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert scr._quit_pending
        await pilot.press("escape")
        await pilot.pause()
        assert not scr._quit_pending
        await pilot.press("q")
        await pilot.pause()
    assert app.return_value is True


# ----- focus nav ------------------------------------------------------------


@pytest.mark.asyncio
async def test_arrow_keys_navigate_focusables(cfg_with_key: Path):
    cfg = load_config(cfg_with_key)
    app = OnboardApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause()
        scr = app.screen
        first = scr.focused
        await pilot.press("down")
        await pilot.pause()
        assert scr.focused is not first
        await pilot.press("up")
        await pilot.pause()
        assert scr.focused is first


# ----- no floating toasts ---------------------------------------------------


@pytest.mark.asyncio
async def test_notify_routes_through_inline_bar(cfg_with_key: Path):
    """Screen.notify (any caller) writes to the #message bar, not a toast."""
    cfg = load_config(cfg_with_key)
    app = OnboardApp(cfg)
    async with app.run_test() as pilot:
        await pilot.pause()
        scr = app.screen
        scr.notify("hello world", severity="warning")
        await pilot.pause()
        from textual.widgets import Static
        bar = scr.query_one("#message", Static)
        # rendered text contains the message somewhere
        assert "hello world" in str(bar.render())
