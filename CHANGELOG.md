# Changelog

All notable changes to neutrix. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/) with the pre-1.0 rule that minor
bumps may include breaking changes (see [release-workflow rule](.claude/rules/release-workflow.md)).

## [v0.4.1] — 2026-05-23

### Fixed
- Onboarding TUI: `Up` / `Down` arrow-key focus navigation regressed in
  v0.4.0 when `OnboardApp` was refactored to `OnboardScreen`. Screen-
  level `priority=True` bindings do not preempt `VerticalScroll`'s own
  arrow-key scroll handling. Switched to widget-level `on_key` handlers
  on `ModelRow` and a new `KeyInput(Input)` subclass — they run first
  in the focus chain and reliably consume the event. `PgUp/PgDn` and
  mouse-wheel scrolling are unaffected.

See [docs/PRDs/v0.4.1-onboard-arrow-nav-fix.md](docs/PRDs/v0.4.1-onboard-arrow-nav-fix.md).

## [v0.4.0] — 2026-05-23

### Added
- `/onboard` slash command in the chat TUI — re-enters the same
  onboarding surface used on first run, so the user can rotate keys,
  add providers, or re-verify models without leaving the chat.

### Changed
- Refactored `OnboardApp` into `OnboardScreen` (Textual `Screen[bool]`)
  plus a thin `OnboardApp` wrapper. Both first-run (`cli.py`) and
  mid-chat (`/onboard`) paths share the same screen. After dismissal,
  `/onboard` reloads the YAML but leaves the live agent on its current
  slot — use `/fast` or `/strong` to adopt new bindings.

See [docs/PRDs/v0.4.0-onboard-slash-command.md](docs/PRDs/v0.4.0-onboard-slash-command.md).

## [v0.3.1] — 2026-05-23

### Added
- Onboarding TUI: `Up` / `Down` arrow keys move focus between api_key
  Inputs and model rows (same widgets `Tab` traverses).
- Onboarding TUI: `Ctrl+C` is now the universal quit, with two-tap
  confirm — first press toasts "press Ctrl+C again to quit, Esc to
  cancel" and arms a 5 s window; second `Ctrl+C` exits, `Esc` cancels,
  or the window simply expires.

See [docs/PRDs/v0.3.1-onboarding-arrow-nav.md](docs/PRDs/v0.3.1-onboarding-arrow-nav.md).

## [v0.3.0] — 2026-05-23

### Added
- First-run onboarding TUI: when neither `fast` nor `strong` resolves
  (both bound providers have empty `api_key`), `neutrix` opens an inline
  onboarding screen instead of exiting with an error. The user pastes a
  key, verifies a model with one keystroke (1-token API call), assigns
  it to a slot, saves, and drops straight into the chat TUI.
- `PROVIDER_DEFAULT_MODELS` catalog in `config.py` — curated model list
  per known provider, shown in onboarding.
- `save_config()` in `config.py` — round-trippable YAML write-back.
- `onboard.py` module hosting the onboarding Textual app.

### Changed
- `cli.py` slot-resolve order: try `fast`, then fall back to `strong`,
  then onboarding. Previously `fast` failure was a hard exit.

See [docs/PRDs/v0.3.0-onboarding-tui.md](docs/PRDs/v0.3.0-onboarding-tui.md).

## [v0.2.0] — 2026-05-23

### Added
- YAML config at `~/.config/neutrix/config.yaml`, auto-bootstrapped on first run.
- Two named model slots — `fast` and `strong` — switchable in-TUI via `/fast`
  and `/strong`.
- `CLAUDE.md` at repo root: SOLID + YAGNI guidance for any AI working in the codebase.
- `.claude/rules/release-workflow.md`: every change requires PRD + CHANGELOG + tag.

### Changed
- `Agent` now takes a resolved `Slot` (base_url + api_key + model) instead of
  a hardcoded `Provider`.
- README rewritten around the YAML / slot model.

### Removed
- All env-var config (`*_API_KEY`, `NEUTRIX_PROVIDER`, `NEUTRIX_MODEL`).
- `python-dotenv` dependency; `.env.example`.
- `--provider` / `--model` CLI flags (and the unimplemented `--fast` / `--strong`).
- Hardcoded `claude` provider entry — Claude is reached via the IHEP gateway in
  the default config; add an `anthropic:` provider manually if you want direct
  API access.

### Dependencies
- + `pyyaml >= 6.0`
- − `python-dotenv`

Breaking change: env-var users must migrate to `~/.config/neutrix/config.yaml`.
See [docs/PRDs/v0.2.0-yaml-config.md](docs/PRDs/v0.2.0-yaml-config.md).

## [v0.1.0] — 2026-05-23

### Added
- Initial release: multi-provider Textual TUI agent over the OpenAI SDK.
- Streaming chat completions for DeepSeek, GLM, and Claude (via Anthropic's
  OpenAI-compat layer).
- OpenAI-style tool calling with built-ins: `read_file`, `write_file`,
  `list_dir`, `run_shell`.
- Runtime provider switching with `/model`, JSON session save/load.
- `pyproject.toml` with `setuptools_scm` dynamic versioning from git tags.
- `neutrix` CLI entry point.
