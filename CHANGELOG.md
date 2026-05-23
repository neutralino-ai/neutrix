# Changelog

All notable changes to neutrix. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/) with the pre-1.0 rule that minor
bumps may include breaking changes (see [release-workflow rule](.claude/rules/release-workflow.md)).

## [v0.6.6] - 2026-05-24

### Changed
- Main chat blocks are now fully borderless and titleless: system prompt,
  user messages/draft, and LLM responses share the same compact block spacing
  and only differ by role color.
- The main chat no longer renders a top header/title/subtitle. Provider,
  model, tool state, and message count now live only on the plain bottom
  status line.
- The main chat now defaults to the strong slot when configured, reserving
  the fast slot as a fallback.
- The dark chat palette now uses warmer ink/graphite surfaces instead of the
  previous blue-toned panel.
- The repository agent workflow now requires a user-based acceptance test
  confirmation before CHANGELOG updates.

### Added
- Keyboard block navigation: Up/Down moves focus by one block, while
  Ctrl+Up/Ctrl+Down moves by a page-sized group of blocks.
- Regression coverage for titleless chat chrome, shared block spacing,
  strong-slot default launch, disabled mouse reporting, and block navigation.

  Total suite: 65 tests.

See [docs/PRDs/v0.6.6-main-chat-borderless-blocks.md](docs/PRDs/v0.6.6-main-chat-borderless-blocks.md) and [docs/PRDs/v0.6.6-agent-workflow-user-acceptance.md](docs/PRDs/v0.6.6-agent-workflow-user-acceptance.md).

## [v0.6.5] - 2026-05-24

### Changed
- Retuned the rendered main chat spacing after inspecting a Textual SVG
  screenshot: the unframed system prompt now has a one-column horizontal
  inset instead of starting at terminal column 0.
- The draft editor now auto-sizes to its content from one to six visible
  rows, so a one-line draft no longer leaves extra empty rows inside the user
  block.

### Added
- Regression coverage for the system prompt inset and draft editor height
  behavior.

  Total suite: 62 tests.

See [docs/PRDs/v0.6.5-main-chat-spacing-retune.md](docs/PRDs/v0.6.5-main-chat-spacing-retune.md).

## [v0.6.4] - 2026-05-24

### Changed
- Main chat blocks are more compact: the chat container no longer adds outer
  padding, transcript blocks keep only a tiny one-row gap between blocks, and
  the notice/status lines use tighter side margins.
- Message and draft block text now sits closer to the block edge by removing
  the extra inner padding column.
- The editable user draft now uses a compact multiline `TextArea` while
  keeping the same outer user-block styling as committed user messages.
- The draft editor keeps one background surface when focused instead of
  showing a second inner focus background.
- The system prompt now renders as warning-colored text on the screen
  background, without a border or title.

### Fixed
- Assistant content is now rendered even when an OpenAI-compatible provider
  emits only the final assistant message and no streamed token deltas.

### Added
- Draft newline shortcuts: Shift+Enter, Alt+Enter, and Ctrl+J insert a line
  break while Enter still submits.
- Regression coverage for final-only assistant responses and multiline draft
  submit/newline behavior.

  Total suite: 61 tests.

See [docs/PRDs/v0.6.4-main-chat-compact-draft.md](docs/PRDs/v0.6.4-main-chat-compact-draft.md).

## [v0.6.3] - 2026-05-23

### Changed
- Main chat transcript blocks now expand to the full terminal width instead
  of a fixed narrow column.
- System, user, and LLM blocks now share the same block shape and differ by
  semantic theme styling. The editable draft uses the same user-block surface
  as committed user messages, with only the inner input focusable/editable.
- The agent internals now follow an onion split:
  `tui.py` owns the view, `agent_loop.py` owns append-only conversation and
  tool continuation, and `llm.py` owns one OpenAI-compatible streaming
  request.

### Fixed
- Model turn failures now surface as error blocks/notices and always restore
  the draft input to an enabled, focused state, avoiding the apparent stuck
  state after submitting a message.

### Added
- Agent-loop tests for plain assistant replies and tool-call follow-up
  sampling.
- LLM adapter coverage for token streaming and final assistant/tool-call
  assembly.
- TUI regression coverage for user/draft block parity and error recovery.

  Total suite: 55 tests.

See [docs/PRDs/v0.6.3-main-onion-blocks.md](docs/PRDs/v0.6.3-main-onion-blocks.md).

## [v0.6.2] - 2026-05-23

### Changed
- Main chat now renders the model-visible context as one block list:
  system prompt, committed user turns, assistant turns, and tool output.
- The composer is now styled as a draft user block inside that same
  list, so typing happens where the next model-visible message will
  appear.
- App-only slash commands and their feedback, including `/onboard`, no
  longer render as transcript blocks. Command feedback now lives in the
  notice line, and `Ctrl+L` clears that notice line.

### Added
- Headless TUI coverage for command feedback staying outside the
  model-visible block list.
- `AGENTS.md` with the required PRD -> code -> changelog -> release
  workflow for implementation tasks.

See [docs/PRDs/v0.6.2-main-visible-block-list.md](docs/PRDs/v0.6.2-main-visible-block-list.md).

## [v0.6.1] - 2026-05-23

### Fixed
- API keys submitted inside `/onboard` no longer bubble into the main
  chat app as user messages. Onboarding now stops handled key-submit
  events, and the chat app only accepts submissions from its own
  composer input.
- This prevents onboarding secrets from being rendered in the chat log,
  appended to `agent.messages`, or sent to the model.

### Added
- Regression coverage that opens onboarding from the main app, submits
  a sentinel API key, and asserts it never reaches the chat transcript
  or model streaming path.

See [docs/PRDs/v0.6.1-onboard-key-submit-isolation.md](docs/PRDs/v0.6.1-onboard-key-submit-isolation.md).

## [v0.6.0] - 2026-05-23

### Changed
- Main chat now opens with a centered composer block instead of a
  bottom-docked full-width input. After the first posted message, the
  composer moves into the chat flow directly under the message list.
- The default system prompt is now the simple chatbot prompt:
  `You are a helpful assistant. Keep it simple.`
- Chat messages now render as compact role-labelled blocks with
  distinct styling for user, assistant, system, tool, and error output.

### Added
- The active system prompt is shown on screen inside the composer.
- Streaming replies now show an animated "assistant is responding"
  indicator and disable the input until the model finishes.
- Headless TUI tests for the visible system prompt, composer placement,
  user submission, assistant streaming, and busy indicator behavior.

  Total suite: 49 tests.

See [docs/PRDs/v0.6.0-main-chat-polish.md](docs/PRDs/v0.6.0-main-chat-polish.md).

## [v0.5.5] — 2026-05-23

### Fixed
- Onboarding api_key fields now restore the committed password mask after
  Enter and the subsequent focus move. This covers the interactive case
  where the key saved correctly but the blurred field could still look
  empty because a late focus/editing event left the visible buffer blank.
- Password fields now render committed secrets with `*` masks, matching
  the expected `****` terminal convention.

### Added
- A regression test that inspects the actual rendered input line after
  typed Enter and requires a mask instead of the `EMPTY` placeholder.
- A regression test that simulates a late empty buffer after submit and
  verifies the onboarding screen restores the committed masked display.

  Total suite: 46 tests.

See [docs/PRDs/v0.5.5-onboard-key-mask-after-enter.md](docs/PRDs/v0.5.5-onboard-key-mask-after-enter.md).

## [v0.5.4] — 2026-05-23

### Fixed
- After typing + Enter on an api_key field, the Input could still flash
  to the `EMPTY ...` placeholder visually in some environments. Root
  cause: a blur from `focus_next()` could race ahead of the screen's
  `on_input_submitted` and read a stale `_committed_value` (still the
  pre-Enter empty buffer), then revert the value to that stale
  baseline. Fix: override `KeyInput.action_submit` to promote the
  buffer to `_committed_value` **before** the Submitted message is
  posted. Any subsequent blur — whatever order Textual chooses for
  events — already sees the up-to-date baseline and does not revert.

### Added
- Three new tests in `tests/test_onboard.py`:
  - `test_real_keystroke_typing_enter_persists` — types each char via
    `pilot.press(ch)` (mirroring real keyboard), confirms value
    persists after Enter and YAML is written.
  - `test_action_submit_commits_baseline_before_message` — directly
    invokes `KeyInput.action_submit()` and then simulates a blur on
    the Input; the blur must not revert.
  - `test_typed_enter_value_renders_dots_not_placeholder` — asserts
    the Input's rendered output does **not** contain the `EMPTY`
    placeholder text after a typed Enter.

  Total suite: 45 tests.

See [docs/PRDs/v0.5.4-onboard-enter-race-fix.md](docs/PRDs/v0.5.4-onboard-enter-race-fix.md).

## [v0.5.3] — 2026-05-23

### Changed
- `KeyInput` now clears its visible buffer on focus, so Tabbing to an
  api_key field with a saved key gives a fresh blank to type into
  (the saved value is preserved internally in `_committed_value` and
  restored on blur). Matches the "ready to enter a new key" mental
  model.
- Enter on an empty api_key field is now treated as "no change":
  the field restores to the committed value, focus advances, and the
  YAML is not touched. Removes the v0.5.2 footgun where Enter on a
  cleared field erased the saved key.

### Added
- Three new tests in `tests/test_onboard.py`:
  - `test_focus_clears_visible_value` — focus blanks the visible
    buffer; `_committed_value` retains the saved key.
  - `test_empty_enter_preserves_committed_value` — empty Enter
    restores, advances focus, leaves YAML unchanged.
  - `test_typed_enter_commits_new_value` — typed Enter saves the new
    value, advances focus, writes YAML.

  Total suite: 42 tests.

See [docs/PRDs/v0.5.3-onboard-key-edit-semantics.md](docs/PRDs/v0.5.3-onboard-key-edit-semantics.md).

## [v0.5.2] — 2026-05-23

### Added
- `tests/test_onboard.py` — 19 headless `OnboardApp.run_test()` tests
  covering every user-facing behavior: api_key Enter persistence and
  focus-advance, blur-revert without Enter, stale-status clear on
  key change, background verify, parallel verify-all, model_status
  YAML round-trip, slot tags rendering from YAML on reopen,
  auto-assign on first verify, auto-assign-never-overrides, q
  dismiss, Ctrl+C×2 hard exit, Esc cancel, arrow nav, and inline
  notify routing. Total suite is now 39 tests.
- Auto-assignment: after any verify, if `fast` / `strong` are unset,
  the first verified model is bound to `fast` (and to `strong`, or
  to a different verified model if available). Never overwrites an
  existing slot binding. Persisted to YAML in the same step.

### Changed
- `OnboardScreen.__init__` now loads `fast` / `strong` from
  `config.slots` so reopening `/onboard` shows the previously-bound
  models with `▸ fast` / `▸ strong` tags and the correct status-bar
  text. Previously both were always `None` on init.
- `on_input_submitted` commits the Input's `_committed_value` baseline
  **before** any other work, then calls `self.focus_next()` so the
  saved indicator becomes visible and the user is unblocked. The race
  where a blur from focus-move read a stale baseline (and reverted to
  empty `EMPTY` placeholder) is gone.
- Overrode `OnboardScreen.notify(...)` to route any Textual-internal
  or future-code call through the inline `#message` bar. No more
  floating toasts on the onboarding screen, period.

See [docs/PRDs/v0.5.2-onboard-state-and-focus.md](docs/PRDs/v0.5.2-onboard-state-and-focus.md).

## [v0.5.1] — 2026-05-23

### Added
- Onboarding now auto-saves on every committing user action: `v`
  (verify), `v` on the `(all)` row (verify-all), `f` / `g` (slot
  assign), and `Enter` in the api_key Input. No more `s` "save and
  launch" step.
- Inline message bar at the bottom of the onboarding screen — single
  line, severity-coloured (`info` dim, `warning` yellow, `error` red,
  `success` green). Replaces the floating toast notifications that
  popped up in the corner.

### Changed
- Onboarding `Ctrl+C` (×2) now **hard-exits the entire app**, not
  just the onboarding screen. When reached via `/onboard` from chat,
  the chat exits too — treat it like SIGINT. `q` keeps its "soft
  done" semantic (dismisses back to chat; auto-save means no data
  is at risk).
- Verification (single + verify-all) runs as a background worker via
  `self.run_worker(...)`. The screen no longer freezes while waiting
  for the upstream API; focus navigation, additional verifies, and
  api_key edits stay responsive.
- api_key Input is borderless and single-line — no more bottom border
  bleeding onto the row below the api_key. Width is `1fr` (grows
  with the card); focus tints the background `$accent 30%`.
- `q` now dismisses with `True` unconditionally (everything's
  auto-saved). The chat-side toast updated to "back from onboarding."
- Removed the `s` keybinding and `action_save_and_launch`.

### Fixed
- `KeyInput` reverts to the last `Enter`-committed value on blur, so
  a half-typed key the user navigates away from never lands in YAML
  and never sticks around as garbage in the Input.

See [docs/PRDs/v0.5.1-onboard-row-and-quit.md](docs/PRDs/v0.5.1-onboard-row-and-quit.md).

## [v0.5.0] — 2026-05-23

### Added
- Onboarding now **persists model verification status** in
  `~/.config/neutrix/config.yaml` per provider, under a new optional
  `model_status: {model: verified|failed}` map. Reopening `/onboard`
  resumes from the last known statuses instead of resetting to `?`.
  Changing an api_key clears its provider's `model_status` (stale).
- **Verify-all in parallel**: each provider now has an `(all)` row at
  the top of its model list. Press `v` on it to verify every model
  via a single `asyncio.gather` (one shared `AsyncOpenAI` client per
  batch). Statuses are written to YAML in one save at the end.
- "saved" indicator next to each api_key Input — lights up after a
  successful submit so the user can see the key is committed.

### Changed
- Onboarding UI redesigned: each provider sits in a rounded bordered
  card; status icons are colour-coded (`○` dim, `✓` green, `✗` red,
  `…` yellow); slot tags render as accent-coloured `▸ fast` / `▸ strong`
  badges; padding generous; intro line trimmed. Uses Textual design
  tokens (`$primary`, `$success`, `$error`, `$accent`, `$text-muted`,
  `$boost`) so the theme follows the user's terminal palette.

### Fixed
- Onboarding arrow nav no longer escapes scope: previously pressing
  `Up`/`Down` past the first/last focusable could land focus on the
  surrounding `VerticalScroll`, whose own `up`/`down` BINDINGS hijacked
  the keys for scrolling. Replaced with a `FocusScroll(VerticalScroll)`
  subclass that sets `can_focus = False`; focus now wraps within the
  ModelRow/KeyInput/VerifyAllRow set as expected. PgUp/PgDn and
  mouse-wheel scrolling are unaffected.
- The api_key Input no longer appears to lose its value after `Enter`
  — defensive re-assign of `event.input.value` and the new "saved"
  affordance remove the ambiguity from password-masked rendering.

See [docs/PRDs/v0.5.0-onboard-polish.md](docs/PRDs/v0.5.0-onboard-polish.md).

## [v0.4.2] — 2026-05-23

### Changed
- `PROVIDER_DEFAULT_MODELS` refreshed to names the upstreams actually
  serve today:
  - `deepseek`: `deepseek-v4-flash`, `deepseek-v4-pro` (was
    `deepseek-chat`, `deepseek-reasoner` — both retired).
  - `glm`: `glm-5.1`, `glm-5.1-highspeed` (was the 4.x line).
  - `ihep`: claude haiku/opus/sonnet 4.x kept; deepseek path renamed
    to `deepseek-ai/deepseek-v4-{pro,flash}`; added
    `openai/gpt-5.5`, `zhipu/glm-5.1`, `kimi/kimi-k2.6`.

Existing user YAML is not migrated — only the onboarding catalog
changes. Default bootstrap slots (`anthropic/claude-haiku-4-5` /
`anthropic/claude-opus-4-7`) are unchanged.

See [docs/PRDs/v0.4.2-deepseek-model-names.md](docs/PRDs/v0.4.2-deepseek-model-names.md).

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
