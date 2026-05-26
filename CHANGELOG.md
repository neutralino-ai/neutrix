# Changelog

All notable changes to neutrix. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/) with the pre-1.0 rule that minor
bumps may include breaking changes (see [release-workflow rule](.claude/rules/release-workflow.md)).

## [v0.9.2] — 2026-05-26

### Added
- ``Esc`` is now the universal "stop and go back to idle" key while a
  turn is in flight. Pressing Esc closes the LLM's HTTP stream
  eagerly, tree-kills every cancellable subprocess registered with
  the executor pool (``run_shell``'s ``sleep 30`` becomes a
  ``killpg`` victim within ~200 ms), rolls
  ``Agent.messages`` back to the pre-turn snapshot — dropping the
  user_turn AND any orphan assistant ``tool_calls`` message that
  would otherwise 400 the next OpenAI request — clears
  ``store.pending_tool_calls`` + ``store.llm_active``, and prints a
  dim-yellow ``interrupted`` notice. Queued user messages (typed
  while busy, per v0.9.1) survive the cancel: the worker loop drains
  them next.
- ``ChatLLM.stop()`` on the protocol + ``OpenAIChatLLM.stop()`` on
  the implementation. Closes the underlying OpenAI SDK
  ``AsyncStream`` so the iterator's next ``__anext__`` exits
  cleanly. Idempotent — no-op when no stream is in flight.
- New ``Executor`` class (``src/neutrix/executor.py``). Owns the
  per-turn rollback snapshot, the cancellable-Popen pool, and the
  single ``cancel()`` entry-point the controller broadcasts to.
  POSIX-only ``_tree_kill`` helper (SIGTERM + 200 ms grace +
  SIGKILL) — Python analog of Claude Code's ``tree-kill``.
- New ``Controller`` class (``src/neutrix/controller.py``). Single
  command surface the view drives. ``cancel()`` broadcasts to
  ``llm.stop()``, ``executor.cancel()``, then ``task.cancel()`` —
  each subordinate is independently idempotent so the controller
  never asks "are you busy?". Designed so v0.11.0's ``Advisor``
  plugs in as a fourth broadcast target with one extra line.
- ``Agent.rollback_to(n)`` — the one new seam the Executor uses;
  trims ``self.messages[n:]`` so a cancelled turn leaves a valid
  history.
- First ``Ctrl+C`` while a turn is in flight now cancels the turn
  WITHOUT arming the v0.9.1 ``press Ctrl+C again to exit`` hint.
  When idle, ``Ctrl+C`` keeps its v0.9.1 arm-or-exit semantics.

### Changed
- ``OpenAIChatLLM.stream_response`` switched to ``stream=True`` —
  token deltas surface as ``LLMEvent("token", str)`` as they
  arrive instead of one final ``assistant`` event per call. Tool
  calls accumulate across streaming deltas (``index``-keyed
  rebuild). Internally this enables the eager-close that
  ``LLM.stop()`` relies on; externally streaming visibly stops
  faster than v0.9.0's final-response wait.
- ``Agent`` now requires an ``llm`` in the constructor. Earlier
  callers that relied on the auto-build path
  (``OpenAIChatLLM(slot)``) must now construct the LLM explicitly
  and pass it. The CLI bootstrap does this in ``cli.py``.
- ``Agent.stream_reply`` accepts an optional ``executor=`` kwarg
  threaded to the tool dispatch shim. ``dispatch(...)`` now grows
  an ``executor=`` keyword that is forwarded only to tools whose
  signature declares it (currently ``run_shell``) — the
  LLM-facing JSON schema is unchanged.
- ``run_shell`` rewritten from ``subprocess.run`` to
  ``subprocess.Popen(start_new_session=True)`` +
  ``communicate``, with the Popen registered with the executor
  pool before the wait and unregistered in ``finally``. On
  cancel, the whole process group dies via ``_tree_kill`` and
  the tool returns ``[cancelled by user]``.
- ``TerminalChat`` constructs an ``Executor`` and ``Controller``
  alongside the agent; ``_send_message`` routes every turn
  through ``controller.send``. The view tracks an outer asyncio
  task and a single ``cancel_hook`` is passed down through
  ``TerminalView`` → ``DraftReader`` → ``build_draft_key_bindings``
  so the key bindings can fire cancel without reaching across
  the layer boundary.

### Removed
- The v0.9.1 ``(escape, enter)`` Alt+Enter newline binding.
  ``eager=True`` on the standalone ``escape`` binding swallows
  the meta-prefix so composed sequences cannot match — a
  deliberate trade-off documented in the PRD. Users insert
  newlines via ``Ctrl+J`` (unchanged).
- ``Alt-*`` word-motion shortcuts (``Alt+B``, ``Alt+F``,
  ``Alt+D``) are unavailable for the same reason — same
  deliberate trade-off.

### Non-changes (deliberately)
- No pure-compute tool cancellation. Tools running via
  ``asyncio.to_thread`` (``read_file``, ``list_dir``, ``TaskCreate``,
  …) keep running on the background thread; their results are
  silently dropped because ``agent.messages`` rolls back. A
  generic ``CancellableTool`` polling protocol is v0.10.x scope.
- No "save partial assistant turn" steering. The cancelled turn
  is dropped whole; the user re-prompts.
- No multi-process / RPC implementation. The service-oriented
  *interfaces* land in v0.9.2; the *transport* stays in-process.
- The legacy Textual ``tui.py`` app is untouched.

See [docs/PRDs/v0.9.2-cancellation.md](docs/PRDs/v0.9.2-cancellation.md).

## [v0.9.1] — 2026-05-26

### Added
- Bash- / Claude-style trailing-backslash line continuation in the
  draft editor. Ending a buffer with ``\`` and pressing ``Enter``
  strips the backslash and inserts a newline instead of submitting.
  Mid-buffer backslashes are ignored (the user is editing earlier in
  the draft) so plain ``Enter`` keeps its split semantics there.
- ``Ctrl+Z`` now suspends ``neutrix`` to the background via
  ``app.suspend_to_background()`` — prompt_toolkit's built-in shim
  that disables raw mode, raises ``SIGTSTP``, then re-enters raw
  mode on ``SIGCONT``. ``fg`` brings the chat back with the
  draft buffer intact.

### Changed
- ``Ctrl+C`` AND empty-buffer ``Ctrl+D`` are now a double-press
  exit with **independent per-chord timers**. The first press of
  either key arms that chord's own 1-second window and renders a
  dim-gray ``press Ctrl+C again to exit`` or
  ``press Ctrl+D again to exit`` hint above the cursor — the
  wording names exactly the key the user must press to confirm.
  The second press of **the same key** within that chord's own
  window exits: ``Ctrl+C`` → ``KeyboardInterrupt``,
  ``Ctrl+D`` → ``EOFError``, both already caught by
  ``_input_loop``. **Cross-key presses are non-destructive** —
  pressing ``Ctrl+D`` while ``Ctrl+C`` was armed arms ``Ctrl+D``
  on its own clock and refreshes the displayed hint to
  ``press Ctrl+D again to exit``, but the ``Ctrl+C`` timer keeps
  running on its original arming time. So ``Ctrl+C → Ctrl+D →
  Ctrl+C`` (all within 1 s of the first press) exits via
  ``KeyboardInterrupt`` — the intervening Ctrl+D did not touch
  c-c's clock. **``Ctrl+D`` only enters the quit dance when the
  buffer is empty**; with text in the draft, the default Emacs
  forward-delete-character is preserved (guarded by a
  ``prompt_toolkit.filters.Condition``). **The hint auto-fades on
  its own** — each arming schedules a background task that
  invalidates the app once that chord's window expires. Hint
  color is ``fg:ansibrightblack`` — the same dim style as queued
  user messages — so the affordance reads as part of the muted
  UI hierarchy, not a warning (an earlier yellow palette was
  rejected at review as visually too loud). Keypresses other than
  the two quit chords neither cancel arming nor extend any
  timer; each chord's window is set by that chord's own most
  recent press alone (pure-timer model — the originally-drafted
  Codex-style "any-key-cancels" rule was rejected at Phase-3
  review). ``handle_sigint=False`` is now passed at each
  ``prompt`` / ``prompt_async`` call so SIGINT reaches the
  binding instead of being translated to ``KeyboardInterrupt``
  upstream.
- Newline keys: ``Ctrl+J`` and ``Alt+Enter`` insert a newline at
  the cursor. ``Shift+Enter`` also inserts a newline on terminals
  that emit ``Shift+Enter`` as ``Ctrl+J`` (gnome-terminal, xterm,
  default macOS Terminal, Linux console). prompt_toolkit 3.0.x has
  no ``Keys.ShiftEnter`` / ``Keys.ControlEnter`` entries, so
  CSI-u-enabled terminals must wait for upstream support — out of
  scope.

### Non-changes (deliberately)
- No mid-stream cancellation. The first ``Ctrl+C`` / ``Ctrl+D``
  only arms the exit shortcut; an in-flight LLM call keeps
  running until it finishes normally or the user double-presses
  to exit the process. Plumbing into ``Agent.stream_reply`` is
  v0.9.2 scope — see ``docs/PRDs/v0.9.2-cancellation.md``.
- ``Ctrl+D`` on a non-empty draft does NOT exit; it keeps the
  default forward-delete behavior because that's the more useful
  thing the chord can do mid-draft.
- ``ChatStore`` and the transcript format are untouched — v0.9.1
  is view-only.

See [docs/PRDs/v0.9.1-keyboard.md](docs/PRDs/v0.9.1-keyboard.md).

## [v0.9.0] — 2026-05-25

### Added
- Two new `AgentEvent` kinds: `llm_request_start` and `llm_request_end`.
  `Agent.stream_reply` now wraps each LLM round with exactly one
  start/end pair, so observers can tell *when* an LLM request is in
  flight, not just what content arrived. Multi-round tool-loop turns
  emit multiple pairs, one per round. Cancellation via
  `.aclose()` (PEP 525 / `GeneratorExit`) is handled by an inner
  `except Exception` plus an explicit `yield` outside any `finally`,
  so `TerminalChat.run_async`'s worker-cancel path on Ctrl-C unwinds
  cleanly without `RuntimeError`.
- `ChatStore.llm_active: bool` (read-only property) — true while an
  LLM request is in flight; flipped by the reducer below.
- Single reducer entry point `ChatStore.apply(event)` that maps
  `llm_request_start`/`llm_request_end` to `llm_active`,
  `tool_call`/`tool_result` to pending-tool-call list mutations, and
  no-ops on `token` / `assistant` / `done` / `error`. Accepts
  `event: Any` (read reflectively) to avoid a circular import with
  `agent_loop`.

### Changed
- `TerminalChat._send_message` now routes every event through
  `self.store.apply(event)` before the existing render switch. The
  view no longer writes to the store mid-event: direct
  `add_pending_tool_call` / `remove_pending_tool_call` calls in
  `_handle_event` are gone, and `_pop_tool_arguments` is replaced
  with a short-lived per-call `tool_arg_cache: dict[str, list[str]]`
  that carries arguments forward from `tool_call` to `tool_result`.
  This is the prerequisite for the v0.10.0 FakeView swap-test: the
  renderer becomes a pure reader of the store.
- `ChatStore.reset()` clears `llm_active` along with the other
  in-flight state.

### Non-changes (deliberately)
- No user-visible behavior change. Same layout, colors, task panel,
  streaming behavior as v0.8.2. The release scope leaked if you can
  spot a difference.
- No rename of `agent_loop.py` / `Agent` / `stream_reply`. Deferred
  to v0.10.0 if still warranted once a second `Agent` consumer
  (FakeView) materializes.
- `/save` / `/load` transcript format unchanged. `llm_active` is
  always `False` at serialize time (saves only happen between turns).

See [docs/PRDs/v0.9.0-lifecycle-events.md](docs/PRDs/v0.9.0-lifecycle-events.md).

## [v0.8.2] — 2026-05-25

### Changed
- Tool-call and tool-result transcript lines now carry a colored
  keyword anchor matching the Anthropic SDK content-block types:
  `-> tool_use   <name> <args>` (keyword in **bold cyan** against the
  existing dim line) and `<- tool_result [tool N] <name> <args> | ...`
  (keyword in **bold bright_green** against the existing yellow
  line). Both keywords are right-padded to width 11 so the body name
  column lines up between the two lines and across successive calls.
  Same vocabulary you see in `/save`'d chatstore JSON and in
  `agent_loop.py`. Fold/expand behavior, `[tool N]` indexing, and the
  `| folded | N lines | ~K tokens` summary are unchanged. Replay paths
  (`/load` and `/tool`) render with the same colored format.

See [docs/PRDs/v0.8.2-tool-keyword-colors.md](docs/PRDs/v0.8.2-tool-keyword-colors.md).

## [v0.8.1] — 2026-05-25

### Added
- Persistent task panel above the input cursor. Shows every task
  (pending, in_progress, completed) the moment any exist, with a
  `… +N pending, M done` overflow line capped at 5 task rows. Hidden
  when no tasks exist. Lives in the same dim-foreground region as the
  queued-message display.
- Task-reminder messages (from v0.8.0's 10-turn trigger) now render
  inline as a single dim "system reminder: task list injected (N done,
  N in progress, N todo)" notice with live counts of the current task
  list — both live (when the controller injects them) and on `/load`
  replay. v0.8.0's trigger algorithm, body text, and persistence
  format are unchanged; only the renderer is new. Folded by default
  because the always-visible task panel above the input already shows
  the task list. Smart, judged reminders (with their own renderer
  treatment) arrive in v0.11.0 — see
  `docs/PRDs/v0.11.0-smart-advisor.md`.

### Changed
- `TaskCreate`, `TaskUpdate`, `TaskList` tool descriptions are now
  ports of Claude Code's V2 task tool prompts (with the agent-swarm
  branches dropped). The load-bearing lifecycle clauses — *"Mark it
  as in_progress BEFORE beginning work"*, *"Always mark your assigned
  tasks as resolved when you finish them"*, *"After resolving, call
  TaskList to find your next task"* — live in the tool schema the
  LLM sees on every turn, matching where Claude Code's own V2 puts
  them. Replaces the v0.8.0 one-line stubs that left the LLM with no
  lifecycle guidance and led to "tasks created but not started" and
  "claimed but no tool call fired" failure modes during manual
  testing.
- Tool result strings now match Claude Code's V2 byte-for-byte:
  `"Task #N created successfully: subject"`,
  `"Updated task #N status, subject"`, `"Updated task #N deleted"`,
  `"Task #N not found"`. (An earlier v0.8.1 draft appended a
  *"Please proceed with the current tasks if applicable"* nudge to
  these — that wording belongs to V1's `TodoWriteTool` and not to
  V2; mixing V1's nudge into V2-shaped tools shaped the LLM
  incorrectly.)
- Input draft placeholder is now dim gray (prompt_toolkit
  FormattedText with `fg:ansibrightblack`) instead of the default
  foreground style, matching the dim hierarchy already used for
  queued user messages.

See [docs/PRDs/v0.8.1-tasks-visible-and-auto-continue.md](docs/PRDs/v0.8.1-tasks-visible-and-auto-continue.md).

## [v0.8.0] - 2026-05-25

### Added
- First-class task tracking. `ChatStore` gains a `tasks` field plus
  `add_task`, `update_task`, `remove_task`, and `replace_tasks`
  mutators. Ids are monotonic strings, scoped per session, and
  preserved across `/save`/`/load` (next id resumes from
  `max(loaded_ids) + 1`, not `len(loaded) + 1`, so deleted-then-saved
  ids never collide).
- Three Claude-Code-shaped LLM-callable tools in `tools.py`:
  - `TaskCreate(subject, description?)`
  - `TaskUpdate(taskId, status?, subject?, description?)` — passing
    `status="deleted"` removes the task, matching Claude exactly.
  - `TaskList()` — read-only JSON dump of every task.
  `dispatch()` now accepts a `store=` keyword that the Task tools
  require and existing tools ignore.
- `Agent` now accepts a `ChatStore` reference and, on every
  `stream_reply`, may inject a Claude-shaped
  `<system-reminder>…</system-reminder>` user message before the
  first LLM round. The trigger algorithm matches Claude Code's
  `TODO_REMINDER_CONFIG`: at least 10 assistant turns since the last
  `TaskCreate`/`TaskUpdate`, at least 10 since the last reminder,
  and at least one actionable (`pending` or `in_progress`) task.
  The reminder is part of the persistent message history and rides
  through `transcript.save`/`load` unchanged.
- New `/tasks` slash command in the terminal chat and the legacy
  Textual TUI. Prints the current task list (read-only) as
  `#{id} [{status}] {subject}` lines, or `"no tasks"` when empty.

### Changed
- Transcript format bumped to **version 2**: the on-disk JSON now
  carries a `tasks` array alongside `messages`. v1 files load
  cleanly with an empty task list. Saves always write v2.
- `TerminalChat` and the legacy `NeutrixApp` construct themselves
  around a `ChatStore` and wire it into the agent
  (`self.agent.store = self.store`), so the Task tools mutate the
  same store the view renders from.
- `/load` now calls `store.replace_tasks(loaded.tasks)` so
  reloading a saved session restores tracked tasks. Without this
  fix `/load` silently discarded them.
- `/clear` resets tasks too — the conversation reset is total.
- `--load PATH` on the CLI plumbs loaded tasks into the
  terminal-chat store the same way `/load` does.

See [docs/PRDs/v0.8.0-tasks.md](docs/PRDs/v0.8.0-tasks.md).

## [v0.7.0] - 2026-05-24

### Added
- New `neutrix.store` module exposing `ChatStore`, the canonical
  in-memory record of a chat session (settled messages, queued user
  inputs, in-progress assistant stream text, pending tool calls).
- `ChatStore.changes()` async iterator for renderers to await store
  mutations; multiple consecutive mutations between yields coalesce
  into a single wake-up.
- Queued user messages now render directly above the input cursor in
  dim foreground, prefixed with `› `, sharing the input area's
  background so the queue and input form one visual region. They go
  through prompt_toolkit's `PromptSession.message`.
- Screen updates are invalidation-driven: a background coroutine
  subscribes to `ChatStore.changes()` and calls `app.invalidate()`
  once per batch of mutations, removing the prior 0.5-s periodic
  refresh and the rhythmic flicker it caused.
- New `/status` command prints the current slot, provider/model,
  tool state, and message count on demand. Replaces the persistent
  bottom toolbar.

### Changed
- Renamed `neutrix.session` → `neutrix.transcript`. The on-disk JSON
  format is unchanged; old `session.py`-written files load cleanly via
  `transcript.load`.
- `transcript.save` / `transcript.load` now operate on a `ChatStore`
  rather than a separate `messages` list. `load` returns
  `(store, metadata)` where `metadata` keeps `raw_messages` for callers
  still feeding the agent's OpenAI-format list.
- `TerminalChat` mirrors agent events into its own `ChatStore`. Pending
  tool calls and the user queue now live on the store rather than as
  private attributes.

### Removed
- The bottom status toolbar. In prompt_toolkit's append-only mode the
  toolbar visibly blinked during streaming output because every stdout
  write triggers a hide-restore cycle of the prompt area. Slot,
  provider, model, tool state, and message count are now available on
  demand via the new `/status` command.
- The `queued:N` counter (the visible queue replaces it).

### Fixed
- IHEP provider's Kimi model is namespaced under `moonshot/`, not
  `kimi/`. The default model catalog now lists `moonshot/kimi-k2.6` so
  the onboarding picker hands the gateway the prefix it actually
  accepts.

See [docs/PRDs/v0.7.0-chatstore.md](docs/PRDs/v0.7.0-chatstore.md).

## [v0.6.8] - 2026-05-24

### Changed
- The main chat now launches as an append-only terminal chat instead of a
  fullscreen Textual screen, so long conversations use normal terminal
  scrollback and short conversations do not reserve a full screen.
- Input now uses a `prompt_toolkit` multiline draft editor with normal
  readline-style editing keys, including Ctrl+A and Ctrl+K.
- The terminal chat now has an explicit view/controller split: the view owns
  prompt rendering and transcript output, while the controller owns commands,
  queueing, agent events, and model/tool work.
- The prompt stays active while the assistant is working; prompts submitted
  during an active response are queued and run in order.
- System, user, and assistant transcript content is distinguished by role
  color instead of `system:`, `user:`, or `assistant:` labels.
- Chat logging now writes to `~/.cache/neutrix/neutrix.log` instead of the
  interactive terminal.
- Tool dispatch now runs off the UI event loop so blocking tools such as
  `run_shell` do not freeze the draft editor.

### Added
- Folded one-line tool-result summaries with exact line counts and approximate
  token counts, plus `/tool` and `/tool N` commands for listing or expanding
  stored tool results.
- Regression coverage for folded tool summaries, loaded-session tool
  summaries, readline-style draft editing helpers, queued input while the
  agent is busy, CLI terminal-chat launch, and non-blocking tool dispatch.

  Total suite: 77 tests.

See [docs/PRDs/v0.6.8-append-only-terminal-chat.md](docs/PRDs/v0.6.8-append-only-terminal-chat.md).

## [v0.6.7] - 2026-05-24

### Changed
- The LLM adapter now uses a final-only Chat Completions request with
  `stream=False`, emitting one final assistant event instead of token deltas.
- Main chat remains in terminal-owned mouse mode with Textual mouse reporting
  disabled, preserving native selection and copy/paste behavior.
- IHEP Anthropic models keep using the OpenAI SDK but send system prompt text
  through SDK `extra_body` instead of an outbound `system` role message.
- IHEP Anthropic models now report `tools:unsupported` and omit OpenAI
  function-tool schemas that the gateway currently rejects.
- Project guidance now reflects the existing `agent_loop.py` / `llm.py`
  responsibilities instead of the older monolithic agent wording.

### Fixed
- Assistant replies now render reliably for final-only model responses.
- The IHEP Claude gateway no longer receives rejected OpenAI-style `system`
  role or `type:function` tool payloads.

### Added
- Regression coverage for final-only OpenAI SDK requests, IHEP Anthropic
  system prompt forwarding, unsupported-tool status, and preserved tool
  continuation for compatible models.

  Total suite: 70 tests.

See [docs/PRDs/v0.6.7-terminal-mouse-final-llm.md](docs/PRDs/v0.6.7-terminal-mouse-final-llm.md).

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
