# Changelog

All notable changes to neutrix. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/) with the pre-1.0 rule that minor
bumps may include breaking changes (see [release-workflow rule](.claude/rules/release-workflow.md)).

## [v0.10.3] — 2026-05-29

### Changed
- **The store is now the single state holder; the view holds none.** The
  folded-tool-result tray moved off `TerminalChat` (`_tool_records`) into
  `ChatStore.folded_tool_results` (with `add_folded_tool_result()`, store-assigned
  index, wiped on `reset()`). `ToolRecord` is now **pure data in
  `neutrix.store`**; its summary-rendering moved to module functions in the view
  (`tool_record_summary[_parts]`). This de-risks the v0.10.4 Advisor's
  store-only-mutator model — no view-private state for a second actor to miss.

### Added
- **`tests/fake_view.py` — a `FakeView` swap-test.** A renderer that reads the
  whole user surface from `ChatStore` alone; `tests/test_fake_view.py` drives the
  scenario set and AST-asserts `fake_view.py` imports `neutrix` **only** via
  `neutrix.store` — the enforceable "a view needs nothing but the store" proof.

### Notes
- The skeleton's module-level "`terminal_chat.py` imports only `store`" check was
  **infeasible** (that module hosts the orchestrator that drives
  `ContextManager`); the boundary proof is scoped to `FakeView` instead (split #1).
  Pure refactor — no transcript-behavior change.

See [docs/PRDs/v0.10.3-tuiview-purify.md](docs/PRDs/v0.10.3-tuiview-purify.md)
and [docs/splits/v0.10.3-tuiview-purify.html](docs/splits/v0.10.3-tuiview-purify.html).

## [v0.10.2] — 2026-05-29

### Added
- **Visibility parity — the transcript now shows the one channel it hid: tool
  schemas.** A folded `[tools] M schemas · folded · K B` block renders at session
  start (when tools are enabled); `/show tools` re-prints the schema list below.
  This closes the only genuine parity gap — ground-truthing the renderer showed
  the system prompt, reminders, and tool results were already surfaced.
- **`.claude/rules/visibility-parity.md`** — the normative rule (every LLM-bound
  channel renders, differing only by fold), worded for the append-only
  expand-by-append reality.
- **`LLMRoundBundle` + `ContextManager.round_bundle()`** — a frozen snapshot of
  the channels sent to the LLM, the single source of truth the invariant test
  enumerates.
- **`tests/test_visibility_parity.py`** — asserts every populated channel of the
  round bundle produces ≥1 render call, iterating the bundle's fields
  dynamically so a future hidden channel trips it.

### Changed
- A **long system prompt now folds** to `[system] · folded · N B` (>200 B);
  short prompts (incl. the default) stay inline. `/show system` expands.
- A subagent (`Agent` tool) result renders with a **`[subagent]` label** instead
  of the generic `tool_result` keyword — discharging v0.10.0's deferred subagent
  rendering with a label (the result already folds/expands via the tool-result
  path; no new fold machinery).

### Non-goal (recorded)
- **No in-place `Ctrl+E` fold/expand toggle** — infeasible in append-only
  scrollback (can't collapse a printed block). Expansion is by re-printing below
  (`/show`, `/tool N`); no new key binding.

See [docs/PRDs/v0.10.2-visibility-parity.md](docs/PRDs/v0.10.2-visibility-parity.md)
and [docs/splits/v0.10.2-visibility-parity.html](docs/splits/v0.10.2-visibility-parity.html).

## [v0.10.1] — 2026-05-29

### Changed
- **Streaming is back (`stream=True`), closing the cancel-as-steer arc.** The
  LLM adapter once again streams token deltas (restored from v0.9.2 and merged
  with v0.9.3's tool-result pairing layer): `stream_response` yields
  `LLMEvent("token", str)` per content delta, rebuilds `tool_calls` index-keyed
  across deltas, and emits one terminal `assistant` event. `stop()` closes the
  active `AsyncStream` (eager teardown). The key win — **Esc mid-response now
  keeps the partial assistant text**: `_call_llm` stashes the streamed bytes and
  `_do_cancel` commits them as a partial assistant turn *before* the
  `[interrupted by user]` marker, so a steer carries the prior assistant intent
  (where before it kept nothing). Hard LLM errors discard the partial (one
  `[LLM error]` message); user-cancel and timeout keep it.

### Non-goal (recorded)
- **Live token-by-token transcript rendering is deferred to v1.x.** The
  append-only scrollback (v0.6.8) can't grow a line in place; the v0.9.4/v0.9.8
  blink heartbeat remains the during-LLM liveness signal. A bounded preview in
  the above-input region is a clean v1.x add. `terminal_chat.py` is unchanged.

See [docs/PRDs/v0.10.1-streaming-steer.md](docs/PRDs/v0.10.1-streaming-steer.md)
and [docs/splits/v0.10.1-streaming-steer.html](docs/splits/v0.10.1-streaming-steer.html).

## [v0.10.0] — 2026-05-29

### Added
- **Subagent framework — the `Agent` tool.** The LLM can now dispatch a
  fresh-context worker — `Agent(description, prompt, subagent_type="general-purpose")`
  — that runs its own LLM/tool loop to completion and returns only its final
  text, so the controller's context grows by one `tool_result` instead of by
  the whole sub-task (the structural answer to context-explode). New
  `src/neutrix/subagent.py` (`run_subagent` + `SubagentResult`) reuses
  `ContextManager` for the loop; the worker gets its own `ChatStore`,
  `Executor`, and a tool allowlist that omits `Agent` (recursion is
  structurally impossible, with a `contextvar` backstop). Runs to completion
  behind the parent's heartbeat; final text capped at 100k chars; bounded by a
  25-round runaway cap. Esc stops the worker's burn via a cross-loop
  `threading.Event`; the `[cancelled by user]` marker reaches the parent
  through the existing v0.9.3 cancel path. First release on the march to v1.0.
  (12 split-point decisions, all autonomous under a delegating `/goal` —
  see [docs/splits/v0.10.0-subagent.html](docs/splits/v0.10.0-subagent.html).)

### Changed
- `ContextManager` gains two opt-in fields used by subagents (both default to
  the prior main-chat behavior): `tool_names` (scope which tool schemas are
  advertised) and `max_rounds` (cap the drive loop). `Executor` carries the
  active `slot` and registers cross-loop cancel tokens; `get_schemas()` and
  `dispatch()` accept an optional tool-name scope / `slot`.

See [docs/PRDs/v0.10.0-subagent.md](docs/PRDs/v0.10.0-subagent.md).

## [v0.9.8] — 2026-05-28

### Changed
- **Heartbeat liveness pulse: blink, not brightness.** The ``●`` above the
  input now winks on/off (present on even ticks, a same-width blank on odd)
  every 600 ms instead of fading through a truecolor gray gradient. On a
  256-color terminal the old fade quantized onto only ~22 distinct grays
  (the xterm 24-step ramp ``#080808..#eeeeee``), so it banded and read as
  ~5 Hz regardless of refresh rate — the palette, not the clock, was the
  ceiling, which is why v0.9.5's jump to 120 Hz didn't help. A 2-state
  presence toggle has nothing to interpolate, so it stays smooth on every
  terminal. Follows Claude Code's tool-use loader (``ToolUseLoader`` +
  ``useBlink``, ``BLINK_INTERVAL_MS = 600``). The blink phase resets to a
  visible dot on each IDLE→busy transition, so a turn never opens on a
  blank dot. Stalled ``AWAITING_LLM`` winks **red** (``LLM (stalled)``) via
  a discrete color swap. (6 split-point decisions, all converging on CC's
  tool-loader wink —
  see [docs/splits/v0.9.8-liveness-motion.html](docs/splits/v0.9.8-liveness-motion.html).)
- **Heartbeat cadence reverts from 120 Hz to a 600 ms toggle**
  (``HEARTBEAT_BLINK_INTERVAL_MS``), strict period (no jitter) — roughly
  70× less heartbeat CPU.

### Removed
- The truecolor brightness machinery: ``HEARTBEAT_BREATH_PERIOD_S``,
  ``HEARTBEAT_REFRESH_HZ``, ``HEARTBEAT_CYCLE_FRAMES``,
  ``HEARTBEAT_TICK_MS``, ``HEARTBEAT_JITTER_RATIO``, the gray/red RGB
  anchors, ``_build_brightness_cycle`` with its two precomputed cycle
  tables, and ``jittered_sleep``.

See [docs/PRDs/v0.9.8-liveness-motion.md](docs/PRDs/v0.9.8-liveness-motion.md).

## [v0.9.7] — 2026-05-28

### Added
- **Rewind / branch from a prior turn.** `ContextManager.rewind_to(index)`
  destructively trims `messages` to an earlier point (Follow CC), rebuilds
  the typed store, and preserves tasks — mirroring `compact()`. Snaps the cut
  to a tool-round boundary so a rewind never leaves a dangling
  `tool_use`/`tool_result`.
- **`/rewind [N]`** — drops the last N user turns (default 1), IDLE-only, and
  prints a forward notice `↶ rewound N turns (back to turn K)`. The dropped
  turns stay in scrollback as history (the append-only renderer can't un-print
  them — v0.9.7 split #7, a scope change from the skeleton's retroactive
  `[rewound]` strikethrough).
- **Up/Down recall.** On an empty draft, `Up` walks prior user turns into the
  buffer (`Down` walks forward; `Esc` clears recall when active, else cancels).
  Decoupled from rewind (split #2): recall only edits the draft — submitting
  appends a new turn, it does not truncate. `RecallState` is the pure, tested
  cursor.

(10 split-point decisions, autonomous under a `/goal`, recorded in
[docs/splits/v0.9.7-rewind.html](docs/splits/v0.9.7-rewind.html); 4 flagged for
user review. See [docs/PRDs/v0.9.7-rewind.md](docs/PRDs/v0.9.7-rewind.md).)

## [v0.9.6] — 2026-05-28

### Added
- **Emergency mechanical ``/compact``.** A slash command that drops the
  oldest ~50 % of ``ContextManager.messages`` with no LLM call — the
  bridge until smart, summary-based compaction lands at v0.10.5. The
  cut preserves the leading ``system`` prefix, snaps *forward* past any
  ``role:tool`` message so the kept tail never begins on an orphan
  ``tool_result`` (which ``llm.py``'s pairing layer does not repair),
  and inserts one ``role:user`` placeholder
  ``<system-compact>{N} earlier turns removed by /compact (no
  summary)</system-compact>`` between the prefix and the kept tail.
  Prints a dim ``compacted N turns (~K tokens dropped)`` notice and
  suppresses re-printing the already-visible tail. ``/compact`` takes
  no arguments, refuses while the assistant is working, and is a clean
  no-op when the conversation is too short to drop a tool-round-safe
  slice. Re-running further-halves; it never stacks placeholders.
  (12 split-point decisions, all autonomous;
  1 Follow CC / 8 Alternative / 3 No CC analog —
  see [docs/splits/v0.9.6-emergency-compact.html](docs/splits/v0.9.6-emergency-compact.html).)
- **``neutrix.compaction`` module.** Pure
  ``compact_messages(messages, *, keep_ratio=0.5) ->
  (new_messages, CompactionOutcome)`` plus an ``is_compact_marker``
  helper (unused by the v0.9.6 renderer; provided for v0.10.2
  visibility-parity, mirroring ``is_task_reminder``). The cut
  computation is isolated so v0.10.5 can reuse it and swap the
  placeholder step for a summarizer.
- **``ContextManager.compact()``.** Async direct method (mirrors
  ``cancel()``) that compacts both ``messages`` and the typed
  ``ChatStore`` so ``/save`` persists the compacted state, while
  **preserving tasks** (compaction trims context, not the live work
  list). The same method is the seam v0.10.5's auto-trigger watchdog
  will call.

### Changed
- **``/compact`` joins the busy-guard set** (``fast``/``strong``/
  ``save``/``load``/``onboard``/``compact``) and the ``/help`` listing.

See [docs/PRDs/v0.9.6-emergency-compact.md](docs/PRDs/v0.9.6-emergency-compact.md).

## [v0.9.5] — 2026-05-28

### Added
- **LLM stall hint (single-knob, derived from the timeout).** While
  ``ContextManager`` is parked in ``AWAITING_LLM`` and no response
  has arrived after ``stall_threshold_for(slot.llm_timeout_s)`` —
  ``max(10.0, llm_timeout_s / 6)``, ≈50 s at the 300 s default — the
  heartbeat glyph palette swaps from the v0.9.4 grayscale gradient to
  a parallel red gradient (``HEARTBEAT_STALLED_CYCLE``, anchors
  ``rgb(60,0,0) → rgb(255,60,60)``, same 40-frame raised-cosine
  breath) and the label flips ``● LLM`` → ``● LLM (stalled)``. UI
  only — no abort. Deriving the stall from the hard timeout means
  raising ``llm_timeout_s`` for a slow model pushes the hint out with
  it, so it stops flickering on healthy-but-slow calls. Renderer-side
  math on a new ``ContextManager.last_progress_at`` field; suppressed
  during ``AWAITING_EXECUTOR`` (tool runs have their own latency
  budget).
- **LLM hard timeout (300 s default).** A background watchdog task
  spawned on every ``AWAITING_LLM`` entry sleeps ``slot.llm_timeout_s``
  then fires ``cm.cancel(reason='timeout')``. The cancel flows through
  the v0.9.3 ``llm.stop()`` machinery; the drive loop's
  ``CancelledError`` catch appends a fresh assistant message
  ``[LLM timeout after Ns]`` and returns CM to ``IDLE``. A
  ``logger.error("LLM call timed out after {}s", elapsed)`` line
  records the event. The 300 s default replaces the SDK's 600 s
  (10 min) default silent hang while leaving headroom for slow hosted
  reasoning models such as deepseek-v4-pro; per-slot override tunes it
  further.
- **Per-slot ``llm_timeout_s`` YAML field.** Added to the slot
  schema; default ``300.0`` when absent. A fast hosted slot can be
  tightened; a slow local model can be given more headroom.
  ``llm_timeout_s: 600.0`` style is parsed as float; non-numeric or
  non-positive values raise ``ConfigError`` at slot resolution.
- **``CancelReason`` literal + ``cancel(reason=...)`` API.** The
  v0.9.3 ``cm.cancel()`` gained a ``reason: Literal['user',
  'timeout']`` kwarg (default ``'user'``); ``CancelEvent`` carries
  the same field. ``reason='user'`` keeps the v0.9.3 cancel-as-steer
  ``[interrupted by user]`` marker; ``reason='timeout'`` skips the
  user marker and lets the drive loop's ``_finalize_cancel`` append
  the timeout assistant message instead.

### Changed
- **``format_heartbeat`` signature.** Added keyword-only
  ``last_progress_at: float | None = None`` and
  ``stall_threshold_s: float`` (default the
  ``HEARTBEAT_STALL_FLOOR_S`` floor). Existing v0.9.4 callers (and
  the v0.9.4 unit tests) work unchanged because the kwargs default
  to "stall hint off." The live caller passes
  ``stall_threshold_for(slot.llm_timeout_s)``.
- **``ContextManager._do_cancel``.** The
  ``_append_user_message(INTERRUPTED_BY_USER_MARKER)`` step is now
  gated on ``self.cancel_reason == 'user'``. Reads/writes to
  ``cancel_reason`` are ordered so a no-op second cancel cannot
  clobber the reason set by the first.
- **Heartbeat refresh 10 Hz → 120 Hz (bundled polish).** The
  breathing glyph now updates 120×/s (``HEARTBEAT_REFRESH_HZ = 120``,
  ``HEARTBEAT_CYCLE_FRAMES = 480``, ~8.33 ms/frame) instead of the
  v0.9.4 10×/s. 10 fps sat below the smooth-motion perception floor,
  so the fade read as discrete steps; 120 Hz makes it a continuous
  glow. The 4 s breath period (resting-calm ~15 BPM) and the
  raised-cosine curve are unchanged. Unrelated to the LLM error
  surface — rides this release per the forward-only versioning rule.

### Notes
- **No retry after a watchdog timeout — by design.** Transient HTTP
  failures (429 / 5xx / 529) are retried by the openai SDK
  (``max_retries=2``) *inside* the timeout envelope, so the cases
  where a retry helps are already covered invisibly. The watchdog
  cancel raises ``asyncio.CancelledError``, which is outside the
  SDK's retry path; a timeout means "the full budget elapsed,"
  which on a generous 300 s budget signals a real hang, not a
  transient blip — auto-retrying would just risk doubling the wait.
  The user resends (optionally after ``/fast`` / ``/strong``).
- The deriving of the stall threshold from ``llm_timeout_s`` keeps
  the SDK's in-envelope retry chain (a 5xx backoff cycle, typically
  <15 s) well under the stall threshold (~50 s at the default), so
  retries no longer paint a transient red hint. The clean
  idle-vs-working distinction still lands with v0.10.1
  streaming-per-chunk ``last_progress_at`` bumping.

See [docs/PRDs/v0.9.5-llm-error-surface.md](docs/PRDs/v0.9.5-llm-error-surface.md)
and [docs/splits/v0.9.5-llm-error-surface.html](docs/splits/v0.9.5-llm-error-surface.html)
(12 split-point decisions).

## [v0.9.4] — 2026-05-27

### Added
- **Heartbeat liveness indicator.** A single ``●`` glyph + short
  label renders at the top of the message-area while the
  ``ContextManager`` is in ``AWAITING_LLM`` (``● LLM``),
  ``AWAITING_EXECUTOR`` (``● tool: {name}`` — name from the head of
  ``store.pending_tool_calls``), or ``CANCELLING``
  (``● cancelling…``). The dot breathes (trough→peak→trough) on a
  **40-frame truecolor gradient at ~100 ms/tick with ±10% jitter,
  giving a 4.0-second cycle** (~15 BPM, middle of the human
  resting-calm respiratory range, 12-20 BPM). The smooth gradient
  interpolates through ~40 distinct hex gray shades between
  ``rgb(60,60,60)`` and ``rgb(255,255,255)`` along a raised-cosine
  curve; the label stays bright. Disappears (renders nothing) when
  state returns to ``IDLE``. Pure renderer change: reads
  ``ContextManager.state`` and ``store.pending_tool_calls``; no new
  store fields. New module-level helpers ``format_heartbeat``,
  ``heartbeat_loop``, and ``jittered_sleep`` in
  ``neutrix.terminal_chat`` are unit-testable in isolation. Stack
  order above the input is now: heartbeat → task panel → queued
  user messages → quit hint → input cursor. See
  [docs/PRDs/v0.9.4-heartbeat.md](docs/PRDs/v0.9.4-heartbeat.md)
  and [docs/splits/v0.9.4-heartbeat.html](docs/splits/v0.9.4-heartbeat.html)
  (14 split-point decisions — 11 original + 3 Phase-2 reopen on
  cycle period, smoothing method, and tick jitter).

### Changed
- **``TaskCreate`` description rewritten to push full-upfront task
  planning.** Bundled fix per the forward-only versioning rule:
  during live use the LLM tended to call ``TaskCreate`` for one
  task at a time instead of capturing the user's full multi-step
  plan upfront. The tool description now opens with *"Use this
  tool proactively to capture the user's full multi-step plan as
  separate tasks, not piece by piece"* and emphasizes
  *"immediately capture every distinct step the request implies as
  its own task, BEFORE starting work on any of them"*. Description
  only — signature, parameters, and lifecycle clauses unchanged.

See [docs/PRDs/v0.9.4-heartbeat.md](docs/PRDs/v0.9.4-heartbeat.md).

## [v0.9.3] — 2026-05-27

### Changed
- **Cancel is now "steer", not "rollback".** Pressing Esc (or first
  Ctrl+C while busy) keeps the interrupted turn in history and
  appends a ``role:user`` message with content
  ``[interrupted by user]`` so the next LLM call sees the prior
  user turn, the partial assistant turn (if any), and the orphan
  ``tool_calls`` — enough context to be steered ("instead, just
  ``ls``") rather than blindly retrying. Follows Claude Code's
  ``createUserInterruptionMessage`` semantic. The v0.9.2
  rollback-to-snapshot behavior is gone; ``Agent.rollback_to`` is
  removed.
- **``Controller`` is substantially reshaped into ``ContextManager``,
  not renamed.** v0.9.2's ``Controller`` was a ~50-line broadcaster
  (``send`` wrapping ``stream_turn`` in a task; ``cancel`` calling
  ``llm.stop`` / ``executor.cancel`` / ``task.cancel``). v0.9.3's
  ``ContextManager`` absorbs the v0.9.2 ``Agent``'s message
  ownership, runs an explicit
  IDLE / AWAITING_LLM / AWAITING_EXECUTOR / CANCELLING state
  machine, owns system-reminder injection, and is the SOLE mutator
  of ``messages`` and ``ChatStore``. The v0.9.2 class is a small
  subset of the v0.9.3 class; framing this as a "rename" would
  understate the role expansion.
- **``Agent`` class dissolved into ``ContextManager``.**
  ``Agent.stream_reply`` (the v0.6.x-era async generator that drove
  the LLM/tool loop) is dismantled and the loop becomes the CM
  state machine. ``src/neutrix/agent.py`` and
  ``src/neutrix/agent_loop.py`` are removed. Helpers
  (``DEFAULT_SYSTEM_PROMPT``, ``build_task_reminder``,
  ``is_task_reminder``, ``format_reminder_notice``,
  ``assistant_turns_since_*``) move to
  ``src/neutrix/context_manager.py``.
- **Streaming is disabled.** ``OpenAIChatLLM.stream_response``
  reverts to ``stream=False`` (rolls back v0.9.2's switch). One
  awaited Chat Completions call returns one
  ``LLMEvent("assistant", LLMResponse(...))``. Token-by-token
  rendering is gone for this release; full-response wait. Streaming
  re-enables in a later PRD with the CC-aligned "keep partial
  text" semantic.
- **``OpenAIChatLLM.stop()`` cancels the awaiting create task**
  instead of closing the SDK's ``AsyncStream``. Same broadcast
  contract; different mechanism aligned with ``stream=False``.
- **UI subscribes to ``ChatStore.changes()`` for the message
  transcript**, not just for the queue/task panel. The v0.9.2
  ``TerminalChat._mirror_new_agent_messages`` and the
  ``AgentEvent`` dispatch path are removed; the renderer walks new
  ``store.messages`` records as they arrive.
- **View-side dim ``interrupted`` notice is removed.** Following
  Claude Code, the rendered ``[interrupted by user]`` message in
  the transcript IS the affordance.
- **``Executor`` surface narrowed.** No more
  ``agent.messages`` mutation, no rollback snapshot. New event
  protocol: ``ToolEvent("tool_started"|"tool_finished", {...})``
  emitted by ``Executor.dispatch_all(tool_calls)``.
  ``Executor.cancel()`` tree-kills the Popen pool only.

### Added
- ``ContextManager`` (``src/neutrix/context_manager.py``) — state
  machine + ``handle_event(event)`` async surface +
  ``cancel() -> bool`` sync convenience for key bindings. Event
  types: ``UserMessageEvent``, ``CancelEvent``, ``SlotSwitchEvent``,
  ``ClearEvent``, ``ReplaceHistoryEvent``.
- ``_ensure_tool_result_pairing(messages)`` in
  ``src/neutrix/llm.py`` — pure transform on the outgoing message
  list. Dedups ``role:tool`` messages by ``tool_call_id`` (first
  wins) and synthesizes a ``role:tool`` placeholder for any orphan
  ``tool_use`` in the latest assistant message. Synthetic content
  is ``[cancelled by user]`` if ``[interrupted by user]`` appears
  after the orphan, otherwise ``[tool result missing]``. Runs at
  API-send time on a copy of the payload — does not mutate
  ``messages`` (preserves the CM-as-sole-mutator rule).
- ``/clear`` and ``/load`` now cancel an in-flight turn first
  (waiting for the drive task to unwind), then reset / replace
  history.

### Removed
- ``src/neutrix/agent.py`` + ``src/neutrix/agent_loop.py``
  (``Agent`` class folded into ``ContextManager``).
- ``src/neutrix/controller.py`` (``Controller.send`` / ``cancel``
  folded into ``ContextManager.handle_event`` / ``cancel``).
- ``src/neutrix/tui.py`` and ``tests/test_tui.py`` — the legacy
  Textual app, dormant for several releases and a standing v0.9.x
  non-goal. Removed rather than rewired through the new
  architecture; recoverable from git history if ever needed.
- ``Agent.rollback_to`` (no callers after cancel-as-steer; can be
  cleanly reintroduced on ``ContextManager.messages`` if ``/undo``
  ever materializes).
- The v0.9.2 dim view-side ``interrupted`` notice.

### Deviations from the PRD worth knowing about
- The PRD says CM owns the queued-input buffer; the implementation
  keeps the queue in ``ChatStore.queued_user_messages`` (mutated by
  the UI's ``_input_loop`` / ``_worker_loop``) so the existing
  queue display above the input keeps working unchanged. CM still
  drains naturally — each user message dequeued by the worker
  becomes one ``UserMessageEvent`` to CM.

### New split point surfaced mid-implementation
- **Queued messages on Esc cancel.** A queued message (typed while
  the assistant was busy) is sent as the next user turn AFTER the
  ``[interrupted by user]`` marker. Follows Claude Code's "Path B"
  (active-response cancel) for both cases. Optimizes for "I
  cancelled to steer with what I already typed." CC's "Path A" —
  popping queued commands back to the editable input on
  idle-state Esc — is deferred (would require injecting text into
  the ``prompt_toolkit`` ``DraftReader`` buffer the input loop
  currently owns). Added as split point #12 in the splits HTML
  during Phase 2 mid-implementation discovery.

See [docs/PRDs/v0.9.3-cancel-steer.md](docs/PRDs/v0.9.3-cancel-steer.md)
and [docs/splits/v0.9.3-cancel-steer.html](docs/splits/v0.9.3-cancel-steer.html).

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
