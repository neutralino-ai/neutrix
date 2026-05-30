# neutrix PRDs

Every neutrix release is gated by a PRD here (**Problem · Goal · Non-goals · Claude
Code reference · Design · Acceptance**) plus a split-point survey under
[`../splits/`](../splits/). The user-visible map is
[`../roadmap.html`](../roadmap.html); shipped releases are git tags
(`setuptools_scm` → `neutrix --version`).

---

## V2 goal — the North Star

> **neutrix v2.0 should be a very handy, practical terminal coding agent for real
> work — concretely, fast and autonomous enough to speed up OMILREC.**
> Not a complete Claude Code clone. *"Keep it simple, otherwise we never ship."*
> — user-directed, 2026-05-30.

The v2 path is deliberately lean. Build only what makes neutrix practical for real
coding, then lock:

| Release | Theme | Why it serves the goal |
|---|---|---|
| **v1.5.3** | Permission as an Executor-only safety layer *(in flight)* | Stops the agent hanging on every destructive command; deterministic deny, no human gate. |
| **v1.6.0** | `/goal` autonomous loop *(planned)* | Unattended multi-step runs — "optimize until the benchmark passes." The autonomy headline. |
| **v1.7.0** | Parallel tools + background Bash *(planned)* | Wall-clock speedup on multi-file work. |
| **v1.8.0** | Usage, cost & timing *(planned)* | Budget long autonomous runs (builds on v1.5.2 `llm_ms`/`tool_ms`). |
| **v2.0.0** | Lock — practical, reliable coding agent *(target)* | Seal the focused surface. |

**Cut from v2** (reopen only if real OMILREC use demands them): hooks · MCP ·
multimodal input · Anthropic-native backend + thinking · sandbox · cross-session
memory · background/scheduled/remote agents.
**Deferred** (useful, not blocking the lock): web fetch/search · LSP.

`.claude/`-compatibility stays **read-only**: neutrix discovers `CLAUDE.md`,
`.claude/skills`·`commands`·`agents`, `.claude/settings.json`; it **never writes into
`~/.claude`** (neutrix-owned state lives under `~/.cache/neutrix/`).

---

## Index

### v2 path — practical coding agent
- [v1.5.3](v1.5.3-permission-executor-only.md) — permission as an Executor-only safety layer *(in flight)*
- v1.6.0 — `/goal` autonomous loop *(planned — PRD TBD)*
- v1.7.0 — parallel tools + background Bash *(planned — PRD TBD)*
- v1.8.0 — usage, cost & timing *(planned — PRD TBD)*
- v2.0.0 — the v2 lock *(target — PRD at lock time, like [v1.0.0](v1.0.0-lock.md))*

### v1.x — the build-out toward v2 (shipped)
- [v1.5.2](v1.5.2-session-resume.md) — conversation resume (`--continue` / `/resume`)
- [v1.5.1](v1.5.1-prompt-no-hang.md) — interactive prompt can't hang the turn
- [v1.5.0](v1.5.0-status-bar.md) — LLM / Exec / Advisor status bar
- [v1.4.9](v1.4.9-turn-resilience.md) — no-progress LLM timeout
- [v1.4.8](v1.4.8-ask-user-question.md) — AskUserQuestion tool
- [v1.4.7](v1.4.7-streaming-render.md) — live streaming render
- [v1.4.0](v1.4.0-permissions.md) — tool permissions (auto / allow-all)
- [v1.3.0](v1.3.0-skills.md) — skills + slash-command framework + hot-reload
- [v1.2.0](v1.2.0-project-context.md) — project context (`CLAUDE.md` / `@` / `/init`)
- [v1.1.0](v1.1.0-coding-tools.md) — CC-shaped coding tools (Read/Edit/Write/Grep/Glob/Bash)

### v1.0 lock
- [v1.0.0](v1.0.0-lock.md) — sealed v1.0 surface

### v0.10.x — lifecycle + agent foundations (shipped)
- [v0.10.5](v0.10.5-compaction.md) — smart conversation compaction (v1.0 gate)
- [v0.10.4](v0.10.4-smart-advisor.md) — smart Advisor (third actor)
- [v0.10.3](v0.10.3-tuiview-purify.md) — TUIView as pure renderer + FakeView swap-test
- [v0.10.2](v0.10.2-visibility-parity.md) — visibility parity
- [v0.10.1](v0.10.1-streaming-steer.md) — streaming + cancel-as-steer
- [v0.10.0](v0.10.0-subagent.md) — subagent framework (`Agent` tool)

### v0.9.x — cancellation, state machine, liveness (shipped)
- [v0.9.8](v0.9.8-liveness-motion.md) · [v0.9.7](v0.9.7-rewind.md) · [v0.9.6](v0.9.6-emergency-compact.md) · [v0.9.5](v0.9.5-llm-error-surface.md) · [v0.9.4](v0.9.4-heartbeat.md) · [v0.9.3](v0.9.3-cancel-steer.md) · [v0.9.2](v0.9.2-cancellation.md) · [v0.9.1](v0.9.1-keyboard.md) · [v0.9.0](v0.9.0-lifecycle-events.md)

### v0.7–v0.8 — store + tasks (shipped)
- [v0.8.2](v0.8.2-tool-keyword-colors.md) · [v0.8.1](v0.8.1-tasks-visible-and-auto-continue.md) · [v0.8.0](v0.8.0-tasks.md) · [v0.7.0](v0.7.0-chatstore.md)

### v0.2–v0.6 — config, onboarding, terminal chat (shipped)
- Main chat: [v0.6.8](v0.6.8-append-only-terminal-chat.md) · [v0.6.7](v0.6.7-terminal-mouse-final-llm.md) · [v0.6.6 (workflow)](v0.6.6-agent-workflow-user-acceptance.md) · [v0.6.6 (blocks)](v0.6.6-main-chat-borderless-blocks.md) · [v0.6.5](v0.6.5-main-chat-spacing-retune.md) · [v0.6.4](v0.6.4-main-chat-compact-draft.md) · [v0.6.3](v0.6.3-main-onion-blocks.md) · [v0.6.2](v0.6.2-main-visible-block-list.md) · [v0.6.1](v0.6.1-onboard-key-submit-isolation.md) · [v0.6.0](v0.6.0-main-chat-polish.md)
- Onboarding: [v0.5.5](v0.5.5-onboard-key-mask-after-enter.md) · [v0.5.4](v0.5.4-onboard-enter-race-fix.md) · [v0.5.3](v0.5.3-onboard-key-edit-semantics.md) · [v0.5.2](v0.5.2-onboard-state-and-focus.md) · [v0.5.1](v0.5.1-onboard-row-and-quit.md) · [v0.5.0](v0.5.0-onboard-polish.md) · [v0.4.2](v0.4.2-deepseek-model-names.md) · [v0.4.1](v0.4.1-onboard-arrow-nav-fix.md) · [v0.4.0](v0.4.0-onboard-slash-command.md) · [v0.3.1](v0.3.1-onboard-arrow-nav.md) · [v0.3.0](v0.3.0-onboarding-tui.md)
- Config: [v0.2.0](v0.2.0-yaml-config.md)

---

*Reorganized 2026-05-30 around the practical-for-OMILREC V2 goal. Cut/deferred items
above are intentionally not on the v2 path; the roadmap's "Descoped" section is the
canonical list.*
