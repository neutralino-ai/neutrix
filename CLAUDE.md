# CLAUDE.md

Project guidance for any AI coding assistant working in this repo.

## Two principles override everything else

### SOLID — one module, one job

| Module | Single responsibility |
|---|---|
| `config.py` | Load + validate `~/.config/neutrix/config.yaml`. Nothing else. |
| `agent.py` | Compatibility facade for the public agent API. |
| `agent_loop.py` | Conversation state and tool-call continuation. |
| `llm.py` | One final OpenAI-compatible chat completion request. |
| `tools.py` | Built-in tools the model can call (read/write/list/shell). |
| `session.py` | JSON save/load of conversations. |
| `tui.py` | Textual UI — rendering, input, slash commands. |
| `cli.py` | Argparse entry point. Build agent, launch TUI, exit. |

Do not blur these. No YAML parsing in `agent.py`. No UI logic in `config.py`.
No agent state owned by `tui.py`. If a change needs to cross these lines,
the design is wrong, not the boundary.

### YAGNI — do not over-engineer

- No abstract base classes, no plugin systems, no registries with one entry.
- No "future extensibility" hooks. Add them when there is a second caller.
- No env-var fallback chains, no auto-discovery. One source of truth:
  `~/.config/neutrix/config.yaml`.
- No backwards-compat shims. Bump the version and break.
- No comments explaining WHAT (the code already says that). Only WHY,
  only when non-obvious.
- No half-finished implementations. If you cannot finish it, do not start it.

If a change needs three new abstractions to land cleanly, the change is wrong.

## Release workflow

Every change requires a PRD, a CHANGELOG entry, and a new SemVer tag.
See `.claude/rules/release-workflow.md` for the full rule.

## Style

- Use `loguru` for logs, never `print`.
- Use `ruff check` — no `flake8`, no `mypy`.
- No emojis in code or docs.
- Type hints on public surfaces; `from __future__ import annotations` at file top.
- Tests at `tests/test_*.py`, no-network smoke style.
