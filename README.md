# neutrix

A practical terminal coding agent for OpenAI-compatible LLM APIs — DeepSeek, GLM,
OpenAI, and Claude models through the IHEP gateway (or any direct provider), all
driven by a single OpenAI SDK client. It drops into a Claude-Code project and
inherits its config (read-only); it never writes into `~/.claude`.

- **Append-only terminal chat** — normal scrollback, not a fullscreen app; live
  token-by-token streaming with a status-bar heartbeat.
- **Claude-Code-shaped tools** — `Read` · `Edit` · `Write` · `Grep` · `Glob` ·
  `Bash`, plus task tracking (`TaskCreate`/`TaskUpdate`/`TaskList`) and sub-agents
  (`Agent`). Folded one-line tool results, expandable with `/tool N`.
- **Project context** — auto-discovers `CLAUDE.md`, `.claude/skills`,
  `.claude/commands`, and `.claude/agents` (read-only), and `@path` file mentions.
- **Permissions** — a non-blocking Executor safety layer blocks destructive shell
  commands by default; `/allow` toggles allow-all.
- **Two model slots** — `fast` and `strong`, switch in-chat with `/fast` /
  `/strong`.
- **Autonomous `/goal`** — drive a task across turns until done; type to take over.
- **Cost / usage / timing** — `/cost` shows session tokens (`hit · miss · out`),
  dollar cost, and API/tool/wall time; prices live in your config YAML.
- **Session resume** — every turn auto-persists to
  `~/.cache/neutrix/sessions/` (never `~/.claude`); `--continue` / `--resume`
  restore it. `/save` · `/load` export a single JSON file.
- **Context management** — `/compact` (summary or mechanical) and `/rewind`
  (round-safe, Up/Down recalls dropped prompts).
- One YAML config at `~/.config/neutrix/config.yaml`, no env vars. pip-installable,
  single `neutrix` CLI.

## Install

```bash
pip install -e .            # from a clone
```

## Configure

Run `neutrix` once — it writes a starter config to
`~/.config/neutrix/config.yaml` and exits. Paste your IHEP gateway `api_key`,
then re-run (an empty key prints "fill your key … and re-run").

```yaml
providers:
  ihep:
    base_url: https://aiapi.ihep.ac.cn/apiv2/
    api_key: ""        # <- paste your key here

fast:
  provider: ihep
  model: anthropic/claude-haiku-4-5
strong:
  provider: ihep
  model: anthropic/claude-opus-4-7

# Cost display (USD per million tokens; `currency` is just a symbol — set "¥" for
# CNY). A model not listed renders "(cost unknown)"; tokens still show.
pricing:
  currency: "$"
  models:
    anthropic/claude-haiku-4-5: { input: 1.0, output: 5.0, cache_read: 0.10, cache_write: 1.25 }
    anthropic/claude-opus-4-7:  { input: 5.0, output: 25.0, cache_read: 0.50, cache_write: 6.25 }
```

Rebind a slot by editing its `provider` / `model`; add a provider under
`providers:`. Edit the `pricing:` numbers to match your billing.

## Run

```bash
neutrix                  # open terminal chat (strong slot preferred)
neutrix --continue       # resume the most recent session in this directory
neutrix --resume <id>    # resume a specific session (id prefix ok)
neutrix --load PATH      # load an exported session JSON
neutrix --no-tools       # disable tool calling
neutrix --version
```

### Slash commands

`/help` lists them all. The common ones:

| Command | Meaning |
|---|---|
| `/fast` · `/strong` · `/model` | switch slot · show current slot/model |
| `/cost` | session tokens, cost, timing |
| `/compact` · `/rewind [N]` | trim history · drop the last N user turns (round-safe) |
| `/save [PATH]` · `/load PATH` | export · import a session JSON |
| `/tools [on\|off]` · `/tool [N]` | list/toggle tools · expand a folded result |
| `/allow` · `/status` | toggle allow-all permissions · show slot/tools/msgs |
| `/clear` · `/quit` | reset conversation · exit |

`Enter` submits; `Ctrl+J` / `Alt+Enter` inserts a newline; readline keys
(`Ctrl+A`, `Ctrl+K`, …) work in the draft editor. Prompts typed while the
assistant is responding queue and run in order. `Esc` cancels an in-flight turn
(keeping partial output); `Ctrl+C` / EOF exits.

## Documentation

- [`docs/architecture.html`](docs/architecture.html) — how it works (component
  graph + the cancel / rewind / save-load / goal flows).
- [`docs/roadmap.html`](docs/roadmap.html) — where it's going.
- [`CHANGELOG.md`](CHANGELOG.md) · per-release design rationale under
  [`docs/PRDs/`](docs/PRDs/).

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check src tests
```

Project conventions live in [`CLAUDE.md`](CLAUDE.md); the release workflow is in
[`.claude/rules/release-workflow.md`](.claude/rules/release-workflow.md).

## License

MIT
