# neutrix

A simple multi-provider TUI agent for DeepSeek, GLM (Zhipu), and Claude — all
driven through a single OpenAI SDK client.

- **Textual** TUI with streaming responses
- Switch provider/model at runtime (`/model claude claude-opus-4-7`)
- OpenAI-style **tool calling** (built-in: `read_file`, `write_file`, `list_dir`, `run_shell`)
- Save / load conversations as JSON
- pip-installable, single `neutrix` CLI

## Install

```bash
pip install -e .            # from a clone
# or, once published:
pip install neutrix
```

## Configure

Copy `.env.example` to `.env` and fill in keys for the providers you use:

```dotenv
DEEPSEEK_API_KEY=sk-...
GLM_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...
NEUTRIX_PROVIDER=deepseek     # default on startup
```

Anthropic's OpenAI-compat layer is used for Claude, so the same SDK works for
all three providers — only `base_url`, `api_key`, and `model` differ.

| Provider  | Base URL                                  | Env var              |
|-----------|-------------------------------------------|----------------------|
| DeepSeek  | `https://api.deepseek.com`                | `DEEPSEEK_API_KEY`   |
| GLM       | `https://open.bigmodel.cn/api/paas/v4/`   | `GLM_API_KEY`        |
| Claude    | `https://api.anthropic.com/v1/`           | `ANTHROPIC_API_KEY`  |

## Run

```bash
neutrix                                  # default provider
neutrix -p claude -m claude-opus-4-7     # pick provider + model
neutrix --load sessions/last.json        # resume a saved session
neutrix --no-tools                       # disable tool calling
```

### Slash commands in the TUI

| Command                       | Meaning                                       |
|-------------------------------|-----------------------------------------------|
| `/help`                       | list commands                                 |
| `/model`                      | show current provider/model                   |
| `/model PROVIDER [MODEL]`     | switch                                        |
| `/save [PATH]`                | save session JSON (default `sessions/<ts>.json`) |
| `/load PATH`                  | load session                                  |
| `/clear`                      | reset conversation                            |
| `/tools` / `/tools on\|off`   | list or toggle tool calling                   |
| `/quit`                       | exit                                          |

`Ctrl+C` also quits.

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check src tests
```

## License

MIT
