# neutrix

A simple multi-provider TUI agent for DeepSeek, GLM, and Claude (via the
IHEP gateway), all driven through a single OpenAI SDK client.

- **Textual** TUI with streaming responses
- Two named model slots — **`fast`** and **`strong`** — switch in-TUI with `/fast` / `/strong`
- OpenAI-style **tool calling** (built-in: `read_file`, `write_file`, `list_dir`, `run_shell`)
- Save / load conversations as JSON
- One YAML config at `~/.config/neutrix/config.yaml`, no env vars
- pip-installable, single `neutrix` CLI

## Install

```bash
pip install -e .            # from a clone
# or, once published:
pip install neutrix
```

## Configure

Run `neutrix` once — it writes a template to `~/.config/neutrix/config.yaml`
and exits. Open the file and paste at least one provider's `api_key`, then
re-run.

```yaml
providers:
  ihep:
    base_url: https://aiapi.ihep.ac.cn/apiv2/
    api_key: ""        # paste here
  deepseek:
    base_url: https://api.deepseek.com
    api_key: ""
  glm:
    base_url: https://open.bigmodel.cn/api/paas/v4/
    api_key: ""

fast:
  provider: ihep
  model: anthropic/claude-haiku-4-5

strong:
  provider: ihep
  model: anthropic/claude-opus-4-7
```

Rebind a slot by editing its `provider` / `model` line. Add a new provider by
appending another entry under `providers:`.

## Run

```bash
neutrix                              # open TUI, fast slot active
neutrix --load sessions/last.json    # resume a saved session
neutrix --no-tools                   # disable tool calling
neutrix --version
```

### Slash commands in the TUI

| Command                  | Meaning                                       |
|--------------------------|-----------------------------------------------|
| `/help`                  | list commands                                 |
| `/fast`                  | switch to fast slot                           |
| `/strong`                | switch to strong slot                         |
| `/model`                 | show current slot/provider/model              |
| `/onboard`               | re-enter onboarding (manage keys / slots)     |
| `/save [PATH]`           | save session JSON (default `sessions/<ts>.json`) |
| `/load PATH`             | load session                                  |
| `/clear`                 | reset conversation                            |
| `/tools` / `/tools on\|off` | list or toggle tool calling                |
| `/quit`                  | exit                                          |

`Ctrl+C` quits. `Ctrl+L` clears the command notice line.

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check src tests
```

Project conventions live in [`CLAUDE.md`](CLAUDE.md); the release workflow is
in [`.claude/rules/release-workflow.md`](.claude/rules/release-workflow.md).

## License

MIT
