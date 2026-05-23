# Changelog

All notable changes to neutrix. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/) with the pre-1.0 rule that minor
bumps may include breaking changes (see [release-workflow rule](.claude/rules/release-workflow.md)).

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
