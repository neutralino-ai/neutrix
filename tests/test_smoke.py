"""Smoke tests — no network calls."""
import json

import pytest

from neutrix import __version__, cli
from neutrix.config import (
    DEFAULT_CONFIG,
    SLOT_NAMES,
    Config,
    ConfigError,
    Slot,
    bootstrap_config,
    load_config,
    resolve_initial_slot,
)
from neutrix.tools import BUILTIN_TOOLS, dispatch, get_schemas


def test_version_string():
    assert isinstance(__version__, str)
    assert __version__


def test_cli_launches_append_only_terminal_chat(monkeypatch, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("ok")
    fast_slot = Slot(
        name="fast",
        provider="test",
        model="fast-model",
        base_url="https://example.test/v1",
        api_key="sk-test",
    )
    strong_slot = Slot(
        name="strong",
        provider="test",
        model="strong-model",
        base_url="https://example.test/v1",
        api_key="sk-test",
    )
    config = Config(
        providers={"test": {"base_url": fast_slot.base_url, "api_key": fast_slot.api_key}},
        slots={
            "fast": {"provider": "test", "model": fast_slot.model},
            "strong": {"provider": "test", "model": strong_slot.model},
        },
        path=path,
    )
    seen: dict[str, object] = {}

    class DummyChat:
        def __init__(self, *args, **kwargs):
            seen["init"] = (args, kwargs)

        async def _ask_user(self, spec):  # v1.4.8: cli wires executor.ask_user
            return None

        def run(self):
            seen["run"] = True

    monkeypatch.setattr(cli, "CONFIG_PATH", path)
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "resolve_initial_slot", lambda _config: (fast_slot, strong_slot))
    monkeypatch.setattr("neutrix.terminal_chat.TerminalChat", DummyChat)

    assert cli.main([]) == 0
    assert seen["run"] is True
    args, kwargs = seen["init"]
    assert args[0].slot is strong_slot
    assert kwargs["config"] is config
    assert kwargs["render_markdown"] is True


# ----- config ----------------------------------------------------------------


def test_bootstrap_writes_template(tmp_path):
    path = tmp_path / "config.yaml"
    written = bootstrap_config(path)
    assert written == path
    assert path.read_text() == DEFAULT_CONFIG
    assert "anthropic/claude-opus-4-7" in path.read_text()


def test_load_config_missing(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_load_config_resolves_slot(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """\
providers:
  ihep:
    base_url: https://aiapi.ihep.ac.cn/apiv2/
    api_key: sk-test-123
fast:
  provider: ihep
  model: anthropic/claude-haiku-4-5
strong:
  provider: ihep
  model: anthropic/claude-opus-4-7
"""
    )
    cfg = load_config(path)
    fast = cfg.slot("fast")
    assert fast.name == "fast"
    assert fast.provider == "ihep"
    assert fast.model == "anthropic/claude-haiku-4-5"
    assert fast.api_key == "sk-test-123"
    assert fast.base_url.endswith("/apiv2/")


def test_slot_missing_api_key(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """\
providers:
  ihep:
    base_url: https://aiapi.ihep.ac.cn/apiv2/
    api_key: ""
fast:
  provider: ihep
  model: anthropic/claude-haiku-4-5
strong:
  provider: ihep
  model: anthropic/claude-opus-4-7
"""
    )
    cfg = load_config(path)
    with pytest.raises(ConfigError, match="no api_key"):
        cfg.slot("fast")


def test_slot_unknown_provider(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """\
providers:
  ihep:
    base_url: https://x/
    api_key: k
fast:
  provider: ghost
  model: m
strong:
  provider: ihep
  model: m
"""
    )
    cfg = load_config(path)
    with pytest.raises(ConfigError, match="unknown provider"):
        cfg.slot("fast")


def test_slot_unknown_name(tmp_path):
    path = tmp_path / "config.yaml"
    bootstrap_config(path)
    cfg = load_config(path)
    with pytest.raises(ConfigError, match="unknown slot"):
        cfg.slot("medium")


def test_slot_names_constant():
    assert SLOT_NAMES == ("fast", "strong")


def test_resolve_initial_slot_one_works(tmp_path):
    """When fast has a key but strong doesn't, fast resolves and strong is None."""
    path = tmp_path / "config.yaml"
    path.write_text(
        """\
providers:
  ihep:
    base_url: https://aiapi.ihep.ac.cn/apiv2/
    api_key: sk-good
  deepseek:
    base_url: https://api.deepseek.com
    api_key: ""
fast:
  provider: ihep
  model: anthropic/claude-haiku-4-5
strong:
  provider: deepseek
  model: deepseek-v4-flash
"""
    )
    cfg = load_config(path)
    fast, strong = resolve_initial_slot(cfg)
    assert fast is not None
    assert fast.api_key == "sk-good"
    assert strong is None


# ----- tools -----------------------------------------------------------------


def test_tool_schemas_well_formed():
    schemas = get_schemas()
    assert len(schemas) == len(BUILTIN_TOOLS)
    for s in schemas:
        assert s["type"] == "function"
        fn = s["function"]
        assert fn["name"] in BUILTIN_TOOLS
        assert "description" in fn
        assert fn["parameters"]["type"] == "object"


def test_tool_dispatch_read_write(tmp_path):
    # v1.1.0: Write (new file) then Read (line-numbered cat -n output).
    target = tmp_path / "hello.txt"
    write_res = dispatch("Write", json.dumps({"path": str(target), "content": "hi"}))
    assert "OK" in write_res
    assert target.read_text() == "hi"
    read_res = dispatch("Read", json.dumps({"path": str(target)}))
    assert "hi" in read_res
    assert "1\t" in read_res  # cat -n style line number


def test_tool_dispatch_unknown():
    result = dispatch("nope", "{}")
    assert result.startswith("ERROR: unknown tool")


def test_tool_dispatch_bad_json():
    result = dispatch("Read", "{not json")
    assert result.startswith("ERROR: invalid JSON args")


def test_tool_dispatch_glob(tmp_path):
    # v1.1.0: list_dir is gone; Glob finds files by pattern.
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("y")
    result = dispatch("Glob", json.dumps({"pattern": "**/*", "path": str(tmp_path)}))
    assert "a.txt" in result
    assert "b.py" in result


# transcript round-trip tests live in tests/test_transcript.py.
