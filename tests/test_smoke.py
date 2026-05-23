"""Smoke tests — no network calls."""
import json

import pytest

from neutrix import __version__
from neutrix.config import (
    DEFAULT_CONFIG,
    SLOT_NAMES,
    ConfigError,
    bootstrap_config,
    load_config,
)
from neutrix.session import dump, load
from neutrix.tools import BUILTIN_TOOLS, dispatch, get_schemas


def test_version_string():
    assert isinstance(__version__, str)
    assert __version__


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
    target = tmp_path / "hello.txt"
    write_res = dispatch(
        "write_file", json.dumps({"path": str(target), "content": "hi"})
    )
    assert "OK" in write_res
    assert target.read_text() == "hi"
    read_res = dispatch("read_file", json.dumps({"path": str(target)}))
    assert read_res == "hi"


def test_tool_dispatch_unknown():
    result = dispatch("nope", "{}")
    assert result.startswith("ERROR: unknown tool")


def test_tool_dispatch_bad_json():
    result = dispatch("read_file", "{not json")
    assert result.startswith("ERROR: invalid JSON args")


def test_tool_dispatch_list_dir(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    result = dispatch("list_dir", json.dumps({"path": str(tmp_path)}))
    assert "f a.txt" in result
    assert "d sub" in result


# ----- session ---------------------------------------------------------------


def test_session_roundtrip(tmp_path):
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    path = tmp_path / "s.json"
    dump(path, provider="ihep", model="anthropic/claude-haiku-4-5", messages=messages)
    payload = load(path)
    assert payload["provider"] == "ihep"
    assert payload["model"] == "anthropic/claude-haiku-4-5"
    assert payload["messages"] == messages
