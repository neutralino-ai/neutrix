"""Smoke tests — no network calls."""
import json

import pytest

from neutrix import __version__
from neutrix.config import PROVIDERS, get_provider
from neutrix.session import dump, load
from neutrix.tools import BUILTIN_TOOLS, dispatch, get_schemas


def test_version_string():
    assert isinstance(__version__, str)
    assert __version__


def test_providers_registered():
    assert {"deepseek", "glm", "claude"} <= set(PROVIDERS)
    for name in ("deepseek", "glm", "claude"):
        p = get_provider(name)
        assert p.base_url.startswith("https://")
        assert p.default_model in p.models


def test_get_provider_unknown():
    with pytest.raises(ValueError):
        get_provider("not-a-thing")


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


def test_session_roundtrip(tmp_path):
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    path = tmp_path / "s.json"
    dump(path, provider="deepseek", model="deepseek-chat", messages=messages)
    payload = load(path)
    assert payload["provider"] == "deepseek"
    assert payload["model"] == "deepseek-chat"
    assert payload["messages"] == messages
