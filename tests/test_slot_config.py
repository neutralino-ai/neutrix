"""Tests for v0.9.5 per-slot ``llm_timeout_s`` config field.

The slot YAML now accepts an optional ``llm_timeout_s: float`` — a
slow local model gets headroom, a hosted-API slot keeps the
60 s default. Absent → 60.0; present → parsed as float. Bad input
raises :class:`ConfigError`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from neutrix.config import ConfigError, Slot, load_config


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _provider_block() -> str:
    return (
        "providers:\n"
        "  test:\n"
        "    base_url: https://example.test/v1\n"
        "    api_key: sk-test\n"
    )


# ---- defaults ------------------------------------------------------------


def test_slot_dataclass_default_llm_timeout_is_300s() -> None:
    """``Slot`` default is the v0.9.5 post-gate 300.0 seconds."""
    slot = Slot(
        name="fast",
        provider="test",
        model="m",
        base_url="https://example.test/v1",
        api_key="sk-test",
    )
    assert slot.llm_timeout_s == 300.0


def test_slot_yaml_absent_field_defaults_to_300s(tmp_path: Path) -> None:
    cfg_path = _write(
        tmp_path / "config.yaml",
        _provider_block()
        + "fast:\n  provider: test\n  model: m\n"
          "strong:\n  provider: test\n  model: m2\n",
    )
    cfg = load_config(cfg_path)
    assert cfg.slot("fast").llm_timeout_s == 300.0
    assert cfg.slot("strong").llm_timeout_s == 300.0


# ---- per-slot override ---------------------------------------------------


def test_slot_yaml_per_slot_override_is_read(tmp_path: Path) -> None:
    cfg_path = _write(
        tmp_path / "config.yaml",
        _provider_block()
        + "fast:\n  provider: test\n  model: m\n  llm_timeout_s: 30.0\n"
          "strong:\n  provider: test\n  model: m2\n  llm_timeout_s: 180\n",
    )
    cfg = load_config(cfg_path)
    assert cfg.slot("fast").llm_timeout_s == 30.0
    # Integer YAML value parses cleanly to float.
    assert cfg.slot("strong").llm_timeout_s == 180.0


def test_slot_yaml_explicit_null_falls_back_to_default(tmp_path: Path) -> None:
    """A user who explicitly nulls the field should still get the default."""
    cfg_path = _write(
        tmp_path / "config.yaml",
        _provider_block()
        + "fast:\n  provider: test\n  model: m\n  llm_timeout_s: null\n"
          "strong:\n  provider: test\n  model: m2\n",
    )
    cfg = load_config(cfg_path)
    assert cfg.slot("fast").llm_timeout_s == 300.0


# ---- validation ----------------------------------------------------------


def test_slot_yaml_non_numeric_raises(tmp_path: Path) -> None:
    cfg_path = _write(
        tmp_path / "config.yaml",
        _provider_block()
        + "fast:\n  provider: test\n  model: m\n  llm_timeout_s: forever\n"
          "strong:\n  provider: test\n  model: m2\n",
    )
    cfg = load_config(cfg_path)
    with pytest.raises(ConfigError, match="llm_timeout_s"):
        cfg.slot("fast")


def test_slot_yaml_non_positive_raises(tmp_path: Path) -> None:
    cfg_path = _write(
        tmp_path / "config.yaml",
        _provider_block()
        + "fast:\n  provider: test\n  model: m\n  llm_timeout_s: 0\n"
          "strong:\n  provider: test\n  model: m2\n",
    )
    cfg = load_config(cfg_path)
    with pytest.raises(ConfigError, match="positive"):
        cfg.slot("fast")
