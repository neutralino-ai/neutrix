"""Tests for the session cost ledger (v1.7.0)."""
from __future__ import annotations

import json
from pathlib import Path

from neutrix import pricing
from neutrix.cost_ledger import CostLedger
from neutrix.llm import Usage


def test_record_accumulates_usage_and_timing():
    led = CostLedger()
    led.record("claude-opus-4-7", Usage(input=100, output=50, cache_read=20), 1000.0, 200.0)
    led.record("claude-opus-4-7", Usage(input=10, output=5), 500.0, 0.0)
    total = led.total_usage()
    assert (total.input, total.output, total.cache_read) == (110, 55, 20)
    assert led.total_llm_ms() == 1500.0
    assert led.total_tool_ms() == 200.0
    assert led.has_usage() is True


def test_cost_sums_priced_entries():
    led = CostLedger()
    led.record("claude-opus-4-7", Usage(input=1_000_000), 0, 0)  # $15
    led.record("claude-opus-4-7", Usage(output=1_000_000), 0, 0)  # $75
    assert led.cost() == 90.0


def test_empty_ledger_cost_is_zero_and_no_usage():
    led = CostLedger()
    assert led.cost() == 0.0
    assert led.has_usage() is False


def test_all_unpriced_cost_is_none():
    led = CostLedger()
    led.record("mystery-model", Usage(input=1000, output=1000), 0, 0)
    assert led.cost() is None  # → surface "(cost unknown)"
    assert led.unpriced_models() == ["mystery-model"]
    assert led.has_usage() is True


def test_mixed_priced_and_unpriced_returns_partial_sum():
    led = CostLedger()
    led.record("claude-opus-4-7", Usage(input=1_000_000), 0, 0)  # $15
    led.record("mystery", Usage(input=1_000_000), 0, 0)  # unknown → 0 contribution
    assert led.cost() == 15.0
    assert led.unpriced_models() == ["mystery"]


def test_by_model_breakdown_insertion_ordered():
    led = CostLedger()
    led.record("a", Usage(input=10), 0, 0)
    led.record("b", Usage(input=20), 0, 0)
    led.record("a", Usage(input=5), 0, 0)
    bm = led.by_model()
    assert bm["a"].input == 15
    assert bm["b"].input == 20
    assert list(bm) == ["a", "b"]


def test_none_usage_records_timing_but_no_tokens():
    # A slot that 400'd on include_usage (or a cancelled turn) → no usage payload,
    # but the timing still counts so the wall/API rollups stay honest.
    led = CostLedger()
    led.record("claude-opus-4-7", None, 300.0, 0.0)
    assert led.total_usage().total == 0
    assert led.total_llm_ms() == 300.0
    assert led.has_usage() is False


def test_from_jsonl_rebuilds_entries(tmp_path: Path):
    p = tmp_path / "s.jsonl"
    lines = [
        {"type": "user", "message": {"role": "user", "content": "hi"}},  # ignored
        {"type": "usage", "model": "claude-opus-4-7", "input": 100, "output": 50,
         "cache_read": 20, "cache_write": 0, "raw": {"prompt_tokens": 120},
         "llm_ms": 1000.0, "tool_ms": 0.0},
        {"type": "usage", "model": "deepseek-v4-pro", "input": 1_000_000, "output": 0,
         "cache_read": 0, "cache_write": 0, "raw": {}, "llm_ms": 200.0, "tool_ms": 50.0},
    ]
    p.write_text("\n".join(json.dumps(o) for o in lines), encoding="utf-8")
    led = CostLedger.from_jsonl(p)
    assert len(led.entries) == 2
    assert led.total_usage().input == 1_000_100
    assert led.total_llm_ms() == 1200.0
    assert led.total_tool_ms() == 50.0
    # Raw payload preserved as the source of truth.
    assert led.entries[0].usage.raw == {"prompt_tokens": 120}


def test_reprice_on_table_change(monkeypatch, tmp_path: Path):
    """Acceptance #5: dollars are computed on read, so editing the price table
    reprices an existing (rebuilt-from-JSONL) session retroactively."""
    p = tmp_path / "s.jsonl"
    line = {"type": "usage", "model": "claude-opus-4-7", "input": 1_000_000,
            "output": 0, "cache_read": 0, "cache_write": 0, "raw": {}}
    p.write_text(json.dumps(line), encoding="utf-8")
    led = CostLedger.from_jsonl(p)
    assert led.cost() == 15.0
    monkeypatch.setitem(
        pricing._PRICES, "claude-opus-4-7", pricing.ModelPrice(30.0, 75.0, 1.5, 18.75)
    )
    assert led.cost() == 30.0  # same ledger, new price


def test_from_jsonl_tolerates_malformed_and_missing_fields(tmp_path: Path):
    p = tmp_path / "s.jsonl"
    p.write_text(
        "\n".join(
            [
                "not json at all",
                json.dumps({"type": "usage", "model": "x"}),  # no counts → zeros
                json.dumps({"type": "tasks", "tasks": []}),  # not a usage line
            ]
        ),
        encoding="utf-8",
    )
    led = CostLedger.from_jsonl(p)
    assert len(led.entries) == 1
    assert led.entries[0].usage.total == 0
    assert led.entries[0].model == "x"


def test_from_jsonl_missing_file_is_empty(tmp_path: Path):
    led = CostLedger.from_jsonl(tmp_path / "nope.jsonl")
    assert led.entries == []
    assert led.cost() == 0.0
