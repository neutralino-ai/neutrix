"""Tests for the pure pricing mechanism (v1.7.1) — prices live in config, not code."""
from __future__ import annotations

import neutrix.pricing as pricing_mod
from neutrix.llm import Usage
from neutrix.pricing import Price, PriceTable, cost


def test_cost_sums_four_classes_per_million():
    # Rates are per MILLION tokens; 1M of each class → the sum of the four rates.
    p = Price(input=5.0, output=25.0, cache_read=0.5, cache_write=6.25)
    u = Usage(input=1_000_000, output=1_000_000, cache_read=1_000_000, cache_write=1_000_000)
    assert cost(u, p) == 5.0 + 25.0 + 0.5 + 6.25


def test_cost_prices_each_class_distinctly():
    p = Price(input=5.0, output=25.0, cache_read=0.5, cache_write=6.25)
    assert cost(Usage(input=1_000_000), p) == 5.0
    assert cost(Usage(output=1_000_000), p) == 25.0
    assert cost(Usage(cache_read=1_000_000), p) == 0.5
    assert cost(Usage(cache_write=1_000_000), p) == 6.25


def test_zero_usage_is_zero():
    assert cost(Usage(), Price(input=5.0, output=25.0)) == 0.0


def test_price_table_lookup_is_exact_by_model_string():
    pt = PriceTable(currency="¥", models={"anthropic/claude-opus-4-7": Price(input=5.0)})
    assert pt.currency == "¥"
    assert pt.price_for("anthropic/claude-opus-4-7") == Price(input=5.0)
    # No prefix-strip / fuzzy match — the config key IS the model string.
    assert pt.price_for("claude-opus-4-7") is None
    assert pt.price_for("unknown") is None


def test_empty_table_defaults_to_dollar_and_no_prices():
    pt = PriceTable()
    assert pt.currency == "$"
    assert pt.price_for("anything") is None


def test_no_price_data_baked_into_the_module():
    # The v1.7.1 invariant (Acceptance #1): no hardcoded rate table, no model
    # lookup helper — those moved to config / PriceTable.
    assert not hasattr(pricing_mod, "_PRICES")
    assert not hasattr(pricing_mod, "price_for")
    assert not hasattr(pricing_mod, "ModelPrice")
