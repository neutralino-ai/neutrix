"""Tests for the curated pricing table (v1.7.0)."""
from __future__ import annotations

from neutrix import pricing
from neutrix.llm import Usage


def test_known_model_prices_four_classes_distinctly():
    # claude-opus-4-7: input 15, output 75, cache_read 1.5, cache_write 18.75 /Mtok.
    one_m = Usage(input=1_000_000, output=1_000_000, cache_read=1_000_000, cache_write=1_000_000)
    assert pricing.cost(one_m, "claude-opus-4-7") == 15.0 + 75.0 + 1.5 + 18.75


def test_cache_priced_relative_to_input():
    price = pricing.price_for("claude-opus-4-7")
    assert price is not None
    assert price.cache_read < price.input  # a cache hit is cheaper than fresh input
    assert price.cache_write > price.input  # cache creation carries a surcharge


def test_provider_prefix_is_stripped():
    bare = pricing.cost(Usage(input=1_000_000), "claude-opus-4-7")
    gateway = pricing.cost(Usage(input=1_000_000), "anthropic/claude-opus-4-7")
    assert bare == gateway == 15.0
    # The direct-provider form and its gateway-prefixed form resolve identically.
    deep_bare = pricing.cost(Usage(input=1_000_000), "deepseek-v4-pro")
    deep_pref = pricing.cost(Usage(input=1_000_000), "deepseek-ai/deepseek-v4-pro")
    assert deep_bare == deep_pref == 0.5


def test_unknown_model_returns_none_not_zero():
    assert pricing.cost(Usage(input=1000, output=1000), "no-such-model") is None
    assert pricing.cost(Usage(input=1000), "vendor/also-unknown") is None


def test_zero_usage_on_priced_model_is_zero_not_none():
    # Distinguishes "free" (0.0) from "unknown" (None) — the caller relies on it.
    assert pricing.cost(Usage(), "claude-opus-4-7") == 0.0


def test_default_slots_are_priced():
    # Acceptance #6 distinguishes "$" from "(cost unknown)"; the common path
    # (the default fast/strong slots) must render a dollar figure.
    for model in ("anthropic/claude-haiku-4-5", "anthropic/claude-opus-4-7"):
        assert pricing.price_for(model) is not None
