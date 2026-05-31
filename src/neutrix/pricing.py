"""Curated per-model pricing for turning token usage into a dollar cost (v1.7.0).

Prices are US dollars **per million tokens**, in four classes that match
:class:`neutrix.llm.Usage`: ``input`` (fresh, non-cached prompt), ``output``
(completion), ``cache_read`` (a cached-prompt hit), and ``cache_write`` (the
cache-creation surcharge — Anthropic only; OpenAI-compatible backends bill no
separate cache-write tier, so it is ``0`` there).

**Provenance (the split-point decision):** the *schema* and the per-million
convention follow LiteLLM's ``model_prices_and_context_window.json`` — used as
**vendored DATA, not the ``litellm`` dependency** ([[project_v2_v3_direction]]).
The table is a curated subset covering only the models neutrix's slots point at;
the numbers come from each provider's published pricing and are deliberately
easy to refresh. Because the ledger stores raw token counts and computes dollars
**on read**, correcting a number here reprices every past session retroactively.

Unknown model → :func:`cost` returns ``None`` → the surface renders
``"(cost unknown)"`` rather than a confidently-wrong figure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a hard import cycle; only needed for the type hint
    from neutrix.llm import Usage


@dataclass(frozen=True)
class ModelPrice:
    """USD per **million** tokens, one figure per :class:`~neutrix.llm.Usage` class."""

    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0


# Keyed by the BARE model id (no ``provider/`` prefix). ``cost`` strips a single
# leading ``provider/`` segment so both the IHEP-gateway form
# (``anthropic/claude-opus-4-7``) and the direct-provider form (``deepseek-v4-pro``)
# resolve to the same entry. Refresh against the provider pricing pages.
_PRICES: dict[str, ModelPrice] = {
    # Anthropic (cache_read ≈ 0.1x input, cache_write ≈ 1.25x input).
    "claude-opus-4-7": ModelPrice(15.0, 75.0, 1.5, 18.75),
    "claude-opus-4-8": ModelPrice(15.0, 75.0, 1.5, 18.75),
    "claude-sonnet-4-6": ModelPrice(3.0, 15.0, 0.3, 3.75),
    "claude-haiku-4-5": ModelPrice(1.0, 5.0, 0.1, 1.25),
    # OpenAI (cached prompt billed at a reduced input rate; no cache-write tier).
    "gpt-5.5": ModelPrice(5.0, 15.0, 2.5, 0.0),
    # DeepSeek (cache hit ≈ 0.1x input; no separate cache-write tier).
    "deepseek-v4-pro": ModelPrice(0.5, 1.5, 0.05, 0.0),
    "deepseek-v4-flash": ModelPrice(0.1, 0.3, 0.01, 0.0),
    # Zhipu GLM.
    "glm-5.1": ModelPrice(0.5, 1.5, 0.05, 0.0),
    "glm-5.1-highspeed": ModelPrice(0.3, 0.9, 0.03, 0.0),
}


def _bare_model(model: str) -> str:
    """Drop a single leading ``provider/`` segment (``a/b/c`` → ``b/c``)."""
    return model.split("/", 1)[1] if "/" in model else model


def price_for(model: str) -> ModelPrice | None:
    """The :class:`ModelPrice` for ``model``, or ``None`` if not in the table.

    Tries the exact id first, then the bare id with any ``provider/`` prefix
    stripped — so ``anthropic/claude-opus-4-7`` and ``claude-opus-4-7`` match.
    """
    if model in _PRICES:
        return _PRICES[model]
    return _PRICES.get(_bare_model(model))


def cost(usage: Usage, model: str) -> float | None:
    """Dollar cost of ``usage`` at ``model``'s rates, or ``None`` if unpriced.

    Sum over the four token classes of ``tokens / 1e6 * price``. ``None`` (not
    ``0.0``) for an unknown model so the caller can distinguish "free" from
    "unknown" and render ``"(cost unknown)"``.
    """
    price = price_for(model)
    if price is None:
        return None
    return (
        usage.input * price.input
        + usage.output * price.output
        + usage.cache_read * price.cache_read
        + usage.cache_write * price.cache_write
    ) / 1_000_000
