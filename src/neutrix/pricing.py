"""Cost mechanism (v1.7.1). **Pure — no price data lives in neutrix code.**

Prices live in the user's config YAML (the ``pricing:`` block, keyed by the exact
model string), parsed into a :class:`PriceTable` by
:meth:`neutrix.config.Config.pricing`. :class:`Price` is one model's rates;
:func:`cost` multiplies usage by them. An unpriced model has no entry → the ledger
renders ``"(cost unknown)"``. To reprice, edit the YAML — nothing in the package
ships a number.

(Earlier versions hard-coded an estimated USD table; live audit showed the
estimates were wrong and the provenance overclaimed. Prices are now config data,
any currency — see ``docs/PRDs/v1.7.1-cost-fix.md``.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle; only needed for the type hint
    from neutrix.llm import Usage


@dataclass(frozen=True)
class Price:
    """Per **million** tokens, one figure per :class:`~neutrix.llm.Usage` class.

    The currency is a display concern carried on :class:`PriceTable`, not encoded
    here — the numbers are whatever the user wrote in config.
    """

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


@dataclass(frozen=True)
class PriceTable:
    """The parsed config ``pricing:`` block: a display ``currency`` symbol + a
    ``model → Price`` map (keyed by the exact slot model string — no name
    mapping)."""

    currency: str = "$"
    models: dict[str, Price] = field(default_factory=dict)

    def price_for(self, model: str) -> Price | None:
        return self.models.get(model)


def cost(usage: Usage, price: Price) -> float:
    """Cost of ``usage`` at ``price`` — Σ over the four token classes
    ``tokens / 1e6 * rate``.

    ``cache_write`` is priced at its own (higher) rate for accuracy, even though
    the *display* folds it into "miss" (the 3-number `hit · miss · output` view).
    """
    return (
        usage.input * price.input
        + usage.output * price.output
        + usage.cache_read * price.cache_read
        + usage.cache_write * price.cache_write
    ) / 1_000_000
