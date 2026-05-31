"""Session-scoped token/cost/timing accumulator (v1.7.0).

The :class:`CostLedger` is a **pure in-memory accumulator** — it does no I/O of
its own (the v3-RL-headless constraint: the core loop and its observers stay
free of TUI/file coupling). It is fed once per completed assistant turn via
:meth:`record`, sums on demand, and computes **dollars lazily on read** through
:mod:`neutrix.pricing` so a price-table correction reprices past turns.

It is a **live accumulator, never a sum over the live message store**: compaction
and ``/rewind`` shrink ``store.messages`` while session cost must stay cumulative.
The append-only session JSONL (dedicated ``{"type": "usage", …}`` lines, written
by :meth:`SessionWriter.append_usage`) is the durable backing — :meth:`from_jsonl`
rebuilds the full ledger on resume even across compactions that dropped the
in-memory turns.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from neutrix import pricing
from neutrix.llm import Usage
from neutrix.pricing import PriceTable
from neutrix.session_store import _read_lines


@dataclass(frozen=True)
class LedgerEntry:
    """One completed assistant turn's accounting."""

    model: str
    usage: Usage
    llm_ms: float = 0.0
    tool_ms: float = 0.0


class CostLedger:
    """Accumulates per-turn usage; computes cost/timing rollups on demand."""

    def __init__(self) -> None:
        self.entries: list[LedgerEntry] = []
        # v1.7.1: prices come from the user's config YAML, injected by
        # terminal_chat after construction / resume. Empty table ⇒ no prices ⇒
        # everything renders "(cost unknown)".
        self.price_table: PriceTable = PriceTable()

    @property
    def currency(self) -> str:
        return self.price_table.currency

    # ---------------------------------------------------------------- feed
    def record(
        self, model: str, usage: Usage | None, llm_ms: float = 0.0, tool_ms: float = 0.0
    ) -> None:
        """Append one turn. A ``None`` usage (e.g. a slot that 400'd on
        ``include_usage``, or a cancelled turn) still records the timing so the
        wall/API/tool rollups stay honest; its token contribution is zero."""
        self.entries.append(
            LedgerEntry(model=model, usage=usage or Usage(), llm_ms=llm_ms, tool_ms=tool_ms)
        )

    # ----------------------------------------------------------- rollups
    def has_usage(self) -> bool:
        return self.total_usage().total > 0

    def total_usage(self) -> Usage:
        total = Usage()
        for e in self.entries:
            total = total + e.usage
        return total

    def by_model(self) -> dict[str, Usage]:
        """Cumulative usage per model, insertion-ordered by first appearance."""
        out: dict[str, Usage] = {}
        for e in self.entries:
            out[e.model] = out.get(e.model, Usage()) + e.usage
        return out

    def model_cost(self, model: str) -> float | None:
        usage = self.by_model().get(model)
        price = self.price_table.price_for(model)
        if usage is None or price is None:
            return None
        return pricing.cost(usage, price)

    def cost(self) -> float | None:
        """Total cost in the configured currency, or ``None`` when entries exist
        but **none** are priced (→ surface ``"(cost unknown)"``). ``0.0`` when
        there are no entries. A mix returns the partial sum of the priced turns;
        pair with :meth:`unpriced_models` to footnote the gap. Prices come from
        the injected :attr:`price_table`, computed on read — editing the config
        YAML reprices the next session."""
        if not self.entries:
            return 0.0
        known: list[float] = []
        for e in self.entries:
            price = self.price_table.price_for(e.model)
            if price is not None:
                known.append(pricing.cost(e.usage, price))
        if not known:
            return None
        return sum(known)

    def unpriced_models(self) -> list[str]:
        """Models in the ledger absent from the config price table (insertion-ordered)."""
        out: list[str] = []
        for model in self.by_model():
            if self.price_table.price_for(model) is None and model not in out:
                out.append(model)
        return out

    def total_llm_ms(self) -> float:
        return sum(e.llm_ms for e in self.entries)

    def total_tool_ms(self) -> float:
        return sum(e.tool_ms for e in self.entries)

    # ----------------------------------------------------------- resume
    @classmethod
    def from_jsonl(cls, path: str | Path) -> CostLedger:
        """Rebuild a ledger from a session log's ``{"type": "usage"}`` lines.

        Reconstructs each :class:`Usage` from the persisted normalized counts +
        raw payload, so dollars recompute against the *current* price table.
        Tolerant of malformed lines (skips them) — best-effort like the writer.
        """
        ledger = cls()
        for obj in _read_lines(path):
            if obj.get("type") != "usage":
                continue
            raw = obj.get("raw")
            usage = Usage(
                input=int(obj.get("input") or 0),
                output=int(obj.get("output") or 0),
                cache_read=int(obj.get("cache_read") or 0),
                cache_write=int(obj.get("cache_write") or 0),
                raw=raw if isinstance(raw, dict) else {},
            )
            ledger.record(
                model=str(obj.get("model") or ""),
                usage=usage,
                llm_ms=float(obj.get("llm_ms") or 0.0),
                tool_ms=float(obj.get("tool_ms") or 0.0),
            )
        return ledger
