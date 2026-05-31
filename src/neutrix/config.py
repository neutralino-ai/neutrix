"""YAML config loader.

Single source of truth: ``~/.config/neutrix/config.yaml``. No env vars.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from neutrix.pricing import Price, PriceTable

CONFIG_PATH = Path("~/.config/neutrix/config.yaml").expanduser()
SLOT_NAMES: tuple[str, ...] = ("fast", "strong")

PROVIDER_DEFAULT_MODELS: dict[str, list[str]] = {
    "ihep": [
        "anthropic/claude-haiku-4-5",
        "anthropic/claude-opus-4-7",
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-5.5",
        "deepseek-ai/deepseek-v4-pro",
        "deepseek-ai/deepseek-v4-flash",
        "zhipu/glm-5.1",
        "moonshot/kimi-k2.6",
    ],
    "deepseek": ["deepseek-v4-flash", "deepseek-v4-pro"],
    "glm": ["glm-5.1", "glm-5.1-highspeed"],
}

DEFAULT_CONFIG = """\
# neutrix config — paste your IHEP gateway api_key below, then re-run `neutrix`.
# Two named slots, `fast` and `strong`, point at (provider, model) pairs.
# Switch between them inside the TUI with /fast and /strong.

providers:
  ihep:
    base_url: https://aiapi.ihep.ac.cn/apiv2/
    api_key: ""        # <- paste your IHEP gateway key here, then re-run

fast:
  provider: ihep
  model: anthropic/claude-haiku-4-5

strong:
  provider: ihep
  model: anthropic/claude-opus-4-7

# Cost display. `currency` is only a display symbol; the numbers are per MILLION
# tokens (frozen USD defaults from LiteLLM's public data — edit to match your
# actual billing, or set `currency: "¥"` and use CNY numbers). A model not
# listed here renders "(cost unknown)" — tokens still shown.
pricing:
  currency: "$"
  models:
    anthropic/claude-haiku-4-5:    { input: 1.0,  output: 5.0,  cache_read: 0.10, cache_write: 1.25 }
    anthropic/claude-opus-4-7:     { input: 5.0,  output: 25.0, cache_read: 0.50, cache_write: 6.25 }
    anthropic/claude-sonnet-4-6:   { input: 3.0,  output: 15.0, cache_read: 0.30, cache_write: 3.75 }
    openai/gpt-5.5:                { input: 5.0,  output: 30.0, cache_read: 0.50 }
    deepseek-ai/deepseek-v4-pro:   { input: 0.28, output: 0.42, cache_read: 0.028 }
    deepseek-ai/deepseek-v4-flash: { input: 0.28, output: 0.42, cache_read: 0.028 }
    zhipu/glm-5.1:                 { input: 1.0,  output: 3.20, cache_read: 0.20 }
"""


class ConfigError(RuntimeError):
    """Raised when the YAML config is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class Slot:
    """A resolved (slot, provider, model, credentials) bundle.

    ``llm_timeout_s`` (v0.9.5) bounds one LLM round end-to-end. A
    background watchdog in :class:`~neutrix.context_manager.ContextManager`
    fires :py:meth:`cm.cancel(reason='timeout')` after this many
    seconds of being parked in ``AWAITING_LLM``. Per-slot so a slow
    local model can be given more headroom than a hosted-API slot.
    """

    name: str
    provider: str
    model: str
    base_url: str
    api_key: str
    llm_timeout_s: float = 300.0
    # v0.10.5: the model's context window in tokens. None ⇒ unknown ⇒
    # automatic (threshold) compaction is disabled for this slot (manual
    # /compact still works). Set per-slot in config.yaml.
    max_context_tokens: int | None = None


@dataclass(frozen=True)
class Config:
    providers: dict[str, dict[str, Any]]
    slots: dict[str, dict[str, Any]]
    path: Path
    # v1.7.1: the raw ``pricing:`` block (currency + model→rates). Default empty
    # ⇒ no prices ⇒ "(cost unknown)". Parsed into a PriceTable by price_table().
    pricing: dict[str, Any] = field(default_factory=dict)

    def price_table(self) -> PriceTable:
        """Build the :class:`~neutrix.pricing.PriceTable` from the ``pricing:``
        block (v1.7.1). ``currency`` is a display symbol; ``models`` maps the
        **exact** slot model string to per-million-token rates. Tolerant of
        missing / partial / malformed entries (→ 0.0)."""
        raw = self.pricing or {}
        currency = str(raw.get("currency") or "$")
        models_raw = raw.get("models")
        models: dict[str, Price] = {}
        if isinstance(models_raw, dict):
            for name, spec in models_raw.items():
                spec = spec or {}
                models[str(name)] = Price(
                    input=_to_float(spec.get("input")),
                    output=_to_float(spec.get("output")),
                    cache_read=_to_float(spec.get("cache_read")),
                    cache_write=_to_float(spec.get("cache_write")),
                )
        return PriceTable(currency=currency, models=models)

    def slot(self, name: str) -> Slot:
        if name not in self.slots:
            raise ConfigError(f"unknown slot {name!r}; choose one of {SLOT_NAMES}")
        spec = self.slots[name] or {}
        prov_name = spec.get("provider", "")
        model = spec.get("model", "")
        if not prov_name or not model:
            raise ConfigError(
                f"slot {name!r} in {self.path} is missing provider or model"
            )
        if prov_name not in self.providers:
            raise ConfigError(
                f"slot {name!r} references unknown provider {prov_name!r} "
                f"(known: {sorted(self.providers)})"
            )
        prov = self.providers[prov_name] or {}
        base_url = (prov.get("base_url") or "").strip()
        api_key = (prov.get("api_key") or "").strip()
        if not base_url:
            raise ConfigError(
                f"provider {prov_name!r} has no base_url in {self.path}"
            )
        if not api_key:
            raise ConfigError(
                f"provider {prov_name!r} has no api_key in {self.path}; "
                f"edit it and re-run"
            )
        llm_timeout_s = _read_llm_timeout(spec, name=name, path=self.path)
        return Slot(
            name=name,
            provider=prov_name,
            model=model,
            base_url=base_url,
            api_key=api_key,
            llm_timeout_s=llm_timeout_s,
            max_context_tokens=_read_max_context_tokens(spec, name=name, path=self.path),
        )


def bootstrap_config(path: Path = CONFIG_PATH) -> Path:
    """Write the default config template. Returns the path written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_CONFIG, encoding="utf-8")
    return path


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Load the YAML config. Raises ConfigError on missing/malformed input."""
    if not path.exists():
        raise ConfigError(f"config not found at {path}")
    try:
        payload: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e
    if not isinstance(payload, dict):
        raise ConfigError(f"{path} top-level must be a mapping")

    providers = payload.get("providers") or {}
    if not isinstance(providers, dict):
        raise ConfigError(f"{path}: `providers` must be a mapping")

    slots = {name: (payload.get(name) or {}) for name in SLOT_NAMES}
    pricing = payload.get("pricing")
    if not isinstance(pricing, dict):
        pricing = {}
    return Config(providers=providers, slots=slots, path=path, pricing=pricing)


def save_config(
    config: Config,
    *,
    fast: dict[str, str] | None = None,
    strong: dict[str, str] | None = None,
    path: Path | None = None,
) -> Path:
    """Round-trippable YAML write-back. Loses any comments in the existing file."""
    out = path or config.path
    data: dict[str, Any] = {
        "providers": {
            name: _serialize_provider(prov)
            for name, prov in config.providers.items()
        },
        "fast": fast if fast is not None else (config.slots.get("fast") or {}),
        "strong": strong if strong is not None else (config.slots.get("strong") or {}),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def _to_float(value: Any) -> float:
    """Coerce a config price value to float; ``None`` / non-numeric → ``0.0``."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_llm_timeout(spec: dict[str, Any], *, name: str, path: Path) -> float:
    """Parse ``llm_timeout_s`` from a slot spec; default 300.0 when absent.

    The 300 s default (v0.9.5 post-gate) gives slow hosted reasoning
    models such as deepseek-v4-pro headroom — a healthy long reply
    isn't killed, while a genuine silent hang still aborts well inside
    the SDK's 600 s read-timeout. Raises :class:`ConfigError` on
    non-numeric or non-positive values.
    """
    raw = spec.get("llm_timeout_s")
    if raw is None:
        return 300.0
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"slot {name!r} in {path}: llm_timeout_s must be numeric, "
            f"got {raw!r}"
        ) from exc
    if value <= 0:
        raise ConfigError(
            f"slot {name!r} in {path}: llm_timeout_s must be positive, "
            f"got {value}"
        )
    return value


def _read_max_context_tokens(
    spec: dict[str, Any], *, name: str, path: Path
) -> int | None:
    """Parse ``max_context_tokens`` from a slot spec; None when absent (v0.10.5).

    None disables automatic compaction for the slot. Raises
    :class:`ConfigError` on non-integer or non-positive values.
    """
    raw = spec.get("max_context_tokens")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"slot {name!r} in {path}: max_context_tokens must be an integer, "
            f"got {raw!r}"
        ) from exc
    if value <= 0:
        raise ConfigError(
            f"slot {name!r} in {path}: max_context_tokens must be positive, "
            f"got {value}"
        )
    return value


def _serialize_provider(prov: Any) -> dict[str, Any]:
    """Provider entries write `model_status` only when non-empty."""
    prov = prov or {}
    out: dict[str, Any] = {
        "base_url": prov.get("base_url", ""),
        "api_key": prov.get("api_key", ""),
    }
    status = prov.get("model_status") or {}
    if status:
        out["model_status"] = {k: v for k, v in status.items() if v in ("verified", "failed")}
    return out


def resolve_initial_slot(config: Config) -> tuple[Slot | None, Slot | None]:
    """Resolve (fast, strong) without raising. Either or both may be None."""
    def _try(name: str) -> Slot | None:
        try:
            return config.slot(name)
        except ConfigError:
            return None
    return _try("fast"), _try("strong")
