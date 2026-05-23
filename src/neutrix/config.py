"""YAML config loader.

Single source of truth: ``~/.config/neutrix/config.yaml``. No env vars.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

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
        "kimi/kimi-k2.6",
    ],
    "deepseek": ["deepseek-v4-flash", "deepseek-v4-pro"],
    "glm": ["glm-5.1", "glm-5.1-highspeed"],
}

DEFAULT_CONFIG = """\
# neutrix config — paste your API keys, then re-run `neutrix`.
# Two named slots, `fast` and `strong`, point at (provider, model) pairs.
# Switch between them inside the TUI with /fast and /strong.

providers:
  ihep:
    base_url: https://aiapi.ihep.ac.cn/apiv2/
    api_key: ""        # paste your IHEP gateway key

  deepseek:
    base_url: https://api.deepseek.com
    api_key: ""

  glm:
    base_url: https://open.bigmodel.cn/api/paas/v4/
    api_key: ""

fast:
  provider: ihep
  model: anthropic/claude-haiku-4-5

strong:
  provider: ihep
  model: anthropic/claude-opus-4-7
"""


class ConfigError(RuntimeError):
    """Raised when the YAML config is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class Slot:
    """A resolved (slot, provider, model, credentials) bundle."""

    name: str
    provider: str
    model: str
    base_url: str
    api_key: str


@dataclass(frozen=True)
class Config:
    providers: dict[str, dict[str, str]]
    slots: dict[str, dict[str, str]]
    path: Path

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
        return Slot(
            name=name,
            provider=prov_name,
            model=model,
            base_url=base_url,
            api_key=api_key,
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
    return Config(providers=providers, slots=slots, path=path)


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
            name: {
                "base_url": (prov or {}).get("base_url", ""),
                "api_key": (prov or {}).get("api_key", ""),
            }
            for name, prov in config.providers.items()
        },
        "fast": fast if fast is not None else (config.slots.get("fast") or {}),
        "strong": strong if strong is not None else (config.slots.get("strong") or {}),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def resolve_initial_slot(config: Config) -> tuple[Slot | None, Slot | None]:
    """Resolve (fast, strong) without raising. Either or both may be None."""
    def _try(name: str) -> Slot | None:
        try:
            return config.slot(name)
        except ConfigError:
            return None
    return _try("fast"), _try("strong")
