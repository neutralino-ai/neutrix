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
