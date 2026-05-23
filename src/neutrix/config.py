"""Provider configuration: base URLs, default models, env-var key names."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key_env: str
    default_model: str
    models: tuple[str, ...]


PROVIDERS: dict[str, Provider] = {
    "deepseek": Provider(
        name="deepseek",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
        models=("deepseek-chat", "deepseek-reasoner"),
    ),
    "glm": Provider(
        name="glm",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key_env="GLM_API_KEY",
        default_model="glm-4.6",
        models=("glm-4.6", "glm-4-plus", "glm-4-air", "glm-4-flash"),
    ),
    "claude": Provider(
        name="claude",
        base_url="https://api.anthropic.com/v1/",
        api_key_env="ANTHROPIC_API_KEY",
        default_model="claude-sonnet-4-6",
        models=(
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ),
    ),
}


def load_env() -> None:
    """Load .env from current directory (no-op if absent)."""
    load_dotenv(override=False)


def get_provider(name: str) -> Provider:
    key = name.lower()
    if key not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {name!r}. Known: {', '.join(PROVIDERS)}"
        )
    return PROVIDERS[key]


def get_api_key(provider: Provider) -> str:
    key = os.environ.get(provider.api_key_env, "").strip()
    if not key:
        raise RuntimeError(
            f"Missing API key. Set ${provider.api_key_env} in your environment or .env file."
        )
    return key


def default_provider_name() -> str:
    return os.environ.get("NEUTRIX_PROVIDER", "deepseek").lower()


def default_model_for(provider: Provider) -> str:
    override = os.environ.get("NEUTRIX_MODEL", "").strip()
    return override or provider.default_model
