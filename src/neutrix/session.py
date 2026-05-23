"""Save / load conversation sessions as JSON."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

SESSION_VERSION = 1


def dump(
    path: str | Path,
    *,
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
) -> Path:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SESSION_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "provider": provider,
        "model": model,
        "messages": messages,
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def load(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser()
    payload = json.loads(p.read_text(encoding="utf-8"))
    if payload.get("version") != SESSION_VERSION:
        raise ValueError(
            f"unsupported session version {payload.get('version')!r}; "
            f"this neutrix expects {SESSION_VERSION}"
        )
    return payload
