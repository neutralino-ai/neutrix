"""Tool permissions — an Executor-only safety layer (v1.5.3). `.claude/`-compatible.

A `PermissionPolicy` (allow / deny / ask rule lists) is loaded from
`.claude/settings.json` + `.claude/settings.local.json` (+ user `~/.claude`), the
same shape Claude Code uses (`permissions.allow/deny/ask`, rules like
`"Bash(git *)"`, `"Write"`, `"Read(~/.ssh/*)"`). `decide()` is consulted by the
`Executor` before each tool call, and the Executor resolves the verdict on its own
— the ContextManager and Advisor never see permission.

**Industrial safety layer (user-directed, 2026-05-30: "if it detect dangerous
action, deny directly … don't ask user question. let's be like industrial
agent").** neutrix never prompts for permission. `decide()` returns only
`"allow"` | `"deny"`:

- **`auto`** (default) — allow normal operations, but **deny clearly destructive
  Bash** (`rm -rf`, force-push, `dd`, fork-bombs, `curl|sh`, …) outright: the
  Executor returns a denied tool_result and the round continues (the model
  adapts). A deterministic danger heuristic, not a sandbox.
- **`allow-all`** — no checks; every tool runs (escape hatch, `/allow`).

Deny rules always win; an explicit `allow` rule overrides the danger heuristic;
the danger heuristic and any settings `ask` rule both resolve to **deny** (neutrix
has no interactive approval — that's deliberate). No plan mode (user-directed).
"""
from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The argument a `Tool(pattern)` rule matches against, per tool.
_PRIMARY_ARG = {
    "Bash": "command",
    "Read": "path",
    "Edit": "path",
    "Write": "path",
    "Glob": "pattern",
    "Grep": "pattern",
    "Agent": "subagent_type",
}

# Auto-mode danger heuristic: classic destructive shell patterns. Not
# exhaustive — a safety net, not a sandbox. An explicit allow rule overrides it.
_DANGEROUS_BASH_RE = re.compile(
    r"""
      \brm\s+-[a-z]*[rf]            # rm -rf / -fr / -r ... / -f ...
    | \bgit\s+push\b.*--force      # force push
    | \bdd\s+if=                   # raw disk write
    | \bmkfs\b                     # format
    | :\(\)\s*\{.*\};\s*:          # fork bomb
    | >\s*/dev/(sd|nvme|disk)      # overwrite a block device
    | \bchmod\s+-R\s+0?777         # world-writable recursive
    | \bsudo\s+rm\b                # sudo rm
    | \b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba|z|fi)?sh  # pipe-to-shell
    | \bgit\s+reset\s+--hard\b     # discard work
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _is_dangerous(tool_name: str, args_json: str) -> bool:
    if tool_name != "Bash":
        return False
    return bool(_DANGEROUS_BASH_RE.search(_primary_value("Bash", args_json)))


@dataclass(frozen=True)
class PermissionPolicy:
    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()
    ask: tuple[str, ...] = ()


def _merge_rules(*settings: dict[str, Any]) -> PermissionPolicy:
    allow: list[str] = []
    deny: list[str] = []
    ask: list[str] = []
    for s in settings:
        perms = s.get("permissions") if isinstance(s, dict) else None
        if not isinstance(perms, dict):
            continue
        for key, bucket in (("allow", allow), ("deny", deny), ("ask", ask)):
            vals = perms.get(key)
            if isinstance(vals, list):
                bucket.extend(str(v) for v in vals)
    return PermissionPolicy(tuple(allow), tuple(deny), tuple(ask))


def load_policy(cwd: str | Path, home: str | Path | None = None) -> PermissionPolicy:
    """Merge permission rules from user + project `.claude/settings*.json`."""
    home_dir = Path(home) if home is not None else Path.home()
    base = Path(cwd).expanduser()
    settings: list[dict[str, Any]] = []
    for path in (
        home_dir / ".claude" / "settings.json",
        base / ".claude" / "settings.json",
        base / ".claude" / "settings.local.json",
    ):
        try:
            settings.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return _merge_rules(*settings)


def _primary_value(tool_name: str, args_json: str) -> str:
    key = _PRIMARY_ARG.get(tool_name)
    if key is None:
        return ""
    try:
        args = json.loads(args_json) if args_json else {}
    except ValueError:
        return ""
    return str(args.get(key, "")) if isinstance(args, dict) else ""


def _matches(rule: str, tool_name: str, args_json: str) -> bool:
    rule = rule.strip()
    if "(" in rule and rule.endswith(")"):
        rname, _, rest = rule.partition("(")
        pattern = rest[:-1]
        if rname.strip() != tool_name:
            return False
        value = _primary_value(tool_name, args_json)
        return fnmatch.fnmatch(value, pattern)
    return rule == tool_name


def decide(
    tool_name: str,
    args_json: str = "{}",
    *,
    mode: str = "auto",
    policy: PermissionPolicy | None = None,
) -> str:
    """Return ``"allow"`` | ``"deny"`` for a tool call (v1.5.3: no interactive ask).

    ``allow-all``/``bypass`` → allow everything. Otherwise: deny rules win, then
    explicit allow rules (which override the auto danger heuristic), then — in
    ``auto`` mode — destructive Bash is **denied** by the safety layer, and any
    settings ``ask`` rule is also **denied** (neutrix never prompts); default is
    allow.
    """
    if mode in ("allow-all", "bypass"):
        return "allow"
    policy = policy or PermissionPolicy()
    if any(_matches(r, tool_name, args_json) for r in policy.deny):
        return "deny"
    if any(_matches(r, tool_name, args_json) for r in policy.allow):
        return "allow"
    if mode == "auto" and _is_dangerous(tool_name, args_json):
        return "deny"
    if any(_matches(r, tool_name, args_json) for r in policy.ask):
        return "deny"
    return "allow"


def block_reason(tool_name: str, args_json: str = "") -> str:
    """Message for a denied tool — distinguishes the auto safety layer from a deny rule.

    Both forms contain "denied" so the model reads it as a hard refusal and adapts.
    """
    if _is_dangerous(tool_name, args_json):
        return (
            f"[denied by safety layer: {tool_name} looks destructive — not run. "
            "Use a non-destructive approach, or the user can /allow to override.]"
        )
    return f"[denied by permission rules: {tool_name} not run]"
