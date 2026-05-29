"""Skills + slash-command framework + custom agents (v1.3.0).

`.claude/`-compatible: discovers markdown skills/commands/agents from the same
dirs Claude Code uses (``~/.claude`` and project ``.claude``). A skill/command
is a markdown file with optional YAML-ish frontmatter (``name``/``description``/
``argument-hint``); invoking ``/name`` enqueues the body (with ``$ARGUMENTS`` /
``$1``… substituted) as a user turn. Custom agents (``.claude/agents/*.md``)
become ``subagent_type`` values the ``Agent`` tool can spawn.

Pure discovery/parsing — the dispatch + hot-reload polling live in
``terminal_chat``; the agent lookup is read at ``Agent``-dispatch time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


@dataclass(frozen=True)
class SkillDef:
    name: str
    description: str
    body: str
    source: str
    argument_hint: str = ""


@dataclass(frozen=True)
class AgentDef:
    name: str
    description: str
    system_prompt: str
    source: str


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text.strip()
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, _, val = line.partition(":")
            meta[key.strip().lower()] = val.strip().strip('"').strip("'")
    return meta, m.group(2).strip()


def _roots(cwd: str | Path, home: str | Path | None) -> list[Path]:
    home_dir = Path(home) if home is not None else Path.home()
    # user first, project second → project overrides user on name collision.
    return [home_dir / ".claude", Path(cwd).expanduser() / ".claude"]


def _description_from_body(body: str) -> str:
    for line in body.splitlines():
        s = line.strip().lstrip("#").strip()
        if s:
            return s[:200]
    return ""


def discover_skills(cwd: str | Path, home: str | Path | None = None) -> dict[str, SkillDef]:
    """Discover ``/name`` skills from ``.claude/skills/<name>/SKILL.md`` and
    flat ``.claude/commands/*.md`` (user then project; project overrides)."""
    found: dict[str, SkillDef] = {}
    for root in _roots(cwd, home):
        skills_dir = root / "skills"
        if skills_dir.is_dir():
            for sub in sorted(skills_dir.iterdir()):
                f = sub / "SKILL.md"
                if sub.is_dir() and f.is_file():
                    _add_skill(found, sub.name, f)
        commands_dir = root / "commands"
        if commands_dir.is_dir():
            for f in sorted(commands_dir.glob("*.md")):
                _add_skill(found, f.stem, f)
    return found


def _add_skill(found: dict[str, SkillDef], default_name: str, path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    meta, body = _parse_frontmatter(text)
    name = (meta.get("name") or default_name).lstrip("/").lower()
    if not name or not body:
        return
    found[name] = SkillDef(
        name=name,
        description=meta.get("description") or _description_from_body(body),
        body=body,
        source=str(path.resolve()),
        argument_hint=meta.get("argument-hint", ""),
    )


def discover_agents(cwd: str | Path, home: str | Path | None = None) -> dict[str, AgentDef]:
    """Discover custom subagent types from ``.claude/agents/*.md`` (user then
    project; project overrides). The body is the agent's system prompt."""
    found: dict[str, AgentDef] = {}
    for root in _roots(cwd, home):
        agents_dir = root / "agents"
        if not agents_dir.is_dir():
            continue
        for f in sorted(agents_dir.glob("*.md")):
            try:
                text = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            meta, body = _parse_frontmatter(text)
            name = (meta.get("name") or f.stem).lstrip("/").lower()
            if not name or not body:
                continue
            found[name] = AgentDef(
                name=name,
                description=meta.get("description") or _description_from_body(body),
                system_prompt=body,
                source=str(f.resolve()),
            )
    return found


def render_skill(skill: SkillDef, args: list[str]) -> str:
    """Substitute ``$ARGUMENTS`` and ``$1``…``$9`` in the skill body."""
    body = skill.body.replace("$ARGUMENTS", " ".join(args))
    for i, arg in enumerate(args[:9], start=1):
        body = body.replace(f"${i}", arg)
    return body


def skills_signature(cwd: str | Path, home: str | Path | None = None) -> tuple:
    """A cheap fingerprint of the skill/command/agent dirs for hot-reload.

    Changes whenever a watched dir or any contained ``.md`` file is added,
    removed, or modified — so a poll can re-discover only on change.
    """
    sig: list[tuple[str, float]] = []
    for root in _roots(cwd, home):
        for kind in ("skills", "commands", "agents"):
            d = root / kind
            if not d.is_dir():
                continue
            try:
                sig.append((str(d), d.stat().st_mtime))
                for f in d.rglob("*.md"):
                    sig.append((str(f), f.stat().st_mtime))
            except OSError:
                continue
    return tuple(sorted(sig))
