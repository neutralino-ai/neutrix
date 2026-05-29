"""Project context (v1.2.0): CLAUDE.md/AGENTS.md auto-load + @-mention inlining.

`.claude/`-compatible — discovers the same memory files Claude Code does
(`~/.claude/CLAUDE.md`, project `CLAUDE.md`/`.claude/CLAUDE.md`/`AGENTS.md`
walking cwd→root, `CLAUDE.local.md`), expands `@import` directives inside them,
and wraps the result with the CC "these instructions OVERRIDE" header before it
is prepended to the system prompt. Also expands `@path` file-mentions a user
types into a turn. Pure functions — the wiring lives in cli.py / terminal_chat.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MEMORY_INSTRUCTION_PROMPT = (
    "Codebase and user instructions are shown below. Be sure to adhere to these "
    "instructions. IMPORTANT: These instructions OVERRIDE any default behavior "
    "and you MUST follow them exactly as written."
)

_MAX_IMPORT_DEPTH = 5
_PROJECT_FILES = ("CLAUDE.md", ".claude/CLAUDE.md", "AGENTS.md")
_IMPORT_LINE_RE = re.compile(r"^\s*@(\S+)\s*$")
_MENTION_RE = re.compile(r'(?:^|\s)@("[^"]+"|[^\s]+)')
_LINE_RANGE_RE = re.compile(r"^(.*)#L(\d+)(?:-(\d+))?$")


@dataclass(frozen=True)
class MemoryFile:
    label: str  # "user" | "project" | "local"
    path: str  # resolved absolute path
    content: str


def discover_memory_files(cwd: str | Path, home: str | Path | None = None) -> list[MemoryFile]:
    """Memory files in inject order (lowest priority first, cwd-most last)."""
    home_dir = Path(home) if home is not None else Path.home()
    base = Path(cwd).expanduser().resolve()
    seen: set[str] = set()
    out: list[MemoryFile] = []

    def add(label: str, p: Path) -> None:
        if not p.is_file():
            return
        rp = str(p.resolve())
        if rp in seen:
            return
        try:
            content = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return
        seen.add(rp)
        out.append(MemoryFile(label, rp, content))

    add("user", home_dir / ".claude" / "CLAUDE.md")
    # Walk root → cwd so the cwd-most file is injected last (highest priority).
    chain = [base, *base.parents]
    for d in reversed(chain):
        for name in _PROJECT_FILES:
            add("project", d / name)
    add("local", base / "CLAUDE.local.md")
    return out


def _resolve_import(raw: str, base_dir: Path) -> Path | None:
    raw = raw.split("#", 1)[0]  # strip #fragment
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    return p


def _expand_imports(content: str, base_dir: Path, *, depth: int = 0, seen: set[str] | None = None) -> str:
    """Expand whole-line ``@path`` imports (recursive, depth ≤ 5, code-safe)."""
    if seen is None:
        seen = set()
    if depth >= _MAX_IMPORT_DEPTH:
        return content
    lines_out: list[str] = []
    in_fence = False
    for line in content.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            lines_out.append(line)
            continue
        if in_fence:
            lines_out.append(line)
            continue
        m = _IMPORT_LINE_RE.match(line)
        if not m:
            lines_out.append(line)
            continue
        target = _resolve_import(m.group(1), base_dir)
        if target is None or not target.is_file() or str(target.resolve()) in seen:
            lines_out.append(line)  # missing/cyclic → keep the directive verbatim
            continue
        seen.add(str(target.resolve()))
        try:
            imported = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            lines_out.append(line)
            continue
        lines_out.append(_expand_imports(imported, target.parent, depth=depth + 1, seen=seen))
    return "\n".join(lines_out)


def compose_memory_block(files: list[MemoryFile]) -> str:
    """The wrapped, import-expanded memory block (empty string if no files)."""
    if not files:
        return ""
    parts = [MEMORY_INSTRUCTION_PROMPT, ""]
    for f in files:
        expanded = _expand_imports(f.content, Path(f.path).parent).strip()
        if not expanded:
            continue
        parts.append(f"Contents of {f.path} ({f.label} memory):")
        parts.append(expanded)
        parts.append("")
    return "\n".join(parts).strip()


def compose_system_prompt(base: str, cwd: str | Path, home: str | Path | None = None) -> str:
    """Base system prompt + the project/user memory block (CLAUDE.md, …)."""
    block = compose_memory_block(discover_memory_files(cwd, home))
    return f"{base}\n\n{block}" if block else base


def _split_line_range(raw: str) -> tuple[str, tuple[int, int] | None]:
    m = _LINE_RANGE_RE.match(raw)
    if not m:
        return raw, None
    start = int(m.group(2))
    end = int(m.group(3)) if m.group(3) else start
    return m.group(1), (start, end)


def expand_at_mentions(text: str, cwd: str | Path) -> str:
    """Inline ``@path`` file-mentions a user typed into the turn.

    Appends ``<file path="…">…</file>`` blocks for each mention that resolves to
    an existing file (or ``<dir>`` listing). ``@path#L10-20`` slices lines.
    Non-existent ``@tokens`` (likely prose) are left untouched — no false noise.
    """
    base = Path(cwd).expanduser()
    blocks: list[str] = []
    for rawhit in _MENTION_RE.findall(text):
        raw = rawhit.strip('"')
        path_part, line_range = _split_line_range(raw)
        p = Path(path_part).expanduser()
        if not p.is_absolute():
            p = base / p
        if not p.exists():
            continue  # not a file → leave the @token alone
        if p.is_dir():
            try:
                names = sorted(c.name + ("/" if c.is_dir() else "") for c in p.iterdir())
            except OSError:
                continue
            listing = "\n".join(names[:200])
            blocks.append(f'<dir path="{path_part}">\n{listing}\n</dir>')
            continue
        try:
            content = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            blocks.append(f'<file path="{path_part}">(not utf-8 text)</file>')
            continue
        if line_range is not None:
            lines = content.splitlines()
            s, e = max(1, line_range[0]), line_range[1]
            content = "\n".join(lines[s - 1 : e])
        blocks.append(f'<file path="{path_part}">\n{content}\n</file>')
    if not blocks:
        return text
    return text + "\n\n" + "\n".join(blocks)
