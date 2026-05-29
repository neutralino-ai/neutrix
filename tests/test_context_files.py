"""Tests for v1.2.0 project context (CLAUDE.md loading + @-mentions)."""
from __future__ import annotations

from pathlib import Path

from neutrix.context_files import (
    MEMORY_INSTRUCTION_PROMPT,
    compose_memory_block,
    compose_system_prompt,
    discover_memory_files,
    expand_at_mentions,
)


def _write(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


# ---- discovery ------------------------------------------------------------


def test_discover_order_user_project_local(tmp_path: Path) -> None:
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    _write(home / ".claude" / "CLAUDE.md", "USER")
    _write(proj / "CLAUDE.md", "PROJECT")
    _write(proj / "CLAUDE.local.md", "LOCAL")
    files = discover_memory_files(proj, home=home)
    labels = [f.label for f in files]
    contents = [f.content for f in files]
    # inject order: user (lowest) → project → local (highest)
    assert labels == ["user", "project", "local"]
    assert contents == ["USER", "PROJECT", "LOCAL"]


def test_discover_walks_up_cwd_most_last(tmp_path: Path) -> None:
    home = tmp_path / "home"  # no user file
    parent = tmp_path / "repo"
    child = parent / "pkg"
    _write(parent / "CLAUDE.md", "PARENT")
    _write(child / "CLAUDE.md", "CHILD")
    files = discover_memory_files(child, home=home)
    # cwd-most (child) injected last → highest priority
    assert [f.content for f in files] == ["PARENT", "CHILD"]


def test_discover_dedups_by_resolved_path(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write(home / ".claude" / "CLAUDE.md", "ONLY")
    # cwd == home so the user file would also be hit as a project walk entry
    files = discover_memory_files(home, home=home)
    assert sum(f.content == "ONLY" for f in files) == 1


def test_discover_includes_agents_md(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write(tmp_path / "p" / "AGENTS.md", "AGENTS")
    files = discover_memory_files(tmp_path / "p", home=home)
    assert any(f.content == "AGENTS" for f in files)


# ---- @import inside memory files ------------------------------------------


def test_import_expands_recursively(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write(tmp_path / "p" / "inner.md", "INNER")
    _write(tmp_path / "p" / "CLAUDE.md", "top\n@inner.md\nbottom")
    block = compose_memory_block(discover_memory_files(tmp_path / "p", home=home))
    assert "INNER" in block and "top" in block and "bottom" in block
    assert "@inner.md" not in block  # directive consumed


def test_import_depth_cap_and_cycle_safe(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write(tmp_path / "p" / "a.md", "A\n@b.md")
    _write(tmp_path / "p" / "b.md", "B\n@a.md")  # cycle
    _write(tmp_path / "p" / "CLAUDE.md", "@a.md")
    block = compose_memory_block(discover_memory_files(tmp_path / "p", home=home))
    assert "A" in block and "B" in block  # terminates, no infinite loop


def test_import_skipped_in_code_block(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write(tmp_path / "p" / "x.md", "SHOULD_NOT_APPEAR")
    _write(tmp_path / "p" / "CLAUDE.md", "```\n@x.md\n```")
    block = compose_memory_block(discover_memory_files(tmp_path / "p", home=home))
    assert "SHOULD_NOT_APPEAR" not in block
    assert "@x.md" in block  # left verbatim inside the fence


# ---- compose_system_prompt ------------------------------------------------


def test_compose_system_prompt_wraps(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write(tmp_path / "p" / "CLAUDE.md", "use ruff not flake8")
    out = compose_system_prompt("BASE", tmp_path / "p", home=home)
    assert out.startswith("BASE")
    assert MEMORY_INSTRUCTION_PROMPT in out
    assert "use ruff not flake8" in out


def test_compose_system_prompt_noop_without_files(tmp_path: Path) -> None:
    home = tmp_path / "home"
    out = compose_system_prompt("BASE", tmp_path / "empty", home=home)
    assert out == "BASE"


# ---- @-mentions in user input ---------------------------------------------


def test_expand_mention_inlines_file(tmp_path: Path) -> None:
    _write(tmp_path / "foo.py", "print('hi')\n")
    out = expand_at_mentions("explain @foo.py please", tmp_path)
    assert "explain @foo.py please" in out  # raw text kept
    assert '<file path="foo.py">' in out
    assert "print('hi')" in out


def test_expand_mention_line_range(tmp_path: Path) -> None:
    _write(tmp_path / "f.txt", "\n".join(f"L{i}" for i in range(1, 11)))
    out = expand_at_mentions("see @f.txt#L3-5", tmp_path)
    assert "L3" in out and "L5" in out
    assert "L2" not in out and "L6" not in out


def test_expand_mention_directory(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    _write(tmp_path / "d" / "a.txt", "x")
    out = expand_at_mentions("look at @d", tmp_path)
    assert '<dir path="d">' in out and "a.txt" in out


def test_expand_mention_nonexistent_left_untouched(tmp_path: Path) -> None:
    out = expand_at_mentions("ping @someone about @missing.py", tmp_path)
    assert out == "ping @someone about @missing.py"  # no blocks, no noise


def test_expand_no_mentions_is_identity(tmp_path: Path) -> None:
    assert expand_at_mentions("just a normal message", tmp_path) == "just a normal message"
