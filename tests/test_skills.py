"""Tests for v1.3.0 skills + slash-command framework + custom agents."""
from __future__ import annotations

from pathlib import Path

from neutrix.skills import (
    discover_agents,
    discover_skills,
    render_skill,
    skills_signature,
)


def _w(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


# ---- discovery ------------------------------------------------------------


def test_discover_skill_dir_and_command_file(tmp_path: Path) -> None:
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    _w(proj / ".claude" / "skills" / "deploy" / "SKILL.md", "---\ndescription: ship it\n---\nrun the deploy")
    _w(proj / ".claude" / "commands" / "lint.md", "run ruff")
    skills = discover_skills(proj, home=home)
    assert "deploy" in skills and skills["deploy"].description == "ship it"
    assert "lint" in skills and skills["lint"].body == "run ruff"


def test_project_overrides_user_on_name_collision(tmp_path: Path) -> None:
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    _w(home / ".claude" / "commands" / "x.md", "USER VERSION")
    _w(proj / ".claude" / "commands" / "x.md", "PROJECT VERSION")
    skills = discover_skills(proj, home=home)
    assert skills["x"].body == "PROJECT VERSION"


def test_frontmatter_name_overrides_filename(tmp_path: Path) -> None:
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    _w(proj / ".claude" / "commands" / "file.md", "---\nname: renamed\n---\nbody")
    skills = discover_skills(proj, home=home)
    assert "renamed" in skills and "file" not in skills


def test_render_skill_substitutes_arguments(tmp_path: Path) -> None:
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    _w(proj / ".claude" / "commands" / "echo.md", "say $ARGUMENTS to $1")
    sk = discover_skills(proj, home=home)["echo"]
    assert render_skill(sk, ["hi", "bob"]) == "say hi bob to hi"


def test_skills_signature_changes_on_new_file(tmp_path: Path) -> None:
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    _w(proj / ".claude" / "commands" / "a.md", "a")
    sig1 = skills_signature(proj, home=home)
    _w(proj / ".claude" / "commands" / "b.md", "b")
    sig2 = skills_signature(proj, home=home)
    assert sig1 != sig2


def test_discover_agents(tmp_path: Path) -> None:
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    _w(proj / ".claude" / "agents" / "reviewer.md", "---\ndescription: reviews code\n---\nYou are a strict reviewer.")
    agents = discover_agents(proj, home=home)
    assert "reviewer" in agents
    assert agents["reviewer"].system_prompt == "You are a strict reviewer."
    assert agents["reviewer"].description == "reviews code"


# ---- custom-agent dispatch via the Agent tool -----------------------------


def test_agent_tool_uses_custom_agent_system_prompt(tmp_path, monkeypatch):
    from neutrix.config import Slot

    _w(tmp_path / ".claude" / "agents" / "reviewer.md", "You are a strict code reviewer.")
    monkeypatch.chdir(tmp_path)

    captured = {}

    async def _fake_run(**kwargs):
        captured.update(kwargs)
        from neutrix.subagent import SubagentResult

        return SubagentResult(final_text="ok", turn_count=1)

    import neutrix.subagent as subagent_mod

    monkeypatch.setattr(subagent_mod, "run_subagent", _fake_run)
    monkeypatch.setattr("neutrix.llm.OpenAIChatLLM", lambda slot: object())

    from neutrix.tools import _agent

    slot = Slot(name="fast", provider="t", model="m", base_url="u", api_key="k")
    out = _agent("review", "look at foo.py", subagent_type="reviewer", slot=slot)
    assert out == "ok"
    assert captured["system_prompt"] == "You are a strict code reviewer."


def test_agent_tool_unknown_custom_agent_lists_available(tmp_path, monkeypatch):
    from neutrix.config import Slot

    monkeypatch.chdir(tmp_path)  # no .claude/agents
    from neutrix.tools import _agent

    slot = Slot(name="fast", provider="t", model="m", base_url="u", api_key="k")
    out = _agent("x", "y", subagent_type="nope", slot=slot)
    assert out.startswith("ERROR") and "general-purpose" in out
