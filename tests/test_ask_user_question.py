"""v1.4.8 / v1.5.3 — bidirectional dispatch protocol for AskUserQuestion.

The Executor is a pure event leaf: it *yields* ``needs_user_input`` and receives
the answer via ``gen.asend(answer)``; it never calls the UI. The ContextManager
owns the ``ask_user`` port and drives the generator. As of v1.5.3 only the
AskUserQuestion tool uses this channel — permission is denied directly inside the
Executor and never yields ``needs_user_input``. These tests exercise both layers.
"""
from __future__ import annotations

import json

import pytest

from neutrix.context_manager import ContextManager
from neutrix.executor import Executor, ToolEvent
from neutrix.permissions import PermissionPolicy
from neutrix.prompts import (
    ASK_NOT_AVAILABLE,
    Answer,
    QuestionAnswer,
    QuestionSpec,
)
from neutrix.store import ChatStore


def _args(**kw) -> str:
    return json.dumps(kw)


def _responder(value: str):
    """A fake answerer: pick the option whose value matches ``value`` (else an
    Other free-text answer of ``value``). Used to reply to needs_user_input."""

    def respond(spec: QuestionSpec) -> Answer:
        answers = []
        for q in spec.questions:
            match = next((o for o in q.options if o.effective_value() == value), None)
            if match is not None:
                answers.append(QuestionAnswer(q.question, match.label, (match.effective_value(),)))
            else:
                answers.append(QuestionAnswer(q.question, value, (), True, value))
        return Answer(per_question=tuple(answers))

    return respond


async def _drive(executor, tool_calls, responder=None):
    """Drive the bidirectional dispatch generator.

    ``responder(spec) -> Answer`` answers each ``needs_user_input`` event;
    ``responder=None`` simulates no interactive consumer (sends ``None`` back,
    as the CM does when ``ask_user`` is None). Returns the non-input events.
    """
    gen = executor.dispatch_all(tool_calls)
    events = []
    send = None
    while True:
        try:
            ev = await gen.asend(send)
        except StopAsyncIteration:
            break
        send = None
        if isinstance(ev, ToolEvent) and ev.kind == "needs_user_input":
            send = responder(ev.data["spec"]) if responder is not None else None
            continue
        events.append(ev)
    return events


def _fin(events):
    return next(e for e in events if e.kind == "tool_finished")


_ASK_ARGS = _args(questions=[{
    "question": "Which DB?",
    "header": "Database",
    "options": [{"label": "Postgres"}, {"label": "SQLite"}],
}])


# ---- Executor protocol: AskUserQuestion ----------------------------------


@pytest.mark.asyncio
async def test_executor_yields_needs_input_for_ask_user_question():
    ex = Executor()
    gen = ex.dispatch_all([{"id": "1", "name": "AskUserQuestion", "arguments": _ASK_ARGS}])
    kinds = []
    send = None
    saw_request = False
    while True:
        try:
            ev = await gen.asend(send)
        except StopAsyncIteration:
            break
        send = None
        kinds.append(ev.kind)
        if ev.kind == "needs_user_input":
            saw_request = True
            assert ev.data["spec"].questions[0].header == "Database"
            send = _responder("Postgres")(ev.data["spec"])  # answer it
    assert saw_request
    assert "needs_user_input" in kinds and "tool_finished" in kinds


@pytest.mark.asyncio
async def test_ask_user_question_answered():
    ex = Executor()
    events = await _drive(
        ex, [{"id": "1", "name": "AskUserQuestion", "arguments": _ASK_ARGS}],
        responder=_responder("Postgres"),
    )
    fin = _fin(events)
    assert fin.data["ok"] is True
    assert json.loads(fin.data["content"]) == {"answers": {"Which DB?": "Postgres"}}


@pytest.mark.asyncio
async def test_ask_user_question_no_consumer_not_available():
    ex = Executor()
    events = await _drive(ex, [{"id": "1", "name": "AskUserQuestion", "arguments": _ASK_ARGS}])
    fin = _fin(events)
    assert fin.data["ok"] is False and fin.data["content"] == ASK_NOT_AVAILABLE


@pytest.mark.asyncio
async def test_ask_user_question_invalid_schema_errors():
    ex = Executor()
    bad = _args(questions=[{"question": "q?", "header": "h", "options": [{"label": "only"}]}])
    events = await _drive(
        ex, [{"id": "1", "name": "AskUserQuestion", "arguments": bad}],
        responder=_responder("x"),
    )
    fin = _fin(events)
    assert fin.data["ok"] is False and "ERROR" in fin.data["content"]


@pytest.mark.asyncio
async def test_ask_user_question_never_hits_thread(monkeypatch):
    monkeypatch.setattr("neutrix.executor.dispatch", lambda *a, **k: "LEAKED TO THREAD")
    ex = Executor()
    events = await _drive(
        ex, [{"id": "1", "name": "AskUserQuestion", "arguments": _ASK_ARGS}],
        responder=_responder("SQLite"),
    )
    assert "LEAKED" not in _fin(events).data["content"]


# ---- Executor protocol: permission is a direct deny (v1.5.3) --------------


@pytest.mark.asyncio
async def test_dangerous_bash_denied_directly_no_prompt(monkeypatch):
    # The safety layer denies dangerous actions outright — it NEVER yields
    # needs_user_input, and the tool does not run.
    monkeypatch.setattr("neutrix.executor.dispatch", lambda *a, **k: "RAN DANGEROUS")
    ex = Executor()  # auto mode
    gen = ex.dispatch_all([{"id": "1", "name": "Bash", "arguments": _args(command="rm -rf build")}])
    events, saw_input, send = [], False, None
    while True:
        try:
            ev = await gen.asend(send)
        except StopAsyncIteration:
            break
        send = None
        if ev.kind == "needs_user_input":
            saw_input = True
        events.append(ev)
    fin = _fin(events)
    assert not saw_input  # no prompt, no park
    assert fin.data["ok"] is False and "denied" in fin.data["content"]
    assert "RAN DANGEROUS" not in fin.data["content"]


@pytest.mark.asyncio
async def test_settings_ask_rule_denied_directly(monkeypatch):
    # A settings `ask` rule resolves to deny — neutrix never prompts.
    monkeypatch.setattr("neutrix.executor.dispatch", lambda *a, **k: "ran")
    ex = Executor()
    ex.policy = PermissionPolicy(ask=("Write",))
    events = await _drive(ex, [{"id": "1", "name": "Write", "arguments": _args(path="a")}])
    fin = _fin(events)
    assert fin.data["ok"] is False and "denied" in fin.data["content"]


@pytest.mark.asyncio
async def test_deny_rule_denied_directly(monkeypatch):
    monkeypatch.setattr("neutrix.executor.dispatch", lambda *a, **k: "ran")
    ex = Executor()
    ex.policy = PermissionPolicy(deny=("Bash(rm *)",))
    events = await _drive(ex, [{"id": "1", "name": "Bash", "arguments": _args(command="rm x")}])
    fin = _fin(events)
    assert fin.data["ok"] is False and "denied" in fin.data["content"]


# ---- ContextManager drive (the real .asend() consumer) -------------------


def _bare_ctx() -> ContextManager:
    from test_terminal_chat import FakeLLM, _make_ctx

    return _make_ctx(FakeLLM())


@pytest.mark.asyncio
async def test_cm_dispatch_relays_ask_user_question():
    ctx = _bare_ctx()

    async def port(spec):
        return _responder("SQLite")(spec)

    ctx.ask_user = port
    await ctx._dispatch_tools([{"id": "1", "name": "AskUserQuestion", "arguments": _ASK_ARGS}])
    tool_msgs = [m for m in ctx.messages if m.get("role") == "tool"]
    assert tool_msgs and json.loads(tool_msgs[-1]["content"]) == {"answers": {"Which DB?": "SQLite"}}


@pytest.mark.asyncio
async def test_cm_dispatch_no_port_not_available():
    ctx = _bare_ctx()  # ask_user is None
    await ctx._dispatch_tools([{"id": "1", "name": "AskUserQuestion", "arguments": _ASK_ARGS}])
    tool_msgs = [m for m in ctx.messages if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[-1]["content"] == ASK_NOT_AVAILABLE


@pytest.mark.asyncio
async def test_cm_dispatch_dangerous_denied_without_port(monkeypatch):
    # The CM never sees permission: a dangerous Bash is denied by the Executor
    # even though ask_user is None, and the round completes (no park).
    monkeypatch.setattr("neutrix.executor.dispatch", lambda *a, **k: "RAN")
    ctx = _bare_ctx()  # ask_user is None
    await ctx._dispatch_tools([{"id": "1", "name": "Bash", "arguments": _args(command="rm -rf x")}])
    tool_msgs = [m for m in ctx.messages if m.get("role") == "tool"]
    assert tool_msgs and "denied" in tool_msgs[-1]["content"]
    assert "RAN" not in tool_msgs[-1]["content"]


def test_ask_user_question_excluded_from_subagents():
    from neutrix.tools import BUILTIN_TOOLS, subagent_tool_names

    assert "AskUserQuestion" in BUILTIN_TOOLS
    assert "AskUserQuestion" not in subagent_tool_names()  # split #7: no humanless deadlock


def test_executor_has_no_ask_user_field():
    # Layering guard: the port lives on the CM, never the Executor.
    assert not hasattr(Executor(store=ChatStore()), "ask_user")
