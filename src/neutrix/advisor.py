"""Smart Advisor (v0.10.4) — a third actor that judges the task list.

Besides the user and the controller, the Advisor watches the store and, on a
trigger (every N completed turns, or `/advise`), makes its own cheap LLM call to
judge progress — then either revises the task list (via the Task tools) or
injects a judged ``<advisor>`` suggestion the main LLM sees next turn. The CC
analog is the turn-end forked actors (auto-dream / extractMemories /
promptSuggestion): observe between turns, own LLM call, surface via message +
state mutation.

``run_once`` is **side-effect-free** — it returns an :class:`AdvisorOutcome`;
the orchestrator applies task mutations (``tools.dispatch(store=…)``) and injects
the suggestion through a ``ContextManager`` method, so the single-mutator
invariant (only CM writes ``messages``/``ChatStore``) is preserved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from neutrix.llm import LLMEvent, LLMResponse
from neutrix.store import Task
from neutrix.tools import get_schemas

ADVISOR_CADENCE_TURNS = 5
ADVISOR_MAX_RUNS = 20
ADVISOR_RECENT_TURNS = 6
ADVISOR_TASK_TOOLS = frozenset({"TaskCreate", "TaskUpdate", "TaskList"})

ADVISOR_SYSTEM_PROMPT = (
    "You are the Advisor — a reviewer that watches a coding assistant's task "
    "list and progress, and keeps the task list honest. You are NOT the "
    "assistant and you do not do the work.\n\n"
    "Each time you are consulted you see the current task list and the last few "
    "turns. Decide if the task list needs revising:\n"
    "- If a task is clearly finished by the recent work, mark it completed "
    "(TaskUpdate status=completed).\n"
    "- If the conversation has drifted and a task is stale, update or delete it.\n"
    "- If the user clearly asked for something new that isn't tracked, add it "
    "(TaskCreate).\n"
    "Use the TaskCreate / TaskUpdate / TaskList tools for any change.\n\n"
    "Separately, you MAY write one short paragraph of advice for the assistant "
    "and user — what's done, what's left, what to focus on. Keep it to a few "
    "sentences. If nothing is worth saying, say nothing.\n\n"
    "Guardrails (non-negotiable): review and suggest, do not execute the work; "
    "do not edit files, run shells, or call any non-Task tool; do not create, "
    "update, or delete a task about yourself or your own scheduling; never ask "
    "to be run again."
)


@dataclass(frozen=True)
class AdvisorOutcome:
    """What one Advisor run produced — applied by the orchestrator, not here.

    ``task_calls`` are ``(tool_name, arguments_json)`` pairs to dispatch against
    the store; ``suggestion`` is the (untagged) advice text to inject, or None.
    """

    task_calls: tuple[tuple[str, str], ...]
    suggestion: str | None


class Advisor:
    """Trigger policy + one-shot LLM judgment. Holds no store/message state."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        cadence_turns: int = ADVISOR_CADENCE_TURNS,
        max_runs: int = ADVISOR_MAX_RUNS,
        recent_turns: int = ADVISOR_RECENT_TURNS,
    ) -> None:
        self.enabled = enabled
        self.cadence_turns = cadence_turns
        self.max_runs = max_runs
        self.recent_turns = recent_turns
        self._turns_since_run = 0
        self._runs = 0
        self._running = False

    @property
    def runs(self) -> int:
        return self._runs

    def note_turn(self) -> None:
        """Record that a controller turn completed (drives the periodic trigger)."""
        self._turns_since_run += 1

    def should_run(self) -> bool:
        """Periodic auto-trigger: cadence reached, under the cap, not re-entrant."""
        return (
            self.enabled
            and not self._running
            and self._runs < self.max_runs
            and self._turns_since_run >= self.cadence_turns
        )

    async def run_once(
        self,
        *,
        tasks: tuple[Task, ...],
        recent_turns: list[dict[str, Any]],
        llm: Any,
        model: str,
    ) -> AdvisorOutcome:
        """Make one advisor LLM call and return the parsed outcome (no mutation).

        Re-entrant-safe at entry: a concurrent call (e.g. ``/advise`` fired
        while a periodic run's LLM await is in flight — the forced path skips
        ``should_run``) is dropped, so the run-lock holds for ALL callers, not
        just the periodic trigger.
        """
        if self._running:
            return AdvisorOutcome(task_calls=(), suggestion=None)
        self._running = True
        self._turns_since_run = 0
        self._runs += 1
        try:
            messages = self._build_messages(tasks, recent_turns)
            tools = get_schemas(ADVISOR_TASK_TOOLS)
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": None}
            async for event in llm.stream_response(
                model=model, messages=messages, tools=tools
            ):
                if not isinstance(event, LLMEvent):  # pragma: no cover - defensive
                    continue
                if event.kind == "assistant":
                    payload = event.data
                    if isinstance(payload, LLMResponse):
                        assistant_msg = payload.message
                    elif isinstance(payload, dict):
                        assistant_msg = payload
            return AdvisorOutcome(
                task_calls=tuple(_extract_task_calls(assistant_msg)),
                suggestion=_extract_suggestion(assistant_msg),
            )
        except Exception as exc:
            logger.warning("Advisor.run_once failed: {}", exc)
            return AdvisorOutcome(task_calls=(), suggestion=None)
        finally:
            self._running = False

    def _build_messages(
        self, tasks: tuple[Task, ...], recent_turns: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": _render_context(tasks, recent_turns)},
        ]


def _render_context(tasks: tuple[Task, ...], recent_turns: list[dict[str, Any]]) -> str:
    lines = ["Current task list:"]
    if tasks:
        for task in tasks:
            lines.append(f"  #{task.id} [{task.status}] {task.subject}")
    else:
        lines.append("  (no tasks)")
    lines.append("")
    lines.append("Recent conversation (oldest first):")
    if recent_turns:
        for msg in recent_turns:
            role = str(msg.get("role", "?"))
            content = str(msg.get("content") or "").strip()
            if not content:
                content = "(no text — tool call)"
            lines.append(f"  {role}: {content[:600]}")
    else:
        lines.append("  (nothing yet)")
    lines.append("")
    lines.append(
        "Revise the task list with the tools if needed, and optionally write a "
        "short paragraph of advice."
    )
    return "\n".join(lines)


def _extract_task_calls(assistant_msg: dict[str, Any]) -> list[tuple[str, str]]:
    raw = assistant_msg.get("tool_calls")
    if not isinstance(raw, list):
        return []
    calls: list[tuple[str, str]] = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "")
        if name in ADVISOR_TASK_TOOLS:
            calls.append((name, str(fn.get("arguments") or "{}")))
    return calls


def _extract_suggestion(assistant_msg: dict[str, Any]) -> str | None:
    content = assistant_msg.get("content")
    if isinstance(content, str):
        text = content.strip()
        if text:
            return text
    return None
