"""Append-only agent loop over a streaming LLM client."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from loguru import logger

from neutrix.config import Slot
from neutrix.llm import LLMEvent, LLMResponse, OpenAIChatLLM
from neutrix.store import ChatStore, Task
from neutrix.tools import dispatch, get_schemas

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Keep it simple."

# Matches Claude Code's TODO_REMINDER_CONFIG exactly.
TURNS_SINCE_WRITE = 10
TURNS_BETWEEN_REMINDERS = 10
TASK_REMINDER_TAG_OPEN = "<system-reminder>"
TASK_REMINDER_TAG_CLOSE = "</system-reminder>"
TASK_REMINDER_MARKER = "Here are the existing tasks:"
TASK_MANAGEMENT_TOOLS = frozenset({"TaskCreate", "TaskUpdate"})


@dataclass(frozen=True)
class AgentEvent:
    """Stream event yielded by Agent.stream_reply."""

    kind: str  # "token" | "tool_call" | "tool_result" | "done" | "error"
    data: Any = None


class ChatLLM(Protocol):
    def switch(self, slot: Slot) -> None: ...

    def stream_response(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMEvent]: ...


@dataclass
class Agent:
    """Conversation state plus the model/tool continuation loop."""

    slot: Slot
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    use_tools: bool = True
    messages: list[dict[str, Any]] = field(default_factory=list)
    llm: ChatLLM | None = field(default=None, repr=False)
    store: ChatStore | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.messages:
            self.messages = [{"role": "system", "content": self.system_prompt}]
        if self.llm is None:
            self.llm = OpenAIChatLLM(self.slot)

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def switch(self, slot: Slot) -> None:
        self.slot = slot
        assert self.llm is not None
        self.llm.switch(slot)

    def supports_tools(self) -> bool:
        return supports_openai_tools(self.slot)

    def effective_tools_enabled(self) -> bool:
        return self.use_tools and self.supports_tools()

    async def stream_reply(self, user_text: str) -> AsyncIterator[AgentEvent]:
        """Append a user turn and continue while tools require follow-up."""

        self.messages.append({"role": "user", "content": user_text})
        self._maybe_inject_task_reminder()

        while True:
            try:
                assistant_msg: dict[str, Any] | None = None
                rendered_tokens = False
                tools = get_schemas() if self.effective_tools_enabled() else None
                assert self.llm is not None
                async for event in self.llm.stream_response(
                    model=self.slot.model,
                    messages=self.messages,
                    tools=tools,
                ):
                    if event.kind == "token":
                        rendered_tokens = True
                        yield AgentEvent("token", event.data)
                    elif event.kind == "assistant":
                        response = event.data
                        if isinstance(response, LLMResponse):
                            assistant_msg = response.message
                        else:
                            assistant_msg = response

                if assistant_msg is None:
                    assistant_msg = {"role": "assistant", "content": None}
                self.messages.append(assistant_msg)
                content = assistant_msg.get("content")
                if isinstance(content, str) and content and not rendered_tokens:
                    yield AgentEvent("assistant", content)

                tool_calls = self._tool_calls(assistant_msg)
                if not tool_calls:
                    yield AgentEvent("done")
                    return

                for tool_call in tool_calls:
                    name = tool_call["name"]
                    arguments = tool_call["arguments"]
                    yield AgentEvent(
                        "tool_call",
                        {"name": name, "arguments": arguments},
                    )
                    result = await asyncio.to_thread(
                        _dispatch_with_store, name, arguments, self.store
                    )
                    yield AgentEvent(
                        "tool_result",
                        {"name": name, "result": result},
                    )
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": result,
                        }
                    )
            except Exception as exc:
                error = compact_error(exc)
                logger.warning("agent loop failed: {}", error)
                yield AgentEvent("error", error)
                return

    def _tool_calls(self, assistant_msg: dict[str, Any]) -> list[dict[str, str]]:
        raw_tool_calls = assistant_msg.get("tool_calls")
        if not isinstance(raw_tool_calls, list):
            return []

        tool_calls: list[dict[str, str]] = []
        for raw_tool_call in raw_tool_calls:
            if not isinstance(raw_tool_call, dict):
                continue
            function = raw_tool_call.get("function")
            if not isinstance(function, dict):
                continue
            tool_calls.append(
                {
                    "id": str(raw_tool_call.get("id") or ""),
                    "name": str(function.get("name") or ""),
                    "arguments": str(function.get("arguments") or ""),
                }
            )
        return tool_calls

    def _maybe_inject_task_reminder(self) -> dict[str, Any] | None:
        """Append a Claude-shaped task reminder if both thresholds met.

        Called once per :py:meth:`stream_reply` invocation, after the
        new user turn is appended and before the first LLM request.
        Tool-driven follow-up rounds inside the same ``stream_reply``
        do NOT re-check — the reminder is a per-turn nudge, not a
        per-round one.
        """
        if self.store is None:
            return None
        reminder = build_task_reminder(self.messages, self.store.tasks)
        if reminder is None:
            return None
        self.messages.append(reminder)
        return reminder


def supports_openai_tools(slot: Slot) -> bool:
    """Whether this slot accepts OpenAI Chat Completions function tools."""
    model = slot.model.lower()
    provider = slot.provider.lower()
    if provider == "ihep" and model.startswith("anthropic/"):
        return False
    return True


def compact_error(exc: Exception, *, limit: int = 600) -> str:
    text = str(exc).replace("\n", " ").strip() or exc.__class__.__name__
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


# ---- task reminder algorithm ------------------------------------------------


def build_task_reminder(
    messages: list[dict[str, Any]],
    tasks: tuple[Task, ...],
) -> dict[str, Any] | None:
    """Return a Claude-shaped ``<system-reminder>`` user message if due.

    Conditions (all must hold):

    1. At least one task is currently ``pending`` or ``in_progress``.
    2. ``TURNS_SINCE_WRITE`` or more assistant turns have elapsed since
       the LLM last called ``TaskCreate`` or ``TaskUpdate``.
    3. ``TURNS_BETWEEN_REMINDERS`` or more assistant turns have elapsed
       since the previous reminder was injected.

    "Assistant turns" here means messages with ``role == "assistant"``
    in ``messages`` — that's what an LLM round emits one of per round,
    and what the user-perceived turn boundary aligns with.
    """
    actionable = [t for t in tasks if t.status in ("pending", "in_progress")]
    if not actionable:
        return None
    if assistant_turns_since_task_management(messages) < TURNS_SINCE_WRITE:
        return None
    if assistant_turns_since_reminder(messages) < TURNS_BETWEEN_REMINDERS:
        return None
    body = _build_task_reminder_body(tasks)
    return {
        "role": "user",
        "content": f"{TASK_REMINDER_TAG_OPEN}\n{body}\n{TASK_REMINDER_TAG_CLOSE}",
    }


def assistant_turns_since_task_management(messages: list[dict[str, Any]]) -> int:
    """Count assistant messages scanning backwards until one is found whose
    ``tool_calls`` includes ``TaskCreate`` or ``TaskUpdate``.

    Returns the total assistant-turn count when no such call exists, so
    a fresh conversation with no prior task management always satisfies
    the threshold.
    """
    seen = 0
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        seen += 1
        if _message_calls_task_management(message):
            return seen - 1
    return seen


def assistant_turns_since_reminder(messages: list[dict[str, Any]]) -> int:
    """Count assistant messages scanning backwards until one is preceded
    or followed by an already-injected reminder.

    More precisely: scan backwards through ``messages`` and count any
    ``role == "assistant"`` entry; if a ``role == "user"`` entry whose
    content starts with the ``<system-reminder>`` tag and contains the
    task-listing marker is seen, return the accumulated assistant count.
    Returns the total assistant-turn count when no prior reminder exists.
    """
    seen = 0
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            seen += 1
            continue
        if role == "user" and _is_task_reminder(message.get("content")):
            return seen
    return seen


def _message_calls_task_management(message: dict[str, Any]) -> bool:
    raw_tool_calls = message.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return False
    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, dict):
            continue
        function = raw_tool_call.get("function")
        if not isinstance(function, dict):
            continue
        if str(function.get("name") or "") in TASK_MANAGEMENT_TOOLS:
            return True
    return False


def _is_task_reminder(content: Any) -> bool:
    if not isinstance(content, str):
        return False
    if not content.startswith(TASK_REMINDER_TAG_OPEN):
        return False
    return TASK_REMINDER_MARKER in content


def _build_task_reminder_body(tasks: tuple[Task, ...]) -> str:
    actionable = [t for t in tasks if t.status in ("pending", "in_progress")]
    lines = [
        "The task tools haven't been used recently. If you're working on "
        "tasks that would benefit from tracking progress, consider using "
        "TaskCreate to add new tasks and TaskUpdate to update task status "
        "(set to in_progress when starting, completed when done). Also "
        "consider cleaning up the task list if it has become stale. Only "
        "use these if relevant to the current work. This is just a gentle "
        "reminder - ignore if not applicable. Make sure that you NEVER "
        "mention this reminder to the user.",
    ]
    if actionable:
        lines.append("")
        lines.append(TASK_REMINDER_MARKER)
        lines.append("")
        for task in actionable:
            lines.append(f"#{task.id}. [{task.status}] {task.subject}")
    return "\n".join(lines)


def _dispatch_with_store(
    name: str,
    arguments: str,
    store: ChatStore | None,
) -> str:
    """asyncio.to_thread shim — dispatch always sees the live store."""
    return dispatch(name, arguments, store=store)
