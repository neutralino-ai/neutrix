"""Append-only agent loop over a streaming LLM client."""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from loguru import logger

from neutrix.config import Slot
from neutrix.llm import LLMEvent, LLMResponse, OpenAIChatLLM
from neutrix.tools import dispatch, get_schemas

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Keep it simple."


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

    async def stream_reply(self, user_text: str) -> AsyncIterator[AgentEvent]:
        """Append a user turn and continue while tools require follow-up."""

        self.messages.append({"role": "user", "content": user_text})

        while True:
            try:
                assistant_msg: dict[str, Any] | None = None
                tools = get_schemas() if self.use_tools else None
                assert self.llm is not None
                async for event in self.llm.stream_response(
                    model=self.slot.model,
                    messages=self.messages,
                    tools=tools,
                ):
                    if event.kind == "token":
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
                    result = dispatch(name, arguments)
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
                logger.exception("agent loop failed")
                yield AgentEvent("error", str(exc))
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
