"""A minimal alternate renderer that reads ONLY ChatStore (v0.10.3 swap proof).

The point isn't a second product surface — it's architectural hygiene. ``FakeView``
imports nothing but :mod:`neutrix.store` (+ stdlib): no ``context_manager``,
``llm``, ``tools``, or ``executor``. If it ever needs another import, the store
is not the single state holder, and the v0.10.4 Advisor's "I only mutate the
store" model has a hole. ``tests/test_fake_view.py`` asserts both the rendered
output across scenarios AND the import boundary.
"""
from __future__ import annotations

from neutrix.store import ChatStore

INTERRUPTED_MARKER = "[interrupted by user]"


class FakeView:
    """Render the whole user-visible surface from a ChatStore snapshot."""

    def render(self, store: ChatStore) -> str:
        lines: list[str] = []
        for msg in store.messages:
            content = msg.content or ""
            if msg.role == "system":
                lines.append(f"[system] {content}")
            elif msg.role == "user":
                if content == INTERRUPTED_MARKER:
                    lines.append("[cancelled] interrupted by user")
                elif content.startswith("<system-reminder>"):
                    lines.append("[reminder] (folded)")
                else:
                    lines.append(f"> {content}")
            elif msg.role == "assistant":
                if content:
                    lines.append(f"< {content}")
                for call in self._tool_calls(msg):
                    lines.append(f"  -> {call}")
            elif msg.role == "tool":
                lines.append(f"  <- tool_result {content[:40]}")
        if store.pending_assistant_text:
            lines.append(f"< {store.pending_assistant_text}…")
        for queued in store.queued_user_messages:
            lines.append(f"» {queued.text}")
        for rec in store.folded_tool_results:
            label = "subagent" if rec.name == "Agent" else "tool_result"
            lines.append(f"[{label} {rec.index}] {rec.name} ({len(rec.result)} B)")
        for task in store.tasks:
            lines.append(f"#{task.id} [{task.status}] {task.subject}")
        return "\n".join(lines)

    @staticmethod
    def _tool_calls(msg: object) -> list[str]:
        extra = getattr(msg, "extra", None) or {}
        calls = extra.get("tool_calls") if isinstance(extra, dict) else None
        names: list[str] = []
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, dict):
                    fn = call.get("function", {})
                    names.append(str(fn.get("name", "?")) if isinstance(fn, dict) else "?")
        return names
