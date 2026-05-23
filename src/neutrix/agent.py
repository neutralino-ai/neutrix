"""Compatibility facade for the public agent API."""
from __future__ import annotations

from neutrix.agent_loop import DEFAULT_SYSTEM_PROMPT, Agent, AgentEvent

__all__ = ["DEFAULT_SYSTEM_PROMPT", "Agent", "AgentEvent"]
