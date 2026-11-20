"""
app/a2a — Agent-to-Agent (A2A) Interoperability Protocol Package

Provides standardized message envelopes, agent capability cards, and an
in-process dispatcher/bridge enabling cross-framework communication between
Google ADK agents and CrewAI specialist agents.
"""
from app.a2a.bridge import (
    A2ABridge,
    A2ADispatcher,
    get_a2a_bridge,
    reset_a2a_bridge,
)
from app.a2a.protocol import (
    A2AAgentCard,
    A2AMessage,
    FrameworkType,
    Performative,
)

__all__ = [
    "A2AAgentCard",
    "A2ABridge",
    "A2ADispatcher",
    "A2AMessage",
    "FrameworkType",
    "Performative",
    "get_a2a_bridge",
    "reset_a2a_bridge",
]
