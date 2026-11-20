"""
app/a2a/protocol.py — Standard Agent-to-Agent (A2A) Protocol Schema

Defines the interoperable message envelope and agent capability card used for
cross-framework communication (e.g., Google ADK <-> CrewAI) in YogaSutra.

Standard envelope fields (Requirement 6):
    - message_id:          Globally unique message identifier
    - sender_agent:        Logical name of the sending agent
    - sender_framework:    Originating framework (google_adk, crewai, etc.)
    - recipient_agent:     Logical name of the target agent
    - recipient_framework: Destination framework (crewai, google_adk, etc.)
    - performative:        FIPA-ACL / A2A communicative act (REQUEST, INFORM, etc.)
    - conversation_id:     Correlation ID linking a multi-turn A2A exchange
    - payload:             Structured JSON-serializable message body
    - timestamp:           UTC ISO-8601 timestamp
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class FrameworkType(StrEnum):
    """Supported agent execution frameworks in the A2A ecosystem."""

    ADK = "google_adk"
    CREWAI = "crewai"
    LANGGRAPH = "langgraph"
    MCP = "mcp"
    CUSTOM = "custom"
    SYSTEM = "system"


class Performative(StrEnum):
    """Standardized A2A / FIPA-ACL communicative performatives."""

    REQUEST = "REQUEST"
    INFORM = "INFORM"
    PROPOSE = "PROPOSE"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    QUERY = "QUERY"
    AUDIT = "AUDIT"
    CRITIQUE = "CRITIQUE"
    ACK = "ACK"
    ERROR = "ERROR"


class A2AAgentCard(BaseModel):
    """Capability advertisement for an A2A-discoverable agent."""

    agent_name: str
    framework: FrameworkType
    role: str
    description: str = ""
    capabilities: list[str] = Field(default_factory=list)
    supported_performatives: list[Performative] = Field(
        default_factory=lambda: [Performative.REQUEST, Performative.INFORM]
    )
    version: str = "1.0"

    model_config = {"use_enum_values": True}


class A2AMessage(BaseModel):
    """
    Standard Agent-to-Agent (A2A) protocol envelope.

    All cross-framework interactions (such as Google ADK delegating a
    safety/pedagogical audit to a CrewAI specialist) are wrapped in this
    validated, JSON-serializable envelope.
    """

    message_id: str = Field(default_factory=lambda: f"a2a_{uuid.uuid4().hex[:12]}")
    sender_agent: str
    sender_framework: FrameworkType
    recipient_agent: str
    recipient_framework: FrameworkType
    performative: Performative
    conversation_id: str = Field(default_factory=lambda: f"conv_{uuid.uuid4().hex[:10]}")
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Optional correlation & protocol metadata
    in_reply_to: str | None = None
    protocol_version: str = "1.0"
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}

    def create_reply(
        self,
        performative: Performative,
        payload: dict[str, Any],
        sender_agent: str | None = None,
        sender_framework: FrameworkType | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> A2AMessage:
        """
        Construct a correlated reply envelope to this message.

        Automatically swaps sender/recipient and preserves conversation_id
        while linking `in_reply_to` to this message's `message_id`.
        """
        return A2AMessage(
            sender_agent=sender_agent or self.recipient_agent,
            sender_framework=sender_framework or FrameworkType(self.recipient_framework),
            recipient_agent=self.sender_agent,
            recipient_framework=FrameworkType(self.sender_framework),
            performative=performative,
            conversation_id=self.conversation_id,
            payload=payload,
            in_reply_to=self.message_id,
            protocol_version=self.protocol_version,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the envelope to a plain JSON-compatible dictionary."""
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Serialize the envelope to a JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> A2AMessage:
        """Deserialize and validate an A2AMessage from a dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, raw: str) -> A2AMessage:
        """Deserialize and validate an A2AMessage from a JSON string."""
        return cls.model_validate_json(raw)
