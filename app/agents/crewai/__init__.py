"""
app/agents/crewai — CrewAI Specialist Agents for YogaSutra (Requirement 7)

Provides cross-framework CrewAI specialist agents (using CrewAI's Agent, Task,
and Crew abstractions with a resilient in-process fallback when the CrewAI
runtime is unavailable on Python 3.14) integrated via the A2A protocol.
"""
from app.agents.crewai.audit_agent import (
    CREWAI_AVAILABLE,
    CrewAIAuditAgent,
    CrewAICriticAgent,
    FallbackAgent,
    FallbackCrew,
    FallbackProcess,
    FallbackTask,
    register_crewai_audit_agent,
)

__all__ = [
    "CREWAI_AVAILABLE",
    "CrewAIAuditAgent",
    "CrewAICriticAgent",
    "FallbackAgent",
    "FallbackCrew",
    "FallbackProcess",
    "FallbackTask",
    "register_crewai_audit_agent",
]
