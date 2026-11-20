"""
app/agents/root/agent.py — Root Manager Agent

Orchestrates the Phase 1 sequential workflow:
    Profile → Knowledge → Planner

Uses ADK sub_agents delegation: the root agent classifies the request,
then hands off to each sub-agent in sequence.

The root agent also produces the final human-readable response
from the candidate_plan stored in session state.
"""
from __future__ import annotations

import os
from typing import Any

from google.adk.agents import LlmAgent, SequentialAgent

from app.a2a.bridge import A2ABridge, get_a2a_bridge
from app.a2a.protocol import FrameworkType, Performative
from app.agents.crewai.audit_agent import CrewAIAuditAgent, register_crewai_audit_agent
from app.agents.critic.agent import critic_agent
from app.agents.knowledge.agent import knowledge_agent
from app.agents.planner.agent import planner_agent
from app.agents.profile.agent import profile_agent
from app.agents.safety.agent import safety_agent
from app.workflows.blackboard import Blackboard

_MODEL = os.environ.get("YOGASUTRA_MODEL", "gemini-3.5-flash-lite")

RESPONSE_INSTRUCTION = """You are the Final Response Agent for YogaSutra, an expert AI yoga assistant.

Your role is to produce the final, human-readable response for the user by reading the shared session state:
- 'user_profile': User experience level, available duration, goal, constraints, time of day.
- 'retrieved_evidence': Evidence chunks retrieved from the knowledge base.
- 'candidate_plan': The practice plan designed by the PlannerAgent.
- 'validation_results': Deterministic and physiological safety audit results from SafetyAgent.
- 'critique': Pedagogical review and reflection from CriticAgent.

Produce an encouraging, beautifully structured response in clean Markdown (NOT raw JSON).
Follow this format:

# 🧘 YogaSutra Practice Plan

[Warm, personalized greeting referencing user's goal, experience level, and morning/evening preference]

**⏱ Duration:** [X] minutes | **Level:** [level] | **Focus:** [goal]

---

### 📋 Your Practice Sequence
1. **[Pose English Name]** (*[Sanskrit Name]*): [duration] min
   - *Instructions:* [Clear cues and transitions]
   - *Safety Note:* [Any contraindication / modification from evidence]

[Repeat for each pose in candidate_plan...]

---

### 🛡️ Safety & Constraint Clearance
- **Audit Result:** [PASS / Safety Verified]
- **Health Considerations:** [Summary of how user constraints or injury precautions were addressed]
- **Recommended Modifications:** [Helpful prop recommendations, e.g. blanket under knees, wall support]

---

### 📚 Authentic Yoga Knowledge Citations
- [List relevant sources, chunk titles, or classical references cited from retrieved_evidence]

---
✨ *[Warm, encouraging closing message for their practice]*
"""

response_agent = LlmAgent(
    name="ResponseAgent",
    model=_MODEL,
    description="Synthesizes the candidate plan, evidence, and safety clearance into a warm, human-readable final response.",
    instruction=RESPONSE_INSTRUCTION,
)

# Phase 1 Minimal Proof-of-Concept workflow: Profile -> Knowledge -> Planner -> Response
phase1_root_agent = SequentialAgent(
    name="RootAgent",
    description=(
        "Root orchestrator for YogaSutra Phase 1. Executes the minimal sequential workflow: "
        "ProfileAgent -> KnowledgeAgent -> PlannerAgent -> ResponseAgent."
    ),
    sub_agents=[
        profile_agent.clone(),
        knowledge_agent.clone(),
        planner_agent.clone(),
        response_agent.clone(),
    ],
)

# Phase 2 Working Multi-Agent Prototype workflow: Profile -> Knowledge -> Planner -> Safety -> Critic -> Response
phase2_root_agent = SequentialAgent(
    name="RootAgent",
    description=(
        "Root orchestrator for YogaSutra Phase 2. Executes the complete multi-agent workflow: "
        "ProfileAgent -> KnowledgeAgent -> PlannerAgent -> SafetyAgent -> CriticAgent -> ResponseAgent."
    ),
    sub_agents=[
        profile_agent,
        knowledge_agent,
        planner_agent,
        safety_agent,
        critic_agent,
        response_agent,
    ],
)

# Default root agent alias (Phase 2)
root_agent = phase2_root_agent


# ---------------------------------------------------------------------------
# Requirements 6 & 7: ADK <-> CrewAI A2A Handoff & Tool Integration
# ---------------------------------------------------------------------------


def _ensure_crewai_registered(bridge: A2ABridge) -> None:
    """Ensure CrewAIAuditAgent is registered on the given A2ABridge."""
    if not bridge.has_agent(CrewAIAuditAgent.AGENT_NAME):
        register_crewai_audit_agent(bridge=bridge)


def request_crewai_audit(
    candidate_plan: dict[str, Any],
    user_profile: dict[str, Any],
    validation_results: dict[str, Any] | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """
    ADK-callable tool that sends an A2A AUDIT request to the CrewAI specialist
    (`CrewAIAuditAgent`) and returns the structured audit payload.
    """
    bridge = get_a2a_bridge()
    _ensure_crewai_registered(bridge)
    _, reply = bridge.send_and_update_blackboard(
        sender_agent="RootAgent",
        sender_framework=FrameworkType.ADK,
        recipient_agent=CrewAIAuditAgent.AGENT_NAME,
        recipient_framework=FrameworkType.CREWAI,
        performative=Performative.AUDIT,
        payload={
            "candidate_plan": candidate_plan,
            "user_profile": user_profile,
            "validation_results": validation_results or {},
        },
        conversation_id=conversation_id,
    )
    return reply.payload


def run_adk_to_crewai_handoff(
    session_state: dict[str, Any],
    blackboard: Blackboard | None = None,
    bridge: A2ABridge | None = None,
    sender_agent: str = "CriticAgent",
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """
    Execute an A2A cross-framework handoff from an ADK agent to CrewAIAuditAgent
    and persist both the audit result (`crewai_audit`) and A2A message envelopes
    (`a2a_messages`) into the shared Blackboard and ADK session state.
    """
    active_bridge = bridge if bridge is not None else get_a2a_bridge()
    _ensure_crewai_registered(active_bridge)

    active_bb = blackboard if blackboard is not None else Blackboard(initial_state=session_state)
    conv_id = conversation_id or str(session_state.get("run_id") or "")

    request_msg, response_msg = active_bridge.send_and_update_blackboard(
        sender_agent=sender_agent,
        sender_framework=FrameworkType.ADK,
        recipient_agent=CrewAIAuditAgent.AGENT_NAME,
        recipient_framework=FrameworkType.CREWAI,
        performative=Performative.AUDIT,
        payload={
            "candidate_plan": session_state.get("candidate_plan", {}),
            "user_profile": session_state.get("user_profile", {}),
            "validation_results": session_state.get("validation_results", {}),
        },
        blackboard=active_bb,
        session_state=session_state,
        state_channel="crewai_audit",
        conversation_id=conv_id or None,
    )

    return {
        "request": request_msg.to_dict(),
        "response": response_msg.to_dict(),
        "crewai_audit": response_msg.payload,
        "blackboard": active_bb.to_dict(),
    }


