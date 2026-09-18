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

from google.adk.agents import LlmAgent, SequentialAgent

from app.agents.knowledge.agent import knowledge_agent
from app.agents.planner.agent import planner_agent
from app.agents.profile.agent import profile_agent

_MODEL = os.environ.get("YOGASUTRA_MODEL", "gemini-3.1-flash-lite")

RESPONSE_INSTRUCTION = """You are the Final Response Agent for YogaSutra, an expert AI yoga assistant.

Your role is to produce the final, human-readable response for the user by reading the shared session state:
- 'user_profile': User experience level, available duration, goal, time of day.
- 'retrieved_evidence': Evidence chunks retrieved from the knowledge base.
- 'candidate_plan': The practice plan designed by the PlannerAgent.

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

### 📚 Authentic Yoga Knowledge Citations
- [List relevant sources and sections cited from retrieved_evidence]

---
✨ *[Warm, encouraging closing message for their practice]*
"""

response_agent = LlmAgent(
    name="ResponseAgent",
    model=_MODEL,
    description="Synthesizes the candidate plan and evidence into a warm, human-readable final response.",
    instruction=RESPONSE_INSTRUCTION,
)

root_agent = SequentialAgent(
    name="RootAgent",
    description=(
        "Root orchestrator for YogaSutra. Executes the sequential Phase 1 workflow: "
        "ProfileAgent -> KnowledgeAgent -> PlannerAgent -> ResponseAgent."
    ),
    sub_agents=[profile_agent, knowledge_agent, planner_agent, response_agent],
)
