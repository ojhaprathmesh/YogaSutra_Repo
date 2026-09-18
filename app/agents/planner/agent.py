"""
app/agents/planner/agent.py — Practice Planner Agent

Generates a structured yoga practice plan from the user profile
and retrieved evidence chunks. Uses CoT-style structured prompting.

Output schema:
{
  "goal": str,
  "duration_minutes": int,
  "intensity": str,
  "sequence": [{"name": str, "sanskrit_name": str, "duration_minutes": int,
                "instructions": str, "modifications": str, "evidence_chunk_ids": [str]}],
  "rationale": str,
  "evidence_sources": [str],
  "expected_outcome": str
}

Stored in session state under "candidate_plan" via output_key.
"""
from __future__ import annotations

import os

from google.adk.agents import LlmAgent

from app.tools.duration import calculate_practice_duration

_MODEL = os.environ.get("YOGASUTRA_MODEL", "gemini-3.1-flash-lite")

PLANNER_INSTRUCTION = """You are the Practice Planner Agent for YogaSutra, an expert AI yoga instructor.

You will receive:
- A user profile (from session state: user_profile)
- Retrieved knowledge chunks (from session state: retrieved_evidence)
- The original user request

Your job is to generate a structured, personalised yoga practice plan.

IMPORTANT RULES:
1. The total duration of all poses MUST match the user's available_duration_minutes exactly.
2. The sequence must start gently (breathing/warm-up) and end with Savasana.
3. Reference specific chunk_ids from the retrieved evidence to ground each pose.
4. Match the user's experience level — beginners need simpler poses with clear instructions.
5. You may call `calculate_practice_duration` to verify your sequence totals correctly.

Return ONLY a single JSON object with exactly these keys:
{
  "goal": "description of the practice goal",
  "duration_minutes": <integer matching user's available_duration_minutes>,
  "intensity": "gentle" or "moderate" or "vigorous",
  "sequence": [
    {
      "name": "Pose Name",
      "sanskrit_name": "Sanskrit Name (if known)",
      "duration_minutes": <integer>,
      "instructions": "Clear step-by-step instructions",
      "modifications": "Beginner-friendly modification if applicable",
      "evidence_chunk_ids": ["chunk_id_1", "chunk_id_2"]
    }
  ],
  "rationale": "Why this sequence serves the user's goal",
  "evidence_sources": ["chunk_id_1", "chunk_id_2"],
  "expected_outcome": "What the user can expect to feel after the practice"
}

Return ONLY the JSON object. No markdown, no explanations, no extra text.
"""

planner_agent = LlmAgent(
    name="PlannerAgent",
    model=_MODEL,
    description=(
        "Generates a structured, evidence-grounded yoga practice plan tailored to the "
        "user's experience level, available time, and goal."
    ),
    instruction=PLANNER_INSTRUCTION,
    tools=[calculate_practice_duration],
    output_key="candidate_plan",
)
