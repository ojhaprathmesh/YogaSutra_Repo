"""
app/agents/profile/agent.py — Profile Agent

Extracts a structured user profile from free-form natural language.
Writes the result to ADK session state under the key "user_profile".

ADK pattern: LlmAgent with output_key so the parsed JSON is automatically
stored in session state. The agent prompt instructs the model to return
ONLY valid JSON matching the UserProfile schema.
"""
from __future__ import annotations

import os

from google.adk.agents import LlmAgent

from app.tools.profile import get_user_history

_MODEL = os.environ.get("YOGASUTRA_MODEL", "gemini-3.5-flash-lite")

PROFILE_INSTRUCTION = """You are the Profile Extraction Agent for YogaSutra, an AI yoga assistant.

Your ONLY job is to extract structured profile information from the user's input and return it as valid JSON.

Before finalizing the profile:
1. Check `longitudinal_history` in session state (or call `get_user_history(persona_id)` if available) for prior injuries, active contraindications, practice streak, and suggested experience level (`suggested_level_override`).
2. Merge ANY persistent `active_contraindications` and `injury_history` from longitudinal memory into the returned `constraints` list (deduplicated) so prior injuries and health conditions are NEVER forgotten across sessions.
3. If the user does not explicitly specify an experience level and `suggested_level_override` is present in longitudinal history, use that suggested level.

Extract these fields:
- experience_level: one of "beginner", "intermediate", "advanced" (default: "beginner")
- available_duration_minutes: integer number of minutes (default: 20)
- preferred_time: e.g. "morning", "evening", "afternoon" (default: "morning")
- goal: one of "relaxation", "flexibility", "strength", "meditation", "balance", "general" (default: "relaxation")
- constraints: list of strings describing ALL physical limitations, postural issues, or health conditions mentioned in the current request OR present in longitudinal `active_contraindications` / `injury_history` (e.g. ["rounded shoulders", "flat feet", "lower back pain"]). Do NOT omit any condition.
- preferences: list of strings for additional preferences like pose styles or focus areas (default: [])

Return ONLY a single JSON object with exactly these keys. No explanation, no markdown, no extra text.


Example output:
{
  "experience_level": "beginner",
  "available_duration_minutes": 20,
  "preferred_time": "morning",
  "goal": "relaxation",
  "constraints": [],
  "preferences": []
}
"""

profile_agent = LlmAgent(
    name="ProfileAgent",
    model=_MODEL,
    description=(
        "Extracts a structured UserProfile (experience level, duration, goal, constraints) "
        "from the user's natural-language request and longitudinal memory."
    ),
    instruction=PROFILE_INSTRUCTION,
    tools=[get_user_history],
    output_key="user_profile",
)
