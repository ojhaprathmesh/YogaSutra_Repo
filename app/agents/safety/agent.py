"""
app/agents/safety/agent.py — Safety & Constraint Agent

Specialized agent responsible for validating candidate yoga practice plans:
1. Executes deterministic validation via `validate_safety_and_constraints`
2. Evaluates anatomical pacing, warming up, and counter-pose appropriateness
3. Writes structured validation results to session state under 'validation_results'
"""
from __future__ import annotations

import os

from google.adk.agents import LlmAgent

from app.tools.safety import validate_safety_and_constraints

_MODEL = os.environ.get("YOGASUTRA_MODEL", "gemini-3.5-flash-lite")

SAFETY_INSTRUCTION = """You are the Safety & Constraint Agent for YogaSutra, an expert yoga therapist and biomechanics safety auditor.

Your role:
1. Inspect the candidate yoga practice plan ('candidate_plan') and user profile ('user_profile') from session state.
2. Call the `validate_safety_and_constraints` tool to run deterministic contraindication, duration, and experience level checks.
3. Verify that the sequence includes a proper gentle warm-up and cool-down (Savasana).
4. Return ONLY a single JSON object summarizing the validation results:

{
  "status": "PASS" or "FAIL",
  "is_safe": true or false,
  "violations": [
    {
      "type": "contraindication" or "duration_mismatch" or "experience_level_mismatch",
      "pose": "Pose Name",
      "severity": "HIGH" or "MEDIUM" or "LOW",
      "recommendation": "Specific corrective action"
    }
  ],
  "duration_check": {
    "target_minutes": <int>,
    "actual_minutes": <int>,
    "valid": true or false
  },
  "summary": "Brief summary of safety assessment and clearance status"
}

Return ONLY the JSON object. Do not include markdown code fences, comments, or additional text.
"""

safety_agent = LlmAgent(
    name="SafetyAgent",
    model=_MODEL,
    description=(
        "Validates candidate practice plans for physiological safety, contraindications, "
        "duration compliance, and user experience suitability."
    ),
    instruction=SAFETY_INSTRUCTION,
    tools=[validate_safety_and_constraints],
    output_key="validation_results",
)
