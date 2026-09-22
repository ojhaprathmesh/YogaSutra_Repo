"""
app/workflows/sequential.py — Sequential Workflow with Critic Reflection Loop

Executes:
    ProfileAgent -> KnowledgeAgent -> PlannerAgent -> SafetyAgent -> CriticAgent
If CriticAgent returns 'REVISE', triggers an automated reflection loop back to
PlannerAgent with the specific critique, repeating until 'PASS' or max_revisions.
Finally synthesizes response via ResponseAgent and spoken cues via SpeechAgent.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("yogasutra.workflows.sequential")


class SequentialWorkflow:
    """Sequential agent workflow engine with automated reflection and revision loop."""

    def __init__(self, max_revisions: int = 2) -> None:
        self.max_revisions = max_revisions

    def should_revise(self, critique_data: Any, current_revision: int) -> bool:
        """Determines if the candidate plan requires revision."""
        if current_revision >= self.max_revisions:
            return False
        if isinstance(critique_data, dict):
            decision = str(critique_data.get("decision", "")).upper()
            return decision == "REVISE"
        if isinstance(critique_data, str):
            return "REVISE" in critique_data.upper()
        return False

    def build_revision_prompt(self, candidate_plan: Any, critique_data: Any) -> str:
        """Constructs an augmented prompt for the PlannerAgent incorporating critic feedback."""
        critique_summary = critique_data
        if isinstance(critique_data, dict):
            critique_summary = critique_data.get("critique") or critique_data.get("summary") or str(critique_data)

        return (
            f"Please revise the candidate yoga plan to address the following critique:\n"
            f"CRITIQUE FEEDBACK:\n{critique_summary}\n\n"
            f"ORIGINAL CANDIDATE PLAN:\n{candidate_plan}\n\n"
            f"Adjust the sequence, asana selections, or timings to resolve all issues."
        )


# Global singleton sequential workflow
sequential_workflow = SequentialWorkflow()
