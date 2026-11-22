"""
app/workflows/sequential.py — Sequential Workflow with Critic Reflection Loop

Executes:
    ProfileAgent -> KnowledgeAgent -> PlannerAgent -> SafetyAgent -> CriticAgent
If CriticAgent returns 'REVISION_REQUIRED' (or 'REVISE' / 'FAIL'), triggers an
automated reflection loop back to PlannerAgent with the specific critique,
repeating until 'PASS' or max_revisions.
Finally synthesizes response via ResponseAgent and spoken cues via SpeechAgent.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("yogasutra.workflows.sequential")

_REVISE_DECISIONS = frozenset({"REVISION_REQUIRED", "REVISE", "FAIL"})


def _try_parse_critique_json(raw: str) -> dict[str, Any] | None:
    """Attempt to parse a JSON object (with or without markdown fences) from critique text."""
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return None


class SequentialWorkflow:
    """Sequential agent workflow engine with automated reflection and revision loop."""

    def __init__(self, max_revisions: int = 2) -> None:
        self.max_revisions = max_revisions

    def should_revise(self, critique_data: Any, current_revision: int) -> bool:
        """Determines if the candidate plan requires revision."""
        if current_revision >= self.max_revisions:
            return False
        if isinstance(critique_data, dict):
            decision = str(critique_data.get("decision", "")).strip().upper()
            return decision in _REVISE_DECISIONS
        if isinstance(critique_data, str):
            parsed = _try_parse_critique_json(critique_data)
            if parsed is not None and "decision" in parsed:
                decision = str(parsed.get("decision", "")).strip().upper()
                return decision in _REVISE_DECISIONS
            upper = critique_data.strip().upper()
            if "REVISION_REQUIRED" in upper or "REVISE" in upper:
                return True
            if re.search(r"\bFAIL\b", upper):
                return True
            return False
        return False

    def build_revision_prompt(self, candidate_plan: Any, critique_data: Any) -> str:
        """Constructs an augmented prompt for the PlannerAgent incorporating critic feedback."""
        critique_summary = critique_data
        if isinstance(critique_data, dict):
            critique_summary = (
                critique_data.get("critique")
                or critique_data.get("summary")
                or str(critique_data)
            )
            mods = critique_data.get("recommended_modifications")
            if mods:
                mods_text = (
                    "\n".join(f"- {m}" for m in mods)
                    if isinstance(mods, list)
                    else str(mods)
                )
                critique_summary = (
                    f"{critique_summary}\n\nRECOMMENDED MODIFICATIONS:\n{mods_text}"
                )

        return (
            f"Please revise the candidate yoga plan to address the following critique:\n"
            f"CRITIQUE FEEDBACK:\n{critique_summary}\n\n"
            f"ORIGINAL CANDIDATE PLAN:\n{candidate_plan}\n\n"
            f"Adjust the sequence, asana selections, or timings to resolve all issues."
        )


# Global singleton sequential workflow
sequential_workflow = SequentialWorkflow()
