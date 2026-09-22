"""
app/workflows/hierarchical.py — Hierarchical Manager Workflow

Implements the Root Manager coordination pattern:
1. Classifies user request intent (full practice vs. safety inquiry vs. knowledge lookup vs. voice cue).
2. Dynamically dispatches only to the relevant specialist agents.
3. Conserves LLM tokens and API calls by avoiding unnecessary full pipeline runs for simple queries.
"""
from __future__ import annotations

import enum
import logging

logger = logging.getLogger("yogasutra.workflows.hierarchical")


class UserIntent(str, enum.Enum):
    PRACTICE_PLANNING = "practice_planning"
    SAFETY_INQUIRY = "safety_inquiry"
    KNOWLEDGE_LOOKUP = "knowledge_lookup"
    VOICE_PRACTICE = "voice_practice"


class HierarchicalWorkflow:
    """Hierarchical Root Manager orchestrator."""

    def classify_intent(self, user_input: str) -> UserIntent:
        """Deterministically classifies user intent from input keywords before LLM delegation."""
        text = user_input.lower().strip()

        # Voice cues request
        if any(w in text for w in ["voice cue", "speak", "cue", "how long do i hold"]):
            return UserIntent.VOICE_PRACTICE

        # Targeted safety question
        if any(w in text for w in ["is it safe", "can i do", "contraindication", "injury", "high bp", "pregnant"]):
            if not any(w in text for w in ["plan", "routine", "session", "morning", "evening"]):
                return UserIntent.SAFETY_INQUIRY

        # Targeted knowledge question
        if any(w in text for w in ["what is", "explain", "sutra", "history of", "benefits of"]):
            if not any(w in text for w in ["plan", "sequence", "routine", "minutes"]):
                return UserIntent.KNOWLEDGE_LOOKUP

        # Default full practice planning
        return UserIntent.PRACTICE_PLANNING

    def get_execution_plan(self, intent: UserIntent) -> list[str]:
        """Returns the list of agent names to invoke for a given intent."""
        if intent == UserIntent.SAFETY_INQUIRY:
            return ["ProfileAgent", "SafetyAgent", "ResponseAgent"]
        elif intent == UserIntent.KNOWLEDGE_LOOKUP:
            return ["KnowledgeAgent", "ResponseAgent"]
        elif intent == UserIntent.VOICE_PRACTICE:
            return ["SpeechAgent"]
        else:
            return [
                "ProfileAgent",
                "KnowledgeAgent",
                "PlannerAgent",
                "SafetyAgent",
                "CriticAgent",
                "ResponseAgent",
                "SpeechAgent",
            ]


# Global singleton hierarchical workflow
hierarchical_workflow = HierarchicalWorkflow()
