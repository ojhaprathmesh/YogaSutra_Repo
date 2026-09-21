"""
tests/test_reflection.py — Unit Tests for Safety Agent, Critic Agent, and Multi-Agent Orchestration
"""
from __future__ import annotations

from app.agents.critic.agent import critic_agent
from app.agents.root.agent import root_agent
from app.agents.safety.agent import safety_agent
from app.state import YogaSessionState


class TestReflectionAndOrchestration:
    """Tests for Phase 2 multi-agent components and reflection setup."""

    def test_safety_agent_configured(self):
        assert safety_agent.name == "SafetyAgent"
        assert safety_agent.output_key == "validation_results"
        assert len(safety_agent.tools) >= 1

    def test_critic_agent_configured(self):
        assert critic_agent.name == "CriticAgent"
        assert critic_agent.output_key == "critique"

    def test_root_agent_includes_all_phase2_sub_agents(self):
        sub_agent_names = [agent.name for agent in root_agent.sub_agents]
        expected = [
            "ProfileAgent",
            "KnowledgeAgent",
            "PlannerAgent",
            "SafetyAgent",
            "CriticAgent",
            "ResponseAgent",
        ]
        assert sub_agent_names == expected

    def test_session_state_records_phase2_fields(self):
        state = YogaSessionState(
            user_input="Test input",
            validation_results={"status": "PASS", "is_safe": True},
            critique="Plan is well balanced.",
            revision_count=1,
        )
        assert state.validation_results["status"] == "PASS"
        assert state.critique == "Plan is well balanced."
        assert state.revision_count == 1
