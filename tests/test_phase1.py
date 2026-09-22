"""
tests/test_phase1.py — Validation tests for Phase 1 Minimal Proof of Concept.

Validates that the Phase 1 implementation aligns with PLAN.md (§35 & §43):
- Phase 1 Minimal Proof of Concept sequential agent: Profile -> Knowledge -> Planner -> Response
- Phase 1 Demo Scenario: "I am a beginner and have 20 minutes in the morning. I want a relaxing yoga practice."
- Phase 1 pipeline visualizer nodes (5 agents, no Safety/Critic in Phase 1)
- Phase 1 session state & execution trace recording
"""
from app.agents.root.agent import phase1_root_agent, phase2_root_agent
from app.state import YogaSessionState
from app.tracing.trace import ExecutionTrace
from app.ui.live_graph import PipelineGraph, PIPELINE_NODES_PHASE1
from main import DEFAULT_INPUT


class TestPhase1Architecture:
    """Verify Phase 1 agent definitions and sequential workflow."""

    def test_phase1_agent_sub_agents(self):
        """Phase 1 RootAgent must orchestrate only Profile, Knowledge, Planner, and Response."""
        expected_names = [
            "ProfileAgent",
            "KnowledgeAgent",
            "PlannerAgent",
            "ResponseAgent",
        ]
        actual_names = [agent.name for agent in phase1_root_agent.sub_agents]
        assert actual_names == expected_names

    def test_phase1_does_not_contain_phase2_agents(self):
        """Phase 1 must NOT include SafetyAgent or CriticAgent."""
        actual_names = [agent.name for agent in phase1_root_agent.sub_agents]
        assert "SafetyAgent" not in actual_names
        assert "CriticAgent" not in actual_names

    def test_phase2_contains_all_agents(self):
        """Phase 2 RootAgent must orchestrate all 6 agents."""
        actual_names = [agent.name for agent in phase2_root_agent.sub_agents]
        assert len(actual_names) == 6
        assert "SafetyAgent" in actual_names
        assert "CriticAgent" in actual_names

    def test_phase1_demo_scenario_default(self):
        """Default demo input matches Phase 1 Demo Scenario from PLAN.md."""
        assert "beginner" in DEFAULT_INPUT
        assert "20 minutes" in DEFAULT_INPUT
        assert "morning" in DEFAULT_INPUT
        assert "relaxing" in DEFAULT_INPUT


class TestPhase1StateAndTracing:
    """Verify Phase 1 session state and trace metadata."""

    def test_session_state_phase_field(self):
        state = YogaSessionState()
        assert hasattr(state, "phase")

    def test_execution_trace_phase_recording(self):
        trace = ExecutionTrace(run_id="run_test", session_id="sess_test", phase=1)
        assert trace.phase == 1
        d = trace.to_dict()
        assert d["phase"] == 1

    def test_execution_trace_record_state_change(self):
        trace = ExecutionTrace(run_id="run_test", session_id="sess_test", phase=1)
        trace.record_state_change("candidate_plan", "Populated by pipeline", "PlannerAgent")
        assert len(trace.state_changes) == 1
        change = trace.state_changes[0]
        assert change["field"] == "candidate_plan"
        assert change["key"] == "candidate_plan"
        assert change["agent"] == "PlannerAgent"
        assert change["summary"] == "Populated by pipeline"


class TestPhase1LiveGraph:
    """Verify Phase 1 live terminal visualizer layout."""

    def test_phase1_pipeline_nodes(self):
        assert PIPELINE_NODES_PHASE1 == [
            "RootAgent",
            "ProfileAgent",
            "KnowledgeAgent",
            "PlannerAgent",
            "ResponseAgent",
        ]

    def test_graph_initializes_with_phase1_nodes(self):
        g = PipelineGraph(user_input=DEFAULT_INPUT, phase=1)
        assert len(g._nodes) == 5
        assert "SafetyAgent" not in g._nodes
        assert "CriticAgent" not in g._nodes

    def test_graph_phase1_render(self):
        g = PipelineGraph(user_input=DEFAULT_INPUT, phase=1)
        rendered = g._render()
        assert rendered is not None
