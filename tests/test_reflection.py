"""
tests/test_reflection.py — Unit Tests for Safety Agent, Critic Agent, and Multi-Agent Orchestration
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from app.agents.critic.agent import critic_agent
from app.agents.root.agent import create_revision_agent, root_agent
from app.agents.safety.agent import safety_agent
from app.runner import _run_revision_cycle, run_yogasutra
from app.state import YogaSessionState
from app.tracing.trace import ExecutionTrace


class _FakeSessionService:
    """Minimal async session service matching ADK InMemoryRunner's interface."""

    def __init__(self) -> None:
        self._sessions: dict[str, SimpleNamespace] = {}

    async def create_session(self, *, app_name: str, user_id: str, state: dict, session_id: str):
        session = SimpleNamespace(state=dict(state), id=session_id)
        self._sessions[session_id] = session
        return session

    async def get_session(self, *, app_name: str, user_id: str, session_id: str):
        return self._sessions.get(session_id)


def _make_event(
    author: str,
    text: str | None = None,
    is_final: bool = False,
    prompt_tokens: int = 10,
    output_tokens: int = 5,
):
    parts = [SimpleNamespace(text=text)] if text is not None else []
    content = SimpleNamespace(parts=parts) if parts else None
    usage = SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=output_tokens,
        total_token_count=prompt_tokens + output_tokens,
    )
    return SimpleNamespace(
        author=author,
        content=content,
        usage_metadata=usage,
        is_final_response=lambda: is_final,
    )


class _FakeRunner:
    """Deterministic fake InMemoryRunner for unit-testing revision cycles without LLM calls."""

    def __init__(self, state_updates: dict, events: list) -> None:
        self.session_service = _FakeSessionService()
        self._state_updates = state_updates
        self._events = events
        self.received_messages: list[str] = []
        self.seeded_states: list[dict] = []

    def run(self, *, user_id: str, session_id: str, new_message):
        session = self.session_service._sessions[session_id]
        self.seeded_states.append(dict(session.state))
        if new_message and getattr(new_message, "parts", None):
            self.received_messages.append(new_message.parts[0].text)
        session.state.update(self._state_updates)
        yield from self._events


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

    def test_create_revision_agent_uses_cloned_sub_agents(self):
        rev_agent = create_revision_agent()
        assert [a.name for a in rev_agent.sub_agents] == [
            "PlannerAgent",
            "SafetyAgent",
            "CriticAgent",
            "ResponseAgent",
        ]
        # Original root_agent sub_agents must remain completely untouched
        assert [a.name for a in root_agent.sub_agents] == [
            "ProfileAgent",
            "KnowledgeAgent",
            "PlannerAgent",
            "SafetyAgent",
            "CriticAgent",
            "ResponseAgent",
        ]

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


class TestCriticPlannerRevisionLoop:
    """Unit tests for the closed-loop Critic -> Planner revision cycle in app/runner.py."""

    def test_run_revision_cycle_seeds_state_and_accumulates_trace(self):
        trace = ExecutionTrace(run_id="run_test", session_id="sess_test")
        trace.record_agent("PlannerAgent", latency_ms=10.0)
        trace.token_usage["PlannerAgent"] = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

        initial_state = {
            "user_profile": {"experience": "beginner", "duration_minutes": 20},
            "retrieved_evidence": [{"chunk_id": "c1", "title": "Safe Basics"}],
            "candidate_plan": {"sequence": [{"name": "Sirsasana"}]},
            "validation_results": {"is_safe": False, "violations": ["Headstand contraindicated"]},
            "critique": json.dumps({"decision": "REVISION_REQUIRED", "critique": "Remove Sirsasana"}),
            "revision_count": 1,
        }

        fake_rev_runner = _FakeRunner(
            state_updates={
                "candidate_plan": {"sequence": [{"name": "Balasana"}]},
                "validation_results": {"is_safe": True, "violations": []},
                "critique": json.dumps({"decision": "PASS", "score": 9, "critique": "Safe now"}),
            },
            events=[
                _make_event("PlannerAgent", prompt_tokens=40, output_tokens=20),
                _make_event("SafetyAgent", prompt_tokens=30, output_tokens=15),
                _make_event("CriticAgent", prompt_tokens=25, output_tokens=10),
                _make_event("ResponseAgent", text="Revised Safe Plan", is_final=True, prompt_tokens=20, output_tokens=30),
            ],
        )

        updated_state, response_text = _run_revision_cycle(
            session_state=initial_state,
            revision_prompt="Please remove Sirsasana",
            user_id="demo_user",
            session_id="sess_test",
            trace=trace,
            runner_factory=lambda: fake_rev_runner,
        )

        assert response_text == "Revised Safe Plan"
        assert updated_state["user_profile"] == initial_state["user_profile"]
        assert updated_state["retrieved_evidence"] == initial_state["retrieved_evidence"]
        assert updated_state["candidate_plan"] == {"sequence": [{"name": "Balasana"}]}
        assert json.loads(updated_state["critique"])["decision"] == "PASS"
        assert fake_rev_runner.received_messages == ["Please remove Sirsasana"]
        assert fake_rev_runner.seeded_states[0]["revision_count"] == 1

        # Trace accumulates tokens for PlannerAgent and records new agents
        assert trace.token_usage["PlannerAgent"] == {
            "input_tokens": 140,
            "output_tokens": 70,
            "total_tokens": 210,
        }
        for expected_agent in ("PlannerAgent", "SafetyAgent", "CriticAgent", "ResponseAgent"):
            assert expected_agent in trace.agents_called

    def test_runner_revises_on_revision_required_and_stops_on_pass(self):
        initial_runner = _FakeRunner(
            state_updates={
                "user_profile": {"experience": "beginner"},
                "retrieved_evidence": [],
                "candidate_plan": {"sequence": [{"name": "Sirsasana"}], "duration_minutes": 20},
                "validation_results": {"is_safe": False},
                "critique": '```json\n{"decision": "REVISION_REQUIRED", "critique": "Remove Sirsasana"}\n```',
            },
            events=[
                _make_event("ProfileAgent"),
                _make_event("KnowledgeAgent"),
                _make_event("PlannerAgent"),
                _make_event("SafetyAgent"),
                _make_event("CriticAgent"),
                _make_event("ResponseAgent", text="Initial Unsafe Plan", is_final=True),
            ],
        )

        rev_runner_1 = _FakeRunner(
            state_updates={
                "candidate_plan": {"sequence": [{"name": "Balasana"}], "duration_minutes": 20},
                "validation_results": {"is_safe": True},
                "critique": json.dumps({"decision": "PASS", "score": 9, "critique": "All clear"}),
            },
            events=[
                _make_event("PlannerAgent"),
                _make_event("SafetyAgent"),
                _make_event("CriticAgent"),
                _make_event("ResponseAgent", text="Revised Safe Plan v1", is_final=True),
            ],
        )

        runners_iter = iter([initial_runner, rev_runner_1])

        with (
            patch("app.runner.InMemoryRunner", side_effect=lambda *a, **kw: next(runners_iter)),
            patch("app.runner.create_revision_agent", return_value=SimpleNamespace(name="RevisionRootAgent")),
        ):
            result = run_yogasutra(
                "I am a beginner wanting 20 min relaxation",
                phase=2,
                enable_crewai_audit=False,
            )

        assert result.trace.revision_count == 1
        assert result.session_state["revision_count"] == 1
        assert result.final_response == "Revised Safe Plan v1"
        assert result.trace.critic_result["decision"] == "PASS"
        assert "Remove Sirsasana" in rev_runner_1.received_messages[0]

    def test_runner_caps_revisions_at_max_revisions_two(self):
        critique_fail = json.dumps({
            "decision": "REVISION_REQUIRED",
            "critique": "Still missing shoulder mobility pose",
        })

        initial_runner = _FakeRunner(
            state_updates={
                "user_profile": {"experience": "beginner"},
                "candidate_plan": {"sequence": [{"name": "Tadasana"}]},
                "validation_results": {"is_safe": True},
                "critique": critique_fail,
            },
            events=[
                _make_event("PlannerAgent"),
                _make_event("SafetyAgent"),
                _make_event("CriticAgent"),
                _make_event("ResponseAgent", text="Initial Plan", is_final=True),
            ],
        )
        rev_runner_1 = _FakeRunner(
            state_updates={
                "candidate_plan": {"sequence": [{"name": "Cat-Cow"}]},
                "critique": critique_fail,
            },
            events=[
                _make_event("PlannerAgent"),
                _make_event("SafetyAgent"),
                _make_event("CriticAgent"),
                _make_event("ResponseAgent", text="Revision 1 Plan", is_final=True),
            ],
        )
        rev_runner_2 = _FakeRunner(
            state_updates={
                "candidate_plan": {"sequence": [{"name": "Child Pose"}]},
                "critique": critique_fail,
            },
            events=[
                _make_event("PlannerAgent"),
                _make_event("SafetyAgent"),
                _make_event("CriticAgent"),
                _make_event("ResponseAgent", text="Revision 2 Plan", is_final=True),
            ],
        )

        created_runners: list[_FakeRunner] = [initial_runner, rev_runner_1, rev_runner_2]
        call_count = {"n": 0}

        def _fake_runner_ctor(*args, **kwargs):
            idx = call_count["n"]
            call_count["n"] += 1
            return created_runners[idx]

        with (
            patch("app.runner.InMemoryRunner", side_effect=_fake_runner_ctor),
            patch("app.runner.create_revision_agent", return_value=SimpleNamespace(name="RevisionRootAgent")),
        ):
            result = run_yogasutra(
                "Fix my rounded shoulders",
                phase=2,
                enable_crewai_audit=False,
            )

        # 1 initial run + 2 revision runs = 3 total runner instantiations
        assert call_count["n"] == 3
        assert result.trace.revision_count == 2
        assert result.session_state["revision_count"] == 2
        assert result.final_response == "Revision 2 Plan"

    def test_runner_does_not_revise_when_initial_critique_passes(self):
        initial_runner = _FakeRunner(
            state_updates={
                "user_profile": {"experience": "beginner"},
                "candidate_plan": {"sequence": [{"name": "Balasana"}]},
                "validation_results": {"is_safe": True},
                "critique": json.dumps({"decision": "PASS", "score": 10}),
            },
            events=[
                _make_event("PlannerAgent"),
                _make_event("SafetyAgent"),
                _make_event("CriticAgent"),
                _make_event("ResponseAgent", text="Great Initial Plan", is_final=True),
            ],
        )

        with (
            patch("app.runner.InMemoryRunner", return_value=initial_runner) as mock_runner_cls,
            patch("app.runner._run_revision_cycle") as mock_rev_cycle,
        ):
            result = run_yogasutra(
                "20 min gentle stretch",
                phase=2,
                enable_crewai_audit=False,
            )

        assert mock_runner_cls.call_count == 1
        mock_rev_cycle.assert_not_called()
        assert result.trace.revision_count == 0
        assert result.final_response == "Great Initial Plan"
