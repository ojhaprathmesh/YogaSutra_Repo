"""
tests/test_a2a_crewai.py — Comprehensive Tests for Requirements 6 & 7

Verifies:
1. A2A Protocol envelope schema, defaults, enum serialization, JSON/dict round-trip,
   correlated replies (`conversation_id`, `in_reply_to`), and `A2AAgentCard`.
2. A2ABridge / A2ADispatcher registration, discovery, synchronous & asynchronous
   routing, error handling for unregistered agents, and conversation history.
3. CrewAI specialist agent (`CrewAIAuditAgent` / `CrewAICriticAgent`) abstractions
   (`Agent`, `Task`, `Crew` + resilient Python 3.14 fallback), safe plan approval,
   contraindication/duration/pedagogical violation detection.
4. End-to-end ADK <-> CrewAI framework handoff via `request_crewai_audit`,
   `run_adk_to_crewai_handoff`, and `execute_a2a_crewai_audit`, updating both the
   shared `Blackboard` and `YogaSessionState` / `session_state`.
"""
from __future__ import annotations

import pytest

from app.a2a import (
    A2AAgentCard,
    A2ABridge,
    A2ADispatcher,
    A2AMessage,
    FrameworkType,
    Performative,
    get_a2a_bridge,
    reset_a2a_bridge,
)
from app.agents.crewai import (
    CrewAIAuditAgent,
    CrewAICriticAgent,
    register_crewai_audit_agent,
)
from app.agents.root.agent import request_crewai_audit, run_adk_to_crewai_handoff
from app.runner import execute_a2a_crewai_audit
from app.state import YogaSessionState
from app.workflows.blackboard import Blackboard

# ---------------------------------------------------------------------------
# Sample Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def safe_beginner_profile() -> dict:
    return {
        "experience_level": "beginner",
        "available_duration_minutes": 15,
        "goal": "relaxation",
        "constraints": [],
    }


@pytest.fixture
def safe_beginner_plan() -> dict:
    return {
        "goal": "relaxation",
        "duration_minutes": 15,
        "intensity": "gentle",
        "sequence": [
            {
                "name": "Easy Pose",
                "sanskrit_name": "Sukhasana",
                "duration_minutes": 3,
                "instructions": "Sit tall and take slow diaphragmatic breaths.",
                "modifications": "Sit on a folded blanket to support the lower back.",
            },
            {
                "name": "Cat-Cow",
                "sanskrit_name": "Marjaryasana-Bitilasana",
                "duration_minutes": 4,
                "instructions": "Move gently between flexion and extension with the breath.",
                "modifications": "Keep range of motion small if lower back feels tight.",
            },
            {
                "name": "Child's Pose",
                "sanskrit_name": "Balasana",
                "duration_minutes": 4,
                "instructions": "Rest hips toward heels and soften the torso.",
                "modifications": "Place a bolster under the chest and blanket under knees.",
            },
            {
                "name": "Corpse Pose",
                "sanskrit_name": "Savasana",
                "duration_minutes": 4,
                "instructions": "Lie supine and release muscular tension completely.",
                "modifications": "Place a bolster under the knees to decompress the lumbar spine.",
            },
        ],
        "rationale": "Gentle restorative sequence tailored for a beginner.",
    }


@pytest.fixture
def unsafe_plan_with_violations() -> dict:
    return {
        "goal": "strength",
        "duration_minutes": 20,
        "intensity": "vigorous",
        "sequence": [
            {
                "name": "Headstand",
                "sanskrit_name": "Sirsasana",
                "duration_minutes": 5,
                "instructions": "Invert onto the crown of the head.",
                "modifications": "",
            },
            {
                "name": "Wheel Pose",
                "sanskrit_name": "Urdhva Dhanurasana",
                "duration_minutes": 5,
                "instructions": "Press into a deep backbend.",
                "modifications": "",
            },
        ],
        "rationale": "High intensity inversion sequence.",
    }


@pytest.fixture(autouse=True)
def _isolate_global_bridge():
    reset_a2a_bridge()
    yield
    reset_a2a_bridge()


# ---------------------------------------------------------------------------
# 1. A2A Protocol Serialization & Schema Tests (Requirement 6)
# ---------------------------------------------------------------------------


class TestA2AProtocol:
    def test_envelope_fields_and_defaults(self):
        msg = A2AMessage(
            sender_agent="CriticAgent",
            sender_framework=FrameworkType.ADK,
            recipient_agent="CrewAIAuditAgent",
            recipient_framework=FrameworkType.CREWAI,
            performative=Performative.AUDIT,
            payload={"plan_id": "plan_01"},
        )

        assert msg.message_id.startswith("a2a_")
        assert msg.conversation_id.startswith("conv_")
        assert msg.sender_agent == "CriticAgent"
        assert msg.sender_framework == "google_adk"
        assert msg.recipient_agent == "CrewAIAuditAgent"
        assert msg.recipient_framework == "crewai"
        assert msg.performative == "AUDIT"
        assert msg.payload == {"plan_id": "plan_01"}
        assert msg.timestamp  # ISO-8601 UTC string
        assert msg.in_reply_to is None

    def test_json_and_dict_roundtrip(self):
        original = A2AMessage(
            sender_agent="PlannerAgent",
            sender_framework=FrameworkType.ADK,
            recipient_agent="CrewAIAuditAgent",
            recipient_framework=FrameworkType.CREWAI,
            performative=Performative.PROPOSE,
            conversation_id="conv_roundtrip_1",
            payload={"poses": ["Sukhasana", "Savasana"], "duration_minutes": 15},
            metadata={"phase": 2},
        )

        raw_json = original.to_json()
        restored_json = A2AMessage.from_json(raw_json)
        assert restored_json == original

        as_dict = original.to_dict()
        restored_dict = A2AMessage.from_dict(as_dict)
        assert restored_dict == original
        assert set(as_dict.keys()) >= {
            "message_id",
            "sender_agent",
            "sender_framework",
            "recipient_agent",
            "recipient_framework",
            "performative",
            "conversation_id",
            "payload",
            "timestamp",
        }

    def test_create_reply_preserves_conversation_and_swaps_endpoints(self):
        request = A2AMessage(
            sender_agent="RootAgent",
            sender_framework=FrameworkType.ADK,
            recipient_agent="CrewAIAuditAgent",
            recipient_framework=FrameworkType.CREWAI,
            performative=Performative.AUDIT,
            conversation_id="conv_xyz",
            payload={"check": "safety"},
        )

        reply = request.create_reply(
            performative=Performative.CRITIQUE,
            payload={"verdict": "APPROVE", "score": 9.5},
        )

        assert reply.message_id != request.message_id
        assert reply.conversation_id == "conv_xyz"
        assert reply.in_reply_to == request.message_id
        assert reply.sender_agent == "CrewAIAuditAgent"
        assert reply.sender_framework == "crewai"
        assert reply.recipient_agent == "RootAgent"
        assert reply.recipient_framework == "google_adk"
        assert reply.performative == "CRITIQUE"
        assert reply.payload["verdict"] == "APPROVE"

    def test_agent_card_schema(self):
        card = A2AAgentCard(
            agent_name="CrewAIAuditAgent",
            framework=FrameworkType.CREWAI,
            role="Yoga Safety Auditor",
            capabilities=["safety_audit", "pedagogical_critique"],
        )
        assert card.agent_name == "CrewAIAuditAgent"
        assert card.framework == "crewai"
        assert "safety_audit" in card.capabilities


# ---------------------------------------------------------------------------
# 2. A2ABridge / Dispatcher Tests
# ---------------------------------------------------------------------------


class TestA2ABridge:
    def test_register_discover_and_dispatch_sync(self):
        bridge = A2ADispatcher()
        card = A2AAgentCard(
            agent_name="EchoCrew",
            framework=FrameworkType.CREWAI,
            role="Echo Tester",
            capabilities=["echo"],
        )

        def echo_handler(msg: A2AMessage) -> dict:
            return {"echoed": msg.payload.get("text"), "from": msg.sender_agent}

        bridge.register_agent(card, echo_handler)
        assert bridge.has_agent("EchoCrew")
        assert len(bridge.list_agents(framework=FrameworkType.CREWAI)) == 1
        assert len(bridge.list_agents(framework=FrameworkType.ADK)) == 0

        req = A2AMessage(
            sender_agent="ProfileAgent",
            sender_framework=FrameworkType.ADK,
            recipient_agent="EchoCrew",
            recipient_framework=FrameworkType.CREWAI,
            performative=Performative.REQUEST,
            payload={"text": "namaste"},
        )
        reply = bridge.dispatch(req)

        assert reply.performative == "INFORM"
        assert reply.in_reply_to == req.message_id
        assert reply.payload == {"echoed": "namaste", "from": "ProfileAgent"}
        assert len(bridge.get_conversation(req.conversation_id)) == 2

    @pytest.mark.asyncio
    async def test_async_dispatch(self):
        bridge = A2ABridge()
        card = A2AAgentCard(
            agent_name="AsyncCrewAgent",
            framework=FrameworkType.CREWAI,
            role="Async Auditor",
        )

        async def async_handler(msg: A2AMessage) -> A2AMessage:
            return msg.create_reply(
                performative=Performative.INFORM,
                payload={"async_ok": True, "received": msg.payload},
            )

        bridge.register_agent(card, async_handler)
        req = A2AMessage(
            sender_agent="PlannerAgent",
            sender_framework=FrameworkType.ADK,
            recipient_agent="AsyncCrewAgent",
            recipient_framework=FrameworkType.CREWAI,
            performative=Performative.QUERY,
            payload={"pose": "Tadasana"},
        )
        reply = await bridge.adispatch(req)

        assert reply.performative == "INFORM"
        assert reply.payload["async_ok"] is True
        assert reply.payload["received"] == {"pose": "Tadasana"}

    def test_unregistered_recipient_returns_error_envelope(self):
        bridge = A2ABridge()
        req = A2AMessage(
            sender_agent="RootAgent",
            sender_framework=FrameworkType.ADK,
            recipient_agent="NonExistentAgent",
            recipient_framework=FrameworkType.CREWAI,
            performative=Performative.REQUEST,
            payload={},
        )
        reply = bridge.dispatch(req)
        assert reply.performative == "ERROR"
        assert "not registered" in reply.payload["error"]

        with pytest.raises(KeyError):
            bridge.dispatch(req, raise_on_missing=True)


# ---------------------------------------------------------------------------
# 3. CrewAI Specialist Agent Tests (Requirement 7)
# ---------------------------------------------------------------------------


class TestCrewAIAuditAgent:
    def test_crewai_abstractions_and_card(self):
        specialist = CrewAICriticAgent(use_fallback=True)
        assert specialist is not None
        assert specialist.agent.role == "Independent Yoga Safety & Pedagogical Auditor"
        assert specialist.task.agent is specialist.agent
        assert specialist.crew.agents == [specialist.agent]
        assert specialist.crew.tasks == [specialist.task]
        assert specialist.runtime_mode == "crewai_in_process_bridge"

        card = specialist.get_agent_card()
        assert card.agent_name == "CrewAIAuditAgent"
        assert card.framework == "crewai"
        assert "safety_audit" in card.capabilities

    def test_audit_approves_safe_beginner_plan(self, safe_beginner_plan, safe_beginner_profile):
        specialist = CrewAIAuditAgent(use_fallback=True)
        result = specialist.audit_plan(
            candidate_plan=safe_beginner_plan,
            user_profile=safe_beginner_profile,
            validation_results={"status": "PASS", "is_safe": True},
        )

        assert result["framework"] == "crewai"
        assert result["verdict"] == "APPROVE"
        assert result["decision"] == "PASS"
        assert result["is_safe"] is True
        assert result["score"] >= 8.5
        assert result["pedagogical_checks"]["has_warmup"] is True
        assert result["pedagogical_checks"]["has_cooldown"] is True
        assert result["cross_framework_agreement"] is True

    def test_audit_flags_unsafe_plan_and_duration_mismatch(self, unsafe_plan_with_violations):
        specialist = CrewAIAuditAgent(use_fallback=True)
        profile = {
            "experience_level": "beginner",
            "available_duration_minutes": 20,
            "goal": "relaxation",
            "constraints": ["high blood pressure", "neck injury"],
        }

        result = specialist.audit_plan(
            candidate_plan=unsafe_plan_with_violations,
            user_profile=profile,
        )

        assert result["verdict"] in ("REVISE", "REJECT")
        assert result["decision"] == "REVISION_REQUIRED"
        assert result["score"] < 7.5
        assert result["pedagogical_checks"]["has_warmup"] is False
        assert result["pedagogical_checks"]["has_cooldown"] is False
        assert len(result["findings"]) >= 2
        assert len(result["recommendations"]) >= 2


# ---------------------------------------------------------------------------
# 4. End-to-End ADK <-> CrewAI Handoff & Blackboard State Updating
# ---------------------------------------------------------------------------


class TestADKCrewAIHandoff:
    def test_register_and_handoff_updates_blackboard_and_session_state(
        self, safe_beginner_plan, safe_beginner_profile
    ):
        bridge = A2ABridge()
        register_crewai_audit_agent(bridge=bridge, use_fallback=True)

        bb = Blackboard()
        session_state = {
            "run_id": "run_test_a2a",
            "user_profile": safe_beginner_profile,
            "candidate_plan": safe_beginner_plan,
            "validation_results": {"status": "PASS", "is_safe": True},
        }

        handoff = run_adk_to_crewai_handoff(
            session_state=session_state,
            blackboard=bb,
            bridge=bridge,
            sender_agent="CriticAgent",
            conversation_id="run_test_a2a",
        )

        # Verify A2A request & response envelopes
        req = A2AMessage.from_dict(handoff["request"])
        resp = A2AMessage.from_dict(handoff["response"])
        assert req.sender_framework == "google_adk"
        assert req.recipient_framework == "crewai"
        assert resp.sender_framework == "crewai"
        assert resp.recipient_framework == "google_adk"
        assert resp.in_reply_to == req.message_id
        assert resp.conversation_id == "run_test_a2a"

        # Verify Blackboard updated
        bb_audit = bb.read("crewai_audit")
        bb_msgs = bb.read("a2a_messages")
        assert bb_audit is not None
        assert bb_audit["verdict"] == "APPROVE"
        assert isinstance(bb_msgs, list) and len(bb_msgs) == 2

        # Verify ADK session_state updated and validates against YogaSessionState
        assert session_state["crewai_audit"]["verdict"] == "APPROVE"
        assert len(session_state["a2a_messages"]) == 2
        validated_state = YogaSessionState(
            run_id=session_state["run_id"],
            crewai_audit=session_state["crewai_audit"],
            a2a_messages=session_state["a2a_messages"],
        )
        assert validated_state.crewai_audit["framework"] == "crewai"
        assert len(validated_state.a2a_messages) == 2

    def test_runner_execute_a2a_crewai_audit_helper(
        self, safe_beginner_plan, safe_beginner_profile
    ):
        bb = Blackboard()
        session_state = {
            "run_id": "run_runner_helper",
            "user_profile": safe_beginner_profile,
            "candidate_plan": safe_beginner_plan,
        }

        outcome = execute_a2a_crewai_audit(
            session_state=session_state,
            blackboard=bb,
            conversation_id="run_runner_helper",
        )

        assert outcome["crewai_audit"]["verdict"] == "APPROVE"
        assert bb.read("crewai_audit")["decision"] == "PASS"
        assert session_state["crewai_audit"]["auditor"] == "CrewAIAuditAgent"
        assert len(get_a2a_bridge().get_conversation("run_runner_helper")) == 2

    def test_adk_tool_request_crewai_audit(self, safe_beginner_plan, safe_beginner_profile):
        audit_payload = request_crewai_audit(
            candidate_plan=safe_beginner_plan,
            user_profile=safe_beginner_profile,
            validation_results={"status": "PASS", "is_safe": True},
            conversation_id="conv_tool_call",
        )

        assert audit_payload["verdict"] == "APPROVE"
        assert audit_payload["framework"] == "crewai"
        history = get_a2a_bridge().get_conversation("conv_tool_call")
        assert len(history) == 2
        assert history[0].sender_agent == "RootAgent"
        assert history[1].sender_agent == "CrewAIAuditAgent"
