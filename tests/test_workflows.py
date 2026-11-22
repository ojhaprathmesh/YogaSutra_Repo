"""
tests/test_workflows.py — Unit tests for Workflow Paradigms (Sequential, Hierarchical, Broadcast, Blackboard)
"""
import pytest
from app.workflows.blackboard import Blackboard
from app.workflows.sequential import SequentialWorkflow
from app.workflows.hierarchical import HierarchicalWorkflow, UserIntent
from app.workflows.broadcast import BroadcastWorkflow


class TestBlackboard:
    def test_blackboard_read_write(self):
        bb = Blackboard()
        bb.write("user_profile", "ProfileAgent", {"experience": "beginner", "duration": 20})
        profile = bb.read("user_profile")
        assert profile["experience"] == "beginner"
        assert profile["duration"] == 20
        assert len(bb.event_log) == 1
        assert bb.event_log[0]["channel"] == "user_profile"
        assert bb.event_log[0]["writer"] == "ProfileAgent"

    def test_blackboard_subscription(self):
        bb = Blackboard()
        events = []
        bb.subscribe(lambda ch, w, val: events.append((ch, w)))
        bb.write("audio_cues", "SpeechAgent", {"cue": "Breathe deeply"})
        assert len(events) == 1
        assert events[0] == ("audio_cues", "SpeechAgent")

    def test_blackboard_to_dict(self):
        bb = Blackboard(initial_state={"test_key": "test_val"})
        d = bb.to_dict()
        assert d["test_key"] == "test_val"
        assert "event_log" in d


class TestSequentialWorkflow:
    def test_should_revise_decision(self):
        wf = SequentialWorkflow(max_revisions=2)
        assert wf.should_revise({"decision": "REVISION_REQUIRED"}, current_revision=0) is True
        assert wf.should_revise({"decision": "REVISION_REQUIRED"}, current_revision=1) is True
        assert wf.should_revise({"decision": "REVISE"}, current_revision=0) is True
        assert wf.should_revise({"decision": "FAIL"}, current_revision=0) is True
        assert wf.should_revise({"decision": "PASS"}, current_revision=0) is False
        assert wf.should_revise({"decision": "REVISE"}, current_revision=2) is False
        assert wf.should_revise({"decision": "REVISION_REQUIRED"}, current_revision=2) is False
        assert wf.should_revise({"decision": "REVISION_REQUIRED"}, current_revision=3) is False

    def test_should_revise_json_and_plain_strings(self):
        wf = SequentialWorkflow(max_revisions=2)
        assert wf.should_revise('{"decision": "REVISION_REQUIRED", "score": 4}', current_revision=0) is True
        assert wf.should_revise('```json\n{"decision": "REVISION_REQUIRED"}\n```', current_revision=0) is True
        assert wf.should_revise('{"decision": "REVISE"}', current_revision=1) is True
        assert wf.should_revise('{"decision": "PASS", "critique": "No need to revise"}', current_revision=0) is False
        assert wf.should_revise("REVISION_REQUIRED: omit headstand", current_revision=0) is True
        assert wf.should_revise("PASS", current_revision=0) is False
        assert wf.should_revise('{"decision": "REVISION_REQUIRED"}', current_revision=2) is False
        assert wf.should_revise(None, current_revision=0) is False

    def test_build_revision_prompt(self):
        wf = SequentialWorkflow()
        prompt = wf.build_revision_prompt(
            candidate_plan="Balasana (5 min), Sirsasana (10 min)",
            critique_data="Headstand is inappropriate for complete beginners",
        )
        assert "Headstand is inappropriate" in prompt
        assert "Balasana" in prompt

    def test_build_revision_prompt_with_structured_critique(self):
        wf = SequentialWorkflow()
        prompt = wf.build_revision_prompt(
            candidate_plan={"sequence": [{"name": "Sirsasana"}]},
            critique_data={
                "decision": "REVISION_REQUIRED",
                "critique": "Sirsasana is unsafe for beginners.",
                "recommended_modifications": ["Replace Sirsasana with Viparita Karani"],
            },
        )
        assert "Sirsasana is unsafe for beginners." in prompt
        assert "Replace Sirsasana with Viparita Karani" in prompt


class TestHierarchicalWorkflow:
    def test_intent_classification(self):
        hw = HierarchicalWorkflow()
        assert hw.classify_intent("Can I do downward dog with high bp?") == UserIntent.SAFETY_INQUIRY
        assert hw.classify_intent("What is the meaning of Sukhasana?") == UserIntent.KNOWLEDGE_LOOKUP
        assert hw.classify_intent("Speak the next voice cue for me") == UserIntent.VOICE_PRACTICE
        assert hw.classify_intent("Plan a 20 min morning yoga session for relaxing") == UserIntent.PRACTICE_PLANNING

    def test_execution_plan_routing(self):
        hw = HierarchicalWorkflow()
        plan_safety = hw.get_execution_plan(UserIntent.SAFETY_INQUIRY)
        assert "SafetyAgent" in plan_safety
        assert "PlannerAgent" not in plan_safety

        plan_full = hw.get_execution_plan(UserIntent.PRACTICE_PLANNING)
        assert len(plan_full) >= 6


class TestBroadcastWorkflow:
    @pytest.mark.asyncio
    async def test_broadcast_specialists_execution(self):
        bw = BroadcastWorkflow()
        bb = Blackboard()

        def task_a():
            return {"knowledge_docs": 3}

        def task_b():
            return {"safety_status": "PASS"}

        tasks = {"retrieved_evidence": task_a, "validation_results": task_b}
        results = await bw.broadcast_specialists(tasks, bb)

        assert results["retrieved_evidence"]["knowledge_docs"] == 3
        assert results["validation_results"]["safety_status"] == "PASS"
        assert bb.read("retrieved_evidence") == {"knowledge_docs": 3}
        assert bb.read("validation_results") == {"safety_status": "PASS"}
