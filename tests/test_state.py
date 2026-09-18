"""
tests/test_state.py — Pydantic model validation tests for app/state.py.

Tests:
- Default values are correct for all models
- Enum coercion works (string → enum value)
- YogaSessionState auto-generates run_id on creation
- TraceStep records timestamp automatically
- PracticePlan and PoseItem validate correctly
- EvidenceChunk validates required fields
"""
import pytest
from pydantic import ValidationError

from app.state import (
    EvidenceChunk,
    ExperienceLevel,
    PoseItem,
    PracticePlan,
    PracticeGoal,
    RequestMode,
    TraceStep,
    UserProfile,
    YogaSessionState,
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TestEnumerations:

    def test_experience_level_values(self):
        assert ExperienceLevel.BEGINNER == "beginner"
        assert ExperienceLevel.INTERMEDIATE == "intermediate"
        assert ExperienceLevel.ADVANCED == "advanced"

    def test_practice_goal_values(self):
        assert PracticeGoal.RELAXATION == "relaxation"
        assert PracticeGoal.FLEXIBILITY == "flexibility"

    def test_request_mode_values(self):
        assert RequestMode.PRACTICE == "practice"
        assert RequestMode.KNOWLEDGE == "knowledge"
        assert RequestMode.UNKNOWN == "unknown"


# ---------------------------------------------------------------------------
# UserProfile
# ---------------------------------------------------------------------------

class TestUserProfile:

    def test_default_values(self):
        profile = UserProfile()
        assert profile.experience_level == "beginner"
        assert profile.available_duration_minutes == 20
        assert profile.preferred_time == "morning"
        assert profile.goal == "relaxation"
        assert profile.constraints == []
        assert profile.preferences == []

    def test_custom_values(self):
        profile = UserProfile(
            experience_level="intermediate",
            available_duration_minutes=30,
            preferred_time="evening",
            goal="flexibility",
            constraints=["lower back pain"],
            preferences=["yin yoga"],
        )
        assert profile.experience_level == "intermediate"
        assert profile.available_duration_minutes == 30
        assert profile.goal == "flexibility"
        assert "lower back pain" in profile.constraints

    def test_enum_coercion_from_string(self):
        profile = UserProfile(experience_level="advanced", goal="strength")
        assert profile.experience_level == "advanced"
        assert profile.goal == "strength"

    def test_invalid_experience_level_raises(self):
        with pytest.raises(ValidationError):
            UserProfile(experience_level="expert")  # Not a valid enum value

    def test_invalid_goal_raises(self):
        with pytest.raises(ValidationError):
            UserProfile(goal="flying")  # Not a valid enum value


# ---------------------------------------------------------------------------
# EvidenceChunk
# ---------------------------------------------------------------------------

class TestEvidenceChunk:

    def test_required_fields(self):
        chunk = EvidenceChunk(
            chunk_id="doc_001_001",
            document_id="doc_001",
            title="Beginner's Guide",
            section="Introduction",
            topic="beginner_practice",
            content="Some yoga content here.",
        )
        assert chunk.chunk_id == "doc_001_001"
        assert chunk.relevance_score == 1.0  # default

    def test_defaults(self):
        chunk = EvidenceChunk(
            chunk_id="x", document_id="d", title="T", section="S",
            topic="t", content="c",
        )
        assert chunk.source_type == "reference"
        assert chunk.license == "curated"
        assert chunk.relevance_score == 1.0

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            EvidenceChunk(chunk_id="x", document_id="d")  # missing required fields


# ---------------------------------------------------------------------------
# PoseItem
# ---------------------------------------------------------------------------

class TestPoseItem:

    def test_minimal_construction(self):
        pose = PoseItem(name="Child's Pose", duration_minutes=3)
        assert pose.name == "Child's Pose"
        assert pose.duration_minutes == 3
        assert pose.sanskrit_name == ""
        assert pose.evidence_chunk_ids == []

    def test_full_construction(self):
        pose = PoseItem(
            name="Legs-up-the-Wall",
            sanskrit_name="Viparita Karani",
            duration_minutes=5,
            instructions="Lie on back, legs up wall.",
            modifications="Use blanket under hips.",
            evidence_chunk_ids=["doc_002_002"],
        )
        assert pose.sanskrit_name == "Viparita Karani"
        assert "doc_002_002" in pose.evidence_chunk_ids


# ---------------------------------------------------------------------------
# PracticePlan
# ---------------------------------------------------------------------------

class TestPracticePlan:

    def _make_sequence(self, total: int = 20) -> list[PoseItem]:
        return [
            PoseItem(name="Breathing", duration_minutes=2),
            PoseItem(name="Cat-Cow", duration_minutes=3),
            PoseItem(name="Child's Pose", duration_minutes=5),
            PoseItem(name="Twist", duration_minutes=4),
            PoseItem(name="Legs-up-the-Wall", duration_minutes=4),
            PoseItem(name="Savasana", duration_minutes=2),
        ]

    def test_basic_construction(self):
        plan = PracticePlan(
            goal="Relaxation",
            duration_minutes=20,
            intensity="gentle",
            sequence=self._make_sequence(),
            rationale="Beginner relaxation sequence.",
        )
        assert plan.goal == "Relaxation"
        assert plan.duration_minutes == 20
        assert len(plan.sequence) == 6

    def test_defaults(self):
        plan = PracticePlan(
            goal="Relaxation", duration_minutes=20, intensity="gentle",
            sequence=[], rationale="test",
        )
        assert plan.evidence_sources == []
        assert plan.expected_outcome == ""


# ---------------------------------------------------------------------------
# TraceStep
# ---------------------------------------------------------------------------

class TestTraceStep:

    def test_default_timestamp_is_set(self):
        step = TraceStep(agent="ProfileAgent", action="extract_profile")
        assert step.timestamp != ""
        assert step.status == "success"

    def test_tools_called_default_empty(self):
        step = TraceStep(agent="KnowledgeAgent", action="retrieve")
        assert step.tools_called == []

    def test_custom_step(self):
        step = TraceStep(
            agent="PlannerAgent",
            action="generate_plan",
            tools_called=["calculate_practice_duration"],
            latency_ms=312.5,
            token_usage={"input": 200, "output": 400},
            status="success",
        )
        assert step.latency_ms == 312.5
        assert "calculate_practice_duration" in step.tools_called


# ---------------------------------------------------------------------------
# YogaSessionState
# ---------------------------------------------------------------------------

class TestYogaSessionState:

    def test_auto_generates_run_id(self):
        state = YogaSessionState()
        assert state.run_id.startswith("run_")
        assert len(state.run_id) > 4

    def test_two_states_have_different_run_ids(self):
        s1 = YogaSessionState()
        s2 = YogaSessionState()
        assert s1.run_id != s2.run_id

    def test_default_values(self):
        state = YogaSessionState()
        assert state.user_input == ""
        assert state.request_mode == "unknown"
        assert state.user_profile is None
        assert state.retrieved_evidence == []
        assert state.candidate_plan is None
        assert state.final_response is None
        assert state.revision_count == 0
        assert state.execution_trace == []

    def test_populating_user_profile(self):
        state = YogaSessionState(user_input="I am a beginner")
        profile = UserProfile(experience_level="beginner", goal="relaxation")
        state.user_profile = profile
        assert state.user_profile.goal == "relaxation"

    def test_populating_evidence(self):
        state = YogaSessionState()
        chunk = EvidenceChunk(
            chunk_id="doc_001_001", document_id="doc_001",
            title="Test", section="S", topic="t", content="c",
        )
        state.retrieved_evidence.append(chunk)
        assert len(state.retrieved_evidence) == 1
        assert state.retrieved_evidence[0].chunk_id == "doc_001_001"

    def test_start_time_is_set(self):
        state = YogaSessionState()
        assert state.start_time != ""
        # Should be a valid ISO format string
        assert "T" in state.start_time or "-" in state.start_time
