"""
tests/test_tools.py — Deterministic unit tests for Phase 1 tools.

Tests:
- retrieve_yoga_knowledge returns correct number of chunks
- retrieve_yoga_knowledge returns relevant chunks for a specific query
- retrieve_yoga_knowledge returns [] for empty query
- calculate_practice_duration sums correctly
- calculate_practice_duration handles edge cases (empty, missing keys)
- get_user_profile returns None for unknown sessions
- store_user_profile + get_user_profile round-trips correctly
"""
import pytest

from app.tools.retrieval import retrieve_yoga_knowledge
from app.tools.duration import calculate_practice_duration
from app.tools.profile import get_user_profile, store_user_profile


# ---------------------------------------------------------------------------
# retrieve_yoga_knowledge
# ---------------------------------------------------------------------------

class TestRetrieveYogaKnowledge:

    def test_returns_up_to_top_k_results(self):
        results = retrieve_yoga_knowledge("beginner relaxation morning yoga", top_k=5)
        assert len(results) <= 5

    def test_returns_at_least_one_result_for_valid_query(self):
        results = retrieve_yoga_knowledge("beginner yoga")
        assert len(results) >= 1

    def test_result_has_required_fields(self):
        results = retrieve_yoga_knowledge("beginner relaxation", top_k=3)
        assert len(results) >= 1
        chunk = results[0]
        required = {"chunk_id", "document_id", "title", "section", "topic", "content",
                    "source_type", "license", "relevance_score"}
        assert required.issubset(chunk.keys()), f"Missing keys: {required - chunk.keys()}"

    def test_relevance_score_is_normalised(self):
        results = retrieve_yoga_knowledge("pranayama breathing", top_k=5)
        for chunk in results:
            assert 0.0 <= chunk["relevance_score"] <= 1.0

    def test_top_result_is_most_relevant_for_pranayama(self):
        results = retrieve_yoga_knowledge("pranayama alternate nostril breathing", top_k=5)
        assert len(results) >= 1
        # Top result should be in pranayama topic
        assert results[0]["topic"] == "pranayama"

    def test_top_result_for_beginners(self):
        results = retrieve_yoga_knowledge("beginner morning yoga relaxation", top_k=1)
        assert len(results) == 1
        # Should be a beginner-related chunk
        chunk = results[0]
        assert "beginner" in chunk["topic"] or "beginner" in chunk["content"].lower()

    def test_returns_empty_for_blank_query(self):
        assert retrieve_yoga_knowledge("") == []
        assert retrieve_yoga_knowledge("   ") == []

    def test_top_k_one_returns_single_result(self):
        results = retrieve_yoga_knowledge("yoga", top_k=1)
        assert len(results) == 1

    def test_top_k_respected(self):
        results = retrieve_yoga_knowledge("yoga poses", top_k=3)
        assert len(results) <= 3

    def test_results_sorted_descending_by_score(self):
        results = retrieve_yoga_knowledge("beginner relaxation", top_k=5)
        scores = [r["relevance_score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# calculate_practice_duration
# ---------------------------------------------------------------------------

class TestCalculatePracticeDuration:

    def test_sums_correctly(self):
        sequence = [
            {"name": "Child's Pose", "duration_minutes": 3},
            {"name": "Savasana", "duration_minutes": 2},
        ]
        assert calculate_practice_duration(sequence) == 5

    def test_twenty_minute_sequence(self):
        sequence = [
            {"name": "Breathing", "duration_minutes": 2},
            {"name": "Neck Rolls", "duration_minutes": 3},
            {"name": "Cat-Cow", "duration_minutes": 2},
            {"name": "Child's Pose", "duration_minutes": 5},
            {"name": "Supine Twist", "duration_minutes": 4},
            {"name": "Legs-up-the-Wall", "duration_minutes": 2},
            {"name": "Savasana", "duration_minutes": 2},
        ]
        assert calculate_practice_duration(sequence) == 20

    def test_empty_sequence_returns_zero(self):
        assert calculate_practice_duration([]) == 0

    def test_missing_duration_key_is_skipped(self):
        sequence = [
            {"name": "Pose A", "duration_minutes": 5},
            {"name": "Pose B"},  # Missing duration_minutes
        ]
        assert calculate_practice_duration(sequence) == 5

    def test_single_item(self):
        assert calculate_practice_duration([{"duration_minutes": 10}]) == 10

    def test_handles_invalid_duration_gracefully(self):
        sequence = [
            {"name": "A", "duration_minutes": 3},
            {"name": "B", "duration_minutes": "invalid"},
        ]
        # Should skip invalid and return 3
        assert calculate_practice_duration(sequence) == 3


# ---------------------------------------------------------------------------
# get_user_profile / store_user_profile
# ---------------------------------------------------------------------------

class TestUserProfile:

    def test_returns_none_for_unknown_session(self):
        result = get_user_profile("session_does_not_exist_xyz")
        assert result is None

    def test_store_and_retrieve_roundtrip(self):
        session_id = "test_session_001"
        profile = {
            "experience_level": "beginner",
            "available_duration_minutes": 20,
            "preferred_time": "morning",
            "goal": "relaxation",
            "constraints": [],
            "preferences": [],
        }
        store_user_profile(session_id, profile)
        retrieved = get_user_profile(session_id)
        assert retrieved == profile

    def test_overwrite_updates_profile(self):
        session_id = "test_session_002"
        profile_v1 = {"experience_level": "beginner", "available_duration_minutes": 20,
                       "preferred_time": "morning", "goal": "relaxation",
                       "constraints": [], "preferences": []}
        profile_v2 = {"experience_level": "intermediate", "available_duration_minutes": 30,
                       "preferred_time": "evening", "goal": "flexibility",
                       "constraints": ["lower back pain"], "preferences": []}
        store_user_profile(session_id, profile_v1)
        store_user_profile(session_id, profile_v2)
        assert get_user_profile(session_id)["experience_level"] == "intermediate"

    def test_different_sessions_are_independent(self):
        store_user_profile("sess_a", {"experience_level": "beginner", "available_duration_minutes": 20,
                                       "preferred_time": "morning", "goal": "relaxation",
                                       "constraints": [], "preferences": []})
        assert get_user_profile("sess_b_not_stored") is None
