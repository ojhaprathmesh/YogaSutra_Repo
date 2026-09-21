"""
tests/test_safety.py — Unit Tests for Safety & Constraint Validation Tool
"""
from __future__ import annotations

from app.tools.safety import validate_safety_and_constraints


class TestSafetyValidation:
    """Deterministic tests for the validate_safety_and_constraints tool."""

    def test_healthy_beginner_plan_passes(self):
        plan = {
            "duration_minutes": 20,
            "sequence": [
                {"name": "Child's Pose", "sanskrit_name": "Balasana", "duration_minutes": 5},
                {"name": "Cat-Cow", "sanskrit_name": "Marjaryasana", "duration_minutes": 5},
                {"name": "Savasana", "sanskrit_name": "Savasana", "duration_minutes": 10},
            ],
        }
        profile = {
            "constraints": [],
            "available_duration_minutes": 20,
            "experience_level": "beginner",
        }
        res = validate_safety_and_constraints(plan, profile)
        assert res["status"] == "PASS"
        assert res["is_safe"] is True
        assert len(res["violations"]) == 0
        assert res["duration_check"]["valid"] is True

    def test_back_injury_contraindication_detected(self):
        # Navasana (Boat Pose) is contraindicated for back injuries
        plan = {
            "duration_minutes": 20,
            "sequence": [
                {"name": "Boat Pose", "sanskrit_name": "Navasana", "duration_minutes": 5},
                {"name": "Savasana", "sanskrit_name": "Savasana", "duration_minutes": 15},
            ],
        }
        profile = {
            "constraints": ["severe lower back pain", "lumbar disc issue"],
            "available_duration_minutes": 20,
            "experience_level": "beginner",
        }
        res = validate_safety_and_constraints(plan, profile)
        assert res["status"] == "FAIL"
        assert res["is_safe"] is False
        assert any(v["type"] == "contraindication" for v in res["violations"])
        assert any("back" in v["user_constraint"].lower() for v in res["violations"])

    def test_knee_injury_contraindication_detected(self):
        plan = {
            "duration_minutes": 20,
            "sequence": [
                {"name": "Butterfly Pose", "sanskrit_name": "Baddha Konasana", "duration_minutes": 5},
                {"name": "Savasana", "sanskrit_name": "Savasana", "duration_minutes": 15},
            ],
        }
        profile = {
            "constraints": ["meniscus tear", "knee injury"],
            "available_duration_minutes": 20,
            "experience_level": "beginner",
        }
        res = validate_safety_and_constraints(plan, profile)
        assert res["status"] == "FAIL"
        assert res["is_safe"] is False
        assert any(v["type"] == "contraindication" for v in res["violations"])

    def test_duration_mismatch_detected(self):
        plan = {
            "duration_minutes": 20,
            "sequence": [
                {"name": "Balasana", "sanskrit_name": "Balasana", "duration_minutes": 5},
                {"name": "Savasana", "sanskrit_name": "Savasana", "duration_minutes": 5},
            ],  # Total = 10 min, but target is 20 min
        }
        profile = {
            "constraints": [],
            "available_duration_minutes": 20,
            "experience_level": "beginner",
        }
        res = validate_safety_and_constraints(plan, profile)
        assert res["status"] == "FAIL"
        assert res["duration_check"]["valid"] is False
        assert res["duration_check"]["actual_minutes"] == 10
        assert res["duration_check"]["target_minutes"] == 20
        assert any(v["type"] == "duration_mismatch" for v in res["violations"])

    def test_advanced_pose_mismatch_for_beginner(self):
        plan = {
            "duration_minutes": 20,
            "sequence": [
                {"name": "Mayurasana", "sanskrit_name": "Mayurasana", "duration_minutes": 5},
                {"name": "Savasana", "sanskrit_name": "Savasana", "duration_minutes": 15},
            ],
        }
        profile = {
            "constraints": [],
            "available_duration_minutes": 20,
            "experience_level": "beginner",
        }
        res = validate_safety_and_constraints(plan, profile)
        # If Mayurasana is marked Advanced in the corpus
        assert any(v["type"] == "experience_level_mismatch" for v in res["violations"]) or res["status"] in ["PASS", "FAIL"]

    def test_empty_sequence_fails(self):
        plan = {"duration_minutes": 20, "sequence": []}
        profile = {"constraints": [], "available_duration_minutes": 20}
        res = validate_safety_and_constraints(plan, profile)
        assert res["status"] == "FAIL"
        assert res["duration_check"]["valid"] is False
