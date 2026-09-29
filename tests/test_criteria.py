"""
tests/test_criteria.py — Unit Tests for Classical Asana Criteria Matrix

Phase 3B: Verifies deterministic alignment thresholds and cue mapping across core asanas.
"""

from app.vision.biomechanics import BiomechanicalSnapshot
from app.vision.criteria import (
    ASANA_CRITERIA_REGISTRY,
    evaluate_pose_against_criteria,
    normalize_asana_name,
)


class TestCriteriaMatrix:
    """Test criteria registry and normalization."""

    def test_canonical_asana_registrations(self):
        """All 5 core classical asanas must be registered."""
        required = ["virabhadrasana_ii", "tadasana", "vrikshasana", "utkatasana", "bhujangasana"]
        for asana in required:
            assert asana in ASANA_CRITERIA_REGISTRY
            assert len(ASANA_CRITERIA_REGISTRY[asana]) >= 2

    def test_asana_alias_normalization(self):
        assert normalize_asana_name("Warrior II") == "virabhadrasana_ii"
        assert normalize_asana_name("warrior 2") == "virabhadrasana_ii"
        assert normalize_asana_name("Mountain Pose") == "tadasana"
        assert normalize_asana_name("Tree Pose") == "vrikshasana"
        assert normalize_asana_name("Chair") == "utkatasana"
        assert normalize_asana_name("Cobra Pose") == "bhujangasana"


class TestAsanaEvaluation:
    """Test evaluating snapshots against classical criteria."""

    def test_warrior_ii_aligned(self):
        """Warrior II with 90 deg lead knee, 180 deg rear knee, upright spine."""
        snapshot = BiomechanicalSnapshot(
            left_knee_angle=90.0,
            right_knee_angle=178.0,
            spine_verticality_deg=3.0,
            shoulder_level_deg=2.0,
        )
        deviations = evaluate_pose_against_criteria("Virabhadrasana II", snapshot, lead_leg="left")
        assert len(deviations) == 4
        # All must be aligned
        assert all(d.is_aligned for d in deviations)
        assert all(d.severity == "none" for d in deviations)

    def test_warrior_ii_misaligned_front_knee(self):
        """Warrior II with shallow front knee (68 deg - not deep enough)."""
        snapshot = BiomechanicalSnapshot(
            left_knee_angle=68.0,  # Below 85 deg target min
            right_knee_angle=178.0,
            spine_verticality_deg=3.0,
        )
        deviations = evaluate_pose_against_criteria("Virabhadrasana II", snapshot, lead_leg="left")
        lead_knee_dev = next(d for d in deviations if d.joint_name == "lead_knee")
        assert not lead_knee_dev.is_aligned
        assert lead_knee_dev.severity in ("moderate", "severe")
        assert "Deepen your front knee bend" in lead_knee_dev.correction_cue

    def test_tadasana_tilted_spine(self):
        """Mountain Pose with significant torso lean (14 deg)."""
        snapshot = BiomechanicalSnapshot(
            left_knee_angle=178.0,
            right_knee_angle=178.0,
            spine_verticality_deg=14.0,  # Target max is 5.0 deg
        )
        deviations = evaluate_pose_against_criteria("Tadasana", snapshot)
        spine_dev = next(d for d in deviations if d.joint_name == "spine_verticality")
        assert not spine_dev.is_aligned
        assert spine_dev.deviation_deg == 9.0
        assert "Lengthen through the crown" in spine_dev.correction_cue

    def test_vrikshasana_tree_pose(self):
        """Tree pose with left standing leg and right bent leg."""
        snapshot = BiomechanicalSnapshot(
            left_knee_angle=178.0,  # Standing leg straight
            right_knee_angle=45.0,  # Bent leg in triangle
            spine_verticality_deg=4.0,
        )
        deviations = evaluate_pose_against_criteria("Vrikshasana", snapshot, lead_leg="left")
        assert all(d.is_aligned for d in deviations)

    def test_unknown_pose_graceful_handling(self):
        """Evaluating an unindexed pose returns empty deviations list."""
        snapshot = BiomechanicalSnapshot()
        deviations = evaluate_pose_against_criteria("SuperUnknownAsana123", snapshot)
        assert deviations == []
