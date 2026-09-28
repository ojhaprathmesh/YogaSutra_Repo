"""
tests/test_biomechanics.py — Comprehensive Unit Tests for 3D Vector Geometry Calculator

Phase 3B: Asserts exact mathematical correctness of the pure-Python 3D vector geometry engine.
Guarantees 0.00° error on known Euclidean geometry coordinates with zero hallucination.
"""
import math
import pytest

from app.vision.biomechanics import (
    Point3D,
    calculate_angle_3d,
    calculate_knee_angle,
    calculate_elbow_angle,
    calculate_spine_verticality,
    calculate_shoulder_symmetry,
    calculate_hip_symmetry,
    extract_biomechanics_from_landmarks,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
)


class TestVectorGeometry3D:
    """Test pure-Python 3D vector geometry angle calculations."""

    def test_exact_90_degree_angle(self):
        """Corner in XY plane: (1, 0, 0) -> (0, 0, 0) -> (0, 1, 0) must be exactly 90.0 deg."""
        a = Point3D(1.0, 0.0, 0.0)
        b = Point3D(0.0, 0.0, 0.0)
        c = Point3D(0.0, 1.0, 0.0)
        angle = calculate_angle_3d(a, b, c)
        assert pytest.approx(angle, abs=1e-5) == 90.0

    def test_exact_180_degree_collinear(self):
        """Straight line: (-1, 0, 0) -> (0, 0, 0) -> (1, 0, 0) must be exactly 180.0 deg."""
        a = Point3D(-1.0, 0.0, 0.0)
        b = Point3D(0.0, 0.0, 0.0)
        c = Point3D(1.0, 0.0, 0.0)
        angle = calculate_angle_3d(a, b, c)
        assert pytest.approx(angle, abs=1e-5) == 180.0

    def test_exact_45_degree_angle(self):
        """45-degree angle in XY plane."""
        a = Point3D(1.0, 0.0, 0.0)
        b = Point3D(0.0, 0.0, 0.0)
        c = Point3D(1.0, 1.0, 0.0)
        angle = calculate_angle_3d(a, b, c)
        assert pytest.approx(angle, abs=1e-5) == 45.0

    def test_3d_out_of_plane_90_degree(self):
        """3D orthogonal vector: (1, 0, 0) -> (0, 0, 0) -> (0, 0, 1) must be 90.0 deg."""
        a = Point3D(1.0, 0.0, 0.0)
        b = Point3D(0.0, 0.0, 0.0)
        c = Point3D(0.0, 0.0, 1.0)
        angle = calculate_angle_3d(a, b, c)
        assert pytest.approx(angle, abs=1e-5) == 90.0

    def test_zero_vector_degeneracy(self):
        """Coincident points must return 0.0 without raising ZeroDivisionError."""
        a = Point3D(0.0, 0.0, 0.0)
        b = Point3D(0.0, 0.0, 0.0)
        c = Point3D(1.0, 0.0, 0.0)
        assert calculate_angle_3d(a, b, c) == 0.0
        assert calculate_angle_3d(b, b, b) == 0.0


class TestBiomechanicalJointCalculations:
    """Test anatomical joint helper functions."""

    def test_knee_flexion_90_degrees(self):
        """Hip at (0.5, 0.2), Knee at (0.5, 0.5), Ankle at (0.8, 0.5) = 90 deg."""
        hip = Point3D(0.5, 0.2, 0.0)
        knee = Point3D(0.5, 0.5, 0.0)
        ankle = Point3D(0.8, 0.5, 0.0)
        knee_angle = calculate_knee_angle(hip, knee, ankle)
        assert pytest.approx(knee_angle, abs=1e-4) == 90.0

    def test_knee_extension_180_degrees(self):
        """Straight leg: Hip at (0.5, 0.2), Knee at (0.5, 0.5), Ankle at (0.5, 0.8) = 180 deg."""
        hip = Point3D(0.5, 0.2, 0.0)
        knee = Point3D(0.5, 0.5, 0.0)
        ankle = Point3D(0.5, 0.8, 0.0)
        knee_angle = calculate_knee_angle(hip, knee, ankle)
        assert pytest.approx(knee_angle, abs=1e-4) == 180.0

    def test_elbow_flexion_90_degrees(self):
        shoulder = Point3D(0.4, 0.3, 0.0)
        elbow = Point3D(0.6, 0.3, 0.0)
        wrist = Point3D(0.6, 0.1, 0.0)
        elbow_angle = calculate_elbow_angle(shoulder, elbow, wrist)
        assert pytest.approx(elbow_angle, abs=1e-4) == 90.0

    def test_spine_verticality_perfectly_vertical(self):
        """Torso vertical in screen coordinates (mid_hip below mid_shoulder)."""
        mid_hip = Point3D(0.5, 0.6, 0.0)
        mid_shoulder = Point3D(0.5, 0.2, 0.0)
        vert = calculate_spine_verticality(mid_hip, mid_shoulder)
        assert pytest.approx(vert, abs=1e-4) == 0.0

    def test_spine_verticality_45_degree_tilt(self):
        """Torso tilted 45 degrees forward."""
        mid_hip = Point3D(0.5, 0.6, 0.0)
        mid_shoulder = Point3D(0.9, 0.2, 0.0)  # delta x = 0.4, delta y = -0.4 -> 45 deg
        vert = calculate_spine_verticality(mid_hip, mid_shoulder)
        assert pytest.approx(vert, abs=1e-4) == 45.0

    def test_shoulder_symmetry_level(self):
        """Shoulders horizontal across the screen."""
        left = Point3D(0.4, 0.3, 0.0)
        right = Point3D(0.6, 0.3, 0.0)
        assert pytest.approx(calculate_shoulder_symmetry(left, right), abs=1e-4) == 0.0

    def test_hip_symmetry_tilted(self):
        """Left hip higher than right hip."""
        left = Point3D(0.4, 0.48, 0.0)
        right = Point3D(0.6, 0.52, 0.0)
        incline = calculate_hip_symmetry(left, right)
        assert incline > 5.0


class TestSnapshotExtraction:
    """Test extracting complete biomechanical snapshot from landmark map."""

    def test_full_landmarks_extraction(self):
        landmarks = {
            LEFT_HIP: Point3D(0.45, 0.5, 0.0),
            RIGHT_HIP: Point3D(0.55, 0.5, 0.0),
            LEFT_KNEE: Point3D(0.45, 0.7, 0.0),
            RIGHT_KNEE: Point3D(0.55, 0.7, 0.0),
            LEFT_ANKLE: Point3D(0.45, 0.9, 0.0),
            RIGHT_ANKLE: Point3D(0.55, 0.9, 0.0),
            LEFT_SHOULDER: Point3D(0.40, 0.3, 0.0),
            RIGHT_SHOULDER: Point3D(0.60, 0.3, 0.0),
            LEFT_ELBOW: Point3D(0.30, 0.3, 0.0),
            RIGHT_ELBOW: Point3D(0.70, 0.3, 0.0),
            LEFT_WRIST: Point3D(0.20, 0.3, 0.0),
            RIGHT_WRIST: Point3D(0.80, 0.3, 0.0),
        }
        snapshot = extract_biomechanics_from_landmarks(landmarks)

        # Both legs straight (180 deg)
        assert pytest.approx(snapshot.left_knee_angle, abs=1e-2) == 180.0
        assert pytest.approx(snapshot.right_knee_angle, abs=1e-2) == 180.0
        # Both arms straight out (180 deg)
        assert pytest.approx(snapshot.left_elbow_angle, abs=1e-2) == 180.0
        assert pytest.approx(snapshot.right_elbow_angle, abs=1e-2) == 180.0
        # Upright spine
        assert pytest.approx(snapshot.spine_verticality_deg, abs=1e-2) == 0.0
        # Shoulders level
        assert pytest.approx(snapshot.shoulder_level_deg, abs=1e-2) == 0.0
        # Hips level
        assert pytest.approx(snapshot.hip_level_deg, abs=1e-2) == 0.0

    def test_partial_landmarks_graceful_handling(self):
        """Missing landmarks should default without crashing."""
        snapshot = extract_biomechanics_from_landmarks({})
        assert snapshot.left_knee_angle == 180.0
        assert snapshot.right_knee_angle == 180.0
        assert snapshot.spine_verticality_deg == 0.0
