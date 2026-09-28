"""
app/vision — Edge Real-Time Computer Vision & Biomechanics Pipeline
Phase 3B: High-efficiency, zero-hallucination pose detection and joint geometry.
"""
from app.vision.biomechanics import (
    Point3D,
    BiomechanicalSnapshot,
    calculate_angle_3d,
    calculate_knee_angle,
    calculate_elbow_angle,
    calculate_spine_verticality,
    calculate_shoulder_symmetry,
    calculate_hip_symmetry,
    extract_biomechanics_from_landmarks,
)
from app.vision.criteria import (
    JointCriterion,
    PostureDeviation,
    ASANA_CRITERIA_REGISTRY,
    evaluate_pose_against_criteria,
)
from app.vision.detector import PoseDetector

__all__ = [
    "Point3D",
    "BiomechanicalSnapshot",
    "calculate_angle_3d",
    "calculate_knee_angle",
    "calculate_elbow_angle",
    "calculate_spine_verticality",
    "calculate_shoulder_symmetry",
    "calculate_hip_symmetry",
    "extract_biomechanics_from_landmarks",
    "JointCriterion",
    "PostureDeviation",
    "ASANA_CRITERIA_REGISTRY",
    "evaluate_pose_against_criteria",
    "PoseDetector",
]
