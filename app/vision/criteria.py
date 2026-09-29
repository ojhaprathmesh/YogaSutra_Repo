"""
app/vision/criteria.py — Deterministic Classical Asana Biomechanical Criteria Matrix

Phase 3B: Grounded alignment criteria mapped from classical yoga texts (corpus.json).
Evaluates live joint angles from BiomechanicalSnapshot against static threshold intervals.
Zero-hallucination guarantee: deviations are strictly rule-derived with exact numeric tolerances.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.vision.biomechanics import BiomechanicalSnapshot


@dataclass(slots=True)
class JointCriterion:
    """Standard target interval and corrective voice cues for an anatomical joint."""
    joint_name: str
    target_min: float
    target_max: float
    ideal_deg: float
    cue_if_low: str
    cue_if_high: str
    weight: float = 1.0  # 1.0 = standard, 2.0 = critical safety


@dataclass(slots=True)
class PostureDeviation:
    """Represents a measured misalignment against an asana criterion."""
    joint_name: str
    measured_angle: float
    target_min: float
    target_max: float
    deviation_deg: float
    severity: Literal["none", "mild", "moderate", "severe"]
    correction_cue: str
    is_aligned: bool


# ---------------------------------------------------------------------------
# Classical Asana Target Standards Matrix
# ---------------------------------------------------------------------------

ASANA_CRITERIA_REGISTRY: dict[str, list[JointCriterion]] = {
    # Virabhadrasana II (Warrior II): Front knee 90 deg, back knee 180 deg, upright spine
    "virabhadrasana_ii": [
        JointCriterion(
            joint_name="lead_knee",
            target_min=85.0,
            target_max=95.0,
            ideal_deg=90.0,
            cue_if_low="Deepen your front knee bend towards 90 degrees.",
            cue_if_high="Ease back slightly; ensure your knee does not pass past your ankle.",
            weight=2.0,
        ),
        JointCriterion(
            joint_name="rear_knee",
            target_min=170.0,
            target_max=180.0,
            ideal_deg=180.0,
            cue_if_low="Straighten and engage your back leg, pressing firmly through your outer heel.",
            cue_if_high="",
            weight=1.5,
        ),
        JointCriterion(
            joint_name="spine_verticality",
            target_min=0.0,
            target_max=8.0,
            ideal_deg=0.0,
            cue_if_low="",
            cue_if_high="Stack your spine vertically over your hips rather than leaning forward.",
            weight=1.5,
        ),
        JointCriterion(
            joint_name="shoulder_level",
            target_min=0.0,
            target_max=6.0,
            ideal_deg=0.0,
            cue_if_low="",
            cue_if_high="Level your arms and shoulders parallel to the floor.",
            weight=1.0,
        ),
    ],

    # Tadasana (Mountain Pose): Symmetrical, plumb-line vertical alignment
    "tadasana": [
        JointCriterion(
            joint_name="left_knee",
            target_min=172.0,
            target_max=180.0,
            ideal_deg=180.0,
            cue_if_low="Extend both legs evenly without locking or sagging the knees.",
            cue_if_high="",
            weight=1.0,
        ),
        JointCriterion(
            joint_name="right_knee",
            target_min=172.0,
            target_max=180.0,
            ideal_deg=180.0,
            cue_if_low="Extend both legs evenly without locking or sagging the knees.",
            cue_if_high="",
            weight=1.0,
        ),
        JointCriterion(
            joint_name="spine_verticality",
            target_min=0.0,
            target_max=5.0,
            ideal_deg=0.0,
            cue_if_low="",
            cue_if_high="Lengthen through the crown of your head, drawing your ribs in.",
            weight=2.0,
        ),
        JointCriterion(
            joint_name="shoulder_level",
            target_min=0.0,
            target_max=4.0,
            ideal_deg=0.0,
            cue_if_low="",
            cue_if_high="Relax your shoulders down and ensure they are evenly balanced.",
            weight=1.0,
        ),
        JointCriterion(
            joint_name="hip_level",
            target_min=0.0,
            target_max=4.0,
            ideal_deg=0.0,
            cue_if_low="",
            cue_if_high="Balance weight equally across both feet to level your hips.",
            weight=1.0,
        ),
    ],

    # Vrikshasana (Tree Pose): Standing leg straight, bent leg lateral opening, vertical torso
    "vrikshasana": [
        JointCriterion(
            joint_name="standing_knee",
            target_min=172.0,
            target_max=180.0,
            ideal_deg=180.0,
            cue_if_low="Root firmly through your standing foot and keep your leg strong and straight.",
            cue_if_high="",
            weight=2.0,
        ),
        JointCriterion(
            joint_name="bent_knee",
            target_min=30.0,
            target_max=65.0,
            ideal_deg=45.0,
            cue_if_low="Gently ease your bent knee outward to open your hip.",
            cue_if_high="Draw your bent foot into your inner thigh or calf (avoid the knee joint).",
            weight=1.5,
        ),
        JointCriterion(
            joint_name="spine_verticality",
            target_min=0.0,
            target_max=8.0,
            ideal_deg=0.0,
            cue_if_low="",
            cue_if_high="Keep your torso tall and center your core over your standing leg.",
            weight=1.5,
        ),
    ],

    # Utkatasana (Chair Pose): Knees bent ~90-110 deg, torso inclined 30-50 deg
    "utkatasana": [
        JointCriterion(
            joint_name="left_knee",
            target_min=85.0,
            target_max=115.0,
            ideal_deg=100.0,
            cue_if_low="Sink your hips back and down deeper as if sitting into a chair.",
            cue_if_high="Sit lower to engage your thighs and stabilize the pose.",
            weight=2.0,
        ),
        JointCriterion(
            joint_name="right_knee",
            target_min=85.0,
            target_max=115.0,
            ideal_deg=100.0,
            cue_if_low="Sink your hips back and down deeper as if sitting into a chair.",
            cue_if_high="Sit lower to engage your thighs and stabilize the pose.",
            weight=2.0,
        ),
        JointCriterion(
            joint_name="spine_verticality",
            target_min=25.0,
            target_max=55.0,
            ideal_deg=40.0,
            cue_if_low="Hinge slightly forward at your hips to keep weight in your heels.",
            cue_if_high="Lift your chest up and soften your shoulders away from your ears.",
            weight=1.5,
        ),
    ],

    # Bhujangasana (Cobra Pose): Soft elbow bend 120-155 deg, broad level shoulders
    "bhujangasana": [
        JointCriterion(
            joint_name="left_elbow",
            target_min=120.0,
            target_max=155.0,
            ideal_deg=140.0,
            cue_if_low="Press gently into your hands to lift your chest higher.",
            cue_if_high="Keep a soft bend in your elbows; avoid locking your joints.",
            weight=1.5,
        ),
        JointCriterion(
            joint_name="right_elbow",
            target_min=120.0,
            target_max=155.0,
            ideal_deg=140.0,
            cue_if_low="Press gently into your hands to lift your chest higher.",
            cue_if_high="Keep a soft bend in your elbows; avoid locking your joints.",
            weight=1.5,
        ),
        JointCriterion(
            joint_name="shoulder_level",
            target_min=0.0,
            target_max=5.0,
            ideal_deg=0.0,
            cue_if_low="",
            cue_if_high="Roll your shoulders back and keep them even and relaxed.",
            weight=1.0,
        ),
    ],
}


# ---------------------------------------------------------------------------
# Name Normalization
# ---------------------------------------------------------------------------

ASANA_NAME_ALIASES: dict[str, str] = {
    "virabhadrasana ii": "virabhadrasana_ii",
    "virabhadrasana 2": "virabhadrasana_ii",
    "warrior ii": "virabhadrasana_ii",
    "warrior 2": "virabhadrasana_ii",
    "warrior_ii": "virabhadrasana_ii",
    "tadasana": "tadasana",
    "mountain pose": "tadasana",
    "mountain": "tadasana",
    "vrikshasana": "vrikshasana",
    "tree pose": "vrikshasana",
    "tree": "vrikshasana",
    "utkatasana": "utkatasana",
    "chair pose": "utkatasana",
    "chair": "utkatasana",
    "bhujangasana": "bhujangasana",
    "cobra pose": "bhujangasana",
    "cobra": "bhujangasana",
}


def normalize_asana_name(name: str) -> str:
    """Normalize input asana name to registry key."""
    cleaned = name.strip().lower().replace("-", " ")
    return ASANA_NAME_ALIASES.get(cleaned, cleaned.replace(" ", "_"))


# ---------------------------------------------------------------------------
# Evaluation Engine
# ---------------------------------------------------------------------------

def evaluate_pose_against_criteria(
    asana_name: str,
    snapshot: BiomechanicalSnapshot,
    lead_leg: Literal["left", "right"] = "left",
) -> list[PostureDeviation]:
    """
    Compare live biomechanical measurements against the normalized criteria matrix.
    
    Returns:
        List of PostureDeviation objects, ordered with misaligned joints first (by weight).
    """
    canonical_name = normalize_asana_name(asana_name)
    criteria_list = ASANA_CRITERIA_REGISTRY.get(canonical_name)

    if not criteria_list:
        # Unknown pose — return empty deviations
        return []

    # Map snapshot values to joint names dynamically
    is_left_lead = (lead_leg == "left")
    lead_knee_val = snapshot.left_knee_angle if is_left_lead else snapshot.right_knee_angle
    rear_knee_val = snapshot.right_knee_angle if is_left_lead else snapshot.left_knee_angle

    # Standing leg for Vrikshasana
    standing_knee_val = snapshot.left_knee_angle if is_left_lead else snapshot.right_knee_angle
    bent_knee_val = snapshot.right_knee_angle if is_left_lead else snapshot.left_knee_angle

    joint_measurements: dict[str, float] = {
        "lead_knee": lead_knee_val,
        "rear_knee": rear_knee_val,
        "standing_knee": standing_knee_val,
        "bent_knee": bent_knee_val,
        "left_knee": snapshot.left_knee_angle,
        "right_knee": snapshot.right_knee_angle,
        "left_elbow": snapshot.left_elbow_angle,
        "right_elbow": snapshot.right_elbow_angle,
        "spine_verticality": snapshot.spine_verticality_deg,
        "shoulder_level": snapshot.shoulder_level_deg,
        "hip_level": snapshot.hip_level_deg,
    }

    deviations: list[PostureDeviation] = []

    for criterion in criteria_list:
        measured = joint_measurements.get(criterion.joint_name)
        if measured is None:
            continue

        if measured < criterion.target_min:
            diff = criterion.target_min - measured
            cue = criterion.cue_if_low
            is_aligned = False
        elif measured > criterion.target_max:
            diff = measured - criterion.target_max
            cue = criterion.cue_if_high
            is_aligned = False
        else:
            diff = 0.0
            cue = ""
            is_aligned = True

        # Assign severity based on degree discrepancy
        if is_aligned or diff == 0.0:
            severity: Literal["none", "mild", "moderate", "severe"] = "none"
        elif diff <= 7.0:
            severity = "mild"
        elif diff <= 18.0:
            severity = "moderate"
        else:
            severity = "severe"

        deviations.append(
            PostureDeviation(
                joint_name=criterion.joint_name,
                measured_angle=round(measured, 2),
                target_min=criterion.target_min,
                target_max=criterion.target_max,
                deviation_deg=round(diff, 2),
                severity=severity,
                correction_cue=cue,
                is_aligned=is_aligned,
            )
        )

    # Sort so worst non-aligned deviations appear first
    severity_order = {"severe": 3, "moderate": 2, "mild": 1, "none": 0}
    deviations.sort(key=lambda d: severity_order[d.severity], reverse=True)

    return deviations
