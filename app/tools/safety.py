"""
app/tools/safety.py — Deterministic Safety & Constraint Validation Tool

Validates candidate practice plans against user profile limitations:
1. Contraindication matching (injury, medical conditions vs pose cautions)
2. Duration verification (total sequence minutes vs available time)
3. Experience level compatibility (beginner safety protection)

Returns structured validation results matching PLAN.md §27 schema.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Load knowledge corpus for pose contraindication lookups
_CORPUS_PATH = Path(__file__).parent.parent.parent / "data" / "knowledge" / "corpus.json"


def _load_pose_database() -> dict[str, dict[str, Any]]:
    """Build a lookup map of pose name/sanskrit_name -> pose metadata."""
    if not _CORPUS_PATH.exists():
        return {}
    try:
        with open(_CORPUS_PATH, encoding="utf-8") as f:
            chunks = json.load(f)
        lookup: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            # Map title, sanskrit_name, english_name
            for key in ["title", "sanskrit_name", "english_name"]:
                val = chunk.get(key)
                if val and isinstance(val, str):
                    lookup[val.strip().lower()] = chunk
        return lookup
    except Exception:
        return {}


_POSE_LOOKUP = _load_pose_database()

# Constraint semantic patterns
CONSTRAINT_KEYWORDS: dict[str, list[str]] = {
    "back": ["back", "spine", "spinal", "lumbar", "disc", "sciatica", "back pain", "back injury"],
    "neck": ["neck", "cervical", "neck pain", "neck injury"],
    "knee": ["knee", "knees", "knee pain", "knee injury", "meniscus"],
    "wrist": ["wrist", "carpal tunnel", "wrist pain"],
    "shoulder": ["shoulder", "rotator cuff"],
    "hypertension": ["high blood pressure", "hypertension", "bp"],
    "heart": ["heart", "cardiac", "angina"],
    "pregnancy": ["pregnant", "pregnancy", "trimester"],
    "hernia": ["hernia"],
    "glaucoma": ["glaucoma", "eye pressure"],
}


def _match_constraint(user_constraint: str, pose_contraindications: list[str]) -> bool:
    """Check if any pose contraindication conflicts with a user constraint."""
    u_lower = user_constraint.lower()
    for pose_contra in pose_contraindications:
        p_lower = pose_contra.lower()
        # Direct substring match
        if u_lower in p_lower or p_lower in u_lower:
            return True
        # Category keyword match
        for keywords in CONSTRAINT_KEYWORDS.values():
            if any(k in u_lower for k in keywords) and any(k in p_lower for k in keywords):
                return True
    return False


def validate_safety_and_constraints(
    candidate_plan: dict[str, Any],
    user_profile: dict[str, Any],
) -> dict[str, Any]:
    """
    Deterministically validate a candidate plan against user profile constraints.

    Args:
        candidate_plan: Dict containing 'sequence', 'duration_minutes', etc.
        user_profile:   Dict containing 'constraints', 'available_duration_minutes',
                        'experience_level', etc.

    Returns:
        Structured validation dict with 'status' (PASS/FAIL), violations, and metrics.
    """
    violations: list[dict[str, Any]] = []

    # 1. Parse user constraints
    raw_constraints = user_profile.get("constraints") or []
    if isinstance(raw_constraints, str):
        raw_constraints = [c.strip() for c in re.split(r"[,;.]", raw_constraints) if c.strip()]
    elif not isinstance(raw_constraints, list):
        raw_constraints = []

    # 2. Extract sequence
    sequence = candidate_plan.get("sequence") or []
    total_minutes = 0

    # 3. Check each pose in sequence
    for pose in sequence:
        name = pose.get("name", "")
        sanskrit = pose.get("sanskrit_name", "")
        duration = pose.get("duration_minutes", 0)
        try:
            total_minutes += int(duration)
        except (ValueError, TypeError):
            pass

        # Lookup pose in knowledge base
        meta = _POSE_LOOKUP.get(name.lower()) or _POSE_LOOKUP.get(sanskrit.lower()) or {}
        pose_contra = meta.get("contraindications") or []
        pose_level = meta.get("level", "Beginner")

        # Check contraindications
        for u_const in raw_constraints:
            if _match_constraint(u_const, pose_contra):
                violations.append({
                    "type": "contraindication",
                    "pose": name,
                    "sanskrit_name": sanskrit,
                    "user_constraint": u_const,
                    "pose_contraindications": pose_contra,
                    "severity": "HIGH",
                    "recommendation": f"Replace {name} due to conflict with '{u_const}'.",
                })

        # Check experience level: beginner should not do advanced postures
        user_level = str(user_profile.get("experience_level", "beginner")).lower()
        if "beginner" in user_level and "advance" in pose_level.lower():
            violations.append({
                "type": "experience_level_mismatch",
                "pose": name,
                "user_level": user_level,
                "pose_level": pose_level,
                "severity": "MEDIUM",
                "recommendation": f"Replace advanced pose '{name}' with a beginner-accessible modification.",
            })

    # 4. Duration Check
    target_duration = user_profile.get("available_duration_minutes") or candidate_plan.get("duration_minutes", 0)
    try:
        target_duration = int(target_duration)
    except (ValueError, TypeError):
        target_duration = 0

    duration_valid = True
    if target_duration > 0 and abs(total_minutes - target_duration) > 1:
        duration_valid = False
        violations.append({
            "type": "duration_mismatch",
            "target_minutes": target_duration,
            "actual_minutes": total_minutes,
            "difference": total_minutes - target_duration,
            "severity": "MEDIUM",
            "recommendation": f"Adjust pose timings so total sequence equals exactly {target_duration} minutes (currently {total_minutes}m).",
        })

    # 5. Overall status determination
    is_safe = len([v for v in violations if v.get("severity") == "HIGH"]) == 0
    status = "PASS" if (is_safe and duration_valid) else "FAIL"

    summary = (
        f"Validation PASSED: {len(sequence)} poses cleared safety checks, total {total_minutes}m matches target {target_duration}m."
        if status == "PASS"
        else f"Validation FAILED: {len(violations)} safety/constraint issues detected ({', '.join(v['type'] for v in violations)})."
    )

    return {
        "status": status,
        "is_safe": is_safe,
        "violations": violations,
        "duration_check": {
            "target_minutes": target_duration,
            "actual_minutes": total_minutes,
            "valid": duration_valid,
        },
        "level_check": {
            "user_level": user_profile.get("experience_level", "beginner"),
            "valid": not any(v.get("type") == "experience_level_mismatch" for v in violations),
        },
        "summary": summary,
    }
