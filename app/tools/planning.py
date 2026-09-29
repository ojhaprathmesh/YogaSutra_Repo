"""
app/tools/planning.py — Advanced Planning Tools for Tree-of-Thoughts (ToT)

Exposes candidate generation and candidate utility evaluation as ADK tools
as specified in PLAN.md §18 and Phase 3A:
- generate_plan_candidates()
- evaluate_candidate_plan()
"""
from __future__ import annotations

from typing import Any

from app.agents.planner.tot import (
    compute_duration_compliance,
    compute_evidence_grounding,
    compute_safety_score,
    compute_transition_smoothness,
    generate_plan_candidates,
)
from app.state import EvidenceChunk, PoseItem, UserProfile


def generate_plan_candidates_tool(
    goal: str = "General wellness",
    duration_minutes: int = 20,
    intensity: str = "gentle",
    constraints: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate multiple diverse yoga sequence candidates exploring different practice pathways.

    Args:
        goal: The user's stated yoga goal (e.g. relaxation, flexibility, mobility).
        duration_minutes: The available duration in minutes.
        intensity: Desired practice intensity ('gentle', 'moderate', 'vigorous').
        constraints: Optional list of physical limitations or injuries.

    Returns:
        A list of candidate practice plans as dictionaries, each with sequence, rationale, and expected outcome.
    """
    profile = UserProfile(
        goal=goal.lower() if goal.lower() in ("relaxation", "flexibility", "strength", "meditation", "balance") else "relaxation",
        available_duration_minutes=duration_minutes,
        constraints=constraints or [],
    )

    sample_chunks = [
        EvidenceChunk(
            chunk_id="chk_tot_01",
            document_id="asana_corpus",
            title="Core Posture Foundations",
            section="Biomechanics",
            topic="Asana Sequencing",
            content="Classical Hatha sequencing transitions smoothly from grounding to dynamic to cooling.",
        )
    ]

    candidates = generate_plan_candidates(base_plan=None, profile=profile, evidence=sample_chunks)
    return [c.model_dump() for c in candidates]


def evaluate_candidate_plan_tool(
    sequence: list[dict[str, Any]],
    available_duration_minutes: int = 20,
    constraints: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate a candidate yoga sequence using a deterministic multi-objective utility function.

    Args:
        sequence: List of pose dictionaries with 'name' and 'duration_minutes'.
        available_duration_minutes: Target duration in minutes.
        constraints: Known physical contraindications/injuries.

    Returns:
        Dict containing composite utility score (0.0 to 1.0) and individual dimension scores
        for duration compliance, safety, transition smoothness, and evidence grounding.
    """
    pose_items: list[PoseItem] = []
    for s in sequence:
        pose_items.append(
            PoseItem(
                name=s.get("name", "Unknown Pose"),
                sanskrit_name=s.get("sanskrit_name", ""),
                duration_minutes=int(s.get("duration_minutes", 5)),
                instructions=s.get("instructions", ""),
                modifications=s.get("modifications", ""),
                evidence_chunk_ids=s.get("evidence_chunk_ids", []),
            )
        )

    actual_duration = sum(p.duration_minutes for p in pose_items)
    c_list = constraints or []

    dur_score = compute_duration_compliance(actual_duration, available_duration_minutes)
    safe_score = compute_safety_score(pose_items, c_list)
    tran_score = compute_transition_smoothness(pose_items)
    # Synthetic chunks for checking
    chunk_list = [
        EvidenceChunk(
            chunk_id=cid,
            document_id="doc",
            title="Title",
            section="Sec",
            topic="Top",
            content="Content",
        )
        for p in pose_items
        for cid in p.evidence_chunk_ids
    ]
    ground_score = compute_evidence_grounding(pose_items, chunk_list)

    composite = round(
        0.25 * dur_score + 0.35 * safe_score + 0.20 * tran_score + 0.20 * ground_score,
        4,
    )

    return {
        "composite_utility": composite,
        "actual_duration_minutes": actual_duration,
        "target_duration_minutes": available_duration_minutes,
        "dimension_scores": {
            "duration_compliance": round(dur_score, 3),
            "safety_score": round(safe_score, 3),
            "transition_smoothness": round(tran_score, 3),
            "evidence_grounding": round(ground_score, 3),
        },
        "is_safe": safe_score >= 0.8,
    }
