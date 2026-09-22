"""
tests/test_tot_planning.py — Unit Tests for Tree-of-Thoughts Exploration & Utility Scorer (Phase 3A)
"""
from __future__ import annotations

from app.agents.planner.tot import (
    compute_duration_compliance,
    compute_evidence_grounding,
    compute_safety_score,
    compute_transition_smoothness,
    evaluate_candidate_plan,
    generate_plan_candidates,
    select_best_candidate,
)
from app.state import EvidenceChunk, PoseItem, PracticePlan, UserProfile
from app.tools.planning import evaluate_candidate_plan_tool, generate_plan_candidates_tool


def test_duration_compliance_math():
    assert compute_duration_compliance(20, 20) == 1.0
    assert compute_duration_compliance(15, 20) == 0.75
    assert compute_duration_compliance(30, 20) == 0.5
    assert compute_duration_compliance(50, 20) == 0.0


def test_safety_score_penalties():
    safe_sequence = [
        PoseItem(name="Sukhasana", duration_minutes=5),
        PoseItem(name="Balasana", duration_minutes=5),
        PoseItem(name="Savasana", duration_minutes=10),
    ]
    # No constraints -> 1.0
    assert compute_safety_score(safe_sequence, []) == 1.0

    # Knee injury with Lotus -> penalized
    unsafe_knee_sequence = [
        PoseItem(name="Padmasana (Lotus Pose)", duration_minutes=5),
        PoseItem(name="Savasana", duration_minutes=15),
    ]
    assert compute_safety_score(unsafe_knee_sequence, ["Knee ligament sprain"]) < 0.5

    # Hypertension with Headstand -> penalized
    unsafe_htn_sequence = [
        PoseItem(name="Sirsasana (Headstand)", duration_minutes=5),
        PoseItem(name="Savasana", duration_minutes=15),
    ]
    assert compute_safety_score(unsafe_htn_sequence, ["Hypertension"]) == 0.0


def test_transition_smoothness_and_savasana_reward():
    # Sequence with warm-up (Sukhasana) and ending with Savasana
    smooth = [
        PoseItem(name="Sukhasana", duration_minutes=5),
        PoseItem(name="Cat-Cow", duration_minutes=5),
        PoseItem(name="Balasana", duration_minutes=5),
        PoseItem(name="Savasana", duration_minutes=5),
    ]
    score_smooth = compute_transition_smoothness(smooth)
    assert score_smooth >= 0.9

    # Sequence without Savasana
    no_savasana = [
        PoseItem(name="Tadasana", duration_minutes=10),
        PoseItem(name="Virabhadrasana II", duration_minutes=10),
    ]
    score_no_sav = compute_transition_smoothness(no_savasana)
    assert score_no_sav < score_smooth


def test_evidence_grounding():
    chunks = [
        EvidenceChunk(
            chunk_id="chunk_1",
            document_id="doc1",
            title="Title",
            section="Sec",
            topic="Top",
            content="Content",
        )
    ]
    grounded_seq = [
        PoseItem(name="Tadasana", duration_minutes=10, evidence_chunk_ids=["chunk_1"]),
        PoseItem(name="Savasana", duration_minutes=10, evidence_chunk_ids=["chunk_1"]),
    ]
    assert compute_evidence_grounding(grounded_seq, chunks) == 1.0

    ungrounded_seq = [
        PoseItem(name="Tadasana", duration_minutes=10),
        PoseItem(name="Savasana", duration_minutes=10),
    ]
    assert compute_evidence_grounding(ungrounded_seq, chunks) < 0.5


def test_tot_candidate_generation_and_selection():
    profile = UserProfile(
        experience_level="beginner",
        available_duration_minutes=20,
        goal="relaxation",
        constraints=["mild back stiffness"],
    )
    chunks = [
        EvidenceChunk(
            chunk_id="chk_tot_01",
            document_id="doc",
            title="Spinal Decompression",
            section="Yoga Therapy",
            topic="Relaxation",
            content="Gentle child's pose decompresses the lumbar spine.",
        )
    ]

    candidates = generate_plan_candidates(base_plan=None, profile=profile, evidence=chunks)
    assert len(candidates) == 3

    best_plan, evaluations, best_idx = select_best_candidate(candidates, profile, chunks)
    assert best_plan is not None
    assert len(evaluations) == 3
    assert 0 <= best_idx < 3
    assert evaluations[best_idx]["composite_utility"] >= max(e["composite_utility"] for e in evaluations)


def test_planning_adk_tools():
    candidates = generate_plan_candidates_tool(goal="relaxation", duration_minutes=20)
    assert len(candidates) == 3
    assert all("sequence" in c for c in candidates)

    eval_result = evaluate_candidate_plan_tool(
        sequence=candidates[0]["sequence"],
        available_duration_minutes=20,
    )
    assert "composite_utility" in eval_result
    assert 0.0 <= eval_result["composite_utility"] <= 1.0
    assert "dimension_scores" in eval_result
