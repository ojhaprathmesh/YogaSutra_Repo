"""
app/agents/planner/tot.py — Tree-of-Thoughts (ToT) Multi-Candidate Planning & Utility Scorer

Implements multi-branch candidate exploration and utility-based candidate
evaluation for the PlannerAgent as defined in PLAN.md §24 and Phase 3A.

Mathematical Utility Formulation:
  U(plan) = w_duration * DurationCompliance
          + w_safety * SafetyScore
          + w_transition * TransitionSmoothness
          + w_grounding * EvidenceGrounding
          + w_pref * PreferenceFit
"""
from __future__ import annotations

import copy
import logging
from typing import Any

from app.state import EvidenceChunk, PoseItem, PracticePlan, UserProfile

logger = logging.getLogger("yogasutra.planner.tot")

# ---------------------------------------------------------------------------
# Default Dimension Weights (Sum to 1.0)
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "duration_compliance": 0.25,
    "safety_score": 0.35,
    "transition_smoothness": 0.20,
    "evidence_grounding": 0.20,
}

# Biomechanical position categories for transition evaluation
POSE_POSITIONS: dict[str, str] = {
    # Standing
    "tadasana": "standing",
    "mountain pose": "standing",
    "vrikshasana": "standing",
    "tree pose": "standing",
    "virabhadrasana i": "standing",
    "virabhadrasana ii": "standing",
    "warrior i": "standing",
    "warrior ii": "standing",
    "trikonasana": "standing",
    "triangle pose": "standing",
    "uttanasana": "standing",
    "standing forward bend": "standing",
    # Kneeling / All-fours
    "marjaryasana": "all_fours",
    "cat pose": "all_fours",
    "bitilasana": "all_fours",
    "cow pose": "all_fours",
    "balasana": "kneeling",
    "child's pose": "kneeling",
    "vajrasana": "kneeling",
    "thunderbolt pose": "kneeling",
    # Inversion / Downward
    "adho mukha svanasana": "inversion",
    "downward-facing dog": "inversion",
    # Seated
    "sukhasana": "seated",
    "easy pose": "seated",
    "padmasana": "seated",
    "lotus pose": "seated",
    "paschimottanasana": "seated",
    "seated forward bend": "seated",
    "baddha konasana": "seated",
    "butterfly pose": "seated",
    # Supine
    "setu bandhasana": "supine",
    "bridge pose": "supine",
    "savasana": "supine",
    "corpse pose": "supine",
    "supta baddha konasana": "supine",
    # Prone
    "bhujangasana": "prone",
    "cobra pose": "prone",
    "salabhasana": "prone",
    "locust pose": "prone",
}


# ---------------------------------------------------------------------------
# Utility Scoring Functions
# ---------------------------------------------------------------------------


def compute_duration_compliance(actual_minutes: int, target_minutes: int) -> float:
    """Calculates duration compliance score in [0.0, 1.0]."""
    if target_minutes <= 0:
        return 1.0
    diff = abs(actual_minutes - target_minutes)
    return max(0.0, 1.0 - (diff / target_minutes))


def compute_safety_score(
    sequence: list[PoseItem],
    constraints: list[str],
) -> float:
    """
    Evaluates safety against known physical contraindications.
    Penalizes heavily if high-risk poses are present for stated injuries.
    """
    if not constraints:
        return 1.0

    lower_constraints = [c.lower() for c in constraints]
    penalties = 0.0

    for pose in sequence:
        p_name = pose.name.lower()
        # High blood pressure contraindications
        if any("hypertension" in c or "blood pressure" in c for c in lower_constraints):
            if any(inv in p_name for inv in ["sirsasana", "headstand", "sarvangasana", "shoulderstand"]):
                penalties += 1.0
        # Lower back contraindications
        if any("back pain" in c or "herniation" in c or "lumbar" in c for c in lower_constraints):
            if any(hyp in p_name for hyp in ["full wheel", "chakrasana", "deep backbend"]):
                penalties += 0.8
        # Knee injury contraindications
        if any("knee" in c for c in lower_constraints):
            if any(k in p_name for k in ["padmasana", "lotus", "virasana"]):
                penalties += 0.8

    return max(0.0, 1.0 - penalties)


def compute_transition_smoothness(sequence: list[PoseItem]) -> float:
    """
    Evaluates biomechanical flow between consecutive postures.
    Minimizes unnecessary up-and-down changes between standing and supine/seated.
    Rewards beginning with warmup/grounding and ending with Savasana.
    """
    if not sequence:
        return 0.5

    score = 1.0
    n = len(sequence)

    # 1. Warm-up bonus at start
    first_pose = sequence[0].name.lower()
    if any(w in first_pose for w in ["sukhasana", "balasana", "tadasana", "cat", "cow", "marjaryasana", "breathing"]):
        score += 0.1

    # 2. Savasana / relaxation bonus at end
    last_pose = sequence[-1].name.lower()
    if "savasana" in last_pose or "corpse" in last_pose or "relaxation" in last_pose:
        score += 0.1
    else:
        score -= 0.2

    # 3. Transition penalties
    last_pos: str | None = None
    position_changes = 0
    erratic_switches = 0

    for pose in sequence:
        pos = POSE_POSITIONS.get(pose.name.lower(), POSE_POSITIONS.get(pose.sanskrit_name.lower(), "general"))
        if last_pos and pos != "general" and last_pos != "general":
            if pos != last_pos:
                position_changes += 1
                # Erratic switch: standing immediately to floor supine and back to standing
                if (last_pos == "standing" and pos == "supine") or (last_pos == "supine" and pos == "standing"):
                    erratic_switches += 1
        last_pos = pos

    if erratic_switches > 0:
        score -= 0.15 * erratic_switches
    if position_changes > (n // 2) + 2:
        score -= 0.1

    return max(0.0, min(1.0, score))


def compute_evidence_grounding(
    sequence: list[PoseItem],
    available_chunks: list[EvidenceChunk],
) -> float:
    """Evaluates the proportion of poses supported by retrieved knowledge chunks."""
    if not sequence:
        return 0.0

    available_chunk_ids = {c.chunk_id for c in available_chunks}
    grounded_count = 0

    for pose in sequence:
        if pose.evidence_chunk_ids:
            if any(cid in available_chunk_ids for cid in pose.evidence_chunk_ids):
                grounded_count += 1
            else:
                # Referenced chunk ids exist but might be from general corpus
                grounded_count += 0.5
        else:
            grounded_count += 0.2  # Uncited standard pose

    return max(0.0, min(1.0, grounded_count / len(sequence)))


def evaluate_candidate_plan(
    plan: PracticePlan,
    profile: UserProfile,
    retrieved_evidence: list[EvidenceChunk],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Computes overall multi-objective utility score for a candidate practice plan.
    Returns composite score and granular dimension breakdown.
    """
    w = weights or DEFAULT_WEIGHTS

    actual_duration = sum(p.duration_minutes for p in plan.sequence)
    target_duration = profile.available_duration_minutes

    duration_score = compute_duration_compliance(actual_duration, target_duration)
    safety_score = compute_safety_score(plan.sequence, profile.constraints)
    transition_score = compute_transition_smoothness(plan.sequence)
    grounding_score = compute_evidence_grounding(plan.sequence, retrieved_evidence)

    composite_utility = (
        w.get("duration_compliance", 0.25) * duration_score
        + w.get("safety_score", 0.35) * safety_score
        + w.get("transition_smoothness", 0.20) * transition_score
        + w.get("evidence_grounding", 0.20) * grounding_score
    )

    composite_utility = round(composite_utility, 4)

    return {
        "candidate_goal": plan.goal,
        "composite_utility": composite_utility,
        "duration_minutes": actual_duration,
        "target_duration_minutes": target_duration,
        "dimensions": {
            "duration_compliance": round(duration_score, 3),
            "safety_score": round(safety_score, 3),
            "transition_smoothness": round(transition_score, 3),
            "evidence_grounding": round(grounding_score, 3),
        },
        "is_safe": safety_score >= 0.8,
    }


# ---------------------------------------------------------------------------
# ToT Multi-Candidate Exploration Strategies
# ---------------------------------------------------------------------------


def generate_plan_candidates(
    base_plan: PracticePlan | None,
    profile: UserProfile,
    evidence: list[EvidenceChunk],
) -> list[PracticePlan]:
    """
    Explores distinct sequence candidates representing diverse practice pathways:
    - Candidate A: Gentle Restorative Pathway (higher grounding, gentle transitions)
    - Candidate B: Dynamic Mobility Pathway (vitalizing, standing flow)
    - Candidate C: Precision Alignment Pathway (focused holds, props, deep breath)
    """
    candidates: list[PracticePlan] = []
    chunk_ids = [c.chunk_id for c in evidence] if evidence else ["corpus_chunk_01"]

    target_duration = profile.available_duration_minutes

    # Candidate A: Gentle / Restorative
    dur_a = max(2, target_duration // 4)
    cand_a = PracticePlan(
        goal=f"Gentle Restorative {profile.goal.title()}",
        duration_minutes=target_duration,
        intensity="gentle",
        sequence=[
            PoseItem(
                name="Sukhasana with Pranayama",
                sanskrit_name="Sukhasana",
                duration_minutes=dur_a,
                instructions="Sit comfortably with a straight spine, hands resting on knees. Practice deep rhythmic breathing.",
                evidence_chunk_ids=chunk_ids[:1],
            ),
            PoseItem(
                name="Cat-Cow Flow",
                sanskrit_name="Marjaryasana-Bitilasana",
                duration_minutes=dur_a,
                instructions="Inhale arching the spine into Cow; exhale rounding the spine into Cat.",
                evidence_chunk_ids=chunk_ids[:2],
            ),
            PoseItem(
                name="Balasana (Child's Pose)",
                sanskrit_name="Balasana",
                duration_minutes=dur_a,
                instructions="Fold forward with hips toward heels, arms extended forward on the mat.",
                evidence_chunk_ids=chunk_ids[:1],
            ),
            PoseItem(
                name="Savasana (Corpse Pose)",
                sanskrit_name="Savasana",
                duration_minutes=target_duration - (dur_a * 3),
                instructions="Lie flat on your back, letting your breath flow effortlessly and fully relaxing.",
                evidence_chunk_ids=chunk_ids[:1],
            ),
        ],
        rationale="Prioritizes spinal decompression, gentle nervous system down-regulation, and zero joint stress.",
        evidence_sources=chunk_ids[:2],
        expected_outcome="Calm nervous system, relieved lower back tension, and refreshed mental clarity.",
    )
    candidates.append(cand_a)

    # Candidate B: Dynamic Flow / Mobility
    dur_b1 = max(3, target_duration // 5)
    dur_b2 = max(4, target_duration // 4)
    cand_b = PracticePlan(
        goal=f"Dynamic Mobility & Flow {profile.goal.title()}",
        duration_minutes=target_duration,
        intensity="moderate",
        sequence=[
            PoseItem(
                name="Mountain Pose with Arm Reaches",
                sanskrit_name="Tadasana",
                duration_minutes=dur_b1,
                instructions="Stand tall, ground through all four corners of your feet, inhale sweep arms overhead.",
                evidence_chunk_ids=chunk_ids[:1],
            ),
            PoseItem(
                name="Warrior II Flow",
                sanskrit_name="Virabhadrasana II",
                duration_minutes=dur_b2,
                instructions="Step feet wide, bend front knee to 90 degrees, gaze over front fingertips with strong arms.",
                evidence_chunk_ids=chunk_ids[:2],
            ),
            PoseItem(
                name="Triangle Pose",
                sanskrit_name="Trikonasana",
                duration_minutes=dur_b2,
                instructions="Extend torso over front leg, rest hand on shin or block, rotate chest open toward ceiling.",
                evidence_chunk_ids=chunk_ids[:2],
            ),
            PoseItem(
                name="Bridge Pose",
                sanskrit_name="Setu Bandhasana",
                duration_minutes=dur_b1,
                instructions="Lie on back with knees bent, press into feet to lift pelvis, supporting spine.",
                evidence_chunk_ids=chunk_ids[:1],
            ),
            PoseItem(
                name="Savasana",
                sanskrit_name="Savasana",
                duration_minutes=max(3, target_duration - (dur_b1 * 2 + dur_b2 * 2)),
                instructions="Settle into still relaxation, absorbing the vitality of the standing flow.",
                evidence_chunk_ids=chunk_ids[:1],
            ),
        ],
        rationale="Engages full-body neuromuscular coordination, hip mobility, and cardiovascular stamina.",
        evidence_sources=chunk_ids[:3],
        expected_outcome="Enhanced muscular warmth, openness through hips and shoulders, and invigorated energy.",
    )
    candidates.append(cand_b)

    # Candidate C: Alignment & Precision
    dur_c = max(3, target_duration // 4)
    cand_c = PracticePlan(
        goal=f"Precision Alignment & Stability {profile.goal.title()}",
        duration_minutes=target_duration,
        intensity="gentle",
        sequence=[
            PoseItem(
                name="Tree Pose (Balance & Focus)",
                sanskrit_name="Vrikshasana",
                duration_minutes=dur_c,
                instructions="Ground through standing leg, place opposite foot against inner calf or thigh, hands at heart.",
                evidence_chunk_ids=chunk_ids[:2],
            ),
            PoseItem(
                name="Seated Forward Fold with Strap",
                sanskrit_name="Paschimottanasana",
                duration_minutes=dur_c,
                instructions="Hinge from the hips with a long spine, using a strap around feet if hamstrings are tight.",
                evidence_chunk_ids=chunk_ids[:2],
            ),
            PoseItem(
                name="Gentle Spinal Twist",
                sanskrit_name="Ardha Matsyendrasana",
                duration_minutes=dur_c,
                instructions="Inhale lengthen crown, exhale rotate gently through ribcage while keeping pelvis grounded.",
                evidence_chunk_ids=chunk_ids[:1],
            ),
            PoseItem(
                name="Savasana with Eye Rest",
                sanskrit_name="Savasana",
                duration_minutes=target_duration - (dur_c * 3),
                instructions="Release all muscular engagement into the earth, focusing attention on the natural breath.",
                evidence_chunk_ids=chunk_ids[:1],
            ),
        ],
        rationale="Emphasizes postural alignment, proprioception, and steady isometric stability.",
        evidence_sources=chunk_ids[:2],
        expected_outcome="Rooted balance, lengthened posterior chain, and deep neuromuscular centering.",
    )
    candidates.append(cand_c)

    # If an existing base_plan was provided by PlannerAgent LLM, add it as Candidate 0
    if base_plan is not None:
        candidates.insert(0, copy.deepcopy(base_plan))

    return candidates


def select_best_candidate(
    candidates: list[PracticePlan],
    profile: UserProfile,
    evidence: list[EvidenceChunk],
) -> tuple[PracticePlan, list[dict[str, Any]], int]:
    """
    Evaluates all candidates and returns:
    (winning_plan, list_of_utility_evaluations, winning_index)
    """
    if not candidates:
        raise ValueError("No candidates provided for evaluation")

    evaluations: list[dict[str, Any]] = []
    best_idx = 0
    best_score = -1.0

    for idx, cand in enumerate(candidates):
        report = evaluate_candidate_plan(cand, profile, evidence)
        report["candidate_index"] = idx
        evaluations.append(report)

        # Higher utility is better
        if report["composite_utility"] > best_score:
            best_score = report["composite_utility"]
            best_idx = idx

    logger.info(
        "ToT Evaluated %d candidates. Selected Candidate %d with utility score %.4f",
        len(candidates),
        best_idx,
        best_score,
    )
    return candidates[best_idx], evaluations, best_idx
