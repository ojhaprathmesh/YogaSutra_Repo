"""
tests/test_longitudinal_memory.py — Unit Tests for Longitudinal Memory Subsystem (Phase 3A)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import app.memory.longitudinal as longitudinal_mod
from app.agents.profile.agent import PROFILE_INSTRUCTION, profile_agent
from app.memory.longitudinal import (
    LongitudinalMemoryStore,
    build_longitudinal_context,
    record_pipeline_completion,
)
from app.tools.profile import get_user_history
from app.workflows.event_driven import (
    FeedbackReceivedEvent,
    SafetyEscalationEvent,
    SessionCompletedEvent,
    get_event_bus,
)

TEST_MEM_DIR = Path("data/memory/test_longitudinal")


def setup_function():
    if TEST_MEM_DIR.exists():
        shutil.rmtree(TEST_MEM_DIR)
    TEST_MEM_DIR.mkdir(parents=True, exist_ok=True)


def teardown_function():
    if TEST_MEM_DIR.exists():
        shutil.rmtree(TEST_MEM_DIR)


def test_record_session_progression():
    store = LongitudinalMemoryStore(storage_dir=TEST_MEM_DIR)
    user_id = "test_user_01"

    # Session 1
    t1 = store.record_session(
        user_id=user_id,
        session_id="s1",
        duration_minutes=20,
        plan_goal="Relaxation",
        completed_poses=["Sukhasana", "Savasana"],
        hold_durations={"Sukhasana": 300, "Savasana": 900},
    )

    assert t1.total_sessions_completed == 1
    assert t1.total_minutes_practiced == 20
    assert t1.current_streak_days == 1
    assert t1.pose_frequency["Sukhasana"] == 1
    assert t1.pose_hold_durations_sec["Sukhasana"] == [300]


def test_record_feedback_and_adaptive_level():
    store = LongitudinalMemoryStore(storage_dir=TEST_MEM_DIR)
    user_id = "test_user_02"

    # Record 5 easy sessions (difficulty 1 out of 5)
    for i in range(1, 6):
        store.record_session(
            user_id=user_id,
            session_id=f"sess_{i}",
            duration_minutes=20,
            plan_goal="Strength",
            completed_poses=["Tadasana", "Warrior II"],
        )
        t = store.record_feedback(
            user_id=user_id,
            session_id=f"sess_{i}",
            difficulty_rating=1,
            fatigue_level="energized",
            comments="Very easy for me.",
        )

    assert t.average_difficulty_rating == 1.0
    assert t.suggested_level_override == "intermediate"

    adjustments = store.get_adaptive_profile_adjustments(user_id)
    assert adjustments["suggested_level"] == "intermediate"
    assert adjustments["total_sessions"] == 5


def test_safety_alert_watchlist():
    store = LongitudinalMemoryStore(storage_dir=TEST_MEM_DIR)
    user_id = "test_user_03"

    t = store.record_safety_alert(user_id, "Rotator cuff tendonitis")
    assert "Rotator cuff tendonitis" in t.active_contraindications
    assert "Rotator cuff tendonitis" in t.injury_history


def test_event_bus_reactive_integration():
    bus = get_event_bus()
    user_id = "test_reactive_user"

    # Publish SessionCompletedEvent
    bus.publish_sync(
        SessionCompletedEvent(
            user_id=user_id,
            session_id="evt_s1",
            duration_minutes=30,
            completed_poses=["Tadasana", "Vrikshasana", "Savasana"],
            plan_goal="Balance & Focus",
        )
    )

    # Publish FeedbackReceivedEvent
    bus.publish_sync(
        FeedbackReceivedEvent(
            user_id=user_id,
            session_id="evt_s1",
            difficulty_rating=2,
            fatigue_level="normal",
            comments="Great focus session",
        )
    )

    # Publish SafetyEscalationEvent
    bus.publish_sync(
        SafetyEscalationEvent(
            user_id=user_id,
            severity="medium",
            trigger_condition="Wrist strain during cat-cow",
        )
    )

    from app.memory.longitudinal import get_longitudinal_store
    live_store = get_longitudinal_store()
    traj = live_store.get_trajectory(user_id)

    assert traj.total_sessions_completed >= 1
    assert "Wrist strain during cat-cow" in traj.active_contraindications


def test_build_longitudinal_context_summary():
    store = LongitudinalMemoryStore(storage_dir=TEST_MEM_DIR)
    user_id = "test_user_ctx"

    goals = ["relaxation", "flexibility", "strength", "balance"]
    pose_batches = [
        ["Tadasana", "Savasana", "Balasana"],
        ["Tadasana", "Vrikshasana", "Savasana"],
        ["Tadasana", "Warrior II", "Cat-Cow", "Cobra"],
        ["Tadasana", "Savasana", "Bridge", "Supine Twist"],
    ]
    for idx, (goal, poses) in enumerate(zip(goals, pose_batches), start=1):
        store.record_session(
            user_id=user_id,
            session_id=f"ctx_s{idx}",
            duration_minutes=20,
            plan_goal=goal,
            completed_poses=poses,
        )

    store.record_safety_alert(user_id, "lower back pain")
    store.record_safety_alert(user_id, "knee meniscus tear")

    ctx = build_longitudinal_context(user_id, store=store)

    assert ctx["user_id"] == user_id
    assert ctx["total_sessions_completed"] == 4
    assert ctx["current_streak_days"] == 1
    assert ctx["active_contraindications"] == ["lower back pain", "knee meniscus tear"]
    assert ctx["injury_history"] == ["lower back pain", "knee meniscus tear"]
    assert ctx["suggested_level_override"] is None
    assert ctx["recent_goals"] == ["flexibility", "strength", "balance"]
    assert len(ctx["frequent_poses"]) == 5
    assert ctx["frequent_poses"][0] == "Tadasana"
    assert ctx["frequent_poses"][1] == "Savasana"


def test_record_pipeline_completion_dict_and_json_string():
    store = LongitudinalMemoryStore(storage_dir=TEST_MEM_DIR)
    user_id = "test_user_pipe"

    # 1) Dict inputs
    dict_state = {
        "user_profile": {
            "experience_level": "beginner",
            "goal": "relaxation",
            "constraints": ["lower back pain", "wrist sensitivity"],
        },
        "candidate_plan": {
            "goal": "relaxation",
            "duration_minutes": 25,
            "sequence": [
                {"name": "Balasana", "duration_minutes": 5},
                {"name": "Savasana", "duration_minutes": 10},
            ],
        },
    }
    t1 = record_pipeline_completion(user_id, "pipe_s1", dict_state, store=store)
    assert t1.total_sessions_completed == 1
    assert t1.total_minutes_practiced == 25
    assert "Balasana" in t1.pose_frequency
    assert "Savasana" in t1.pose_frequency
    assert t1.active_contraindications == ["lower back pain", "wrist sensitivity"]

    # 2) Markdown-fenced JSON string inputs + constraint deduplication
    profile_json = json.dumps(
        {
            "experience_level": "beginner",
            "goal": "flexibility",
            "constraints": ["Lower Back Pain", "tight hamstrings"],
        }
    )
    plan_json = json.dumps(
        {
            "goal": "flexibility",
            "duration_minutes": 15,
            "sequence": [{"name": "Cat-Cow", "duration_minutes": 5}],
        }
    )
    str_state = {
        "user_profile": f"```json\n{profile_json}\n```",
        "candidate_plan": f"```json\n{plan_json}\n```",
    }
    t2 = record_pipeline_completion(user_id, "pipe_s2", str_state, store=store)
    assert t2.total_sessions_completed == 2
    assert t2.total_minutes_practiced == 40
    assert t2.pose_frequency["Cat-Cow"] == 1
    assert t2.active_contraindications == [
        "lower back pain",
        "wrist sensitivity",
        "tight hamstrings",
    ]

    # Verify persisted to disk
    reloaded = store.get_trajectory(user_id)
    assert reloaded.total_sessions_completed == 2
    assert "tight hamstrings" in reloaded.active_contraindications


def test_record_pipeline_completion_no_plan_still_saves_constraints():
    store = LongitudinalMemoryStore(storage_dir=TEST_MEM_DIR)
    user_id = "test_user_no_plan"

    state = {
        "user_profile": {"constraints": ["cervical spondylosis"]},
        "candidate_plan": {},
    }
    traj = record_pipeline_completion(user_id, "pipe_empty", state, store=store)
    assert traj.total_sessions_completed == 0
    assert traj.active_contraindications == ["cervical spondylosis"]

    reloaded = store.get_trajectory(user_id)
    assert reloaded.active_contraindications == ["cervical spondylosis"]


def test_get_user_history_tool_and_profile_agent_wiring(monkeypatch):
    store = LongitudinalMemoryStore(storage_dir=TEST_MEM_DIR)
    store.record_safety_alert("persona_p01", "herniated disc")
    store.record_session(
        user_id="persona_p01",
        session_id="hist_s1",
        duration_minutes=20,
        plan_goal="relaxation",
        completed_poses=["Savasana"],
    )

    monkeypatch.setattr(longitudinal_mod, "_GLOBAL_LONGITUDINAL_STORE", store)

    history = get_user_history("persona_p01")
    assert history["user_id"] == "persona_p01"
    assert history["total_sessions_completed"] == 1
    assert "herniated disc" in history["active_contraindications"]
    assert "herniated disc" in history["injury_history"]
    assert history["recent_goals"] == ["relaxation"]
    assert history["frequent_poses"] == ["Savasana"]

    # Verify ProfileAgent wiring
    assert get_user_history in (profile_agent.tools or [])
    assert "longitudinal_history" in PROFILE_INSTRUCTION
    assert "get_user_history" in PROFILE_INSTRUCTION
    assert "active_contraindications" in PROFILE_INSTRUCTION
