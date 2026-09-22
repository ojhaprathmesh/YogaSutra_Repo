"""
tests/test_longitudinal_memory.py — Unit Tests for Longitudinal Memory Subsystem (Phase 3A)
"""
from __future__ import annotations

import shutil
from pathlib import Path

from app.memory.longitudinal import LongitudinalMemoryStore
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
