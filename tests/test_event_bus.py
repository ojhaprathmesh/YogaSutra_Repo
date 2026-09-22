"""
tests/test_event_bus.py — Unit Tests for Asynchronous Pub/Sub Event Bus (Phase 3A)
"""
from __future__ import annotations

import pytest

from app.state import YogaSessionState
from app.workflows.event_driven import (
    BaseEvent,
    EventBus,
    FeedbackReceivedEvent,
    PostureDeviationEvent,
    SafetyEscalationEvent,
    SessionCompletedEvent,
    get_event_bus,
)


def test_event_bus_subscribe_and_publish_sync():
    bus = EventBus()
    received = []

    def handler(event: BaseEvent):
        received.append(event)

    bus.subscribe("SessionCompleted", handler)

    event = SessionCompletedEvent(
        session_id="sess_123",
        user_id="P01",
        duration_minutes=25,
        completed_poses=["Tadasana", "Savasana"],
    )
    bus.publish_sync(event)

    assert len(received) == 1
    assert received[0].session_id == "sess_123"
    assert received[0].duration_minutes == 25


@pytest.mark.asyncio
async def test_event_bus_publish_async():
    bus = EventBus()
    async_received = []

    async def async_handler(event: BaseEvent):
        async_received.append(event)
        return "ok"

    bus.subscribe("PostureDeviation", async_handler)

    dev_event = PostureDeviationEvent(
        session_id="sess_pose_1",
        user_id="P01",
        pose_name="Virabhadrasana II",
        joint_name="left_knee",
        measured_angle=70.0,
        deviation_deg=20.0,
        correction_cue="Bend deeper to reach 90 degrees.",
    )
    results = await bus.publish(dev_event)

    assert len(async_received) == 1
    assert results == ["ok"]
    assert async_received[0].deviation_deg == 20.0


def test_event_bus_wildcard_subscription():
    bus = EventBus()
    all_events = []

    bus.subscribe("*", lambda e: all_events.append(e.event_type))

    bus.publish_sync(FeedbackReceivedEvent(difficulty_rating=4))
    bus.publish_sync(SafetyEscalationEvent(severity="critical"))

    assert all_events == ["FeedbackReceived", "SafetyEscalation"]


def test_event_bus_journal_and_history():
    bus = EventBus(max_journal_size=3)

    bus.publish_sync(BaseEvent(event_type="Evt1"))
    bus.publish_sync(BaseEvent(event_type="Evt2"))
    bus.publish_sync(BaseEvent(event_type="Evt3"))
    bus.publish_sync(BaseEvent(event_type="Evt4"))

    history = bus.get_history()
    assert len(history) == 3
    assert [e.event_type for e in history] == ["Evt2", "Evt3", "Evt4"]


def test_event_bus_record_to_session_state():
    bus = EventBus()
    state = YogaSessionState()

    event = FeedbackReceivedEvent(
        user_id="P02",
        difficulty_rating=5,
        fatigue_level="exhausted",
        comments="Too fast",
    )
    bus.publish_sync(event, state=state)

    assert len(state.event_journal) == 1
    assert state.event_journal[0]["event_type"] == "FeedbackReceived"
    assert state.event_journal[0]["user_id"] == "P02"


def test_global_event_bus_default_handlers():
    bus = get_event_bus()

    # Safety escalation handler test
    safe_evt = SafetyEscalationEvent(
        user_id="P03",
        severity="critical",
        trigger_condition="Acute sharp pain in shoulder",
        action_required="halt_practice",
    )
    results = bus.publish_sync(safe_evt)
    assert any(r.get("circuit_breaker") is True for r in results if isinstance(r, dict))

    # Posture deviation handler test
    posture_evt = PostureDeviationEvent(
        pose_name="Trikonasana",
        joint_name="spine",
        deviation_deg=15.0,
        correction_cue="Lengthen through crown, do not collapse lower ribs.",
    )
    posture_results = bus.publish_sync(posture_evt)
    assert any("Lengthen" in r.get("cue", "") for r in posture_results if isinstance(r, dict))
