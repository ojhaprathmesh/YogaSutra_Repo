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
        user_id="test_user_p02",
        difficulty_rating=5,
        fatigue_level="exhausted",
        comments="Too fast",
    )
    bus.publish_sync(event, state=state)

    assert len(state.event_journal) == 1
    assert state.event_journal[0]["event_type"] == "FeedbackReceived"
    assert state.event_journal[0]["user_id"] == "test_user_p02"


def test_global_event_bus_default_handlers():
    bus = get_event_bus()

    # Safety escalation handler test
    safe_evt = SafetyEscalationEvent(
        user_id="test_user_p03",
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


def test_posture_deviation_handler_synthesizes_tts(monkeypatch):
    from app.audio.tts import SpokenCue, tts_engine

    called_with: list[str] = []

    def fake_synthesize(text: str, voice_id: str | None = None, prefer_elevenlabs: bool = True) -> SpokenCue:
        called_with.append(text)
        return SpokenCue(
            text=text,
            audio_path="data/audio_cache/knee_cue_123.mp3",
            provider="elevenlabs",
            voice_name="Sarah",
        )

    monkeypatch.setattr(tts_engine, "synthesize_cue", fake_synthesize)

    bus = get_event_bus()
    evt = PostureDeviationEvent(
        pose_name="Virabhadrasana II",
        joint_name="right_knee",
        measured_angle=112.4,
        deviation_deg=22.4,
        severity="moderate",
        correction_cue="Stack your right knee over your ankle.",
    )
    results = bus.publish_sync(evt)
    handler_res = next(r for r in results if isinstance(r, dict) and r.get("handled"))

    assert called_with == ["Stack your right knee over your ankle."]
    assert handler_res["audio_url"] == "/api/audio/file/knee_cue_123.mp3"
    assert handler_res["provider"] == "elevenlabs"
    assert handler_res["voice_name"] == "Sarah"
    assert handler_res["cue"] == "Stack your right knee over your ankle."


def test_posture_deviation_handler_fallback_on_tts_exception(monkeypatch):
    from app.audio.tts import tts_engine

    def broken_synthesize(*args, **kwargs):
        raise RuntimeError("ElevenLabs offline")

    monkeypatch.setattr(tts_engine, "synthesize_cue", broken_synthesize)

    bus = get_event_bus()
    evt = PostureDeviationEvent(
        pose_name="Trikonasana",
        joint_name="left_hip",
        deviation_deg=12.0,
        correction_cue="Open your top hip slightly.",
    )
    results = bus.publish_sync(evt)
    handler_res = next(r for r in results if isinstance(r, dict) and r.get("handled"))

    assert handler_res["audio_url"] is None
    assert handler_res["provider"] == "webspeech"
    assert handler_res["cue"] == "Open your top hip slightly."


@pytest.mark.asyncio
async def test_stream_listener_receives_posture_and_safety_events(monkeypatch):
    import asyncio
    import threading
    from app.audio.tts import SpokenCue, tts_engine

    monkeypatch.setattr(
        tts_engine,
        "synthesize_cue",
        lambda text, **kw: SpokenCue(
            text=text,
            audio_path="data/audio_cache/stream_test.mp3",
            provider="elevenlabs",
            voice_name="Sarah",
        ),
    )

    bus = get_event_bus()
    queue = bus.add_stream_listener()
    try:
        # Publish from a background thread to verify call_soon_threadsafe delivery
        evt = PostureDeviationEvent(
            pose_name="Virabhadrasana II",
            joint_name="left_knee",
            measured_angle=72.0,
            deviation_deg=18.0,
            severity="moderate",
            correction_cue="Align left knee directly over ankle.",
        )
        t = threading.Thread(target=bus.publish_sync, args=(evt,))
        t.start()
        t.join(timeout=2.0)

        payload = await asyncio.wait_for(queue.get(), timeout=2.0)
        assert payload["event_type"] == "PostureDeviation"
        assert payload["pose_name"] == "Virabhadrasana II"
        assert payload["joint_name"] == "left_knee"
        assert payload["deviation_deg"] == 18.0
        assert payload["severity"] == "moderate"
        assert payload["correction_cue"] == "Align left knee directly over ankle."
        assert payload["audio_url"] == "/api/audio/file/stream_test.mp3"
        assert payload["provider"] == "elevenlabs"
        assert "timestamp" in payload

        # Also verify SafetyEscalation broadcast via async publish
        safety_evt = SafetyEscalationEvent(
            severity="critical",
            trigger_condition="Sharp knee pain",
            action_required="halt_practice",
        )
        await bus.publish(safety_evt)
        safety_payload = await asyncio.wait_for(queue.get(), timeout=2.0)
        assert safety_payload["event_type"] == "SafetyEscalation"
        assert safety_payload["severity"] == "critical"
        assert safety_payload["trigger_condition"] == "Sharp knee pain"
    finally:
        bus.remove_stream_listener(queue)

    # Ensure unsubscribed queue receives nothing further
    await bus.publish(PostureDeviationEvent(correction_cue="After unsubscribe"))
    assert queue.empty()
