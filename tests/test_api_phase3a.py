"""
tests/test_api_phase3a.py — Integration tests for Phase 3A API endpoints
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.server import app

client = TestClient(app)


def test_api_events_publish_and_history():
    # Publish an event
    pub_res = client.post(
        "/api/events/publish",
        json={
            "event_type": "FeedbackReceived",
            "user_id": "test_api_user",
            "session_id": "test_api_sess",
            "payload": {
                "difficulty_rating": 4,
                "fatigue_level": "normal",
                "comments": "Challenging but good",
            },
        },
    )
    assert pub_res.status_code == 200
    data = pub_res.json()
    assert data["status"] == "published"
    assert data["event_type"] == "FeedbackReceived"

    # Query history
    hist_res = client.get("/api/events/history?event_type=FeedbackReceived")
    assert hist_res.status_code == 200
    events = hist_res.json()
    assert len(events) >= 1
    assert any(e.get("session_id") == "test_api_sess" for e in events)


def test_api_memory_trajectory():
    res = client.get("/api/memory/trajectory/test_api_user")
    assert res.status_code == 200
    data = res.json()
    assert "trajectory" in data
    assert "adaptive_adjustments" in data
    assert data["trajectory"]["user_id"] == "test_api_user"


def test_api_planning_candidates():
    res = client.post(
        "/api/planning/candidates",
        json={
            "goal": "flexibility",
            "duration_minutes": 25,
            "intensity": "gentle",
            "constraints": ["mild knee stiffness"],
        },
    )
    assert res.status_code == 200
    candidates = res.json()
    assert len(candidates) == 3
    assert all("sequence" in c for c in candidates)


def test_api_planning_evaluate():
    test_sequence = [
        {"name": "Sukhasana", "duration_minutes": 5},
        {"name": "Balasana", "duration_minutes": 5},
        {"name": "Savasana", "duration_minutes": 10},
    ]
    res = client.post(
        "/api/planning/evaluate",
        json={
            "sequence": test_sequence,
            "available_duration_minutes": 20,
            "constraints": [],
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "composite_utility" in data
    assert data["actual_duration_minutes"] == 20
    assert data["is_safe"] is True


def test_api_publish_posture_deviation_returns_tts_payload(monkeypatch):
    from app.audio.tts import SpokenCue, tts_engine

    monkeypatch.setattr(
        tts_engine,
        "synthesize_cue",
        lambda text, **kw: SpokenCue(
            text=text,
            audio_path="data/audio_cache/api_posture_cue.mp3",
            provider="elevenlabs",
            voice_name="Sarah",
        ),
    )

    resp = client.post(
        "/api/events/publish",
        json={
            "event_type": "PostureDeviation",
            "user_id": "StudioUser",
            "session_id": "sess_cv_1",
            "payload": {
                "pose_name": "Virabhadrasana II",
                "joint_name": "right_knee",
                "measured_angle": 112.4,
                "deviation_deg": 22.4,
                "severity": "moderate",
                "correction_cue": "Stack right knee directly over ankle.",
            },
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "published"
    assert data["event_type"] == "PostureDeviation"
    assert data["cue"] == "Stack right knee directly over ankle."
    assert data["audio_url"] == "/api/audio/file/api_posture_cue.mp3"
    assert data["provider"] == "elevenlabs"
    assert data["voice_name"] == "Sarah"


def test_api_events_stream_yields_sse_posture_event(monkeypatch):
    import asyncio
    import json
    from app.api.server import stream_events
    from app.audio.tts import SpokenCue, tts_engine
    from app.workflows.event_driven import PostureDeviationEvent, get_event_bus

    monkeypatch.setattr(
        tts_engine,
        "synthesize_cue",
        lambda text, **kw: SpokenCue(
            text=text,
            audio_path="data/audio_cache/sse_cue.mp3",
            provider="elevenlabs",
            voice_name="Sarah",
        ),
    )

    class _FakeRequest:
        def __init__(self) -> None:
            self.checks = 0

        async def is_disconnected(self) -> bool:
            self.checks += 1
            return self.checks > 1

    async def _run_sse_check() -> list[str]:
        req = _FakeRequest()
        response = await stream_events(req)  # type: ignore[arg-type]
        assert response.media_type == "text/event-stream"

        gen = response.body_iterator
        first_chunk = await gen.__anext__()
        assert "event: connected" in first_chunk

        bus = get_event_bus()
        bus.publish_sync(
            PostureDeviationEvent(
                pose_name="Virabhadrasana II",
                joint_name="right_knee",
                measured_angle=110.0,
                deviation_deg=20.0,
                severity="moderate",
                correction_cue="Soften shoulders and align front knee.",
            )
        )

        event_chunk = await gen.__anext__()
        await gen.aclose()
        return [first_chunk, event_chunk]

    chunks = asyncio.run(_run_sse_check())
    assert chunks[1].startswith("data: ")
    payload = json.loads(chunks[1].removeprefix("data: ").strip())
    assert payload["event_type"] == "PostureDeviation"
    assert payload["correction_cue"] == "Soften shoulders and align front knee."
    assert payload["audio_url"] == "/api/audio/file/sse_cue.mp3"
    assert payload["provider"] == "elevenlabs"
