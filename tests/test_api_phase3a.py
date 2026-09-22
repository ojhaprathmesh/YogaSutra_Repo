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
            "user_id": "P01",
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
    res = client.get("/api/memory/trajectory/P01")
    assert res.status_code == 200
    data = res.json()
    assert "trajectory" in data
    assert "adaptive_adjustments" in data
    assert data["trajectory"]["user_id"] == "P01"


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
