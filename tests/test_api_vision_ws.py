"""
tests/test_api_vision_ws.py — Unit & integration tests for WS /ws/vision and /api/models NVIDIA default.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.api.server import app
from app.vision.biomechanics import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    BiomechanicalSnapshot,
    Point3D,
)

client = TestClient(app)


def _make_valid_jpeg_data_url(width: int = 64, height: int = 48) -> str:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _make_sample_landmarks() -> dict[int, Point3D]:
    return {
        NOSE: Point3D(0.50, 0.15, 0.0, 0.99),
        LEFT_SHOULDER: Point3D(0.42, 0.28, 0.0, 0.98),
        RIGHT_SHOULDER: Point3D(0.58, 0.28, 0.0, 0.98),
        LEFT_ELBOW: Point3D(0.32, 0.28, 0.0, 0.96),
        RIGHT_ELBOW: Point3D(0.68, 0.28, 0.0, 0.96),
        LEFT_WRIST: Point3D(0.22, 0.28, 0.0, 0.95),
        RIGHT_WRIST: Point3D(0.78, 0.28, 0.0, 0.95),
        LEFT_HIP: Point3D(0.45, 0.52, 0.0, 0.99),
        RIGHT_HIP: Point3D(0.55, 0.52, 0.0, 0.99),
        LEFT_KNEE: Point3D(0.38, 0.72, 0.0, 0.99),
        RIGHT_KNEE: Point3D(0.64, 0.72, 0.0, 0.99),
        LEFT_ANKLE: Point3D(0.38, 0.92, 0.0, 0.97),
        RIGHT_ANKLE: Point3D(0.72, 0.92, 0.0, 0.97),
    }


def test_models_endpoint_uses_nvidia_model_env(monkeypatch):
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    resp = client.get("/api/models")
    assert resp.status_code == 200
    assert resp.json()["models"]["nvidia_nim"] == ["nvidia/nemotron-3.5-bolt-e4b"]

    monkeypatch.setenv("NVIDIA_MODEL", "nvidia/custom-vision-model")
    resp2 = client.get("/api/models")
    assert resp2.status_code == 200
    assert resp2.json()["models"]["nvidia_nim"] == ["nvidia/custom-vision-model"]


def test_ws_vision_config_ack():
    with client.websocket_connect("/ws/vision") as ws:
        ws.send_json({
            "type": "config",
            "asana": "Vrikshasana",
            "lead_leg": "right",
            "user_id": "P07",
        })
        ack = ws.receive_json()
        assert ack["type"] == "config_ack"
        assert ack["asana"] == "Vrikshasana"
        assert ack["lead_leg"] == "right"
        assert ack["user_id"] == "P07"


def test_ws_vision_frame_telemetry_and_fired_cues(monkeypatch):
    import app.api.server as server_mod
    from app.audio.tts import SpokenCue, tts_engine

    deviated_snapshot = BiomechanicalSnapshot(
        left_knee_angle=122.0,  # Deviated from Virabhadrasana II target 85-95 deg
        right_knee_angle=176.5,
        left_elbow_angle=175.0,
        right_elbow_angle=175.0,
        spine_verticality_deg=3.0,
        shoulder_level_deg=2.0,
        hip_level_deg=1.5,
        landmarks=_make_sample_landmarks(),
    )

    class _FakeDetector:
        def extract_biomechanics(self, frame):
            assert frame is not None and frame.shape[0] > 0
            return deviated_snapshot

    monkeypatch.setattr(server_mod, "_get_pose_detector", lambda: _FakeDetector())
    monkeypatch.setattr(
        tts_engine,
        "synthesize_cue",
        lambda text, **kw: SpokenCue(
            text=text,
            audio_path="data/audio_cache/ws_vision_cue.mp3",
            provider="elevenlabs",
            voice_name="Sarah",
        ),
    )

    frame_data_url = _make_valid_jpeg_data_url()

    with client.websocket_connect("/ws/vision") as ws:
        # Configure zero debounce for deterministic immediate cue firing in test
        ws.send_json({
            "type": "config",
            "asana": "Virabhadrasana II",
            "lead_leg": "left",
            "user_id": "P01",
            "debounce_seconds": 0.0,
        })
        ack = ws.receive_json()
        assert ack["type"] == "config_ack"

        ws.send_json({
            "type": "frame",
            "image": frame_data_url,
            "asana": "Virabhadrasana II",
            "lead_leg": "left",
        })
        telemetry = ws.receive_json()

        assert telemetry["type"] == "telemetry"
        assert telemetry["pose_detected"] is True
        assert telemetry["asana"] == "Virabhadrasana II"
        assert isinstance(telemetry["latency_ms"], (int, float))
        assert len(telemetry["landmarks"]) == 33
        assert telemetry["angles"]["left_knee"] == 122.0
        assert telemetry["angles"]["right_knee"] == 176.5

        lead_dev = next(d for d in telemetry["deviations"] if d["joint_name"] == "lead_knee")
        assert lead_dev["is_aligned"] is False
        assert lead_dev["severity"] == "severe"

        assert len(telemetry["fired_cues"]) >= 1
        cue = telemetry["fired_cues"][0]
        assert cue["joint_name"] == "lead_knee"
        assert cue["audio_url"] == "/api/audio/file/ws_vision_cue.mp3"
        assert cue["provider"] == "elevenlabs"


def test_ws_vision_invalid_frame_does_not_crash():
    with client.websocket_connect("/ws/vision") as ws:
        ws.send_json({
            "type": "frame",
            "image": "data:image/jpeg;base64,!!!not_valid_jpeg_bytes!!!",
            "asana": "Tadasana",
            "lead_leg": "left",
        })
        resp = ws.receive_json()
        assert resp["type"] == "telemetry"
        assert resp["pose_detected"] is False
        assert resp["landmarks"] == []
        assert resp["fired_cues"] == []

        # Connection remains open and healthy for subsequent messages
        ws.send_json({"type": "config", "asana": "Tadasana", "lead_leg": "left"})
        ack = ws.receive_json()
        assert ack["type"] == "config_ack"
        assert ack["asana"] == "Tadasana"
