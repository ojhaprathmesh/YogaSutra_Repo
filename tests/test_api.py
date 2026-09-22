"""
tests/test_api.py — Unit and integration tests for FastAPI Web Server & Speech Studio
"""
from fastapi.testclient import TestClient
from app.api.server import app

client = TestClient(app)


class TestYogaSutraAPI:
    def test_health_endpoint(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["app"] == "YogaSutra"

    def test_models_status_endpoint(self):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "providers" in data
        assert "gemini" in data["providers"]
        assert "ollama_cloud" in data["providers"]
        assert "nvidia_nim" in data["providers"]

    def test_mcp_tools_endpoint(self):
        resp = client.get("/api/mcp/tools")
        assert resp.status_code == 200
        tools = resp.json()
        tool_names = [t["name"] for t in tools]
        assert "search_yoga_knowledge" in tool_names
        assert "get_asana_metadata" in tool_names

    def test_audio_transcribe_endpoint(self):
        payload = {"audio_payload": "Gentle morning flow for 20 minutes"}
        resp = client.post("/api/audio/transcribe", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["transcript"] == "Gentle morning flow for 20 minutes"
        assert "confidence" in data

    def test_audio_synthesize_endpoint(self):
        payload = {"text": "Inhale, reach arms high, softening shoulders"}
        resp = client.post("/api/audio/synthesize", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert "provider" in data

    def test_serve_static_index(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "YogaSutra" in resp.text
        assert "Practice Studio" in resp.text
