"""
tests/test_audio.py — Unit tests for Speech Subsystem, ElevenLabs, and Audio Tools
"""
import json

from app.audio.cache import AudioCache
from app.audio.elevenlabs import ElevenLabsClient
from app.audio.stt import SpeechToTextAdapter
from app.audio.tts import HybridTTSEngine
from app.tools.audio_tools import synthesize_speech_feedback, transcribe_speech_audio
from app.agents.speech.agent import speech_agent


class TestAudioCache:
    def test_audio_cache_put_and_get(self, tmp_path):
        cache = AudioCache(cache_dir=tmp_path)
        sample_audio = b"ID3FAKEAUDIOBYTES12345"
        text = "Inhale and reach up"
        voice_id = "test_voice"

        # Initially empty
        assert cache.get(text, voice_id) is None

        # Store
        saved = cache.put(text, voice_id, sample_audio, ext="mp3")
        assert saved.exists()
        assert saved.read_bytes() == sample_audio

        # Retrieve
        cached = cache.get(text, voice_id)
        assert cached is not None
        assert cached == saved
        assert cache.hits == 1


class TestElevenLabsClient:
    def test_unconfigured_client(self):
        client = ElevenLabsClient(api_key="")
        assert not client.is_configured
        quota = client.check_quota()
        assert quota["status"] == "unconfigured"
        assert quota["remaining_characters"] == 0

    def test_client_configured_with_valid_key(self):
        client = ElevenLabsClient()
        if client.is_configured:
            quota = client.check_quota()
            assert "remaining_characters" in quota
            if quota.get("status") != "error":
                assert quota["character_limit"] >= 10000


class TestTTSEngine:
    def test_synthesize_cue_webspeech_fallback(self):
        engine = HybridTTSEngine()
        cue = engine.synthesize_cue("Gently roll shoulders", prefer_elevenlabs=False)
        assert cue.text == "Gently roll shoulders"
        assert cue.provider == "webspeech"
        assert cue.speaking_rate == 0.88

    def test_synthesize_empty_text(self):
        engine = HybridTTSEngine()
        cue = engine.synthesize_cue("   ")
        assert cue.text == ""


class TestSTTAdapter:
    def test_transcribe_text_payload(self):
        adapter = SpeechToTextAdapter()
        res = adapter.transcribe_payload("I want 20 min morning practice")
        assert res.transcript == "I want 20 min morning practice"
        assert res.confidence >= 0.9
        assert res.source == "webspeech"

    def test_transcribe_audio_bytes(self):
        adapter = SpeechToTextAdapter()
        res = adapter.transcribe_payload(b"\x00\x01\x02")
        assert len(res.transcript) > 0
        assert res.source == "audio_payload"


class TestAudioTools:
    def test_transcribe_tool(self):
        output_str = transcribe_speech_audio("Gentle stretching for lower back")
        data = json.loads(output_str)
        assert "transcript" in data
        assert "confidence" in data
        assert data["transcript"] == "Gentle stretching for lower back"

    def test_synthesize_tool(self):
        output_str = synthesize_speech_feedback("Relax your jaw and breathe smoothly")
        data = json.loads(output_str)
        assert "text" in data
        assert data["text"] == "Relax your jaw and breathe smoothly"
        assert "provider" in data


class TestSpeechAgent:
    def test_speech_agent_metadata(self):
        assert speech_agent.name == "SpeechAgent"
        assert len(speech_agent.tools) == 2
        tool_names = [t.__name__ for t in speech_agent.tools]
        assert "transcribe_speech_audio" in tool_names
        assert "synthesize_speech_feedback" in tool_names


class TestElevenLabsStreaming:
    def test_synthesize_stream_from_disk_cache_zero_api_calls(self, tmp_path, monkeypatch):
        import urllib.request
        import app.audio.elevenlabs as el_mod

        cache = AudioCache(cache_dir=tmp_path)
        monkeypatch.setattr(el_mod, "audio_cache", cache)

        sample_mp3 = b"ID3" + (b"A" * 10000) + (b"B" * 6500)
        text = "Breathe deeply into Warrior Two"
        voice_id = "EXAVITQu4vr4xnSDxMaL"
        cache.put(text, voice_id, sample_mp3, ext="mp3", model=el_mod.DEFAULT_MODEL_ID)

        def _fail_urlopen(*args, **kwargs):
            raise AssertionError("urlopen must not be called on cache hit")

        monkeypatch.setattr(urllib.request, "urlopen", _fail_urlopen)

        client = ElevenLabsClient(api_key="sk_test_dummy_key")
        chunks = list(client.synthesize_stream(text, voice_id=voice_id))

        assert len(chunks) == 3  # 8192 + 8192 + 119
        assert all(len(c) <= 8192 for c in chunks)
        assert b"".join(chunks) == sample_mp3
        assert cache.hits == 1

    def test_synthesize_stream_live_http_and_caches_to_disk(self, tmp_path, monkeypatch):
        import urllib.request
        import app.audio.elevenlabs as el_mod

        cache = AudioCache(cache_dir=tmp_path)
        monkeypatch.setattr(el_mod, "audio_cache", cache)

        stream_chunks = [b"ID3_CHUNK_1_", b"CHUNK_2_", b"CHUNK_3_END"]
        captured_urls: list[str] = []

        class _FakeStreamResponse:
            def __init__(self, chunks):
                self._iter = iter(chunks)

            def read(self, size=-1):
                return next(self._iter, b"")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        def _fake_urlopen(req, timeout=20, context=None):
            captured_urls.append(req.full_url)
            return _FakeStreamResponse(stream_chunks)

        monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

        client = ElevenLabsClient(api_key="sk_test_dummy_key")
        monkeypatch.setattr(
            client,
            "check_quota",
            lambda: {"status": "free", "remaining_characters": 5000},
        )

        text = "Soften your shoulders and lengthen the spine"
        voice_id = "EXAVITQu4vr4xnSDxMaL"
        yielded = list(client.synthesize_stream(text, voice_id=voice_id))

        assert yielded == stream_chunks
        assert len(captured_urls) == 1
        assert captured_urls[0].endswith(f"/text-to-speech/{voice_id}/stream")

        cached_file = cache.get(text, voice_id, model=el_mod.DEFAULT_MODEL_ID)
        assert cached_file is not None
        assert cached_file.read_bytes() == b"".join(stream_chunks)

    def test_synthesize_stream_unconfigured_yields_nothing(self, tmp_path, monkeypatch):
        import app.audio.elevenlabs as el_mod

        monkeypatch.setattr(el_mod, "audio_cache", AudioCache(cache_dir=tmp_path))
        client = ElevenLabsClient(api_key="")
        assert list(client.synthesize_stream("Relax your jaw")) == []


class TestAudioStreamAndPracticeAudioAPI:
    def test_post_audio_stream_returns_mpeg_bytes(self, monkeypatch):
        from fastapi.testclient import TestClient
        import app.api.server as server_mod

        monkeypatch.setattr(server_mod.elevenlabs_client, "api_key", "sk_test_stream_key")
        monkeypatch.setattr(
            server_mod.elevenlabs_client,
            "synthesize_stream",
            lambda text, voice_id="EXAVITQu4vr4xnSDxMaL", **kw: iter([b"ID3_PART1", b"_PART2"]),
        )

        client = TestClient(server_mod.app)
        resp = client.post(
            "/api/audio/stream",
            json={"text": "Exhale slowly into Child's Pose", "prefer_elevenlabs": True},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("audio/mpeg")
        assert resp.content == b"ID3_PART1_PART2"

    def test_post_audio_stream_fallback_when_unavailable(self, monkeypatch):
        from fastapi.testclient import TestClient
        import app.api.server as server_mod

        client = TestClient(server_mod.app)
        resp = client.post(
            "/api/audio/stream",
            json={"text": "Exhale slowly", "prefer_elevenlabs": False},
        )
        assert resp.status_code == 503
        data = resp.json()
        assert data["fallback"]["provider"] == "webspeech"
        assert data["fallback"]["text"] == "Exhale slowly"

    def test_practice_generate_includes_audio_cue(self, monkeypatch):
        from types import SimpleNamespace
        from fastapi.testclient import TestClient
        import app.api.server as server_mod
        from app.audio.tts import SpokenCue

        class _FakeTrace:
            def to_dict(self):
                return {"run_id": "run_test_audio", "steps": []}

        fake_result = SimpleNamespace(
            run_id="run_test_audio",
            session_id="sess_test_audio",
            final_response="Welcome to your gentle 15-minute evening flow. Start in Sukhasana and breathe deeply.",
            phase=2,
            session_state={
                "candidate_plan": {
                    "goal": "evening relaxation",
                    "duration_minutes": 15,
                    "sequence": [{"name": "Sukhasana"}, {"name": "Balasana"}],
                }
            },
            trace=_FakeTrace(),
        )

        captured_cue_texts: list[str] = []

        def _fake_synthesize_cue(text: str, **kw):
            captured_cue_texts.append(text)
            return SpokenCue(
                text=text,
                audio_path="data/audio_cache/plan_intro_test.mp3",
                provider="elevenlabs",
                voice_name="Sarah",
            )

        monkeypatch.setattr(server_mod, "run_yogasutra", lambda **kw: fake_result)
        monkeypatch.setattr(server_mod.tts_engine, "synthesize_cue", _fake_synthesize_cue)

        client = TestClient(server_mod.app)
        resp = client.post(
            "/api/practice/generate",
            json={
                "user_input": "I want a gentle 15 minute evening yoga routine",
                "persona_id": "P00",
                "phase": 2,
                "include_audio": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["audio_cue"] is not None
        assert data["audio_cue"]["provider"] == "elevenlabs"
        assert data["audio_cue"]["voice_name"] == "Sarah"
        assert data["audio_cue"]["audio_url"] == "/api/audio/file/plan_intro_test.mp3"
        assert len(data["audio_cue"]["text"]) <= 220
        assert "Sukhasana" in data["audio_cue"]["text"]
        assert len(captured_cue_texts) == 1

        # Verify include_audio=False skips synthesis
        resp_no_audio = client.post(
            "/api/practice/generate",
            json={
                "user_input": "I want a gentle 15 minute evening yoga routine",
                "persona_id": "P00",
                "phase": 2,
                "include_audio": False,
            },
        )
        assert resp_no_audio.status_code == 200
        assert resp_no_audio.json()["audio_cue"] is None
        assert len(captured_cue_texts) == 1
