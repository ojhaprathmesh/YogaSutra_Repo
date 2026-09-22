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
