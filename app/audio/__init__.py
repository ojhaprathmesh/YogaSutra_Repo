"""
app/audio — Studio-Grade Speech Subsystem with ElevenLabs & WebSpeech
"""
from app.audio.cache import AudioCache, audio_cache
from app.audio.elevenlabs import ElevenLabsClient, elevenlabs_client
from app.audio.stt import SpeechToTextAdapter, TranscriptResult, stt_adapter
from app.audio.tts import HybridTTSEngine, SpokenCue, tts_engine

__all__ = [
    "audio_cache",
    "AudioCache",
    "elevenlabs_client",
    "ElevenLabsClient",
    "stt_adapter",
    "SpeechToTextAdapter",
    "TranscriptResult",
    "tts_engine",
    "HybridTTSEngine",
    "SpokenCue",
]
