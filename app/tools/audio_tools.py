"""
app/tools/audio_tools.py — ADK Speech & Audio Tools

Tools for:
1. Speech perception (transcription).
2. Spoken cue synthesis via ElevenLabs Turbo v2.5 with audio caching & WebSpeech fallback.
"""
from __future__ import annotations

import json

from app.audio.stt import stt_adapter
from app.audio.tts import tts_engine


def transcribe_speech_audio(audio_payload: str, language: str = "en") -> str:
    """
    Transcribes spoken user audio or transcript payload into clean text.

    Args:
        audio_payload: Spoken transcript string or base64 audio payload.
        language: Language code (default 'en').

    Returns:
        JSON string with transcript, confidence, and source.
    """
    res = stt_adapter.transcribe_payload(audio_payload, language=language)
    return json.dumps({
        "transcript": res.transcript,
        "confidence": res.confidence,
        "source": res.source,
        "language": res.language,
    })


def synthesize_speech_feedback(
    text_cue: str,
    voice_name: str = "Rachel",
    speaking_rate: float = 0.9,
) -> str:
    """
    Synthesizes a realistic spoken yoga instructor cue via ElevenLabs or WebSpeech.

    Args:
        text_cue: The verbal alignment or pacing cue to speak (e.g. "Gently lengthen your spine").
        voice_name: Desired soothing voice name (default "Rachel").
        speaking_rate: Cadence rate (default 0.9 for serene yoga pace).

    Returns:
        JSON string with cue text, audio_path (if saved to disk), and provider info.
    """
    cue = tts_engine.synthesize_cue(text_cue)
    return json.dumps({
        "text": cue.text,
        "audio_path": cue.audio_path,
        "provider": cue.provider,
        "voice_name": cue.voice_name,
        "speaking_rate": cue.speaking_rate,
    })
