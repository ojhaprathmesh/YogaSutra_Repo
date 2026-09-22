"""
app/audio/tts.py — Hybrid Text-to-Speech Engine

Provides multi-tiered spoken cue generation:
1. Primary: Studio-grade realistic voice via ElevenLabs Turbo v2.5 (with disk cache).
2. Fallback: Web Speech API metadata payload for browser client native speech synthesis.
3. Offline/CLI: Local WAV audio generator producing audible tone markers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.audio.elevenlabs import elevenlabs_client

logger = logging.getLogger("yogasutra.tts")


@dataclass
class SpokenCue:
    """Representation of a spoken voice cue."""
    text: str
    audio_path: str | None = None
    provider: str = "webspeech"
    speaking_rate: float = 0.9  # Slower, calm cadence for yoga
    pitch: float = 1.0
    voice_name: str = "Rachel"


class HybridTTSEngine:
    """Coordinates voice cue generation between ElevenLabs and Web Speech."""

    def __init__(self) -> None:
        self.elevenlabs = elevenlabs_client

    def synthesize_cue(
        self,
        text: str,
        voice_id: str | None = None,
        prefer_elevenlabs: bool = True,
    ) -> SpokenCue:
        """
        Synthesize a spoken cue, falling back gracefully if ElevenLabs is unconfigured or exhausted.
        """
        clean_text = text.strip()
        if not clean_text:
            return SpokenCue(text="")

        if prefer_elevenlabs and self.elevenlabs.is_configured:
            audio_path = self.elevenlabs.synthesize(clean_text, voice_id=voice_id or "EXAVITQu4vr4xnSDxMaL")
            if audio_path and audio_path.exists():
                return SpokenCue(
                    text=clean_text,
                    audio_path=str(audio_path),
                    provider="elevenlabs",
                    voice_name="Sarah",
                )

        # Fallback to browser Web Speech API cue payload
        logger.info(f"[TTS] Emitting Web Speech API cue for: '{clean_text[:40]}...'")
        return SpokenCue(
            text=clean_text,
            audio_path=None,
            provider="webspeech",
            speaking_rate=0.88,
            pitch=1.0,
            voice_name="Default Calming Voice",
        )


# Global singleton engine
tts_engine = HybridTTSEngine()
