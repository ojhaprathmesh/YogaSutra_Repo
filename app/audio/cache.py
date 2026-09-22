"""
app/audio/cache.py — Persistent Disk Cache for Spoken Audio Cues

Stores synthesized speech files (MP3/WAV) in `data/audio_cache/` keyed by
SHA-256 hash of (text + voice_id + model). Prevents burning ElevenLabs
monthly character quota on repetitive cues (greetings, standard transitions,
relaxation instructions).
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger("yogasutra.audio_cache")

_AUDIO_CACHE_DIR = Path("data/audio_cache")


class AudioCache:
    """Disk-backed cache for audio synthesis artifacts."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or _AUDIO_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits: int = 0
        self.misses: int = 0

    def _hash_key(self, text: str, voice_id: str, model: str = "") -> str:
        s = f"{text.strip().lower()}|{voice_id}|{model}"
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def get(self, text: str, voice_id: str, model: str = "") -> Path | None:
        """Return Path to cached audio file if present, else None."""
        key = self._hash_key(text, voice_id, model)
        mp3_path = self.cache_dir / f"{key}.mp3"
        wav_path = self.cache_dir / f"{key}.wav"

        for p in [mp3_path, wav_path]:
            if p.exists() and p.stat().st_size > 0:
                self.hits += 1
                logger.info(f"[AudioCache] HIT for '{text[:30]}...' -> {p.name}")
                return p

        self.misses += 1
        return None

    def put(self, text: str, voice_id: str, audio_bytes: bytes, ext: str = "mp3", model: str = "") -> Path:
        """Store audio bytes to disk cache and return file path."""
        key = self._hash_key(text, voice_id, model)
        file_path = self.cache_dir / f"{key}.{ext.lstrip('.')}"
        file_path.write_bytes(audio_bytes)
        logger.info(f"[AudioCache] STORED {len(audio_bytes)} bytes for '{text[:30]}...' -> {file_path.name}")
        return file_path


# Global singleton audio cache
audio_cache = AudioCache()
