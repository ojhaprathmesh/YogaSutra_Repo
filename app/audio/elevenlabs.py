"""
app/audio/elevenlabs.py — Ultra-Realistic Studio Voice Synthesis Client

Integrates ElevenLabs Turbo v2.5 with:
1. Automated remaining character quota verification (/v1/user/subscription).
2. Disk caching integration (0-character consumption on repeat cues).
3. Yoga-tailored voice styling (calm cadence, stability=0.75).
4. Graceful degradation when quota is depleted or offline.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

from app.audio.cache import audio_cache

logger = logging.getLogger("yogasutra.elevenlabs")

# Default voice: "Sarah" (EXAVITQu4vr4xnSDxMaL) — reassuring, mature, confident yoga guide (Free Tier Supported)
DEFAULT_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
import ssl

DEFAULT_MODEL_ID = "eleven_turbo_v2_5"


def _make_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class ElevenLabsClient:
    """Client for ElevenLabs Text-to-Speech API with quota protection and caching."""

    def __init__(self, api_key: str | None = None) -> None:
        raw_key = os.environ.get("ELEVENLABS_API_KEY", "") if api_key is None else api_key
        self.api_key = raw_key.strip()
        self.base_url = "https://api.elevenlabs.io/v1"
        self._ssl_ctx = _make_ssl_context()

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and not self.api_key.startswith("your_"))

    def check_quota(self) -> dict[str, int | str]:
        """Check remaining character count and subscription status."""
        if not self.is_configured:
            return {"status": "unconfigured", "remaining_characters": 0}

        url = f"{self.base_url}/user/subscription"
        req = urllib.request.Request(url, headers={"xi-api-key": self.api_key})
        try:
            with urllib.request.urlopen(req, timeout=10, context=self._ssl_ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            used = data.get("character_count", 0)
            limit = data.get("character_limit", 10000)
            remaining = max(0, limit - used)
            return {
                "status": data.get("status", "free"),
                "character_count": used,
                "character_limit": limit,
                "remaining_characters": remaining,
            }
        except Exception as e:
            logger.warning(f"[ElevenLabs] Failed to fetch subscription quota: {e}")
            return {"status": "error", "remaining_characters": 0, "error": str(e)}

    def synthesize(
        self,
        text: str,
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = DEFAULT_MODEL_ID,
    ) -> Path | None:
        """
        Synthesize text cue into an MP3 file.
        Checks disk cache first. Returns file Path or None if unconfigured/exhausted.
        """
        if not text.strip():
            return None

        # 1. Check audio cache (0-character consumption)
        cached_path = audio_cache.get(text, voice_id, model=model_id)
        if cached_path:
            return cached_path

        if not self.is_configured:
            logger.info("[ElevenLabs] API key not configured. Skipping synthesis.")
            return None

        # 2. Check quota
        quota = self.check_quota()
        remaining = int(quota.get("remaining_characters", 0))
        if remaining < len(text):
            logger.warning(
                f"[ElevenLabs] Insufficient character quota ({remaining} < {len(text)}). "
                f"Skipping ElevenLabs call to prevent overage."
            )
            return None

        # 3. Request audio synthesis
        url = f"{self.base_url}/text-to-speech/{voice_id}"
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.75,
                "similarity_boost": 0.75,
                "style": 0.20,
                "use_speaker_boost": True,
            },
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "xi-api-key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=15, context=self._ssl_ctx) as resp:
                audio_bytes = resp.read()

            # Store in cache and return path
            saved_path = audio_cache.put(text, voice_id, audio_bytes, ext="mp3", model=model_id)
            logger.info(f"[ElevenLabs] Successfully synthesized {len(text)} chars -> {saved_path.name}")
            return saved_path
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            logger.error(f"[ElevenLabs] HTTP {e.code} error: {err_body}")
            return None
        except Exception as e:
            logger.error(f"[ElevenLabs] Synthesis error: {e}")
            return None


# Global singleton client
elevenlabs_client = ElevenLabsClient()
