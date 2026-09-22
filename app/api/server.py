"""
app/api/server.py — YogaSutra FastAPI Web & Audio Server

Provides REST and Static UI endpoints for:
1. Practice Plan Generation across Multi-Model Providers (Gemini, Ollama, NVIDIA).
2. Spoken Voice Synthesis (ElevenLabs Turbo v2.5 + WebSpeech fallback).
3. Speech-to-Text Transcription.
4. Audio File Serving from `data/audio_cache/`.
5. Execution Trace and MCP Tool Inspection.
6. Interactive Hands-Free Practice Studio Web App.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.audio.elevenlabs import elevenlabs_client
from app.audio.tts import tts_engine
from app.audio.stt import stt_adapter
from app.audio.cache import audio_cache
from app.mcp.server import mcp_server
from app.models.gateway import model_gateway
from app.models.router import model_router, TaskComplexity
from app.runner import run_yogasutra

logger = logging.getLogger("yogasutra.api")

app = FastAPI(
    title="YogaSutra API",
    description="Multi-Agent AI Yoga Planner with Multi-Model Routing, ElevenLabs Voice, and MCP",
    version="0.2.0",
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)
_AUDIO_DIR = Path("data/audio_cache")
_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Request & Response Models
# ---------------------------------------------------------------------------

class PracticeRequest(BaseModel):
    user_input: str
    persona_id: str = "P00"
    phase: int = 2
    preferred_provider: str | None = None


class SynthesisRequest(BaseModel):
    text: str
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    prefer_elevenlabs: bool = True


class TranscriptionRequest(BaseModel):
    audio_payload: str
    language: str = "en"


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "app": "YogaSutra", "version": "0.2.0"}


@app.get("/api/models")
def get_model_status() -> dict[str, Any]:
    """Returns availability and status of configured model providers and speech."""
    quota = elevenlabs_client.check_quota() if elevenlabs_client.is_configured else {"status": "unconfigured"}
    return {
        "providers": {
            "gemini": bool(model_gateway.google_api_key),
            "ollama_cloud": bool(model_gateway.ollama_api_key),
            "nvidia_nim": bool(model_gateway.nvidia_api_key),
            "elevenlabs": elevenlabs_client.is_configured,
        },
        "elevenlabs_quota": quota,
        "models": {
            "gemini": ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
            "ollama_cloud": ["gpt-oss:120b", "gemma4:31b", "nemotron-3-nano:30b"],
            "nvidia_nim": ["meta/llama-3.2-11b-vision-instruct"],
        },
    }


@app.post("/api/practice/generate")
def generate_practice(req: PracticeRequest) -> dict[str, Any]:
    """Runs the multi-agent pipeline and returns the practice plan and execution trace."""
    try:
        result = run_yogasutra(
            user_input=req.user_input,
            persona_id=req.persona_id,
            phase=req.phase,
        )
        return {
            "run_id": result.run_id,
            "session_id": result.session_id,
            "final_response": result.final_response,
            "phase": result.phase,
            "trace": result.trace.to_dict(),
        }
    except Exception as e:
        logger.error(f"[API] Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/audio/synthesize")
def synthesize_speech(req: SynthesisRequest) -> dict[str, Any]:
    """Synthesize voice cue via ElevenLabs with audio caching and WebSpeech fallback."""
    cue = tts_engine.synthesize_cue(
        text=req.text,
        voice_id=req.voice_id,
        prefer_elevenlabs=req.prefer_elevenlabs,
    )
    audio_url = None
    if cue.audio_path:
        filename = Path(cue.audio_path).name
        audio_url = f"/api/audio/file/{filename}"

    return {
        "text": cue.text,
        "provider": cue.provider,
        "voice_name": cue.voice_name,
        "audio_url": audio_url,
        "speaking_rate": cue.speaking_rate,
    }


@app.post("/api/audio/transcribe")
def transcribe_speech(req: TranscriptionRequest) -> dict[str, Any]:
    """Transcribes user speech payload into text."""
    res = stt_adapter.transcribe_payload(req.audio_payload, language=req.language)
    return {
        "transcript": res.transcript,
        "confidence": res.confidence,
        "source": res.source,
    }


@app.get("/api/audio/file/{filename}")
def get_audio_file(filename: str):
    """Serves synthesized audio files from the disk cache."""
    safe_name = Path(filename).name
    file_path = _AUDIO_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/mpeg")


@app.get("/api/trace/{run_id}")
def get_trace(run_id: str) -> dict[str, Any]:
    """Fetch stored execution trace JSON by run ID."""
    trace_path = Path("data/traces") / f"{run_id}.json"
    if not trace_path.exists():
        raise HTTPException(status_code=404, detail=f"Trace {run_id} not found")
    return json.loads(trace_path.read_text(encoding="utf-8"))


@app.get("/api/mcp/tools")
def get_mcp_tools() -> list[dict[str, Any]]:
    """List exposed MCP tools."""
    return mcp_server.list_tools()


# ---------------------------------------------------------------------------
# Static Studio UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_index() -> HTMLResponse:
    index_file = _STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>YogaSutra API is running. Studio index.html loading...</h1>")
