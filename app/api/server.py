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
from pydantic import BaseModel

from app.audio.elevenlabs import elevenlabs_client
from app.audio.tts import tts_engine
from app.audio.stt import stt_adapter
from app.mcp.server import mcp_server
from app.models.gateway import model_gateway
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
    voice_id: str = "EXAVITQu4vr4xnSDxMaL"
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
# Phase 3A: Event Bus, Longitudinal Memory, and ToT Planning Endpoints
# ---------------------------------------------------------------------------

from app.memory.longitudinal import get_longitudinal_store
from app.tools.planning import evaluate_candidate_plan_tool, generate_plan_candidates_tool
from app.workflows.event_driven import (
    BaseEvent,
    FeedbackReceivedEvent,
    PostureDeviationEvent,
    SafetyEscalationEvent,
    SessionCompletedEvent,
    get_event_bus,
)


class EventPublishRequest(BaseModel):
    event_type: str
    user_id: str = "P01"
    session_id: str = ""
    payload: dict[str, Any] = {}


class EvaluatePlanRequest(BaseModel):
    sequence: list[dict[str, Any]]
    available_duration_minutes: int = 20
    constraints: list[str] = []


class CandidateGenerationRequest(BaseModel):
    goal: str = "relaxation"
    duration_minutes: int = 20
    intensity: str = "gentle"
    constraints: list[str] = []


@app.get("/api/events/history")
def get_event_history(event_type: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve recent events from the in-memory Event Bus journal."""
    bus = get_event_bus()
    events = bus.get_history(event_type=event_type, limit=limit)
    return [e.model_dump() for e in events]


@app.post("/api/events/publish")
def publish_event(req: EventPublishRequest) -> dict[str, Any]:
    """Publish an event to the YogaSutra Event Bus."""
    bus = get_event_bus()
    evt: BaseEvent

    if req.event_type == "FeedbackReceived":
        evt = FeedbackReceivedEvent(
            session_id=req.session_id,
            user_id=req.user_id,
            difficulty_rating=req.payload.get("difficulty_rating", 3),
            fatigue_level=req.payload.get("fatigue_level", "normal"),
            comments=req.payload.get("comments", ""),
            pose_ratings=req.payload.get("pose_ratings", {}),
        )
    elif req.event_type == "SafetyEscalation":
        evt = SafetyEscalationEvent(
            session_id=req.session_id,
            user_id=req.user_id,
            severity=req.payload.get("severity", "medium"),
            trigger_condition=req.payload.get("trigger_condition", "Reported pain"),
            action_required=req.payload.get("action_required", "halt_practice"),
            recommended_alternatives=req.payload.get("recommended_alternatives", []),
        )
    elif req.event_type == "PostureDeviation":
        evt = PostureDeviationEvent(
            session_id=req.session_id,
            user_id=req.user_id,
            pose_name=req.payload.get("pose_name", "Unknown Pose"),
            joint_name=req.payload.get("joint_name", "knee"),
            measured_angle=float(req.payload.get("measured_angle", 0.0)),
            target_angle_range=req.payload.get("target_angle_range", [85.0, 95.0]),
            deviation_deg=float(req.payload.get("deviation_deg", 0.0)),
            severity=req.payload.get("severity", "moderate"),
            correction_cue=req.payload.get("correction_cue", "Adjust alignment"),
        )
    elif req.event_type == "SessionCompleted":
        evt = SessionCompletedEvent(
            session_id=req.session_id,
            user_id=req.user_id,
            duration_minutes=int(req.payload.get("duration_minutes", 20)),
            completed_poses=req.payload.get("completed_poses", []),
            plan_goal=req.payload.get("plan_goal", "Practice"),
        )
    else:
        evt = BaseEvent(
            event_type=req.event_type,
            session_id=req.session_id,
            payload=req.payload,
        )

    results = bus.publish_sync(evt)
    return {
        "status": "published",
        "event_id": evt.event_id,
        "event_type": evt.event_type,
        "handled_results_count": len(results),
    }


@app.get("/api/memory/trajectory/{user_id}")
def get_user_trajectory_endpoint(user_id: str) -> dict[str, Any]:
    """Fetch longitudinal progress profile and adaptive difficulty metrics."""
    store = get_longitudinal_store()
    trajectory = store.get_trajectory(user_id)
    adjustments = store.get_adaptive_profile_adjustments(user_id)
    return {
        "trajectory": trajectory.model_dump(),
        "adaptive_adjustments": adjustments,
    }


@app.post("/api/planning/evaluate")
def evaluate_plan_endpoint(req: EvaluatePlanRequest) -> dict[str, Any]:
    """Evaluate candidate sequence using the ToT multi-objective utility scorer."""
    return evaluate_candidate_plan_tool(
        sequence=req.sequence,
        available_duration_minutes=req.available_duration_minutes,
        constraints=req.constraints,
    )


@app.post("/api/planning/candidates")
def generate_candidates_endpoint(req: CandidateGenerationRequest) -> list[dict[str, Any]]:
    """Generate multiple sequence candidate branches for ToT exploration."""
    return generate_plan_candidates_tool(
        goal=req.goal,
        duration_minutes=req.duration_minutes,
        intensity=req.intensity,
        constraints=req.constraints,
    )


# ---------------------------------------------------------------------------
# Static Studio UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_index() -> HTMLResponse:
    index_file = _STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>YogaSutra API is running. Studio index.html loading...</h1>")
