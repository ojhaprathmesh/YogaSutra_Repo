"""
app/api/server.py — YogaSutra FastAPI Web & Audio Server

Provides REST and Static UI endpoints for:
1. Practice Plan Generation across Multi-Model Providers (Gemini, Ollama, NVIDIA).
2. Spoken Voice Synthesis (ElevenLabs Turbo v2.5 + WebSpeech fallback).
3. Speech-to-Text Transcription.
4. Audio File Serving from `data/audio_cache/`.
5. Execution Trace and MCP Tool Inspection.
6. Interactive Hands-Free Practice Studio Web App.
7. Security Guardrails — input validation and prompt-injection protection (§8).
8. Prompt Strategy Variants — ReAct, Few-Shot, CoT via /api/practice/generate (§8).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.audio.elevenlabs import elevenlabs_client
from app.audio.tts import tts_engine
from app.audio.stt import stt_adapter
from app.mcp.server import mcp_server
from app.models.gateway import model_gateway
from app.runner import run_yogasutra
from app.security.guardrails import GuardrailResult, run_input_guardrails
from app.security.callbacks import validate_tool_return, validate_state_transition
from app.prompts.factory import PromptStrategy, build_agent_instruction, build_variant_agents
from app.ui.dashboard import collect_dashboard_stats, record_guardrail_audit, render_dashboard_html

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
    prompt_strategy: str = "default"   # "react" | "few_shot" | "cot" | "default"
    include_audio: bool = True


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
            "nvidia_nim": [os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3.5-bolt-e4b")],
        },
    }


def _build_plan_audio_cue_text(result: Any, max_chars: int = 220) -> str:
    """Extract a calm 1–2 sentence spoken intro cue from an approved practice plan."""
    candidate_text = ""
    session_state = getattr(result, "session_state", None)
    if isinstance(session_state, dict):
        raw_plan = session_state.get("candidate_plan")
        plan: dict[str, Any] | None = None
        if isinstance(raw_plan, dict):
            plan = raw_plan
        elif isinstance(raw_plan, str) and raw_plan.strip():
            cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw_plan.strip())
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    plan = parsed
            except Exception:
                plan = None

        if plan:
            summary = plan.get("summary") or plan.get("opening_cue") or plan.get("intro")
            if isinstance(summary, str) and summary.strip():
                candidate_text = summary.strip()
            else:
                goal = plan.get("goal") or plan.get("title") or plan.get("session_title") or "yoga"
                duration = plan.get("duration_minutes") or plan.get("total_duration_minutes")
                seq = plan.get("sequence") or plan.get("poses") or []
                first_pose = None
                if isinstance(seq, list) and seq:
                    first_item = seq[0]
                    if isinstance(first_item, dict):
                        first_pose = first_item.get("name") or first_item.get("asana") or first_item.get("pose")
                    elif isinstance(first_item, str):
                        first_pose = first_item
                dur_part = f"{duration}-minute " if duration else ""
                intro = f"Welcome to your {dur_part}{goal} practice."
                if first_pose:
                    intro += f" Take a calm breath as we begin with {first_pose}."
                candidate_text = intro

    if not candidate_text:
        raw_resp = getattr(result, "final_response", "") or ""
        plain = re.sub(r"[#*_`>]+", " ", str(raw_resp))
        plain = re.sub(r"\s+", " ", plain).strip()
        if plain:
            sentences = re.split(r"(?<=[.!?])\s+", plain)
            candidate_text = " ".join(sentences[:2]).strip()

    candidate_text = re.sub(r"\s+", " ", candidate_text).strip()
    if len(candidate_text) <= max_chars:
        return candidate_text

    truncated = candidate_text[: max_chars - 3].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return f"{truncated}..." if truncated else candidate_text[: max_chars - 3] + "..."


@app.post("/api/practice/generate")
def generate_practice(req: PracticeRequest) -> dict[str, Any]:
    """Runs the multi-agent pipeline and returns the practice plan and execution trace.

    Security:
      - Passes user_input through the guardrail engine BEFORE the pipeline starts.
      - Logs any guardrail violations as SafetyEscalationEvents.
    Prompt Variances:
      - Accepts prompt_strategy: "react" | "few_shot" | "cot" | "default".
      - The strategy is recorded in the trace for evaluation comparisons.
    """
    # ── Layer 1: Input Guardrail ──────────────────────────────────────────
    guard = run_input_guardrails(req.user_input)
    if not guard.is_safe:
        # Log violation to event bus for observability and audit trail
        try:
            record_guardrail_audit(
                input_text=req.user_input,
                violations=[v.value for v in guard.violations],
                risk_score=guard.risk_score,
                status="BLOCKED",
            )
            from app.workflows.event_driven import SafetyEscalationEvent, get_event_bus
            bus = get_event_bus()
            bus.publish_sync(SafetyEscalationEvent(
                session_id="pre-pipeline",
                user_id=req.persona_id,
                severity="medium" if guard.risk_score < 0.9 else "critical",
                trigger_condition=f"Input guardrail violation: {guard.violations}",
                action_required="halt_practice",
            ))
        except Exception:
            pass
        logger.warning("[API] Guardrail blocked request. violations=%s", guard.violations)
        raise HTTPException(status_code=400, detail=guard.rejection_reason)

    # ── Layer 2: Resolve Prompt Strategy ──────────────────────────────────
    try:
        strategy = PromptStrategy(req.prompt_strategy)
    except ValueError:
        strategy = PromptStrategy.DEFAULT

    try:
        result = run_yogasutra(
            user_input=guard.sanitized_input,
            persona_id=req.persona_id,
            phase=req.phase,
            prompt_strategy=strategy,
        )
        audio_cue_payload: dict[str, Any] | None = None
        if req.include_audio:
            try:
                cue_text = _build_plan_audio_cue_text(result)
                if cue_text:
                    cue = tts_engine.synthesize_cue(text=cue_text)
                    audio_url = f"/api/audio/file/{Path(cue.audio_path).name}" if cue.audio_path else None
                    audio_cue_payload = {
                        "text": cue.text,
                        "audio_url": audio_url,
                        "provider": cue.provider,
                        "voice_name": cue.voice_name,
                    }
            except Exception as audio_err:
                logger.warning("[API] Plan audio cue synthesis failed (non-fatal): %s", audio_err)

        return {
            "run_id": result.run_id,
            "session_id": result.session_id,
            "final_response": result.final_response,
            "phase": result.phase,
            "prompt_strategy": strategy.value,
            "guardrail": guard.to_dict(),
            "audio_cue": audio_cue_payload,
            "trace": result.trace.to_dict(),
        }
    except Exception as e:
        logger.error(f"[API] Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/audio/stream")
def stream_speech(req: SynthesisRequest):
    """Stream synthesized MP3 audio directly via ElevenLabs (or cache), with clean fallback."""
    clean_text = req.text.strip()
    if req.prefer_elevenlabs and elevenlabs_client.is_configured and clean_text:
        stream = elevenlabs_client.synthesize_stream(
            text=clean_text,
            voice_id=req.voice_id,
        )
        first_chunk = next(stream, None)
        if first_chunk is not None:
            def _chunk_iter():
                yield first_chunk
                yield from stream

            return StreamingResponse(
                _chunk_iter(),
                media_type="audio/mpeg",
                headers={"Cache-Control": "no-cache"},
            )

    return JSONResponse(
        status_code=503,
        content={
            "detail": "ElevenLabs audio streaming unavailable",
            "fallback": {
                "text": clean_text,
                "provider": "webspeech",
                "speaking_rate": 0.88,
                "voice_name": "Default Calming Voice",
            },
        },
    )


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


@app.get("/api/events/stream")
async def stream_events(request: Request) -> StreamingResponse:
    """Server-Sent Events (SSE) stream of live PostureDeviation and SafetyEscalation events."""
    bus = get_event_bus()
    queue = bus.add_stream_listener()

    async def event_generator():
        try:
            yield f"event: connected\ndata: {json.dumps({'status': 'connected'})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            bus.remove_stream_listener(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
    serializable_results = [r for r in results if isinstance(r, dict)]
    response: dict[str, Any] = {
        "status": "published",
        "event_id": evt.event_id,
        "event_type": evt.event_type,
        "handled_results_count": len(results),
        "results": serializable_results,
    }
    if req.event_type == "PostureDeviation":
        for res in serializable_results:
            if res.get("handled"):
                response["cue"] = res.get("cue")
                response["audio_url"] = res.get("audio_url")
                response["provider"] = res.get("provider")
                response["voice_name"] = res.get("voice_name")
                break
    return response


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
# Security & Prompt Variant Endpoints (§8)
# ---------------------------------------------------------------------------

class SecurityValidateRequest(BaseModel):
    user_input: str


@app.post("/api/security/validate")
def security_validate(req: SecurityValidateRequest) -> dict[str, Any]:
    """
    Inspect a user input string through the guardrail pipeline without
    running the full multi-agent system. Returns a GuardrailResult.

    Useful for the Studio UI to show a live safety indicator before submission.
    """
    result = run_input_guardrails(req.user_input)
    return result.to_dict()


class PromptVariantsRequest(BaseModel):
    persona_id: str = "P00"


@app.post("/api/prompts/variants")
def get_prompt_variants(req: PromptVariantsRequest) -> dict[str, str]:
    """
    Return all three architectural prompt variants (ReAct, Few-Shot, CoT)
    for the PlannerAgent, parameterised by persona_id.

    Used by the evaluation benchmark to run side-by-side architecture comparisons.
    """
    return build_variant_agents(persona_id=req.persona_id)


@app.get("/api/prompts/strategies")
def list_prompt_strategies() -> list[dict[str, str]]:
    """List the available prompt strategy variants with descriptions."""
    return [
        {
            "value": "default",
            "label": "Default / Chain-of-Thought",
            "description": "Production prompt. Structured CoT-style output directly to JSON.",
        },
        {
            "value": "react",
            "label": "ReAct (Reason + Act)",
            "description": "Explicit Thought → Action → Observation cycle over tool calls. "
                           "Best for transparency and tool-heavy agents.",
        },
        {
            "value": "few_shot",
            "label": "Few-Shot In-Context",
            "description": "In-context exemplars of reference input/output pairs. "
                           "Best for structured output reliability.",
        },
        {
            "value": "cot",
            "label": "Chain-of-Thought (Explicit)",
            "description": "Narrates reasoning inside <thinking> tags before final answer. "
                           "Balances latency vs interpretability.",
        },
    ]


# ---------------------------------------------------------------------------
# Phase 3C: Observability & Trace Analytics Dashboard Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/dashboard/stats")
def get_dashboard_stats_endpoint() -> dict[str, Any]:
    """Retrieve aggregated observability statistics, latencies, tokens, and event bus data."""
    return collect_dashboard_stats()


@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard() -> HTMLResponse:
    """Serve interactive Observability & Trace Analytics Dashboard."""
    return HTMLResponse(content=render_dashboard_html())


# ---------------------------------------------------------------------------
# Phase 3B: Real-Time Vision WebSocket Endpoint (/ws/vision)
# ---------------------------------------------------------------------------

_POSE_DETECTOR: Any = None


def _get_pose_detector() -> Any:
    """Lazy-initialize the MediaPipe PoseDetector singleton."""
    global _POSE_DETECTOR
    if _POSE_DETECTOR is None:
        from app.vision.detector import PoseDetector

        _POSE_DETECTOR = PoseDetector()
    return _POSE_DETECTOR


def _decode_base64_frame(image_data: str, max_width: int = 640, max_height: int = 480) -> np.ndarray | None:
    """Decode a base64 (or data-URI) JPEG/PNG frame into a 480p-clamped BGR numpy array."""
    if not image_data or not isinstance(image_data, str):
        return None
    try:
        payload = image_data.split(",", 1)[1] if "," in image_data else image_data
        raw_bytes = base64.b64decode(payload, validate=False)
        if not raw_bytes:
            return None
        np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        if w > max_width or h > max_height:
            scale = min(max_width / w, max_height / h)
            frame = cv2.resize(
                frame,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return frame
    except Exception:
        return None


@app.websocket("/ws/vision")
async def vision_websocket(websocket: WebSocket) -> None:
    """Real-time webcam vision WebSocket for 33-point pose extraction & spoken cues."""
    from app.agents.pose.agent import PoseAgent

    await websocket.accept()

    asana_name: str = "Virabhadrasana II"
    lead_leg: Literal["left", "right"] = "left"
    user_id: str = "P00"
    pose_agent = PoseAgent(
        debounce_seconds=1.5,
        alert_cooldown_seconds=4.0,
        user_id=user_id,
    )

    try:
        while True:
            try:
                raw_text = await websocket.receive_text()
                msg = json.loads(raw_text)
                if not isinstance(msg, dict):
                    await websocket.send_json({
                        "type": "telemetry",
                        "pose_detected": False,
                        "asana": asana_name,
                        "latency_ms": 0.0,
                        "landmarks": [],
                        "angles": {},
                        "deviations": [],
                        "fired_cues": [],
                    })
                    continue
            except WebSocketDisconnect:
                break
            except (ValueError, TypeError):
                await websocket.send_json({
                    "type": "telemetry",
                    "pose_detected": False,
                    "asana": asana_name,
                    "latency_ms": 0.0,
                    "landmarks": [],
                    "angles": {},
                    "deviations": [],
                    "fired_cues": [],
                })
                continue

            msg_type = msg.get("type")

            if msg_type == "config":
                new_asana = msg.get("asana")
                if isinstance(new_asana, str) and new_asana.strip():
                    asana_name = new_asana.strip()
                new_lead = msg.get("lead_leg")
                if new_lead in ("left", "right"):
                    lead_leg = new_lead
                new_uid = msg.get("user_id")
                if isinstance(new_uid, str) and new_uid.strip():
                    user_id = new_uid.strip()
                    pose_agent.user_id = user_id
                if isinstance(msg.get("debounce_seconds"), (int, float)):
                    pose_agent.debounce_seconds = max(0.0, float(msg["debounce_seconds"]))
                if isinstance(msg.get("alert_cooldown_seconds"), (int, float)):
                    pose_agent.alert_cooldown_seconds = max(0.0, float(msg["alert_cooldown_seconds"]))
                pose_agent.reset_state()
                await websocket.send_json({
                    "type": "config_ack",
                    "asana": asana_name,
                    "lead_leg": lead_leg,
                    "user_id": user_id,
                })
                continue

            if msg_type == "frame":
                t0 = time.perf_counter()

                frame_asana = msg.get("asana")
                if isinstance(frame_asana, str) and frame_asana.strip() and frame_asana.strip() != asana_name:
                    asana_name = frame_asana.strip()
                    pose_agent.reset_state()

                frame_lead = msg.get("lead_leg")
                if frame_lead in ("left", "right") and frame_lead != lead_leg:
                    lead_leg = frame_lead
                    pose_agent.reset_state()

                frame = _decode_base64_frame(msg.get("image", ""))
                if frame is None:
                    latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                    await websocket.send_json({
                        "type": "telemetry",
                        "pose_detected": False,
                        "asana": asana_name,
                        "latency_ms": latency_ms,
                        "landmarks": [],
                        "angles": {},
                        "deviations": [],
                        "fired_cues": [],
                    })
                    continue

                try:
                    detector = _get_pose_detector()
                    snapshot = await asyncio.to_thread(detector.extract_biomechanics, frame)
                except Exception as exc:
                    logger.warning("[WS /ws/vision] Pose extraction failed: %s", exc)
                    latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                    await websocket.send_json({
                        "type": "telemetry",
                        "pose_detected": False,
                        "asana": asana_name,
                        "latency_ms": latency_ms,
                        "landmarks": [],
                        "angles": {},
                        "deviations": [],
                        "fired_cues": [],
                    })
                    continue

                if not snapshot or not snapshot.landmarks:
                    latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                    await websocket.send_json({
                        "type": "telemetry",
                        "pose_detected": False,
                        "asana": asana_name,
                        "latency_ms": latency_ms,
                        "landmarks": [],
                        "angles": {},
                        "deviations": [],
                        "fired_cues": [],
                    })
                    continue

                deviations, fired_events = pose_agent.evaluate_snapshot(
                    snapshot,
                    asana_name=asana_name,
                    lead_leg=lead_leg,
                )

                # Build ordered 33-point landmark array
                max_idx = max(32, max(snapshot.landmarks.keys()))
                landmarks_list: list[dict[str, Any]] = []
                for idx in range(max_idx + 1):
                    pt = snapshot.landmarks.get(idx)
                    if pt is not None:
                        landmarks_list.append({
                            "id": idx,
                            "x": round(float(pt.x), 4),
                            "y": round(float(pt.y), 4),
                            "visibility": round(float(pt.visibility), 3),
                        })
                    else:
                        landmarks_list.append({
                            "id": idx,
                            "x": 0.0,
                            "y": 0.0,
                            "visibility": 0.0,
                        })

                deviations_payload = [
                    {
                        "joint_name": d.joint_name,
                        "measured_angle": d.measured_angle,
                        "target_min": d.target_min,
                        "target_max": d.target_max,
                        "deviation_deg": d.deviation_deg,
                        "is_aligned": d.is_aligned,
                        "severity": d.severity,
                        "correction_cue": d.correction_cue,
                    }
                    for d in deviations
                ]

                fired_cues: list[dict[str, Any]] = []
                if fired_events:
                    recent_cues = pose_agent.event_bus.get_recent_cue_events(limit=20)
                    recent_by_id = {
                        c.get("event_id"): c for c in recent_cues if isinstance(c, dict) and c.get("event_id")
                    }
                    for evt in fired_events:
                        meta = recent_by_id.get(evt.event_id, {})
                        fired_cues.append({
                            "event_id": evt.event_id,
                            "joint_name": evt.joint_name,
                            "correction_cue": evt.correction_cue,
                            "severity": evt.severity,
                            "measured_angle": evt.measured_angle,
                            "deviation_deg": evt.deviation_deg,
                            "audio_url": meta.get("audio_url"),
                            "provider": meta.get("provider", "webspeech"),
                        })

                latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                await websocket.send_json({
                    "type": "telemetry",
                    "pose_detected": True,
                    "asana": asana_name,
                    "latency_ms": latency_ms,
                    "landmarks": landmarks_list,
                    "angles": snapshot.summary(),
                    "deviations": deviations_payload,
                    "fired_cues": fired_cues,
                })
    except WebSocketDisconnect:
        logger.debug("[WS /ws/vision] Client disconnected cleanly.")


# ---------------------------------------------------------------------------
# Static Studio UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_index() -> HTMLResponse:
    index_file = _STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>YogaSutra API is running. Studio index.html loading...</h1>")
