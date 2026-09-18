"""
app/runner.py — YogaSutra ADK InMemoryRunner

Sets up and executes a YogaSutra run:
1. Creates an InMemoryRunner with the root agent
2. Creates a session with initial state
3. Runs the root agent with the user message
4. Collects the execution trace
5. Returns the final response text and trace

Usage (programmatic)::

    from app.runner import run_yogasutra
    result = run_yogasutra("I am a beginner and have 20 minutes...")
    print(result.final_response)
    print(result.trace_path)
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from app.agents.root.agent import root_agent
from app.tracing.trace import ExecutionTrace

# Load .env if present
load_dotenv()

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class YogaSutraResult:
    """Holds the outcome of a single YogaSutra run."""

    run_id: str
    session_id: str
    user_input: str
    final_response: str
    session_state: dict
    trace: ExecutionTrace
    trace_path: Path | None = None


# ---------------------------------------------------------------------------
# Main runner function
# ---------------------------------------------------------------------------


def run_yogasutra(user_input: str, persona_id: str = "P00") -> YogaSutraResult:
    """
    Run the full YogaSutra multi-agent pipeline for a given user input.

    Args:
        user_input:  The user's natural-language yoga request.
        persona_id:  Optional persona identifier for tracing (default "P00").

    Returns:
        YogaSutraResult with final_response, session_state, and trace.
    """
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    user_id = "demo_user"

    # Initialise trace
    trace = ExecutionTrace(
        run_id=run_id,
        session_id=session_id,
        persona_id=persona_id,
        user_input=user_input,
    )

    # Create runner and session
    runner = InMemoryRunner(agent=root_agent, app_name="yogasutra")
    session_service = InMemorySessionService()

    # Initial session state (blackboard)
    initial_state = {
        "run_id": run_id,
        "session_id": session_id,
        "persona_id": persona_id,
        "user_input": user_input,
    }

    session = session_service.create_session(
        app_name="yogasutra",
        user_id=user_id,
        state=initial_state,
        session_id=session_id,
    )

    # Build the user message
    user_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_input)],
    )

    # Run the agent and collect events
    final_response_text = ""
    events_collected: list = []

    t_start = time.monotonic()
    try:
        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=user_message,
        ):
            events_collected.append(event)

            # Track agent calls
            if event.author and event.author not in trace.agents_called:
                trace.record_agent(
                    agent_name=event.author,
                    latency_ms=0.0,  # Fine-grained latency tracked separately if needed
                )

            # Capture token usage if available
            if event.usage_metadata and event.author:
                trace.token_usage[event.author] = {
                    "input_tokens": getattr(event.usage_metadata, "prompt_token_count", 0) or 0,
                    "output_tokens": getattr(event.usage_metadata, "candidates_token_count", 0) or 0,
                    "total_tokens": getattr(event.usage_metadata, "total_token_count", 0) or 0,
                }

            # Capture the final response
            if event.is_final_response():
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_response_text += part.text

    except Exception as exc:  # noqa: BLE001
        final_response_text = f"[Error during run: {exc}]"

    total_ms = (time.monotonic() - t_start) * 1000

    # Retrieve the final session state
    final_session = session_service.get_session(
        app_name="yogasutra",
        user_id=user_id,
        session_id=session_id,
    )
    session_state = dict(final_session.state) if final_session else {}

    # Record retrieved evidence in trace
    retrieved_evidence = session_state.get("retrieved_evidence")
    if retrieved_evidence:
        # May be a JSON string or a list depending on ADK output_key handling
        if isinstance(retrieved_evidence, str):
            try:
                retrieved_evidence = json.loads(retrieved_evidence)
            except json.JSONDecodeError:
                retrieved_evidence = []
        if isinstance(retrieved_evidence, list):
            for chunk in retrieved_evidence:
                if isinstance(chunk, dict):
                    trace.record_retrieved_document(
                        chunk_id=chunk.get("chunk_id", ""),
                        title=chunk.get("title", ""),
                        topic=chunk.get("topic", ""),
                    )

    # Record state changes
    for key in ["user_profile", "retrieved_evidence", "candidate_plan"]:
        if session_state.get(key):
            trace.record_state_change(key, f"Populated by pipeline")

    # Finalise trace
    trace.set_final_output(final_response_text)
    trace.total_latency_ms = round(total_ms, 2)

    # Save trace to disk
    trace_path: Path | None = None
    try:
        trace_path = trace.save()
    except Exception:  # noqa: BLE001
        pass  # Non-fatal — trace saving failure doesn't break the run

    return YogaSutraResult(
        run_id=run_id,
        session_id=session_id,
        user_input=user_input,
        final_response=final_response_text,
        session_state=session_state,
        trace=trace,
        trace_path=trace_path,
    )
