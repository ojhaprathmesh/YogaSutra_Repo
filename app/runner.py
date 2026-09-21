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

import asyncio
import json
import logging
import re
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Suppress framework noise and experimental ADK warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("google.adk").setLevel(logging.ERROR)

from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from app.agents.root.agent import phase1_root_agent, phase2_root_agent, root_agent
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
    phase: int = 2


# ---------------------------------------------------------------------------
# Main runner function
# ---------------------------------------------------------------------------


def run_yogasutra(
    user_input: str,
    persona_id: str = "P00",
    phase: int = 2,
    on_agent_start: Any | None = None,
    on_agent_done: Any | None = None,
    on_agent_error: Any | None = None,
) -> YogaSutraResult:
    """
    Run the YogaSutra multi-agent pipeline for a given user input.

    Args:
        user_input:      The user's natural-language yoga request.
        persona_id:      Optional persona identifier for tracing (default "P00").
        phase:           Workflow phase: 1 (Phase 1 Minimal PoC) or 2 (Phase 2 Full MAS).
        on_agent_start:  Optional callback(agent_name: str) fired when an agent begins.
        on_agent_done:   Optional callback(agent_name: str) fired when an agent completes.
        on_agent_error:  Optional callback(agent_name: str, msg: str) fired on error.

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
        phase=phase,
    )

    # Select agent based on phase
    chosen_agent = phase1_root_agent if phase == 1 else phase2_root_agent

    # Create runner and session
    runner = InMemoryRunner(agent=chosen_agent, app_name="yogasutra")

    # Initial session state (blackboard)
    initial_state = {
        "run_id": run_id,
        "session_id": session_id,
        "persona_id": persona_id,
        "user_input": user_input,
        "phase": phase,
    }

    asyncio.run(
        runner.session_service.create_session(
            app_name="yogasutra",
            user_id=user_id,
            state=initial_state,
            session_id=session_id,
        )
    )

    # Build the user message
    user_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=user_input)],
    )

    # Run the agent and collect events
    final_response_text = ""
    events_collected: list = []

    # Live graph tracking: fire callbacks on author transitions
    _current_author: list[str] = [""]  # mutable container for closure

    def _on_new_author(author: str) -> None:
        """Called when a new agent begins emitting events."""
        prev = _current_author[0]
        if prev and prev != author:
            # Previous agent has finished — mark it done
            if on_agent_done:
                try:
                    on_agent_done(prev)
                except Exception:  # noqa: BLE001
                    pass
        if author and author != prev:
            _current_author[0] = author
            if on_agent_start:
                try:
                    on_agent_start(author)
                except Exception:  # noqa: BLE001
                    pass

    # Activate RootAgent immediately
    if on_agent_start:
        try:
            on_agent_start("RootAgent")
        except Exception:  # noqa: BLE001
            pass

    t_start = time.monotonic()
    try:
        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=user_message,
        ):
            events_collected.append(event)

            # Track agent calls and fire live graph callbacks
            if event.author:
                _on_new_author(event.author)
                if event.author not in trace.agents_called:
                    trace.record_agent(
                        agent_name=event.author,
                        latency_ms=0.0,
                    )

            # Capture token usage if available
            if event.usage_metadata and event.author:
                trace.token_usage[event.author] = {
                    "input_tokens": getattr(event.usage_metadata, "prompt_token_count", 0) or 0,
                    "output_tokens": getattr(event.usage_metadata, "candidates_token_count", 0) or 0,
                    "total_tokens": getattr(event.usage_metadata, "total_token_count", 0) or 0,
                }

            # Capture the final response from the terminal agent
            if event.is_final_response() and event.content and event.content.parts:
                response_parts = [
                    p.text for p in event.content.parts if hasattr(p, "text") and p.text
                ]
                if response_parts:
                    final_response_text = "\n".join(response_parts)

    except Exception as exc:  # noqa: BLE001
        final_response_text = f"[Error during run: {exc}]"
        if on_agent_error and _current_author[0]:
            try:
                on_agent_error(_current_author[0], str(exc))
            except Exception:  # noqa: BLE001
                pass

    # Mark the last active sub-agent as done
    if on_agent_done and _current_author[0]:
        try:
            on_agent_done(_current_author[0])
        except Exception:  # noqa: BLE001
            pass

    # RootAgent was activated at the start but never auto-completed via
    # event.author transitions (sub-agents are the actual authors in events).
    # Explicitly mark it done now that the full pipeline has finished.
    if on_agent_done:
        try:
            on_agent_done("RootAgent")
        except Exception:  # noqa: BLE001
            pass

    total_ms = (time.monotonic() - t_start) * 1000

    # Retrieve the final session state
    final_session = asyncio.run(
        runner.session_service.get_session(
            app_name="yogasutra",
            user_id=user_id,
            session_id=session_id,
        )
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
    state_agent_map = {
        "user_profile": "ProfileAgent",
        "retrieved_evidence": "KnowledgeAgent",
        "candidate_plan": "PlannerAgent",
        "validation_results": "SafetyAgent",
        "critique": "CriticAgent",
    }
    for key, writer_agent in state_agent_map.items():
        if session_state.get(key):
            trace.record_state_change(
                field_name=key,
                summary="Populated by pipeline",
                agent=writer_agent,
            )

    # Clean and record validation results in trace
    def _clean_json_dict(val: Any) -> dict:
        if isinstance(val, dict):
            return val
        if not isinstance(val, str):
            return {}
        s = val.strip()
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return {"summary": val}

    val_res = session_state.get("validation_results")
    if val_res:
        trace.validation_results = _clean_json_dict(val_res)

    critique_res = session_state.get("critique")
    if critique_res:
        trace.critic_result = _clean_json_dict(critique_res)

    trace.revision_count = int(session_state.get("revision_count", 0))

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
        phase=phase,
    )
