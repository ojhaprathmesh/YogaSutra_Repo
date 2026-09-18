"""
app/tracing/trace.py — Execution Trace Collector

Writes structured JSON traces to data/traces/ per PLAN.md §27.

Trace schema fields:
    run_id, persona_id, session_id, input, agents_called, tools_called,
    retrieved_documents, state_changes, agent_latency_ms, total_latency_ms,
    token_usage, validation_results, critic_result, revision_count,
    final_output, evaluation
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Trace directory
# ---------------------------------------------------------------------------

_TRACE_DIR = Path(
    os.environ.get("YOGASUTRA_TRACE_DIR", "data/traces")
)


# ---------------------------------------------------------------------------
# ExecutionTrace — collected during a run, written at the end
# ---------------------------------------------------------------------------


@dataclass
class ExecutionTrace:
    """
    Collector for a single YogaSutra run trace.

    Usage::

        trace = ExecutionTrace(run_id="run_abc123", session_id="s1", persona_id="P00")
        trace.record_agent("ProfileAgent", latency_ms=412.3, tokens={"input": 110, "output": 45})
        trace.record_tool("retrieve_yoga_knowledge", input_summary="beginner 20min relaxation", output_summary="5 chunks")
        trace.record_retrieved_document("doc_001_002", "20-Minute Structure")
        trace.set_final_output("Here is your 20-minute practice plan...")
        path = trace.save()
    """

    run_id: str
    session_id: str
    persona_id: str = "P00"
    user_input: str = ""

    # Accumulated during the run
    agents_called: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    retrieved_documents: list[dict[str, str]] = field(default_factory=list)
    state_changes: list[dict[str, Any]] = field(default_factory=list)
    agent_latency_ms: dict[str, float] = field(default_factory=dict)
    token_usage: dict[str, dict[str, int]] = field(default_factory=dict)
    validation_results: dict[str, Any] = field(default_factory=dict)
    critic_result: dict[str, Any] = field(default_factory=dict)
    revision_count: int = 0
    final_output: str = ""
    evaluation: dict[str, Any] = field(default_factory=dict)

    # Internal timing
    _start_time: float = field(default_factory=time.monotonic, repr=False, compare=False)
    total_latency_ms: float = 0.0

    # -----------------------------------------------------------------------
    # Recording helpers
    # -----------------------------------------------------------------------

    def record_agent(
        self,
        agent_name: str,
        latency_ms: float = 0.0,
        tokens: dict[str, int] | None = None,
    ) -> None:
        """Record that an agent was invoked."""
        if agent_name not in self.agents_called:
            self.agents_called.append(agent_name)
        self.agent_latency_ms[agent_name] = round(latency_ms, 2)
        if tokens:
            self.token_usage[agent_name] = tokens

    def record_tool(
        self,
        tool_name: str,
        input_summary: str = "",
        output_summary: str = "",
    ) -> None:
        """Record a tool invocation."""
        self.tools_called.append(
            {"tool": tool_name, "input": input_summary, "output": output_summary}
        )

    def record_retrieved_document(self, chunk_id: str, title: str, topic: str = "") -> None:
        """Record a document retrieved by the RAG pipeline."""
        self.retrieved_documents.append(
            {"chunk_id": chunk_id, "title": title, "topic": topic}
        )

    def record_state_change(self, field_name: str, summary: str) -> None:
        """Record a state field being populated by an agent."""
        self.state_changes.append({"field": field_name, "summary": summary})

    def set_final_output(self, output: str) -> None:
        """Set the final response text."""
        self.final_output = output
        self.total_latency_ms = round(
            (time.monotonic() - self._start_time) * 1000, 2
        )

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the trace as a plain dict matching PLAN.md §27."""
        return {
            "run_id": self.run_id,
            "persona_id": self.persona_id,
            "session_id": self.session_id,
            "input": self.user_input,
            "agents_called": self.agents_called,
            "tools_called": self.tools_called,
            "retrieved_documents": self.retrieved_documents,
            "state_changes": self.state_changes,
            "agent_latency_ms": self.agent_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "token_usage": self.token_usage,
            "validation_results": self.validation_results,
            "critic_result": self.critic_result,
            "revision_count": self.revision_count,
            "final_output": self.final_output,
            "evaluation": self.evaluation,
        }

    def save(self, trace_dir: Path | None = None) -> Path:
        """
        Write the trace to a JSON file in the trace directory.

        Returns:
            Path to the written file.
        """
        directory = Path(trace_dir) if trace_dir else _TRACE_DIR
        # Make the path relative to the project root if it isn't absolute
        if not directory.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            directory = project_root / directory
        directory.mkdir(parents=True, exist_ok=True)

        # File named by run_id
        out_path = directory / f"{self.run_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        return out_path
