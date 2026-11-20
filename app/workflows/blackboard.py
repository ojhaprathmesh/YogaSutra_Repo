"""
app/workflows/blackboard.py — Typed Blackboard Architecture for Shared Multi-Agent State

Partitions the ADK session state into distinct, typed channels:
- 'user_profile': user experience, goal, constraints, duration
- 'retrieved_evidence': grounded knowledge chunks with provenance
- 'candidate_plan': sequenced asanas, timings, props
- 'validation_results': deterministic + physiological safety audit
- 'critique': pedagogical review and reflection
- 'audio_cues': spoken voice feedback and ElevenLabs audio URLs
- 'trace_events': chronological state change history
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("yogasutra.blackboard")


@dataclass
class BlackboardChannel:
    """A named channel on the blackboard with history."""
    name: str
    value: Any = None
    last_writer: str = ""
    updated_at: float = field(default_factory=time.monotonic)


class Blackboard:
    """Structured blackboard coordinating multi-agent state transfers."""

    def __init__(self, initial_state: dict[str, Any] | None = None) -> None:
        self.channels: dict[str, BlackboardChannel] = {
            "user_profile": BlackboardChannel(name="user_profile"),
            "retrieved_evidence": BlackboardChannel(name="retrieved_evidence"),
            "candidate_plan": BlackboardChannel(name="candidate_plan"),
            "validation_results": BlackboardChannel(name="validation_results"),
            "critique": BlackboardChannel(name="critique"),
            "crewai_audit": BlackboardChannel(name="crewai_audit"),
            "a2a_messages": BlackboardChannel(name="a2a_messages"),
            "audio_cues": BlackboardChannel(name="audio_cues"),
        }
        self.event_log: list[dict[str, Any]] = []
        self._subscribers: list[Callable[[str, str, Any], None]] = []

        if initial_state:
            for k, v in initial_state.items():
                self.write(channel=k, writer="initial_setup", value=v)

    def write(self, channel: str, writer: str, value: Any) -> None:
        """Write a value to a named channel and log the event."""
        if channel not in self.channels:
            self.channels[channel] = BlackboardChannel(name=channel)

        ch = self.channels[channel]
        ch.value = value
        ch.last_writer = writer
        ch.updated_at = time.monotonic()

        event = {
            "channel": channel,
            "writer": writer,
            "timestamp": time.time(),
            "summary": f"Updated by {writer}",
        }
        self.event_log.append(event)
        logger.info(f"[Blackboard] Channel '{channel}' updated by {writer}")

        for callback in self._subscribers:
            try:
                callback(channel, writer, value)
            except Exception as e:
                logger.warning(f"[Blackboard] Subscriber error: {e}")

    def read(self, channel: str) -> Any:
        """Read current value of a channel."""
        ch = self.channels.get(channel)
        return ch.value if ch else None

    def subscribe(self, callback: Callable[[str, str, Any], None]) -> None:
        """Subscribe to blackboard state changes."""
        self._subscribers.append(callback)

    def to_dict(self) -> dict[str, Any]:
        """Convert current blackboard state to standard dictionary."""
        d = {k: ch.value for k, ch in self.channels.items() if ch.value is not None}
        d["event_log"] = list(self.event_log)
        return d
