"""
app/workflows/event_driven.py — Asynchronous Event-Driven Bus for YogaSutra MAS

Implements a thread-safe, asynchronous Pub/Sub Event Bus architecture
connecting agents, memory systems, CV workers, and UI components.
Declared in PLAN.md §30 and Phase 3A specifications.

Core Events:
- SessionCompletedEvent: triggers longitudinal memory updates and streaks.
- FeedbackReceivedEvent: adapts personalization and difficulty scaling.
- SafetyEscalationEvent: immediate circuit breaker for injuries / severe pain.
- PostureDeviationEvent: asynchronous stream for computer vision corrections.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from datetime import datetime, UTC
from typing import Any, Callable

from pydantic import BaseModel, Field

from app.state import YogaSessionState

logger = logging.getLogger("yogasutra.events")


# ---------------------------------------------------------------------------
# Typed Event Schemas
# ---------------------------------------------------------------------------


class BaseEvent(BaseModel):
    """Base event contract for all YogaSutra event messages."""

    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:10]}")
    event_type: str
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    session_id: str = ""
    source_agent: str = "System"
    payload: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class SessionCompletedEvent(BaseEvent):
    """Fired when a user finishes their yoga practice sequence."""

    event_type: str = "SessionCompleted"
    user_id: str = "P01"
    duration_minutes: int = 20
    completed_poses: list[str] = Field(default_factory=list)
    plan_goal: str = "General wellness"


class FeedbackReceivedEvent(BaseEvent):
    """Fired when post-session subjective feedback is submitted."""

    event_type: str = "FeedbackReceived"
    user_id: str = "P01"
    difficulty_rating: int = 3  # 1 (too easy) to 5 (too hard)
    fatigue_level: str = "normal"  # "energized", "normal", "exhausted"
    comments: str = ""
    pose_ratings: dict[str, int] = Field(default_factory=dict)


class SafetyEscalationEvent(BaseEvent):
    """Fired when a critical contraindication, acute pain, or medical alert occurs."""

    event_type: str = "SafetyEscalation"
    user_id: str = "P01"
    severity: str = "medium"  # "low", "medium", "critical"
    trigger_condition: str = "Reported joint pain"
    action_required: str = "halt_practice"  # "halt_practice", "modify_pose", "medical_alert"
    recommended_alternatives: list[str] = Field(default_factory=list)


class PostureDeviationEvent(BaseEvent):
    """Fired by the Computer Vision engine when a joint angle deviates from criteria."""

    event_type: str = "PostureDeviation"
    user_id: str = "P01"
    pose_name: str = "Virabhadrasana II"
    joint_name: str = "left_knee"
    measured_angle: float = 72.0
    target_angle_range: list[float] = Field(default_factory=lambda: [85.0, 95.0])
    deviation_deg: float = 18.0
    severity: str = "moderate"  # "mild", "moderate", "severe"
    correction_cue: str = "Align your front knee directly over your ankle at 90 degrees."


# ---------------------------------------------------------------------------
# Asynchronous Pub/Sub Event Bus
# ---------------------------------------------------------------------------


class EventBus:
    """
    Central Pub/Sub Event Dispatcher for YogaSutra.

    Supports:
    - Typed event registration and dispatch
    - Both synchronous and asynchronous subscriber handlers
    - Wildcard ('*') subscriptions for logging and observability
    - In-memory historical event journal
    - Shared blackboard state integration
    """

    def __init__(self, max_journal_size: int = 500) -> None:
        self._subscribers: dict[str, list[Callable[..., Any]]] = {}
        self._journal: list[BaseEvent] = []
        self._max_journal_size = max_journal_size
        self._lock = asyncio.Lock() if asyncio else None

    def subscribe(self, event_type: str, handler: Callable[..., Any]) -> None:
        """Register a subscriber callback for a given event type or '*'."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)
            logger.debug("Subscribed %s to %s", getattr(handler, "__name__", str(handler)), event_type)

    def unsubscribe(self, event_type: str, handler: Callable[..., Any]) -> None:
        """Unregister a subscriber callback."""
        if event_type in self._subscribers and handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)

    def _record_event(self, event: BaseEvent) -> None:
        """Record the event in the journal with bounded memory size."""
        self._journal.append(event)
        if len(self._journal) > self._max_journal_size:
            self._journal.pop(0)

    def publish_sync(self, event: BaseEvent, state: YogaSessionState | None = None) -> list[Any]:
        """
        Synchronous event publication.
        Executes synchronous handlers immediately and schedules coroutines on running loop.
        """
        self._record_event(event)
        if state is not None:
            self.record_to_state(state, event)

        results: list[Any] = []
        handlers = list(self._subscribers.get(event.event_type, [])) + list(self._subscribers.get("*", []))

        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    try:
                        loop = asyncio.get_running_loop()
                        task = loop.create_task(handler(event))
                        results.append(task)
                    except RuntimeError:
                        # No running loop, run in a temporary loop
                        res = asyncio.run(handler(event))
                        results.append(res)
                else:
                    res = handler(event)
                    results.append(res)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error executing handler for %s: %s", event.event_type, exc)
                results.append({"error": str(exc)})

        return results

    async def publish(self, event: BaseEvent, state: YogaSessionState | None = None) -> list[Any]:
        """
        Asynchronous event publication.
        Awaits coroutine subscribers and calls synchronous subscribers directly.
        """
        self._record_event(event)
        if state is not None:
            self.record_to_state(state, event)

        results: list[Any] = []
        handlers = list(self._subscribers.get(event.event_type, [])) + list(self._subscribers.get("*", []))

        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    res = await handler(event)
                else:
                    res = handler(event)
                results.append(res)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error executing async handler for %s: %s", event.event_type, exc)
                results.append({"error": str(exc)})

        return results

    def record_to_state(self, state: YogaSessionState, event: BaseEvent) -> None:
        """Appends the event to the shared blackboard's event journal."""
        state.event_journal.append(event.model_dump())

    def get_history(self, event_type: str | None = None, limit: int = 50) -> list[BaseEvent]:
        """Retrieve recent events from the in-memory journal."""
        if event_type:
            filtered = [e for e in self._journal if e.event_type == event_type]
            return filtered[-limit:]
        return self._journal[-limit:]

    def clear(self) -> None:
        """Clear all subscribers and history (useful for test isolation)."""
        self._subscribers.clear()
        self._journal.clear()


# ---------------------------------------------------------------------------
# Global Singleton Access & Standard Handlers
# ---------------------------------------------------------------------------

_GLOBAL_EVENT_BUS: EventBus | None = None


def get_event_bus() -> EventBus:
    """Retrieve or initialize the global EventBus singleton."""
    global _GLOBAL_EVENT_BUS
    if _GLOBAL_EVENT_BUS is None:
        _GLOBAL_EVENT_BUS = EventBus()
        _register_default_handlers(_GLOBAL_EVENT_BUS)
    return _GLOBAL_EVENT_BUS


def _register_default_handlers(bus: EventBus) -> None:
    """Attach standard core listeners for system-level event processing."""

    def handle_safety_escalation(event: BaseEvent) -> dict[str, Any]:
        if isinstance(event, SafetyEscalationEvent):
            logger.warning(
                "🚨 SAFETY ESCALATION [%s]: %s (Action: %s)",
                event.severity.upper(),
                event.trigger_condition,
                event.action_required,
            )
            return {
                "handled": True,
                "status": "escalated",
                "circuit_breaker": event.severity == "critical",
            }
        return {}

    def handle_posture_deviation(event: BaseEvent) -> dict[str, Any]:
        if isinstance(event, PostureDeviationEvent):
            logger.info(
                "👁️ CV POSTURE DEVIATION: %s - %s off by %.1f° -> Cue: '%s'",
                event.pose_name,
                event.joint_name,
                event.deviation_deg,
                event.correction_cue,
            )
            return {
                "handled": True,
                "cue": event.correction_cue,
                "severity": event.severity,
            }
        return {}

    bus.subscribe("SafetyEscalation", handle_safety_escalation)
    bus.subscribe("PostureDeviation", handle_posture_deviation)
