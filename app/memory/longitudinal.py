"""
app/memory/longitudinal.py — Longitudinal Multi-Session Memory Subsystem

Provides cross-session progression tracking, practice streak computation,
pose hold duration tracking, adaptive difficulty scaling, and persistent
injury/contraindication watchlists as specified in PLAN.md Phase 3A.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, date, UTC
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.workflows.event_driven import (
    BaseEvent,
    FeedbackReceivedEvent,
    SafetyEscalationEvent,
    SessionCompletedEvent,
    get_event_bus,
)

logger = logging.getLogger("yogasutra.memory.longitudinal")

DEFAULT_MEMORY_DIR = Path("data/memory/longitudinal")


class SessionRecord(BaseModel):
    """Historical snapshot of a single completed practice session."""

    session_id: str
    date_iso: str
    duration_minutes: int
    plan_goal: str
    completed_poses: list[str] = Field(default_factory=list)
    difficulty_rating: int | None = None
    fatigue_level: str | None = None
    notes: str = ""


class UserTrajectory(BaseModel):
    """Longitudinal progression profile for an individual practitioner."""

    user_id: str
    total_sessions_completed: int = 0
    total_minutes_practiced: int = 0
    current_streak_days: int = 0
    longest_streak_days: int = 0
    last_practice_date: str | None = None  # YYYY-MM-DD

    # Progression tracking
    pose_hold_durations_sec: dict[str, list[int]] = Field(default_factory=dict)
    pose_frequency: dict[str, int] = Field(default_factory=dict)
    average_difficulty_rating: float = 3.0

    # Safety & personalization watchlist
    active_contraindications: list[str] = Field(default_factory=list)
    injury_history: list[str] = Field(default_factory=list)
    suggested_level_override: str | None = None

    # History
    session_history: list[SessionRecord] = Field(default_factory=list)


class LongitudinalMemoryStore:
    """Persistent store for user multi-session trajectories."""

    def __init__(self, storage_dir: Path | str = DEFAULT_MEMORY_DIR) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, user_id: str) -> Path:
        sanitized = "".join(c for c in user_id if c.isalnum() or c in ("-", "_"))
        return self.storage_dir / f"{sanitized}.json"

    def get_trajectory(self, user_id: str) -> UserTrajectory:
        """Load the user's trajectory or initialize a new profile."""
        path = self._file_path(user_id)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return UserTrajectory.model_validate(data)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load trajectory for %s: %s", user_id, exc)
        return UserTrajectory(user_id=user_id)

    def save_trajectory(self, trajectory: UserTrajectory) -> None:
        """Persist user trajectory to disk."""
        path = self._file_path(trajectory.user_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trajectory.model_dump(), f, indent=2)

    def record_session(
        self,
        user_id: str,
        session_id: str,
        duration_minutes: int,
        plan_goal: str,
        completed_poses: list[str],
        hold_durations: dict[str, int] | None = None,
    ) -> UserTrajectory:
        """Update user trajectory with a completed session and streak calculation."""
        trajectory = self.get_trajectory(user_id)
        today_str = date.today().isoformat()

        # Update streak
        if trajectory.last_practice_date:
            try:
                last_date = date.fromisoformat(trajectory.last_practice_date)
                today = date.today()
                diff = (today - last_date).days
                if diff == 1:
                    trajectory.current_streak_days += 1
                elif diff > 1:
                    trajectory.current_streak_days = 1
                # if diff == 0, already practiced today, keep current streak
            except Exception:  # noqa: BLE001
                trajectory.current_streak_days = 1
        else:
            trajectory.current_streak_days = 1

        if trajectory.current_streak_days > trajectory.longest_streak_days:
            trajectory.longest_streak_days = trajectory.current_streak_days

        trajectory.last_practice_date = today_str
        trajectory.total_sessions_completed += 1
        trajectory.total_minutes_practiced += duration_minutes

        # Pose frequency & hold durations
        for pose in completed_poses:
            trajectory.pose_frequency[pose] = trajectory.pose_frequency.get(pose, 0) + 1

        if hold_durations:
            for pose, sec in hold_durations.items():
                if pose not in trajectory.pose_hold_durations_sec:
                    trajectory.pose_hold_durations_sec[pose] = []
                trajectory.pose_hold_durations_sec[pose].append(sec)

        # Append record
        rec = SessionRecord(
            session_id=session_id,
            date_iso=datetime.now(UTC).isoformat(),
            duration_minutes=duration_minutes,
            plan_goal=plan_goal,
            completed_poses=completed_poses,
        )
        trajectory.session_history.append(rec)
        self.save_trajectory(trajectory)
        logger.info(
            "Recorded session for %s: %d total sessions, %d-day streak",
            user_id,
            trajectory.total_sessions_completed,
            trajectory.current_streak_days,
        )
        return trajectory

    def record_feedback(
        self,
        user_id: str,
        session_id: str,
        difficulty_rating: int,
        fatigue_level: str,
        comments: str = "",
    ) -> UserTrajectory:
        """Update trajectory with post-practice feedback and recompute difficulty trends."""
        trajectory = self.get_trajectory(user_id)

        # Update last session record if matches
        for rec in reversed(trajectory.session_history):
            if rec.session_id == session_id or not rec.difficulty_rating:
                rec.difficulty_rating = difficulty_rating
                rec.fatigue_level = fatigue_level
                rec.notes = comments
                break

        # Recompute average difficulty
        ratings = [r.difficulty_rating for r in trajectory.session_history if r.difficulty_rating is not None]
        if ratings:
            trajectory.average_difficulty_rating = round(sum(ratings) / len(ratings), 2)

        # Adaptive level recommendation
        if len(ratings) >= 3:
            recent_avg = sum(ratings[-3:]) / 3.0
            if recent_avg <= 1.8 and trajectory.total_sessions_completed >= 5:
                trajectory.suggested_level_override = "intermediate"
            elif recent_avg >= 4.2:
                trajectory.suggested_level_override = "beginner"

        self.save_trajectory(trajectory)
        return trajectory

    def record_safety_alert(self, user_id: str, condition: str) -> UserTrajectory:
        """Add reported condition or injury to user active watchlist."""
        trajectory = self.get_trajectory(user_id)
        if condition not in trajectory.active_contraindications:
            trajectory.active_contraindications.append(condition)
        if condition not in trajectory.injury_history:
            trajectory.injury_history.append(condition)
        self.save_trajectory(trajectory)
        return trajectory

    def get_adaptive_profile_adjustments(self, user_id: str) -> dict[str, Any]:
        """Compute personalization adjustments to merge into the user profile."""
        trajectory = self.get_trajectory(user_id)
        adjustments: dict[str, Any] = {
            "streak_days": trajectory.current_streak_days,
            "total_sessions": trajectory.total_sessions_completed,
            "active_contraindications": list(trajectory.active_contraindications),
            "suggested_level": trajectory.suggested_level_override,
            "average_difficulty": trajectory.average_difficulty_rating,
        }
        return adjustments


# ---------------------------------------------------------------------------
# Singleton & EventBus Wire-Up
# ---------------------------------------------------------------------------

_GLOBAL_LONGITUDINAL_STORE: LongitudinalMemoryStore | None = None


def get_longitudinal_store() -> LongitudinalMemoryStore:
    """Retrieve global LongitudinalMemoryStore instance."""
    global _GLOBAL_LONGITUDINAL_STORE
    if _GLOBAL_LONGITUDINAL_STORE is None:
        _GLOBAL_LONGITUDINAL_STORE = LongitudinalMemoryStore()
        _wire_event_subscribers(_GLOBAL_LONGITUDINAL_STORE)
    return _GLOBAL_LONGITUDINAL_STORE


def _wire_event_subscribers(store: LongitudinalMemoryStore) -> None:
    """Connect LongitudinalMemoryStore to EventBus for automatic reactive updates."""
    bus = get_event_bus()

    def on_session_completed(event: BaseEvent) -> None:
        if isinstance(event, SessionCompletedEvent):
            existing = store.get_trajectory(event.user_id)
            if any(rec.session_id == event.session_id for rec in existing.session_history):
                return
            store.record_session(
                user_id=event.user_id,
                session_id=event.session_id,
                duration_minutes=event.duration_minutes,
                plan_goal=event.plan_goal,
                completed_poses=event.completed_poses,
            )

    def on_feedback_received(event: BaseEvent) -> None:
        if isinstance(event, FeedbackReceivedEvent):
            store.record_feedback(
                user_id=event.user_id,
                session_id=event.session_id,
                difficulty_rating=event.difficulty_rating,
                fatigue_level=event.fatigue_level,
                comments=event.comments,
            )

    def on_safety_escalation(event: BaseEvent) -> None:
        if isinstance(event, SafetyEscalationEvent):
            store.record_safety_alert(
                user_id=event.user_id,
                condition=event.trigger_condition,
            )

    bus.subscribe("SessionCompleted", on_session_completed)
    bus.subscribe("FeedbackReceived", on_feedback_received)
    bus.subscribe("SafetyEscalation", on_safety_escalation)


# ---------------------------------------------------------------------------
# Pipeline Integration Helpers
# ---------------------------------------------------------------------------


def _parse_state_dict(value: Any) -> dict[str, Any]:
    """Safely parse a dict or (possibly markdown-fenced) JSON string into a dict."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    cleaned = value.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def build_longitudinal_context(
    user_id: str,
    store: LongitudinalMemoryStore | None = None,
) -> dict[str, Any]:
    """Build a concise longitudinal history summary for ProfileAgent and session state."""
    active_store = store or get_longitudinal_store()
    trajectory = active_store.get_trajectory(user_id)

    recent_goals = [
        rec.plan_goal
        for rec in trajectory.session_history[-3:]
        if rec.plan_goal
    ]
    frequent_poses = [
        pose
        for pose, _ in sorted(
            trajectory.pose_frequency.items(),
            key=lambda item: (-item[1], item[0]),
        )[:5]
    ]

    return {
        "user_id": trajectory.user_id,
        "total_sessions_completed": trajectory.total_sessions_completed,
        "current_streak_days": trajectory.current_streak_days,
        "active_contraindications": list(trajectory.active_contraindications),
        "injury_history": list(trajectory.injury_history),
        "suggested_level_override": trajectory.suggested_level_override,
        "recent_goals": recent_goals,
        "frequent_poses": frequent_poses,
    }


def record_pipeline_completion(
    user_id: str,
    session_id: str,
    session_state: dict[str, Any],
    store: LongitudinalMemoryStore | None = None,
) -> UserTrajectory:
    """Persist completed pipeline session and newly extracted user constraints."""
    active_store = store or get_longitudinal_store()
    profile_dict = _parse_state_dict(session_state.get("user_profile"))
    plan_dict = _parse_state_dict(session_state.get("candidate_plan"))

    # Extract pose names from candidate_plan.sequence
    completed_poses: list[str] = []
    raw_sequence = plan_dict.get("sequence", [])
    if isinstance(raw_sequence, list):
        for item in raw_sequence:
            if isinstance(item, dict):
                name = item.get("name") or item.get("pose_name") or item.get("pose")
                if isinstance(name, str) and name.strip():
                    completed_poses.append(name.strip())
            elif isinstance(item, str) and item.strip():
                completed_poses.append(item.strip())

    # Extract duration_minutes safely
    raw_duration = plan_dict.get("duration_minutes")
    if raw_duration is None:
        raw_duration = plan_dict.get("total_duration_minutes", 0)
    try:
        duration_minutes = max(0, int(raw_duration))
    except (ValueError, TypeError):
        duration_minutes = 0

    plan_goal = str(
        plan_dict.get("goal")
        or profile_dict.get("goal")
        or "general"
    )

    if completed_poses or duration_minutes > 0:
        trajectory = active_store.record_session(
            user_id=user_id,
            session_id=session_id,
            duration_minutes=duration_minutes,
            plan_goal=plan_goal,
            completed_poses=completed_poses,
        )
    else:
        trajectory = active_store.get_trajectory(user_id)

    # Merge newly extracted constraints into active_contraindications (deduplicated)
    raw_constraints = profile_dict.get("constraints", [])
    updated_contraindications = False
    if isinstance(raw_constraints, list):
        existing_lower = {c.lower() for c in trajectory.active_contraindications}
        for constraint in raw_constraints:
            if isinstance(constraint, str):
                cleaned = constraint.strip()
                if cleaned and cleaned.lower() not in existing_lower:
                    trajectory.active_contraindications.append(cleaned)
                    existing_lower.add(cleaned.lower())
                    updated_contraindications = True

    if updated_contraindications:
        active_store.save_trajectory(trajectory)

    return trajectory
