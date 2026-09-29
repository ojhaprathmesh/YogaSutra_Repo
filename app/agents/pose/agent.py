"""
app/agents/pose/agent.py — Pose Understanding & Real-Time Alignment Correction Agent

Phase 3B: Connects the deterministic 3D biomechanics engine to the YogaSutra MAS.
- Evaluates live joint angles against classical asana criteria.
- Implements a 1.5s temporal debounce filter to eliminate sensor jitter and false alarms.
- Dispatches PostureDeviationEvent into the EventBus for closed-loop SpeechAgent voice guidance.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Literal

from google.adk.agents import LlmAgent

from app.vision.biomechanics import BiomechanicalSnapshot
from app.vision.criteria import (
    PostureDeviation,
    evaluate_pose_against_criteria,
    normalize_asana_name,
)
from app.workflows.event_driven import PostureDeviationEvent, get_event_bus

logger = logging.getLogger("yogasutra.agents.pose")

_MODEL = os.environ.get("YOGASUTRA_MODEL", "gemini-2.5-flash")

# Debounce configuration (PLAN.md §31 specification)
DEFAULT_DEBOUNCE_SEC = 1.5
ALERT_COOLDOWN_SEC = 4.0


class PoseAgent:
    """
    Stateful real-time pose evaluation agent with temporal debounce filtering.
    """

    def __init__(
        self,
        debounce_seconds: float = DEFAULT_DEBOUNCE_SEC,
        alert_cooldown_seconds: float = ALERT_COOLDOWN_SEC,
        user_id: str = "P01",
    ) -> None:
        self.debounce_seconds = debounce_seconds
        self.alert_cooldown_seconds = alert_cooldown_seconds
        self.user_id = user_id

        # Joint tracking states:
        # { joint_key: timestamp_when_first_detected }
        self._deviation_start_times: dict[str, float] = {}
        # { joint_key: timestamp_when_last_alert_fired }
        self._last_alert_times: dict[str, float] = {}

        self.event_bus = get_event_bus()

    def reset_state(self) -> None:
        """Reset temporal tracking buffers (e.g. when changing asanas)."""
        self._deviation_start_times.clear()
        self._last_alert_times.clear()

    def evaluate_snapshot(
        self,
        snapshot: BiomechanicalSnapshot,
        asana_name: str,
        lead_leg: Literal["left", "right"] = "left",
        current_time: float | None = None,
    ) -> tuple[list[PostureDeviation], list[PostureDeviationEvent]]:
        """
        Evaluate a biomechanical snapshot against asana criteria with temporal debounce.
        
        Returns:
            (all_deviations, fired_events)
        """
        now = time.time() if current_time is None else current_time
        canonical_pose = normalize_asana_name(asana_name)
        deviations = evaluate_pose_against_criteria(canonical_pose, snapshot, lead_leg=lead_leg)

        fired_events: list[PostureDeviationEvent] = []
        currently_deviated_joints: set[str] = set()

        for dev in deviations:
            joint_key = f"{canonical_pose}:{dev.joint_name}"

            if not dev.is_aligned:
                currently_deviated_joints.add(joint_key)

                # Record first detection if not already tracking
                if joint_key not in self._deviation_start_times:
                    self._deviation_start_times[joint_key] = now

                duration_persisted = now - self._deviation_start_times[joint_key]
                time_since_last_alert = now - self._last_alert_times.get(joint_key, 0.0)

                # Temporal debounce condition:
                # Must persist >= debounce_seconds AND satisfy cooldown
                if duration_persisted >= self.debounce_seconds and time_since_last_alert >= self.alert_cooldown_seconds:
                    event = PostureDeviationEvent(
                        user_id=self.user_id,
                        pose_name=asana_name,
                        joint_name=dev.joint_name,
                        measured_angle=dev.measured_angle,
                        target_angle_range=[dev.target_min, dev.target_max],
                        deviation_deg=dev.deviation_deg,
                        severity=dev.severity,
                        correction_cue=dev.correction_cue,
                    )
                    self._last_alert_times[joint_key] = now
                    fired_events.append(event)

                    # Publish to reactive EventBus for SpeechAgent consumption
                    try:
                        self.event_bus.publish_sync(event)
                        logger.info("Published PostureDeviationEvent for %s: %s", dev.joint_name, dev.correction_cue)
                    except Exception as e:
                        logger.warning("Could not publish PostureDeviationEvent to EventBus: %s", e)
            else:
                # Joint is aligned: clear tracking
                self._deviation_start_times.pop(joint_key, None)

        # Clear any tracked joints that are no longer reported
        for tracked_key in list(self._deviation_start_times.keys()):
            if tracked_key not in currently_deviated_joints:
                del self._deviation_start_times[tracked_key]

        return deviations, fired_events


# ---------------------------------------------------------------------------
# ADK Agent Definition (for MAS Planner & Blackboard Integration)
# ---------------------------------------------------------------------------

POSE_INSTRUCTION = """You are the Pose Understanding & Biomechanical Alignment Agent for YogaSutra.

Your role:
1. Ground posture feedback in physical 3D vector geometry and classical criteria tables.
2. Emphasize joint safety, spinal elongation, and stable foundation over extreme flexibility.
3. Formulate concise, encouraging corrective voice cues under 15 words.
4. When joint misalignment occurs, explain both the physical adjustment and the safety rationale.
"""

pose_agent = LlmAgent(
    name="PoseAgent",
    model=_MODEL,
    description="Biomechanical specialist verifying joint alignment and formulating physical correction cues.",
    instruction=POSE_INSTRUCTION,
)
