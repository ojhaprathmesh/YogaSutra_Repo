"""
tests/test_pose_agent.py — Unit Tests for Pose Understanding & Alignment Agent

Phase 3B: Tests 1.5s temporal debounce filter, closed-loop event emission, and EventBus wiring.
"""
import pytest

from app.agents.pose.agent import PoseAgent
from app.vision.biomechanics import BiomechanicalSnapshot
from app.workflows.event_driven import PostureDeviationEvent, get_event_bus


class TestPoseAgentDebounce:
    """Test the 1.5-second temporal debounce filter to avoid sensor jitter false alarms."""

    def test_instant_jitter_does_not_fire_event(self):
        """A deviation persisting for only 0.4s must NOT fire a PostureDeviationEvent."""
        agent = PoseAgent(debounce_seconds=1.5)
        # Front knee at 60 deg (misaligned)
        bad_snapshot = BiomechanicalSnapshot(left_knee_angle=60.0)

        # Frame 1 at t = 0.0s
        deviations, events = agent.evaluate_snapshot(
            bad_snapshot, "Virabhadrasana II", lead_leg="left", current_time=100.0
        )
        assert len(events) == 0  # No event fired yet

        # Frame 2 at t = 0.5s (duration 0.5s < 1.5s)
        deviations, events = agent.evaluate_snapshot(
            bad_snapshot, "Virabhadrasana II", lead_leg="left", current_time=100.5
        )
        assert len(events) == 0  # Still debouncing

    def test_sustained_deviation_fires_event_at_1_5s(self):
        """A deviation persisting for >= 1.5s triggers a PostureDeviationEvent."""
        bus = get_event_bus()
        received_bus_events: list[PostureDeviationEvent] = []

        def on_deviation(event: PostureDeviationEvent) -> None:
            received_bus_events.append(event)

        bus.subscribe("PostureDeviation", on_deviation)

        agent = PoseAgent(debounce_seconds=1.5, alert_cooldown_seconds=4.0)
        bad_snapshot = BiomechanicalSnapshot(left_knee_angle=60.0)

        # t = 0.0s
        agent.evaluate_snapshot(bad_snapshot, "Virabhadrasana II", lead_leg="left", current_time=200.0)
        # t = 1.0s
        agent.evaluate_snapshot(bad_snapshot, "Virabhadrasana II", lead_leg="left", current_time=201.0)
        # t = 1.6s -> threshold (1.5s) passed!
        deviations, events = agent.evaluate_snapshot(
            bad_snapshot, "Virabhadrasana II", lead_leg="left", current_time=201.6
        )

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, PostureDeviationEvent)
        assert event.joint_name == "lead_knee"
        assert event.measured_angle == 60.0
        assert "Deepen your front knee bend" in event.correction_cue

        # Verify EventBus publication
        assert len(received_bus_events) >= 1
        assert received_bus_events[-1].joint_name == "lead_knee"

    def test_cooldown_prevents_spamming_events(self):
        """After firing, the agent must wait through the cooldown period before re-alerting."""
        agent = PoseAgent(debounce_seconds=1.5, alert_cooldown_seconds=4.0)
        bad_snapshot = BiomechanicalSnapshot(left_knee_angle=60.0)

        # t = 0.0s
        agent.evaluate_snapshot(bad_snapshot, "Virabhadrasana II", current_time=300.0)
        # t = 1.6s -> fires first event
        _, events1 = agent.evaluate_snapshot(bad_snapshot, "Virabhadrasana II", current_time=301.6)
        assert len(events1) == 1

        # t = 2.5s (within 4.0s cooldown) -> must NOT fire again
        _, events2 = agent.evaluate_snapshot(bad_snapshot, "Virabhadrasana II", current_time=302.5)
        assert len(events2) == 0

        # t = 6.0s (after cooldown has elapsed: 6.0 - 1.6 = 4.4s >= 4.0s) -> fires follow-up
        _, events3 = agent.evaluate_snapshot(bad_snapshot, "Virabhadrasana II", current_time=306.0)
        assert len(events3) == 1

    def test_reset_state_clears_timers(self):
        """Switching asanas resets tracking history."""
        agent = PoseAgent(debounce_seconds=1.5)
        bad_snapshot = BiomechanicalSnapshot(left_knee_angle=60.0)

        agent.evaluate_snapshot(bad_snapshot, "Virabhadrasana II", current_time=400.0)
        agent.reset_state()

        # After reset, duration starts from 0 again
        _, events = agent.evaluate_snapshot(bad_snapshot, "Virabhadrasana II", current_time=401.6)
        assert len(events) == 0
