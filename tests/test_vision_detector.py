"""
tests/test_vision_detector.py — Unit Tests for Vision Frame Ingestion & Pose Detector

Phase 3B: Tests 480p ingestion clamping, MediaPipe landmark extraction, and skeleton rendering.
"""
import numpy as np

from app.vision.biomechanics import BiomechanicalSnapshot, Point3D, LEFT_KNEE, RIGHT_KNEE
from app.vision.criteria import PostureDeviation
from app.vision.detector import PoseDetector, BoundedVideoWorker


class TestPoseDetectorIngestion:
    """Test frame clamping and edge optimizations."""

    def test_ingestion_clamping_4k_and_1080p(self):
        """High-resolution camera frames must be clamped to 480p (<= 640x480)."""
        detector = PoseDetector()

        # Synthetic 1080p frame (1920x1080)
        hd_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        clamped_hd = detector.clamp_frame(hd_frame)
        h, w = clamped_hd.shape[:2]
        assert w <= 640
        assert h <= 480

        # Synthetic 4K frame (3840x2160)
        four_k = np.zeros((2160, 3840, 3), dtype=np.uint8)
        clamped_4k = detector.clamp_frame(four_k)
        h, w = clamped_4k.shape[:2]
        assert w <= 640
        assert h <= 480

    def test_native_480p_not_enlarged(self):
        """Standard 640x480 frame remains at native resolution."""
        detector = PoseDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        clamped = detector.clamp_frame(frame)
        assert clamped.shape == (480, 640, 3)

    def test_extract_biomechanics_returns_valid_snapshot(self):
        """Extracting biomechanics from an image returns a valid BiomechanicalSnapshot object."""
        detector = PoseDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        snapshot = detector.extract_biomechanics(frame)
        assert isinstance(snapshot, BiomechanicalSnapshot)
        assert hasattr(snapshot, "left_knee_angle")
        assert hasattr(snapshot, "spine_verticality_deg")

    def test_draw_skeleton_rendering(self):
        """draw_skeleton overlays bones and circles on frame without shape mutation."""
        detector = PoseDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = {
            LEFT_KNEE: Point3D(0.4, 0.6, 0.0, visibility=0.9),
            RIGHT_KNEE: Point3D(0.6, 0.6, 0.0, visibility=0.9),
        }
        snapshot = BiomechanicalSnapshot(landmarks=landmarks)
        deviations = [
            PostureDeviation(
                joint_name="left_knee",
                measured_angle=70.0,
                target_min=85.0,
                target_max=95.0,
                deviation_deg=15.0,
                severity="moderate",
                correction_cue="Bend deeper",
                is_aligned=False,
            )
        ]
        rendered = detector.draw_skeleton(frame, snapshot, deviations)
        assert rendered.shape == frame.shape
        # Should have drawn non-zero colored pixels on the blank canvas
        assert np.any(rendered > 0)


class TestBoundedBufferQueue:
    """Test Queue(maxsize=1) dropping stale frames for zero-lag performance."""

    def test_bounded_worker_initialization(self):
        worker = BoundedVideoWorker(capture_source=0)
        assert worker.frame_queue.maxsize == 1
        assert not worker.running
