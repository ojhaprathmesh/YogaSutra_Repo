#!/usr/bin/env python3
"""
scripts/run_pose_detector.py — Interactive Standalone Posture Detector & Telemetry HUD Runner

Phase 3B: Manual verification CLI for the computer vision biomechanics pipeline.
Provides a live visual feed with:
- 33-point MediaPipe skeletal overlay with color-coded joint status.
- Real-time Telemetry HUD: FPS, inference latency (ms), and active RAM consumption (MB).
- Dynamic asana alignment audit against classical criteria matrix.
- Closed-loop speech cue dispatch simulation.

Usage:
    # Run with default webcam (index 0) on Warrior II
    python3 scripts/run_pose_detector.py

    # Run on a static image
    python3 scripts/run_pose_detector.py --image path/to/pose.jpg --asana "Tadasana"

    # Run headless (for CI/automated test verification)
    python3 scripts/run_pose_detector.py --image path/to/pose.jpg --headless
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import psutil

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.pose.agent import PoseAgent
from app.vision.criteria import ASANA_CRITERIA_REGISTRY, normalize_asana_name
from app.vision.detector import PoseDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YogaSutra — Real-Time Posture Detection & Biomechanical Telemetry HUD",
    )
    parser.add_argument("--webcam", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--image", type=str, default=None, help="Path to static image file")
    parser.add_argument("--video", type=str, default=None, help="Path to recorded video file")
    parser.add_argument("--asana", type=str, default="Virabhadrasana II", help="Target asana to verify")
    parser.add_argument("--lead-leg", choices=["left", "right"], default="left", help="Lead leg for asymmetric poses")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without cv2.imshow GUI")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process before exiting")
    return parser.parse_args()


def draw_telemetry_hud(
    frame: np.ndarray,
    fps: float,
    latency_ms: float,
    ram_mb: float,
    asana_name: str,
    lead_leg: str,
    deviations: list,
    active_cue: str = "",
) -> np.ndarray:
    """Render high-contrast, professional telemetry HUD overlay."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Semi-transparent HUD background card (top-left)
    overlay = out.copy()
    cv2.rectangle(overlay, (10, 10), (320, 180), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
    cv2.rectangle(out, (10, 10), (320, 180), (80, 80, 90), 1)

    # Status check
    has_misalignment = any(not d.is_aligned for d in deviations)
    status_text = "ALIGNMENT NEEDED" if has_misalignment else "ALIGNED"
    status_color = (0, 70, 255) if has_misalignment else (0, 230, 70)  # Red vs Green

    # Telemetry metrics
    cv2.putText(out, f"YOGASUTRA CV ENGINE (480p)", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 200), 1, cv2.LINE_AA)
    cv2.putText(out, f"Pose: {asana_name} ({lead_leg})", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"Status: {status_text}", (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)

    # Performance vitals
    cv2.putText(out, f"FPS: {fps:.1f} | Latency: {latency_ms:.1f} ms", (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(out, f"RAM: {ram_mb:.1f} MB (Target < 100 MB)", (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    # Key Joint Readouts
    y_offset = 140
    for dev in deviations[:2]:
        j_color = (0, 230, 70) if dev.is_aligned else (0, 70, 255)
        text = f"{dev.joint_name}: {dev.measured_angle:.0f} deg [{dev.target_min:.0f}-{dev.target_max:.0f}]"
        cv2.putText(out, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, j_color, 1, cv2.LINE_AA)
        y_offset += 18

    # Bottom Spoken Cue Ticker (if corrective cue is active)
    if active_cue:
        cue_overlay = out.copy()
        cv2.rectangle(cue_overlay, (10, h - 55), (w - 10, h - 10), (15, 25, 45), -1)
        cv2.addWeighted(cue_overlay, 0.85, out, 0.15, 0, out)
        cv2.rectangle(out, (10, h - 55), (w - 10, h - 10), (0, 180, 255), 1)
        cv2.putText(out, f"[AUDIO CUE]: {active_cue}", (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def main() -> None:
    args = parse_args()
    process = psutil.Process(os.getpid())

    print("\n" + "=" * 65)
    print("🧘 YogaSutra — Standalone Posture Detector & Telemetry HUD")
    print("   Edge Optimization: 480p Ingestion Clamp | Zero-Lag Buffer")
    print(f"   Target Asana: {args.asana} (Lead Leg: {args.lead_leg})")
    print("=" * 65)

    detector = PoseDetector()
    pose_agent = PoseAgent()

    # Determine input source
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Error: Image file '{args.image}' not found.")
            sys.exit(1)
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"❌ Error: Could not decode image '{args.image}'.")
            sys.exit(1)

        t0 = time.perf_counter()
        snapshot = detector.extract_biomechanics(frame)
        latency_ms = (time.perf_counter() - t0) * 1000
        deviations, events = pose_agent.evaluate_snapshot(snapshot, args.asana, lead_leg=args.lead_leg)
        ram_mb = process.memory_info().rss / 1024 / 1024

        rendered = detector.draw_skeleton(frame, snapshot, deviations)
        cue = events[0].correction_cue if events else (deviations[0].correction_cue if deviations and not deviations[0].is_aligned else "")
        final_hud = draw_telemetry_hud(rendered, fps=0.0, latency_ms=latency_ms, ram_mb=ram_mb, asana_name=args.asana, lead_leg=args.lead_leg, deviations=deviations, active_cue=cue)

        print(f"\n📊 Image Analysis Summary:")
        print(f"   Inference Latency: {latency_ms:.2f} ms")
        print(f"   Active RAM:        {ram_mb:.1f} MB")
        print(f"   Detected Joints:   {len(snapshot.landmarks)} landmarks")
        for dev in deviations:
            mark = "✅" if dev.is_aligned else "⚠️"
            print(f"   {mark} {dev.joint_name}: {dev.measured_angle:.1f}° (Target: {dev.target_min:.0f}°–{dev.target_max:.0f}°) — Severity: {dev.severity}")
            if not dev.is_aligned:
                print(f"      Cue: \"{dev.correction_cue}\"")

        if not args.headless:
            cv2.imshow("YogaSutra Pose Detector", final_hud)
            print("\nPress any key in image window to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # Video or Live Webcam stream
    source = args.video if args.video else args.webcam
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video capture source: {source}")
        sys.exit(1)

    fps_history = deque(maxlen=20)
    active_cue = ""
    active_asana = args.asana
    lead_leg = args.lead_leg
    frame_count = 0

    print("\n▶️ Starting live posture detection loop (Press 'q' or ESC to quit)...")
    print("   Shortcuts: [1] Warrior II | [2] Mountain | [3] Tree | [4] Chair | [5] Cobra | [l] Flip Leg\n")

    try:
        while cap.isOpened():
            t_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break

            # 1. Immediate 480p Ingestion Clamp
            clamped = detector.clamp_frame(frame)

            # 2. 3D Pose Extraction & Biomechanical Calculation
            t_infer = time.perf_counter()
            snapshot = detector.extract_biomechanics(clamped)
            latency_ms = (time.perf_counter() - t_infer) * 1000

            # 3. Posture Evaluation & Debounce
            deviations, events = pose_agent.evaluate_snapshot(snapshot, active_asana, lead_leg=lead_leg)
            if events:
                active_cue = events[0].correction_cue

            # 4. Rendering & Telemetry HUD
            t_end = time.perf_counter()
            fps_history.append(1.0 / max(1e-4, t_end - t_start))
            current_fps = sum(fps_history) / len(fps_history)
            ram_mb = process.memory_info().rss / 1024 / 1024

            rendered = detector.draw_skeleton(clamped, snapshot, deviations)
            final_frame = draw_telemetry_hud(
                rendered,
                fps=current_fps,
                latency_ms=latency_ms,
                ram_mb=ram_mb,
                asana_name=active_asana,
                lead_leg=lead_leg,
                deviations=deviations,
                active_cue=active_cue,
            )

            if not args.headless:
                cv2.imshow("YogaSutra Pose Detector", final_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or ESC
                    break
                elif key == ord("1"):
                    active_asana = "Virabhadrasana II"
                    pose_agent.reset_state()
                elif key == ord("2"):
                    active_asana = "Tadasana"
                    pose_agent.reset_state()
                elif key == ord("3"):
                    active_asana = "Vrikshasana"
                    pose_agent.reset_state()
                elif key == ord("4"):
                    active_asana = "Utkatasana"
                    pose_agent.reset_state()
                elif key == ord("5"):
                    active_asana = "Bhujangasana"
                    pose_agent.reset_state()
                elif key == ord("l"):
                    lead_leg = "right" if lead_leg == "left" else "left"
                    pose_agent.reset_state()

    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print("\nSession completed cleanly.")


if __name__ == "__main__":
    main()
