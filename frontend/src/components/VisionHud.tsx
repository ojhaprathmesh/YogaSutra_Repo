"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Camera,
  CameraOff,
  CheckCircle2,
  FlaskConical,
  Radio,
  Volume2,
  VolumeX,
  Zap,
} from "lucide-react";
import {
  API_BASE_URL,
  getWebSocketUrl,
  publishEvent,
  type FiredCue,
  type LandmarkPoint,
  type PoseDeviation,
  type SSEEventPayload,
  type VisionTelemetry,
} from "@/lib/api";
import { playSpokenCue, stopSpokenAudio } from "@/lib/audio";

export const SUPPORTED_ASANAS = [
  { value: "Virabhadrasana II", label: "Virabhadrasana II (Warrior II)" },
  { value: "Tadasana", label: "Tadasana (Mountain Pose)" },
  { value: "Vrikshasana", label: "Vrikshasana (Tree Pose)" },
  { value: "Utkatasana", label: "Utkatasana (Chair Pose)" },
  { value: "Bhujangasana", label: "Bhujangasana (Cobra Pose)" },
] as const;

export type SupportedAsana = (typeof SUPPORTED_ASANAS)[number]["value"];

// MediaPipe Pose 33-point skeleton topology
const POSE_CONNECTIONS: Array<[number, number]> = [
  // Face
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
  // Torso
  [11, 12], [11, 23], [12, 24], [23, 24],
  // Left arm
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],
  // Right arm
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22],
  // Left leg
  [23, 25], [25, 27], [27, 29], [29, 31], [27, 31],
  // Right leg
  [24, 26], [26, 28], [28, 30], [30, 32], [28, 32],
];

const SEVERITY_COLORS: Record<string, string> = {
  aligned: "#10b981",
  mild: "#f59e0b",
  moderate: "#f59e0b",
  severe: "#ef4444",
  critical: "#ef4444",
};

function mapJointToLandmarkIndices(
  jointName: string,
  leadLeg: "left" | "right"
): number[] {
  const j = jointName.toLowerCase();
  if (j.includes("left_knee")) return [25];
  if (j.includes("right_knee")) return [26];
  if (j.includes("front_knee")) return leadLeg === "left" ? [25] : [26];
  if (j.includes("back_knee")) return leadLeg === "left" ? [26] : [25];
  if (j.includes("knee")) return [25, 26];

  if (j.includes("left_hip")) return [23];
  if (j.includes("right_hip")) return [24];
  if (j.includes("front_hip")) return leadLeg === "left" ? [23] : [24];
  if (j.includes("hip")) return [23, 24];

  if (j.includes("left_shoulder")) return [11];
  if (j.includes("right_shoulder")) return [12];
  if (j.includes("shoulder")) return [11, 12];

  if (j.includes("left_elbow")) return [13];
  if (j.includes("right_elbow")) return [14];
  if (j.includes("elbow")) return [13, 14];

  if (j.includes("left_ankle")) return [27];
  if (j.includes("right_ankle")) return [28];
  if (j.includes("ankle")) return [27, 28];

  if (j.includes("spine") || j.includes("torso") || j.includes("back")) {
    return [11, 12, 23, 24];
  }
  return [];
}

function getSeverityColor(severity: string, isAligned: boolean): string {
  if (isAligned) return SEVERITY_COLORS.aligned;
  return SEVERITY_COLORS[severity.toLowerCase()] ?? "#f59e0b";
}

interface CueBannerState {
  text: string;
  severity: string;
  joint: string;
  provider?: string;
  timestamp: string;
}

interface VisionHudProps {
  asana: SupportedAsana;
  onAsanaChange: (asana: SupportedAsana) => void;
  leadLeg: "left" | "right";
  onLeadLegChange: (leg: "left" | "right") => void;
  personaId: string;
}

export function VisionHud({
  asana,
  onAsanaChange,
  leadLeg,
  onLeadLegChange,
  personaId,
}: VisionHudProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const inFlightRef = useRef<boolean>(false);

  const asanaRef = useRef<SupportedAsana>(asana);
  const leadLegRef = useRef<"left" | "right">(leadLeg);
  const autoSpeakRef = useRef<boolean>(true);

  const [cameraActive, setCameraActive] = useState(false);
  const [wsStatus, setWsStatus] = useState<"disconnected" | "connecting" | "connected">("disconnected");
  const [sseConnected, setSseConnected] = useState(false);
  const [autoSpeak, setAutoSpeak] = useState(true);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [simulating, setSimulating] = useState(false);

  const [poseDetected, setPoseDetected] = useState(false);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [angles, setAngles] = useState<Record<string, number>>({});
  const [deviations, setDeviations] = useState<PoseDeviation[]>([]);
  const [activeBanner, setActiveBanner] = useState<CueBannerState | null>(null);
  const [recentAlerts, setRecentAlerts] = useState<CueBannerState[]>([]);

  useEffect(() => {
    asanaRef.current = asana;
  }, [asana]);

  useEffect(() => {
    leadLegRef.current = leadLeg;
  }, [leadLeg]);

  useEffect(() => {
    autoSpeakRef.current = autoSpeak;
  }, [autoSpeak]);

  // Send config update over active WS when asana / leadLeg / personaId changes
  useEffect(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: "config",
          asana,
          lead_leg: leadLeg,
          user_id: personaId,
        })
      );
    }
  }, [asana, leadLeg, personaId]);

  const triggerCueBannerAndAudio = useCallback(
    (cue: {
      text: string;
      severity?: string;
      joint?: string;
      audioUrl?: string | null;
      provider?: string;
    }) => {
      const entry: CueBannerState = {
        text: cue.text,
        severity: cue.severity ?? "moderate",
        joint: cue.joint ?? "alignment",
        provider: cue.provider,
        timestamp: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        }),
      };
      setActiveBanner(entry);
      setRecentAlerts((prev) => [entry, ...prev].slice(0, 6));

      if (autoSpeakRef.current && cue.text) {
        void playSpokenCue({
          text: cue.text,
          audioUrl: cue.audioUrl,
        });
      }
    },
    []
  );

  // Connect to SSE /api/events/stream for live PostureDeviation & SafetyEscalation alerts
  useEffect(() => {
    let es: EventSource | null = null;
    try {
      es = new EventSource(`${API_BASE_URL}/api/events/stream`);
      es.addEventListener("connected", () => {
        setSseConnected(true);
      });
      es.onopen = () => {
        setSseConnected(true);
      };
      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as SSEEventPayload;
          const evtType = data.event_type ?? "";
          if (evtType === "PostureDeviation" || data.correction_cue) {
            triggerCueBannerAndAudio({
              text: data.correction_cue || "Adjust your alignment gently.",
              severity: data.severity || "moderate",
              joint: data.joint_name || data.pose_name || "posture",
              audioUrl: data.audio_url,
              provider: data.provider || "elevenlabs",
            });
          } else if (evtType === "SafetyEscalation" || data.trigger_condition) {
            triggerCueBannerAndAudio({
              text:
                data.trigger_condition ||
                "Safety escalation triggered. Pause and return to neutral posture.",
              severity: data.severity || "critical",
              joint: "safety_guard",
              audioUrl: data.audio_url,
              provider: data.provider,
            });
          }
        } catch {
          // Ignore keepalive or non-JSON SSE frames
        }
      };
      es.onerror = () => {
        setSseConnected(false);
      };
    } catch {
      setSseConnected(false);
    }

    return () => {
      if (es) es.close();
    };
  }, [triggerCueBannerAndAudio]);

  // Draw 33-point MediaPipe skeleton overlay on canvas
  const drawSkeleton = useCallback(
    (landmarks: LandmarkPoint[], currentDeviations: PoseDeviation[]) => {
      const canvas = overlayCanvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video) return;

      const width = video.videoWidth || 640;
      const height = video.videoHeight || 480;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, width, height);
      if (!landmarks.length) return;

      // Build per-landmark severity color map
      const landmarkColorMap = new Map<number, string>();
      for (const dev of currentDeviations) {
        const color = getSeverityColor(dev.severity, dev.is_aligned);
        const indices = mapJointToLandmarkIndices(
          dev.joint_name,
          leadLegRef.current
        );
        for (const idx of indices) {
          // Prioritize worse severities if multiple deviations touch a landmark
          const existing = landmarkColorMap.get(idx);
          if (existing === "#ef4444") continue;
          if (existing === "#f59e0b" && color === "#10b981") continue;
          landmarkColorMap.set(idx, color);
        }
      }

      // Draw bone connections
      ctx.lineCap = "round";
      for (const [startIdx, endIdx] of POSE_CONNECTIONS) {
        const a = landmarks[startIdx];
        const b = landmarks[endIdx];
        if (!a || !b || a.visibility < 0.35 || b.visibility < 0.35) continue;

        const cA = landmarkColorMap.get(startIdx);
        const cB = landmarkColorMap.get(endIdx);
        let strokeColor = "rgba(16, 185, 129, 0.85)"; // emerald default
        if (cA === "#ef4444" || cB === "#ef4444") {
          strokeColor = "rgba(239, 68, 68, 0.92)";
        } else if (cA === "#f59e0b" || cB === "#f59e0b") {
          strokeColor = "rgba(245, 158, 11, 0.92)";
        }

        ctx.beginPath();
        ctx.moveTo(a.x * width, a.y * height);
        ctx.lineTo(b.x * width, b.y * height);
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 3;
        ctx.stroke();
      }

      // Draw 33 landmark joints
      for (let i = 0; i < landmarks.length; i++) {
        const pt = landmarks[i];
        if (!pt || pt.visibility < 0.35) continue;
        const x = pt.x * width;
        const y = pt.y * height;
        const color = landmarkColorMap.get(i) ?? "#10b981";
        const isMajorJoint = [11, 12, 13, 14, 23, 24, 25, 26, 27, 28].includes(i);

        ctx.beginPath();
        ctx.arc(x, y, isMajorJoint ? 5.5 : 3.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.shadowColor = color;
        ctx.shadowBlur = isMajorJoint ? 10 : 4;
        ctx.fill();
        ctx.shadowBlur = 0;

        if (isMajorJoint) {
          ctx.beginPath();
          ctx.arc(x, y, 2, 0, Math.PI * 2);
          ctx.fillStyle = "#ffffff";
          ctx.fill();
        }
      }
    },
    []
  );

  const stopCamera = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    inFlightRef.current = false;

    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    setWsStatus("disconnected");

    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop();
      }
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    const canvas = overlayCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }

    setCameraActive(false);
    setPoseDetected(false);
    setLatencyMs(null);
  }, []);

  useEffect(() => {
    return () => {
      stopCamera();
      stopSpokenAudio();
    };
  }, [stopCamera]);

  const startCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user",
        },
        audio: false,
      });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);

      // Connect WS /ws/vision
      setWsStatus("connecting");
      const ws = new WebSocket(getWebSocketUrl("/ws/vision"));
      wsRef.current = ws;

      ws.onopen = () => {
        setWsStatus("connected");
        ws.send(
          JSON.stringify({
            type: "config",
            asana: asanaRef.current,
            lead_leg: leadLegRef.current,
            user_id: personaId,
          })
        );
      };

      ws.onmessage = (event) => {
        inFlightRef.current = false;
        try {
          const data = JSON.parse(event.data) as VisionTelemetry;
          if (data.type !== "telemetry") return;

          setPoseDetected(Boolean(data.pose_detected));
          if (typeof data.latency_ms === "number") {
            setLatencyMs(data.latency_ms);
          }
          if (data.angles) {
            setAngles(data.angles);
          }
          const nextDeviations = data.deviations ?? [];
          setDeviations(nextDeviations);
          drawSkeleton(data.landmarks ?? [], nextDeviations);

          if (data.fired_cues && data.fired_cues.length > 0) {
            const topCue: FiredCue = data.fired_cues[0];
            triggerCueBannerAndAudio({
              text: topCue.correction_cue,
              severity: topCue.severity,
              joint: topCue.joint_name,
              audioUrl: topCue.audio_url,
              provider: topCue.provider,
            });
          }
        } catch {
          // Ignore malformed WS payloads
        }
      };

      ws.onerror = () => {
        inFlightRef.current = false;
        setWsStatus("disconnected");
        setCameraError("Vision WebSocket connection failed. Ensure FastAPI backend is running.");
      };

      ws.onclose = () => {
        inFlightRef.current = false;
        setWsStatus("disconnected");
      };

      // Capture & send 480p JPEG frames at ~6 FPS (166ms) with in-flight guard
      if (!captureCanvasRef.current) {
        captureCanvasRef.current = document.createElement("canvas");
      }
      timerRef.current = setInterval(() => {
        const video = videoRef.current;
        const captureCanvas = captureCanvasRef.current;
        const activeWs = wsRef.current;
        if (
          !video ||
          !captureCanvas ||
          !activeWs ||
          activeWs.readyState !== WebSocket.OPEN ||
          video.readyState < 2 ||
          inFlightRef.current
        ) {
          return;
        }

        const targetW = Math.min(video.videoWidth || 640, 640);
        const targetH = Math.min(video.videoHeight || 480, 480);
        captureCanvas.width = targetW;
        captureCanvas.height = targetH;

        const ctx = captureCanvas.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(video, 0, 0, targetW, targetH);
        const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.68);

        inFlightRef.current = true;
        activeWs.send(
          JSON.stringify({
            type: "frame",
            image: dataUrl,
            asana: asanaRef.current,
            lead_leg: leadLegRef.current,
          })
        );
      }, 166);
    } catch (err) {
      const msg =
        err instanceof Error
          ? err.message
          : "Unable to access webcam. Check browser permissions.";
      setCameraError(msg);
      stopCamera();
    }
  };

  const handleSimulateDeviation = async () => {
    setSimulating(true);
    try {
      const res = await publishEvent({
        event_type: "PostureDeviation",
        user_id: personaId,
        session_id: "studio-hud-sim",
        payload: {
          pose_name: asana,
          joint_name: `${leadLeg}_knee`,
          measured_angle: 118.0,
          target_angle_range: [85.0, 95.0],
          deviation_deg: 23.0,
          severity: "moderate",
          correction_cue: `In ${asana}, bend your ${leadLeg} front knee slightly deeper toward 90 degrees, stacked directly over your ankle.`,
        },
      });

      const simulatedDev: PoseDeviation = {
        joint_name: `${leadLeg}_knee`,
        measured_angle: 118,
        target_min: 85,
        target_max: 95,
        deviation_deg: 23,
        is_aligned: false,
        severity: "moderate",
        correction_cue:
          res.cue ||
          `In ${asana}, bend your ${leadLeg} front knee slightly deeper toward 90 degrees.`,
      };
      setDeviations([simulatedDev]);
      setAngles((prev) => ({ ...prev, [`${leadLeg}_knee`]: 118 }));

      // If SSE didn't already trigger audio (e.g. SSE disconnected), play directly
      if (!sseConnected) {
        triggerCueBannerAndAudio({
          text: simulatedDev.correction_cue,
          severity: "moderate",
          joint: `${leadLeg}_knee`,
          audioUrl: res.audio_url,
          provider: res.provider || "elevenlabs",
        });
      }
    } catch (err) {
      setCameraError(
        err instanceof Error ? err.message : "Failed to publish simulated deviation."
      );
    } finally {
      setSimulating(false);
    }
  };

  const angleEntries = Object.entries(angles);

  return (
    <section className="studio-card rounded-2xl p-5 flex flex-col gap-4">
      {/* HUD Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="h-8 w-8 rounded-xl bg-cyan-500/15 border border-cyan-400/30 flex items-center justify-center">
            <Activity className="h-4 w-4 text-cyan-300" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-white tracking-tight">
              Live Biomechanics Vision HUD
            </h2>
            <p className="text-xs text-slate-400">
              33-point MediaPipe Pose stream &amp; real-time ElevenLabs coaching
            </p>
          </div>
        </div>

        {/* Telemetry Status Badges */}
        <div className="flex items-center gap-2 flex-wrap">
          <span
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono border ${
              wsStatus === "connected"
                ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
                : wsStatus === "connecting"
                  ? "bg-amber-500/10 border-amber-500/30 text-amber-300"
                  : "bg-white/[0.03] border-white/[0.08] text-slate-400"
            }`}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                wsStatus === "connected"
                  ? "bg-emerald-400 animate-pulse"
                  : wsStatus === "connecting"
                    ? "bg-amber-400"
                    : "bg-slate-600"
              }`}
            />
            WS: {wsStatus}
          </span>

          <span
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono border ${
              sseConnected
                ? "bg-violet-500/10 border-violet-500/30 text-violet-300"
                : "bg-white/[0.03] border-white/[0.08] text-slate-500"
            }`}
          >
            <Radio className="h-3 w-3" />
            SSE {sseConnected ? "Live" : "Idle"}
          </span>

          {latencyMs !== null && (
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-mono bg-cyan-500/10 border border-cyan-500/25 text-cyan-300">
              <Zap className="h-3 w-3" />
              {latencyMs.toFixed(0)} ms
            </span>
          )}
        </div>
      </div>

      {/* Controls Bar */}
      <div className="grid grid-cols-1 sm:grid-cols-12 gap-2.5 items-center bg-white/[0.02] border border-white/[0.06] rounded-xl p-3">
        {/* Target Asana */}
        <div className="sm:col-span-5">
          <label className="block text-[10px] font-mono uppercase tracking-wider text-slate-400 mb-1">
            Target Asana
          </label>
          <select
            value={asana}
            onChange={(e) => onAsanaChange(e.target.value as SupportedAsana)}
            className="w-full bg-[#0d111a] border border-white/[0.1] rounded-lg px-2.5 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-violet-400/60"
          >
            {SUPPORTED_ASANAS.map((a) => (
              <option key={a.value} value={a.value}>
                {a.label}
              </option>
            ))}
          </select>
        </div>

        {/* Lead Leg */}
        <div className="sm:col-span-3">
          <label className="block text-[10px] font-mono uppercase tracking-wider text-slate-400 mb-1">
            Lead Leg
          </label>
          <div className="grid grid-cols-2 gap-1 p-0.5 rounded-lg bg-[#0d111a] border border-white/[0.1]">
            {(["left", "right"] as const).map((leg) => (
              <button
                key={leg}
                type="button"
                onClick={() => onLeadLegChange(leg)}
                className={`py-1 rounded-md text-xs font-medium capitalize transition-colors ${
                  leadLeg === leg
                    ? "bg-violet-500/25 text-violet-100 border border-violet-400/35"
                    : "text-slate-400 hover:text-slate-200 border border-transparent"
                }`}
              >
                {leg}
              </button>
            ))}
          </div>
        </div>

        {/* Auto-Speak Toggle */}
        <div className="sm:col-span-4 flex items-end justify-end gap-2 pt-3 sm:pt-4">
          <button
            type="button"
            onClick={() => {
              const next = !autoSpeak;
              setAutoSpeak(next);
              if (!next) stopSpokenAudio();
            }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              autoSpeak
                ? "bg-emerald-500/15 border-emerald-500/30 text-emerald-200"
                : "bg-white/[0.03] border-white/[0.08] text-slate-400"
            }`}
          >
            {autoSpeak ? (
              <Volume2 className="h-3.5 w-3.5" />
            ) : (
              <VolumeX className="h-3.5 w-3.5" />
            )}
            <span>Auto-Speak {autoSpeak ? "On" : "Off"}</span>
          </button>
        </div>
      </div>

      {/* Video + Skeleton Canvas Stage */}
      <div className="relative aspect-[4/3] w-full rounded-xl overflow-hidden bg-[#06080d] border border-white/[0.08]">
        <video
          ref={videoRef}
          playsInline
          muted
          className="w-full h-full object-cover"
        />
        <canvas
          ref={overlayCanvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />

        {/* Idle Placeholder */}
        {!cameraActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center bg-gradient-to-b from-[#0b0f19]/80 to-[#06080d]/95">
            <div className="h-14 w-14 rounded-2xl bg-violet-500/10 border border-violet-400/25 flex items-center justify-center mb-3">
              <Camera className="h-6 w-6 text-violet-300" />
            </div>
            <p className="text-sm font-medium text-slate-200">
              480p Real-Time Pose Camera Ready
            </p>
            <p className="text-xs text-slate-400 max-w-sm mt-1">
              Start your webcam to stream 6 FPS frames to MediaPipe Pose and receive
              live joint-alignment overlays and ElevenLabs spoken corrections.
            </p>
          </div>
        )}

        {/* Top-left HUD Overlay Pill */}
        {cameraActive && (
          <div className="absolute top-3 left-3 flex items-center gap-2">
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-mono backdrop-blur-md border ${
                poseDetected
                  ? "bg-emerald-950/70 border-emerald-400/35 text-emerald-200"
                  : "bg-slate-950/75 border-white/10 text-slate-300"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  poseDetected ? "bg-emerald-400" : "bg-amber-400 animate-pulse"
                }`}
              />
              {poseDetected ? `Tracking ${asana}` : "Step into frame..."}
            </span>
          </div>
        )}

        {/* Bottom Live Corrective Cue Banner */}
        {activeBanner && (
          <div
            className={`absolute bottom-3 left-3 right-3 rounded-xl px-3.5 py-2.5 backdrop-blur-xl border flex items-start gap-2.5 shadow-lg ${
              activeBanner.severity === "critical" || activeBanner.severity === "severe"
                ? "bg-rose-950/85 border-rose-400/40 text-rose-100"
                : "bg-amber-950/85 border-amber-400/40 text-amber-100"
            }`}
          >
            <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 text-[11px] font-mono uppercase tracking-wider opacity-85">
                <span>{activeBanner.joint.replace(/_/g, " ")}</span>
                <span>•</span>
                <span>{activeBanner.severity}</span>
                {activeBanner.provider && (
                  <>
                    <span>•</span>
                    <span>{activeBanner.provider}</span>
                  </>
                )}
              </div>
              <p className="text-xs font-medium mt-0.5 leading-relaxed">
                {activeBanner.text}
              </p>
            </div>
            <span className="text-[10px] font-mono opacity-75 shrink-0">
              {activeBanner.timestamp}
            </span>
          </div>
        )}
      </div>

      {/* Camera Action Row */}
      <div className="flex flex-wrap items-center justify-between gap-2.5">
        <div className="flex items-center gap-2">
          {!cameraActive ? (
            <button
              type="button"
              onClick={() => void startCamera()}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white shadow-[0_0_20px_rgba(124,58,237,0.3)] transition-all"
            >
              <Camera className="h-3.5 w-3.5" />
              <span>Start Live Camera</span>
            </button>
          ) : (
            <button
              type="button"
              onClick={stopCamera}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-rose-500/20 hover:bg-rose-500/30 border border-rose-500/40 text-rose-200 transition-colors"
            >
              <CameraOff className="h-3.5 w-3.5" />
              <span>Stop Camera</span>
            </button>
          )}
        </div>

        <button
          type="button"
          onClick={() => void handleSimulateDeviation()}
          disabled={simulating}
          className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-medium bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 text-amber-200 transition-colors disabled:opacity-50"
        >
          <FlaskConical className="h-3.5 w-3.5" />
          <span>{simulating ? "Simulating..." : "Simulate Knee Deviation"}</span>
        </button>
      </div>

      {cameraError && (
        <div className="rounded-xl bg-rose-500/10 border border-rose-500/30 px-3.5 py-2.5 text-xs text-rose-200">
          {cameraError}
        </div>
      )}

      {/* Joint Angle Telemetry & Deviation Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {/* Joint Deviations */}
        <div className="rounded-xl bg-white/[0.02] border border-white/[0.06] p-3">
          <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-2">
            Alignment Telemetry ({deviations.length})
          </div>
          {deviations.length === 0 ? (
            <p className="text-xs text-slate-500 py-2">
              No active joint deviations. Start camera or click &ldquo;Simulate Knee Deviation&rdquo;.
            </p>
          ) : (
            <div className="space-y-2">
              {deviations.map((d, idx) => (
                <div
                  key={`${d.joint_name}-${idx}`}
                  className="flex items-center justify-between gap-2 rounded-lg bg-[#0d111a] border border-white/[0.06] px-2.5 py-2"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    {d.is_aligned ? (
                      <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                    ) : (
                      <AlertTriangle
                        className={`h-3.5 w-3.5 shrink-0 ${
                          d.severity === "severe" || d.severity === "critical"
                            ? "text-rose-400"
                            : "text-amber-400"
                        }`}
                      />
                    )}
                    <div className="truncate">
                      <div className="text-xs font-medium text-slate-200 capitalize">
                        {d.joint_name.replace(/_/g, " ")}
                      </div>
                      <div className="text-[10px] font-mono text-slate-400">
                        Target {d.target_min}°–{d.target_max}°
                      </div>
                    </div>
                  </div>
                  <div className="text-right shrink-0">
                    <div className="text-xs font-mono font-semibold text-white">
                      {Math.round(d.measured_angle)}°
                    </div>
                    <div
                      className={`text-[10px] font-mono uppercase ${
                        d.is_aligned
                          ? "text-emerald-400"
                          : d.severity === "severe" || d.severity === "critical"
                            ? "text-rose-400"
                            : "text-amber-400"
                      }`}
                    >
                      {d.is_aligned ? "Aligned" : d.severity}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Raw Joint Angles + Recent Alerts */}
        <div className="rounded-xl bg-white/[0.02] border border-white/[0.06] p-3">
          <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 mb-2">
            Measured Joint Angles
          </div>
          {angleEntries.length === 0 ? (
            <p className="text-xs text-slate-500 py-2">
              Awaiting pose landmarks from MediaPipe...
            </p>
          ) : (
            <div className="grid grid-cols-2 gap-1.5">
              {angleEntries.slice(0, 8).map(([name, deg]) => (
                <div
                  key={name}
                  className="flex items-center justify-between rounded-lg bg-[#0d111a] border border-white/[0.05] px-2.5 py-1.5"
                >
                  <span className="text-[11px] text-slate-400 capitalize truncate">
                    {name.replace(/_/g, " ")}
                  </span>
                  <span className="text-xs font-mono text-cyan-300 font-medium">
                    {typeof deg === "number" ? `${Math.round(deg)}°` : String(deg)}
                  </span>
                </div>
              ))}
            </div>
          )}

          {recentAlerts.length > 0 && (
            <div className="mt-3 pt-2.5 border-t border-white/[0.06]">
              <div className="text-[10px] font-mono uppercase tracking-wider text-slate-500 mb-1.5">
                Recent Spoken Cues
              </div>
              <div className="space-y-1 max-h-24 overflow-y-auto pr-1">
                {recentAlerts.map((alert, idx) => (
                  <div
                    key={`${alert.timestamp}-${idx}`}
                    className="text-[11px] text-slate-300 flex items-baseline justify-between gap-2"
                  >
                    <span className="truncate">{alert.text}</span>
                    <span className="text-[10px] font-mono text-slate-500 shrink-0">
                      {alert.timestamp}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
