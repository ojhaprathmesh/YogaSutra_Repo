"use client";

import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Cpu,
  Loader2,
  Mic,
  MicOff,
  Play,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Square,
  Volume2,
} from "lucide-react";
import {
  generatePractice,
  inspectSecurity,
  synthesizeAudio,
  type AudioCuePayload,
  type GuardrailResult,
  type PracticeResponse,
} from "@/lib/api";
import { playSpokenCue, stopSpokenAudio } from "@/lib/audio";
import { VisionHud, type SupportedAsana } from "./VisionHud";

// Minimal browser SpeechRecognition types for TypeScript strict mode
interface SpeechRecognitionEventLike {
  results: ArrayLike<{
    isFinal: boolean;
    0: { transcript: string };
  }>;
}

interface SpeechRecognitionErrorEventLike {
  error: string;
}

interface SpeechRecognitionLike {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start: () => void;
  stop: () => void;
  onresult: ((ev: SpeechRecognitionEventLike) => void) | null;
  onerror: ((ev: SpeechRecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
}

type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

const PERSONAS = [
  { id: "P00", label: "P00 — Balanced Practitioner" },
  { id: "P01", label: "P01 — Knee & Joint Sensitivity" },
  { id: "P02", label: "P02 — Prenatal / Gentle Mobility" },
  { id: "P03", label: "P03 — Athletic Endurance" },
];

const STRATEGIES = [
  { value: "default", label: "Default (Direct)" },
  { value: "react", label: "ReAct (Reason + Act)" },
  { value: "few_shot", label: "Few-Shot" },
  { value: "cot", label: "Chain-of-Thought" },
] as const;

const QUICK_PROMPTS = [
  "20-minute energizing morning flow focusing on Warrior II and hip stability, mild left knee sensitivity.",
  "15-minute gentle restorative yoga for lower back release and calm breathing.",
  "10-minute balance and posture session with Mountain Pose and Tree Pose.",
];

function parseVoiceCommand(transcript: string): {
  asana?: SupportedAsana;
  leadLeg?: "left" | "right";
  matchedLabel?: string;
} | null {
  const lower = transcript.toLowerCase();
  let asana: SupportedAsana | undefined;
  let leadLeg: "left" | "right" | undefined;
  const labels: string[] = [];

  if (lower.includes("warrior") || lower.includes("virabhadrasana")) {
    asana = "Virabhadrasana II";
    labels.push("Warrior II");
  } else if (lower.includes("tree") || lower.includes("vrikshasana")) {
    asana = "Vrikshasana";
    labels.push("Tree Pose");
  } else if (lower.includes("mountain") || lower.includes("tadasana")) {
    asana = "Tadasana";
    labels.push("Tadasana");
  } else if (lower.includes("chair") || lower.includes("utkatasana")) {
    asana = "Utkatasana";
    labels.push("Chair Pose");
  } else if (lower.includes("cobra") || lower.includes("bhujangasana")) {
    asana = "Bhujangasana";
    labels.push("Cobra Pose");
  }

  if (lower.includes("left leg") || lower.includes("left side")) {
    leadLeg = "left";
    labels.push("Left Leg");
  } else if (lower.includes("right leg") || lower.includes("right side")) {
    leadLeg = "right";
    labels.push("Right Leg");
  }

  if (asana || leadLeg) {
    return { asana, leadLeg, matchedLabel: labels.join(" • ") };
  }
  return null;
}

export function PracticeStudio() {
  const [userInput, setUserInput] = useState(QUICK_PROMPTS[0]);
  const [personaId, setPersonaId] = useState("P00");
  const [promptStrategy, setPromptStrategy] = useState<
    "default" | "react" | "few_shot" | "cot"
  >("default");

  // Synced Vision HUD pose state (also controllable via Voice Commands)
  const [asana, setAsana] = useState<SupportedAsana>("Virabhadrasana II");
  const [leadLeg, setLeadLeg] = useState<"left" | "right">("left");

  // Guardrail state
  const [guardrail, setGuardrail] = useState<GuardrailResult | null>(null);
  const [guardLoading, setGuardLoading] = useState(false);

  // Voice recognition state
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const [listening, setListening] = useState(false);
  const [voiceFeedback, setVoiceFeedback] = useState<string | null>(null);

  // Practice generation & audio playback state
  const [generating, setGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [result, setResult] = useState<PracticeResponse | null>(null);

  const [audioCue, setAudioCue] = useState<AudioCuePayload | null>(null);
  const [synthesizingAudio, setSynthesizingAudio] = useState(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [activeAudioSource, setActiveAudioSource] = useState<string | null>(null);

  // Debounced live guardrail inspection
  useEffect(() => {
    const trimmed = userInput.trim();
    if (!trimmed) {
      setGuardrail(null);
      return;
    }

    const handle = setTimeout(async () => {
      setGuardLoading(true);
      try {
        const res = await inspectSecurity(trimmed);
        setGuardrail(res);
      } catch {
        // Keep silent if backend is offline while typing
      } finally {
        setGuardLoading(false);
      }
    }, 380);

    return () => clearTimeout(handle);
  }, [userInput]);

  useEffect(() => {
    return () => {
      recognitionRef.current?.stop();
      stopSpokenAudio();
    };
  }, []);

  const toggleVoiceInput = () => {
    if (listening) {
      recognitionRef.current?.stop();
      setListening(false);
      return;
    }

    const win = window as unknown as {
      SpeechRecognition?: SpeechRecognitionCtor;
      webkitSpeechRecognition?: SpeechRecognitionCtor;
    };
    const Ctor = win.SpeechRecognition || win.webkitSpeechRecognition;
    if (!Ctor) {
      setVoiceFeedback("Browser SpeechRecognition API is unavailable in this browser.");
      return;
    }

    const recognition = new Ctor();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognitionRef.current = recognition;

    recognition.onresult = (event) => {
      let fullTranscript = "";
      let isFinal = false;
      for (let i = 0; i < event.results.length; i++) {
        fullTranscript += event.results[i][0].transcript;
        if (event.results[i].isFinal) isFinal = true;
      }
      setVoiceFeedback(`Heard: "${fullTranscript}"`);

      if (isFinal) {
        const cmd = parseVoiceCommand(fullTranscript);
        if (cmd && fullTranscript.split(/\s+/).length <= 8) {
          if (cmd.asana) setAsana(cmd.asana);
          if (cmd.leadLeg) setLeadLeg(cmd.leadLeg);
          const ack = `Switched Vision HUD to ${cmd.matchedLabel}`;
          setVoiceFeedback(ack);
          void playSpokenCue({ text: ack });
        } else {
          setUserInput(fullTranscript.trim());
          if (cmd?.asana) setAsana(cmd.asana);
          if (cmd?.leadLeg) setLeadLeg(cmd.leadLeg);
          setVoiceFeedback("Dictated practice goal into input.");
        }
      }
    };

    recognition.onerror = (ev) => {
      setVoiceFeedback(`Voice input error: ${ev.error}`);
      setListening(false);
    };

    recognition.onend = () => {
      setListening(false);
    };

    setVoiceFeedback("Listening... Say a pose command (e.g. 'Switch to Tree Pose') or dictate a goal.");
    setListening(true);
    recognition.start();
  };

  const handleGenerate = async () => {
    if (!userInput.trim() || (guardrail && !guardrail.is_safe)) return;
    setGenerating(true);
    setGenerateError(null);
    stopSpokenAudio();
    setIsPlayingAudio(false);

    try {
      const res = await generatePractice({
        user_input: userInput.trim(),
        persona_id: personaId,
        phase: 2,
        prompt_strategy: promptStrategy,
        include_audio: true,
      });
      setResult(res);
      setAudioCue(res.audio_cue ?? null);
    } catch (err) {
      setGenerateError(
        err instanceof Error ? err.message : "Failed to generate practice plan."
      );
    } finally {
      setGenerating(false);
    }
  };

  const handlePlayGuidance = async () => {
    if (isPlayingAudio) {
      stopSpokenAudio();
      setIsPlayingAudio(false);
      return;
    }
    if (!result) return;

    // Extract a concise spoken summary if no pre-generated audio_cue exists
    let cueToPlay = audioCue;
    if (!cueToPlay) {
      const cleanSummary = result.final_response
        .replace(/[#*_`>-]/g, " ")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 420);

      setSynthesizingAudio(true);
      try {
        cueToPlay = await synthesizeAudio(cleanSummary, {
          prefer_elevenlabs: true,
        });
        setAudioCue(cueToPlay);
      } catch {
        cueToPlay = { text: cleanSummary, provider: "webspeech" };
      } finally {
        setSynthesizingAudio(false);
      }
    }

    await playSpokenCue({
      text: cueToPlay.text,
      audioUrl: cueToPlay.audio_url,
      onStart: (source) => {
        setIsPlayingAudio(true);
        setActiveAudioSource(source);
      },
      onEnd: () => {
        setIsPlayingAudio(false);
      },
    });
  };

  // Compute guardrail badge state
  const guardrailState = (() => {
    if (guardLoading) return { label: "Inspecting...", tone: "neutral" as const };
    if (!guardrail) return { label: "Guardrail Ready", tone: "neutral" as const };
    if (!guardrail.is_safe) {
      return {
        label: `Blocked (Risk ${(guardrail.risk_score * 100).toFixed(0)}%)`,
        tone: "blocked" as const,
      };
    }
    if (guardrail.risk_score >= 0.35 || guardrail.violations.length > 0) {
      return {
        label: `Warning (Risk ${(guardrail.risk_score * 100).toFixed(0)}%)`,
        tone: "warning" as const,
      };
    }
    return {
      label: `Safe (Risk ${(guardrail.risk_score * 100).toFixed(0)}%)`,
      tone: "safe" as const,
    };
  })();

  const agentLatencies = result?.trace?.agent_latency_ms
    ? Object.entries(result.trace.agent_latency_ms)
    : [];

  return (
    <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
      {/* LEFT PANEL: Multimodal Goal, Voice Input & Generated Plan */}
      <div className="xl:col-span-6 flex flex-col gap-5">
        <section className="studio-card rounded-2xl p-5 flex flex-col gap-4">
          {/* Header + Guardrail Badge */}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2.5">
              <div className="h-8 w-8 rounded-xl bg-violet-500/15 border border-violet-400/30 flex items-center justify-center">
                <Sparkles className="h-4 w-4 text-violet-300" />
              </div>
              <div>
                <h1 className="text-sm font-semibold text-white tracking-tight">
                  Multimodal Practice Planner
                </h1>
                <p className="text-xs text-slate-400">
                  Natural language, voice commands &amp; security guardrail inspection
                </p>
              </div>
            </div>

            <div
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono border ${
                guardrailState.tone === "safe"
                  ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
                  : guardrailState.tone === "warning"
                    ? "bg-amber-500/10 border-amber-500/30 text-amber-300"
                    : guardrailState.tone === "blocked"
                      ? "bg-rose-500/15 border-rose-500/40 text-rose-300"
                      : "bg-white/[0.04] border-white/[0.08] text-slate-400"
              }`}
            >
              {guardrailState.tone === "safe" ? (
                <ShieldCheck className="h-3.5 w-3.5" />
              ) : guardrailState.tone === "blocked" ? (
                <ShieldAlert className="h-3.5 w-3.5" />
              ) : guardrailState.tone === "warning" ? (
                <AlertTriangle className="h-3.5 w-3.5" />
              ) : (
                <CheckCircle2 className="h-3.5 w-3.5" />
              )}
              <span>{guardrailState.label}</span>
            </div>
          </div>

          {/* Persona & Prompt Strategy Controls */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="block text-[10px] font-mono uppercase tracking-wider text-slate-400 mb-1">
                Practitioner Persona
              </label>
              <select
                value={personaId}
                onChange={(e) => setPersonaId(e.target.value)}
                className="w-full bg-[#0d111a] border border-white/[0.1] rounded-xl px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-violet-400/60"
              >
                {PERSONAS.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-[10px] font-mono uppercase tracking-wider text-slate-400 mb-1">
                Prompt Strategy
              </label>
              <select
                value={promptStrategy}
                onChange={(e) =>
                  setPromptStrategy(
                    e.target.value as "default" | "react" | "few_shot" | "cot"
                  )
                }
                className="w-full bg-[#0d111a] border border-white/[0.1] rounded-xl px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-violet-400/60"
              >
                {STRATEGIES.map((s) => (
                  <option key={s.value} value={s.value}>
                    {s.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Goal Textarea */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
                Practice Goal &amp; Physical Constraints
              </label>
              <span className="text-[10px] font-mono text-slate-500">
                Live Guardrail Active
              </span>
            </div>
            <textarea
              rows={3}
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              placeholder="Describe your yoga goals, duration, energy level, or any injuries..."
              className="w-full rounded-xl bg-[#0b0f17] border border-white/[0.1] focus:border-violet-400/60 focus:outline-none px-3.5 py-2.5 text-sm text-slate-100 placeholder:text-slate-500 resize-none leading-relaxed"
            />

            {/* Quick Presets */}
            <div className="flex flex-wrap gap-1.5 mt-2">
              {QUICK_PROMPTS.map((preset, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => setUserInput(preset)}
                  className="text-[11px] px-2.5 py-1 rounded-lg bg-white/[0.03] hover:bg-white/[0.07] border border-white/[0.07] text-slate-400 hover:text-slate-200 transition-colors truncate max-w-[240px]"
                >
                  {preset}
                </button>
              ))}
            </div>
          </div>

          {/* Guardrail Rejection Details */}
          {guardrail && !guardrail.is_safe && (
            <div className="rounded-xl bg-rose-500/10 border border-rose-500/35 p-3 text-xs text-rose-200 flex items-start gap-2.5">
              <ShieldAlert className="h-4 w-4 shrink-0 mt-0.5 text-rose-400" />
              <div>
                <div className="font-semibold">
                  Blocked by Input Security Guardrail
                </div>
                <p className="text-rose-200/85 mt-0.5">
                  {guardrail.rejection_reason ||
                    "Unsafe or out-of-domain instruction detected."}
                </p>
                {guardrail.violations.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-1.5">
                    {guardrail.violations.map((v) => (
                      <span
                        key={v}
                        className="px-1.5 py-0.5 rounded bg-rose-500/20 font-mono text-[10px]"
                      >
                        {v}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Action Row: Hands-Free Voice + Generate Button */}
          <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
            <button
              type="button"
              onClick={toggleVoiceInput}
              className={`inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-medium border transition-all ${
                listening
                  ? "bg-cyan-500/20 border-cyan-400/50 text-cyan-200 shadow-[0_0_20px_rgba(6,182,212,0.25)]"
                  : "bg-white/[0.04] hover:bg-white/[0.08] border-white/[0.1] text-slate-200"
              }`}
            >
              {listening ? (
                <MicOff className="h-3.5 w-3.5 animate-pulse text-cyan-300" />
              ) : (
                <Mic className="h-3.5 w-3.5 text-cyan-300" />
              )}
              <span>
                {listening ? "Stop Listening" : "Hands-Free Voice Input"}
              </span>
            </button>

            <button
              type="button"
              onClick={() => void handleGenerate()}
              disabled={
                generating ||
                !userInput.trim() ||
                Boolean(guardrail && !guardrail.is_safe)
              }
              className="inline-flex items-center gap-2 px-5 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white shadow-[0_0_24px_rgba(124,58,237,0.32)] transition-all disabled:opacity-45 disabled:cursor-not-allowed"
            >
              {generating ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Sparkles className="h-3.5 w-3.5" />
              )}
              <span>
                {generating ? "Orchestrating Agents..." : "Generate Practice Plan"}
              </span>
            </button>
          </div>

          {voiceFeedback && (
            <div className="rounded-lg bg-cyan-500/10 border border-cyan-500/25 px-3 py-2 text-xs text-cyan-200 font-mono">
              {voiceFeedback}
            </div>
          )}

          {generateError && (
            <div className="rounded-xl bg-rose-500/10 border border-rose-500/30 px-3.5 py-2.5 text-xs text-rose-200">
              {generateError}
            </div>
          )}
        </section>

        {/* Generated Practice Plan Output + Spoken Guidance Player */}
        {result && (
          <section className="studio-card rounded-2xl p-5 flex flex-col gap-4">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.08] pb-3.5">
              <div>
                <h2 className="text-sm font-semibold text-white">
                  Personalized Practice Plan
                </h2>
                <div className="flex items-center gap-2 text-[11px] font-mono text-slate-400 mt-0.5">
                  <span>Run {result.run_id.slice(0, 12)}</span>
                  <span>•</span>
                  <span className="uppercase text-violet-300">
                    {result.prompt_strategy}
                  </span>
                  {typeof result.trace?.total_latency_ms === "number" && (
                    <>
                      <span>•</span>
                      <span className="inline-flex items-center gap-1 text-emerald-300">
                        <Clock className="h-3 w-3" />
                        {result.trace.total_latency_ms.toFixed(0)} ms
                      </span>
                    </>
                  )}
                </div>
              </div>

              {/* ElevenLabs Spoken Guidance Audio Player */}
              <button
                type="button"
                onClick={() => void handlePlayGuidance()}
                disabled={synthesizingAudio}
                className={`inline-flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold border transition-all ${
                  isPlayingAudio
                    ? "bg-emerald-500/20 border-emerald-400/40 text-emerald-200"
                    : "bg-violet-500/15 hover:bg-violet-500/25 border-violet-400/30 text-violet-200"
                }`}
              >
                {synthesizingAudio ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : isPlayingAudio ? (
                  <Square className="h-3.5 w-3.5 fill-current" />
                ) : (
                  <Play className="h-3.5 w-3.5 fill-current" />
                )}
                <Volume2 className="h-3.5 w-3.5" />
                <span>
                  {synthesizingAudio
                    ? "Synthesizing..."
                    : isPlayingAudio
                      ? `Stop (${activeAudioSource ?? "Audio"})`
                      : "Play Spoken Guidance"}
                </span>
              </button>
            </div>

            {/* Agent Trace Pills */}
            {agentLatencies.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {agentLatencies.map(([agentName, ms]) => (
                  <span
                    key={agentName}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white/[0.03] border border-white/[0.07] text-[11px] font-mono text-slate-300"
                  >
                    <Cpu className="h-3 w-3 text-violet-400" />
                    <span>{agentName}</span>
                    <span className="text-cyan-300">{Math.round(ms)}ms</span>
                  </span>
                ))}
              </div>
            )}

            {/* Plan Body */}
            <div className="rounded-xl bg-[#0b0f17] border border-white/[0.07] p-4 text-sm text-slate-200 leading-relaxed whitespace-pre-wrap max-h-[520px] overflow-y-auto">
              {result.final_response}
            </div>
          </section>
        )}
      </div>

      {/* RIGHT PANEL: Live Biomechanics Vision HUD */}
      <div className="xl:col-span-6">
        <VisionHud
          asana={asana}
          onAsanaChange={setAsana}
          leadLeg={leadLeg}
          onLeadLegChange={setLeadLeg}
          personaId={personaId}
        />
      </div>
    </div>
  );
}
