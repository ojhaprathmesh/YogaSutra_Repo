export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export function getWebSocketUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  if (API_BASE_URL.startsWith("https://")) {
    return `${API_BASE_URL.replace(/^https:\/\//, "wss://")}${normalizedPath}`;
  }
  if (API_BASE_URL.startsWith("http://")) {
    return `${API_BASE_URL.replace(/^http:\/\//, "ws://")}${normalizedPath}`;
  }
  return `ws://localhost:8000${normalizedPath}`;
}

export function resolveApiUrl(pathOrUrl: string | null | undefined): string | null {
  if (!pathOrUrl) return null;
  if (/^https?:\/\//i.test(pathOrUrl)) return pathOrUrl;
  const normalized = pathOrUrl.startsWith("/") ? pathOrUrl : `/${pathOrUrl}`;
  return `${API_BASE_URL}${normalized}`;
}

// ---------------------------------------------------------------------------
// API Types
// ---------------------------------------------------------------------------

export interface HealthStatus {
  status: string;
  app: string;
  version: string;
}

export interface ModelStatus {
  providers: {
    gemini: boolean;
    ollama_cloud: boolean;
    nvidia_nim: boolean;
    elevenlabs: boolean;
  };
  elevenlabs_quota?: Record<string, unknown>;
  models?: Record<string, string[]>;
}

export interface GuardrailResult {
  is_safe: boolean;
  risk_score: number;
  violations: string[];
  rejection_reason?: string | null;
  sanitized_input?: string;
}

export interface AudioCuePayload {
  text: string;
  audio_url?: string | null;
  provider?: string;
  voice_name?: string;
  speaking_rate?: number;
}

export interface TracePayload {
  run_id?: string;
  total_latency_ms?: number;
  agent_latency_ms?: Record<string, number>;
  token_usage?: Record<string, { input?: number; output?: number }>;
  [key: string]: unknown;
}

export interface PracticeRequestPayload {
  user_input: string;
  persona_id: string;
  phase?: number;
  prompt_strategy: "default" | "react" | "few_shot" | "cot";
  include_audio?: boolean;
}

export interface PracticeResponse {
  run_id: string;
  session_id: string;
  final_response: string;
  phase: number;
  prompt_strategy: string;
  guardrail?: GuardrailResult;
  trace?: TracePayload;
  audio_cue?: AudioCuePayload | null;
}

export interface LandmarkPoint {
  id?: number;
  x: number;
  y: number;
  visibility: number;
}

export interface PoseDeviation {
  joint_name: string;
  measured_angle: number;
  target_min: number;
  target_max: number;
  deviation_deg?: number;
  is_aligned: boolean;
  severity: "aligned" | "mild" | "moderate" | "severe" | "critical" | string;
  correction_cue: string;
}

export interface FiredCue {
  event_id?: string;
  joint_name: string;
  correction_cue: string;
  severity: string;
  measured_angle?: number;
  deviation_deg?: number;
  audio_url?: string | null;
  provider?: string;
  voice_name?: string;
}

export interface VisionTelemetry {
  type: "telemetry" | "config_ack";
  pose_detected?: boolean;
  asana: string;
  lead_leg?: "left" | "right";
  latency_ms?: number;
  landmarks?: LandmarkPoint[];
  angles?: Record<string, number>;
  deviations?: PoseDeviation[];
  fired_cues?: FiredCue[];
}

export interface SSEEventPayload {
  event_id?: string;
  event_type?: string;
  pose_name?: string;
  joint_name?: string;
  deviation_deg?: number;
  measured_angle?: number;
  severity?: string;
  correction_cue?: string;
  trigger_condition?: string;
  action_required?: string;
  audio_url?: string | null;
  provider?: string;
  voice_name?: string;
  timestamp?: string;
}

export interface WaterfallStage {
  agent: string;
  role?: string;
  duration_ms: number;
  start_offset_ms: number;
  end_offset_ms: number;
  pct_of_total: number;
}

export interface ProviderTokenMetric {
  display_name: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  calls: number;
  actual_cost_usd: number;
  commercial_equiv_usd: number;
  is_free_tier: boolean;
}

export interface PromptStrategyMetric {
  strategy: string;
  name: string;
  description: string;
  avg_latency_ms: number;
  p50_latency_ms: number;
  p90_latency_ms: number;
  avg_input_tokens: number;
  avg_output_tokens: number;
  avg_total_tokens: number;
  constraint_adherence_pct: number;
  safety_clearance_pct: number;
  interpretability_score: number;
  tradeoff: string;
}

export interface GuardrailAuditEntry {
  timestamp: string;
  input_snippet: string;
  violations: string[];
  risk_score: number;
  status: string;
}

export interface EventBusEntry {
  event_id?: string;
  event_type: string;
  timestamp?: string;
  session_id?: string;
  user_id?: string;
  pose_name?: string;
  joint_name?: string;
  severity?: string;
  correction_cue?: string;
  trigger_condition?: string;
  [key: string]: unknown;
}

export interface DashboardStatsResponse {
  timestamp?: string;
  summary: {
    total_runs: number;
    avg_latency_ms: number;
    total_tokens: number;
    actual_total_cost_usd: number;
    commercial_savings_usd: number;
    safety_clearance_rate_pct: number;
  };
  latency_waterfall?: {
    total_pipeline_ms: number;
    stages: WaterfallStage[];
  };
  waterfall?: WaterfallStage[];
  cost_and_tokens?: {
    providers: Record<string, ProviderTokenMetric>;
    total_commercial_value_usd: number;
    total_actual_cost_usd: number;
    rate_tier: string;
  };
  provider_tokens?: Record<string, ProviderTokenMetric>;
  prompt_strategies?: PromptStrategyMetric[];
  guardrail_security?: {
    total_inspections: number;
    total_blocked: number;
    clearance_rate_pct: number;
    violation_breakdown: Record<string, number>;
    recent_audit_log: GuardrailAuditEntry[];
  };
  guardrail_audit?: GuardrailAuditEntry[];
  event_bus_stream?: {
    total_buffered_events: number;
    events: EventBusEntry[];
  };
  recent_events?: EventBusEntry[];
  recent_traces?: TracePayload[];
}

// ---------------------------------------------------------------------------
// API Fetchers
// ---------------------------------------------------------------------------

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      // Ignore parse error on non-JSON error responses
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export async function getHealth(): Promise<HealthStatus> {
  return fetchJson<HealthStatus>(`${API_BASE_URL}/api/health`);
}

export async function getModelStatus(): Promise<ModelStatus> {
  return fetchJson<ModelStatus>(`${API_BASE_URL}/api/models`);
}

export async function inspectSecurity(userInput: string): Promise<GuardrailResult> {
  const payload = JSON.stringify({ user_input: userInput });
  const primary = await fetch(`${API_BASE_URL}/api/security/inspect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: payload,
  });

  if (primary.ok) {
    return (await primary.json()) as GuardrailResult;
  }

  // Graceful fallback to /api/security/validate if /inspect is not exposed
  if (primary.status === 404 || primary.status === 405) {
    return fetchJson<GuardrailResult>(`${API_BASE_URL}/api/security/validate`, {
      method: "POST",
      body: payload,
    });
  }

  throw new Error(`Guardrail inspection failed (HTTP ${primary.status})`);
}

export async function generatePractice(
  req: PracticeRequestPayload
): Promise<PracticeResponse> {
  return fetchJson<PracticeResponse>(`${API_BASE_URL}/api/practice/generate`, {
    method: "POST",
    body: JSON.stringify({
      phase: 2,
      include_audio: true,
      ...req,
    }),
  });
}

export async function synthesizeAudio(
  text: string,
  options?: { voice_id?: string; prefer_elevenlabs?: boolean }
): Promise<AudioCuePayload> {
  return fetchJson<AudioCuePayload>(`${API_BASE_URL}/api/audio/synthesize`, {
    method: "POST",
    body: JSON.stringify({
      text,
      prefer_elevenlabs: options?.prefer_elevenlabs ?? true,
      ...(options?.voice_id ? { voice_id: options.voice_id } : {}),
    }),
  });
}

export async function publishEvent(body: {
  event_type: string;
  user_id?: string;
  session_id?: string;
  payload: Record<string, unknown>;
}): Promise<{
  status: string;
  event_id: string;
  event_type: string;
  cue?: string;
  audio_url?: string | null;
  provider?: string;
  voice_name?: string;
}> {
  return fetchJson(`${API_BASE_URL}/api/events/publish`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function getDashboardStats(): Promise<DashboardStatsResponse> {
  return fetchJson<DashboardStatsResponse>(`${API_BASE_URL}/api/dashboard/stats`, {
    cache: "no-store",
  });
}
