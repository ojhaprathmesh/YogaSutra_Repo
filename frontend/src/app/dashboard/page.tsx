"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  Clock,
  Coins,
  Cpu,
  Radio,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { getDashboardStats, type DashboardStatsResponse } from "@/lib/api";

const WATERFALL_COLORS = [
  "from-violet-500 to-indigo-500",
  "from-cyan-500 to-sky-500",
  "from-emerald-500 to-teal-500",
  "from-amber-500 to-orange-500",
  "from-fuchsia-500 to-pink-500",
  "from-indigo-400 to-violet-500",
];

function formatNumber(n: number): string {
  return new Intl.NumberFormat("en-US").format(Math.round(n));
}

export default function DashboardPage() {
  const [stats, setStats] = useState<DashboardStatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<string>("");

  const loadStats = useCallback(async () => {
    try {
      const data = await getDashboardStats();
      setStats(data);
      setError(null);
      setLastUpdated(
        new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        })
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to reach /api/dashboard/stats."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadStats();
  }, [loadStats]);

  useEffect(() => {
    if (!autoRefresh) return;
    const id = setInterval(() => {
      void loadStats();
    }, 10000);
    return () => clearInterval(id);
  }, [autoRefresh, loadStats]);

  const stages = stats?.latency_waterfall?.stages ?? stats?.waterfall ?? [];
  const totalPipelineMs =
    stats?.latency_waterfall?.total_pipeline_ms ??
    stages.reduce((acc, s) => acc + s.duration_ms, 0);

  const providersMap =
    stats?.cost_and_tokens?.providers ?? stats?.provider_tokens ?? {};
  const providerEntries = Object.entries(providersMap);
  const maxProviderTokens = Math.max(
    1,
    ...providerEntries.map(([, p]) => p.total_tokens)
  );

  const promptStrategies = stats?.prompt_strategies ?? [];
  const auditEntries =
    stats?.guardrail_security?.recent_audit_log ?? stats?.guardrail_audit ?? [];
  const violationBreakdown =
    stats?.guardrail_security?.violation_breakdown ?? {};
  const recentEvents =
    stats?.event_bus_stream?.events ?? stats?.recent_events ?? [];

  return (
    <div className="flex flex-col gap-6">
      {/* Dashboard Top Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-violet-500/15 border border-violet-400/30 flex items-center justify-center">
            <BarChart3 className="h-5 w-5 text-violet-300" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-white tracking-tight">
              MAS Observability &amp; Trace Analytics
            </h1>
            <p className="text-xs text-slate-400">
              Multi-agent latency waterfall, zero-cost token telemetry, prompt
              benchmarks &amp; live security audit
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          {lastUpdated && (
            <span className="text-[11px] font-mono text-slate-500 hidden sm:inline">
              Updated {lastUpdated}
            </span>
          )}
          <button
            type="button"
            onClick={() => setAutoRefresh((prev) => !prev)}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium border transition-colors ${
              autoRefresh
                ? "bg-emerald-500/15 border-emerald-500/30 text-emerald-200"
                : "bg-white/[0.03] border-white/[0.08] text-slate-400"
            }`}
          >
            <Radio className={`h-3.5 w-3.5 ${autoRefresh ? "animate-pulse" : ""}`} />
            <span>Auto-Refresh {autoRefresh ? "10s" : "Off"}</span>
          </button>

          <button
            type="button"
            onClick={() => void loadStats()}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-white/[0.05] hover:bg-white/[0.09] border border-white/[0.1] text-slate-200 transition-colors"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-xl bg-rose-500/10 border border-rose-500/30 px-4 py-3 text-xs text-rose-200">
          {error}
        </div>
      )}

      {/* KPI Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="studio-card rounded-2xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Total Pipeline Runs</span>
            <Activity className="h-4 w-4 text-violet-400" />
          </div>
          <div className="mt-2 text-2xl font-semibold font-mono text-white">
            {stats ? formatNumber(stats.summary.total_runs) : "—"}
          </div>
          <div className="mt-1 text-[11px] text-slate-500">
            Traced multi-agent sessions
          </div>
        </div>

        <div className="studio-card rounded-2xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Avg Pipeline Latency</span>
            <Clock className="h-4 w-4 text-cyan-400" />
          </div>
          <div className="mt-2 text-2xl font-semibold font-mono text-white">
            {stats ? `${Math.round(stats.summary.avg_latency_ms)} ms` : "—"}
          </div>
          <div className="mt-1 text-[11px] text-slate-500">
            End-to-end 6-agent flow
          </div>
        </div>

        <div className="studio-card rounded-2xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Total Tokens</span>
            <Cpu className="h-4 w-4 text-indigo-400" />
          </div>
          <div className="mt-2 text-2xl font-semibold font-mono text-white">
            {stats ? formatNumber(stats.summary.total_tokens) : "—"}
          </div>
          <div className="mt-1 text-[11px] text-slate-500">
            Across 3 model providers
          </div>
        </div>

        <div className="studio-card rounded-2xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Actual Spend</span>
            <Coins className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="mt-2 text-2xl font-semibold font-mono text-emerald-300">
            ${stats ? stats.summary.actual_total_cost_usd.toFixed(2) : "0.00"}
          </div>
          <div className="mt-1 text-[11px] text-emerald-400/80">
            Est. ${stats ? stats.summary.commercial_savings_usd.toFixed(3) : "0.000"} saved
          </div>
        </div>

        <div className="studio-card rounded-2xl p-4">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Safety Clearance</span>
            <ShieldCheck className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="mt-2 text-2xl font-semibold font-mono text-white">
            {stats ? `${stats.summary.safety_clearance_rate_pct.toFixed(1)}%` : "—"}
          </div>
          <div className="mt-1 text-[11px] text-slate-500">
            Guardrail pass-through rate
          </div>
        </div>
      </div>

      {/* Latency Waterfall + Multi-Provider Breakdown */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        {/* Per-Agent Latency Waterfall */}
        <section className="xl:col-span-7 studio-card rounded-2xl p-5 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-white">
                Per-Agent Latency Waterfall
              </h2>
              <p className="text-xs text-slate-400">
                Sequential stage execution offsets across the 6-agent pipeline
              </p>
            </div>
            <span className="text-xs font-mono px-2.5 py-1 rounded-lg bg-cyan-500/10 border border-cyan-500/25 text-cyan-300">
              Total: {Math.round(totalPipelineMs)} ms
            </span>
          </div>

          <div className="space-y-3">
            {stages.map((stage, idx) => {
              const leftPct =
                totalPipelineMs > 0
                  ? (stage.start_offset_ms / totalPipelineMs) * 100
                  : 0;
              const widthPct =
                totalPipelineMs > 0
                  ? Math.max(4, (stage.duration_ms / totalPipelineMs) * 100)
                  : 10;
              const gradient = WATERFALL_COLORS[idx % WATERFALL_COLORS.length];

              return (
                <div key={stage.agent} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="font-mono font-medium text-slate-200">
                        {stage.agent}
                      </span>
                      {stage.role && (
                        <span className="text-[11px] text-slate-500 truncate hidden sm:inline">
                          — {stage.role}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2 font-mono text-[11px] shrink-0">
                      <span className="text-white font-medium">
                        {stage.duration_ms.toFixed(1)} ms
                      </span>
                      <span className="text-slate-500">
                        ({stage.pct_of_total}%)
                      </span>
                    </div>
                  </div>
                  <div className="h-3 w-full rounded-full bg-white/[0.04] overflow-hidden relative">
                    <div
                      style={{
                        marginLeft: `${Math.min(leftPct, 95)}%`,
                        width: `${Math.min(widthPct, 100 - leftPct)}%`,
                      }}
                      className={`h-full rounded-full bg-gradient-to-r ${gradient} shadow-sm`}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* Multi-Provider Token & Cost Breakdown */}
        <section className="xl:col-span-5 studio-card rounded-2xl p-5 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-white">
                Multi-Provider Token &amp; Cost Breakdown
              </h2>
              <p className="text-xs text-slate-400">
                {stats?.cost_and_tokens?.rate_tier ??
                  "Free-Tier Pooled Zero-Cost Architecture"}
              </p>
            </div>
            <Sparkles className="h-4 w-4 text-violet-400" />
          </div>

          <div className="space-y-3">
            {providerEntries.map(([key, metric]) => {
              const pct = Math.round(
                (metric.total_tokens / maxProviderTokens) * 100
              );
              return (
                <div
                  key={key}
                  className="rounded-xl bg-[#0b0f17] border border-white/[0.07] p-3.5 space-y-2"
                >
                  <div className="flex items-center justify-between gap-2">
                    <div>
                      <div className="text-xs font-semibold text-white">
                        {metric.display_name}
                      </div>
                      <div className="text-[11px] font-mono text-slate-400">
                        {metric.calls} calls • In {formatNumber(metric.input_tokens)} / Out{" "}
                        {formatNumber(metric.output_tokens)}
                      </div>
                    </div>
                    <div className="text-right shrink-0">
                      <div className="text-xs font-mono font-semibold text-emerald-300">
                        ${metric.actual_cost_usd.toFixed(2)}
                      </div>
                      <div className="text-[10px] font-mono text-slate-500">
                        Equiv ${metric.commercial_equiv_usd.toFixed(4)}
                      </div>
                    </div>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-white/[0.05] overflow-hidden">
                    <div
                      style={{ width: `${Math.max(6, pct)}%` }}
                      className="h-full rounded-full bg-gradient-to-r from-violet-500 to-cyan-400"
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      </div>

      {/* Prompt Strategy Benchmark Comparison Table */}
      <section className="studio-card rounded-2xl p-5 flex flex-col gap-4">
        <div>
          <h2 className="text-sm font-semibold text-white">
            Prompt Strategy Benchmark Comparison
          </h2>
          <p className="text-xs text-slate-400">
            Latency, token footprint, constraint adherence &amp; interpretability across architectural variants
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-white/[0.08] text-[11px] font-mono uppercase tracking-wider text-slate-400">
                <th className="py-2.5 pr-4">Strategy</th>
                <th className="py-2.5 px-3">Avg / P90 Latency</th>
                <th className="py-2.5 px-3">Avg Tokens</th>
                <th className="py-2.5 px-3">Adherence</th>
                <th className="py-2.5 px-3">Safety</th>
                <th className="py-2.5 px-3">Interpretability</th>
                <th className="py-2.5 pl-3">Tradeoff</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.05] text-xs">
              {promptStrategies.map((row) => (
                <tr
                  key={row.strategy}
                  className="hover:bg-white/[0.02] transition-colors"
                >
                  <td className="py-3 pr-4">
                    <div className="font-semibold text-white">{row.name}</div>
                    <div className="text-[11px] font-mono text-violet-300">
                      {row.strategy}
                    </div>
                  </td>
                  <td className="py-3 px-3 font-mono text-slate-200 whitespace-nowrap">
                    {row.avg_latency_ms.toFixed(0)} ms{" "}
                    <span className="text-slate-500">
                      / {row.p90_latency_ms.toFixed(0)} ms
                    </span>
                  </td>
                  <td className="py-3 px-3 font-mono text-slate-200 whitespace-nowrap">
                    {formatNumber(row.avg_total_tokens)}
                  </td>
                  <td className="py-3 px-3 font-mono text-emerald-300">
                    {row.constraint_adherence_pct.toFixed(1)}%
                  </td>
                  <td className="py-3 px-3 font-mono text-cyan-300">
                    {row.safety_clearance_pct.toFixed(1)}%
                  </td>
                  <td className="py-3 px-3">
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 rounded-full bg-white/[0.07] overflow-hidden">
                        <div
                          style={{
                            width: `${(row.interpretability_score / 5) * 100}%`,
                          }}
                          className="h-full rounded-full bg-violet-400"
                        />
                      </div>
                      <span className="font-mono text-slate-200">
                        {row.interpretability_score.toFixed(1)}/5
                      </span>
                    </div>
                  </td>
                  <td className="py-3 pl-3 text-slate-400 max-w-xs">
                    {row.tradeoff}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Bottom Row: Guardrail Security Audit + Event Bus Feed */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Guardrail Audit */}
        <section className="studio-card rounded-2xl p-5 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <ShieldAlert className="h-4 w-4 text-rose-400" />
              <h2 className="text-sm font-semibold text-white">
                Guardrail Security Audit
              </h2>
            </div>
            {stats?.guardrail_security && (
              <span className="text-[11px] font-mono text-slate-400">
                {stats.guardrail_security.total_blocked} blocked /{" "}
                {stats.guardrail_security.total_inspections} inspected
              </span>
            )}
          </div>

          {/* Violation Breakdown Chips */}
          {Object.keys(violationBreakdown).length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(violationBreakdown).map(([type, count]) => (
                <span
                  key={type}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-rose-500/10 border border-rose-500/20 text-[11px] font-mono text-rose-200"
                >
                  <span>{type}</span>
                  <span className="text-rose-400 font-semibold">{count}</span>
                </span>
              ))}
            </div>
          )}

          <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
            {auditEntries.length === 0 ? (
              <p className="text-xs text-slate-500">No guardrail violations logged.</p>
            ) : (
              auditEntries.map((entry, idx) => (
                <div
                  key={`${entry.timestamp}-${idx}`}
                  className="rounded-xl bg-[#0b0f17] border border-white/[0.06] p-3 flex items-start justify-between gap-3"
                >
                  <div className="min-w-0">
                    <p className="text-xs text-slate-200 truncate">
                      &ldquo;{entry.input_snippet}&rdquo;
                    </p>
                    <div className="flex flex-wrap gap-1.5 mt-1.5">
                      {entry.violations.map((v) => (
                        <span
                          key={v}
                          className="px-1.5 py-0.5 rounded bg-rose-500/15 text-rose-300 font-mono text-[10px]"
                        >
                          {v}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="text-right shrink-0">
                    <span className="inline-block px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-rose-500/20 text-rose-300">
                      {entry.status}
                    </span>
                    <div className="text-[10px] font-mono text-slate-500 mt-1">
                      Risk {(entry.risk_score * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </section>

        {/* Event Bus Journal */}
        <section className="studio-card rounded-2xl p-5 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Radio className="h-4 w-4 text-cyan-400" />
              <h2 className="text-sm font-semibold text-white">
                Recent Event Bus Stream
              </h2>
            </div>
            <span className="text-[11px] font-mono text-slate-400">
              {recentEvents.length} buffered events
            </span>
          </div>

          <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
            {recentEvents.length === 0 ? (
              <p className="text-xs text-slate-500">
                No events recorded yet. Trigger a posture deviation or practice run from Studio.
              </p>
            ) : (
              recentEvents.map((evt, idx) => (
                <div
                  key={`${evt.event_id ?? idx}`}
                  className="rounded-xl bg-[#0b0f17] border border-white/[0.06] p-3 flex items-start justify-between gap-3"
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold ${
                          evt.event_type === "SafetyEscalation"
                            ? "bg-rose-500/20 text-rose-300"
                            : evt.event_type === "PostureDeviation"
                              ? "bg-amber-500/20 text-amber-300"
                              : "bg-cyan-500/20 text-cyan-300"
                        }`}
                      >
                        {evt.event_type}
                      </span>
                      {evt.pose_name && (
                        <span className="text-xs text-slate-300 font-medium">
                          {evt.pose_name}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-slate-400 mt-1 truncate">
                      {evt.correction_cue ||
                        evt.trigger_condition ||
                        `User ${evt.user_id ?? "system"}`}
                    </p>
                  </div>
                  {evt.severity && (
                    <span className="text-[10px] font-mono uppercase text-slate-400 shrink-0">
                      {evt.severity}
                    </span>
                  )}
                </div>
              ))
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
