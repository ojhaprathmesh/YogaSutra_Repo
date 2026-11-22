"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  BarChart3,
  RefreshCw,
  Sparkles,
  Wifi,
  WifiOff,
} from "lucide-react";
import { getModelStatus, type ModelStatus } from "@/lib/api";

interface ProviderPill {
  key: keyof ModelStatus["providers"];
  label: string;
  accent: string;
}

const PROVIDERS: ProviderPill[] = [
  { key: "gemini", label: "Gemini", accent: "text-violet-300" },
  { key: "nvidia_nim", label: "NVIDIA NIM", accent: "text-emerald-300" },
  { key: "ollama_cloud", label: "Ollama", accent: "text-cyan-300" },
  { key: "elevenlabs", label: "ElevenLabs", accent: "text-amber-300" },
];

export function Header() {
  const pathname = usePathname();
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [apiReachable, setApiReachable] = useState<boolean | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const loadModels = async () => {
    setRefreshing(true);
    try {
      const data = await getModelStatus();
      setStatus(data);
      setApiReachable(true);
    } catch {
      setApiReachable(false);
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    void loadModels();
    const interval = setInterval(() => {
      void loadModels();
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { href: "/", label: "Practice Studio", icon: Activity },
    { href: "/dashboard", label: "Observability", icon: BarChart3 },
  ];

  return (
    <header className="sticky top-0 z-50 border-b border-white/[0.08] bg-[#090b10]/85 backdrop-blur-xl">
      <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
        {/* Brand + Primary Nav */}
        <div className="flex items-center gap-6">
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-violet-500/25 via-indigo-500/20 to-cyan-500/25 border border-violet-400/30 flex items-center justify-center shadow-[0_0_20px_rgba(139,92,246,0.2)] group-hover:border-violet-400/50 transition-colors">
              <Sparkles className="h-4.5 w-4.5 text-violet-300" />
            </div>
            <div className="leading-tight">
              <div className="text-sm font-semibold tracking-tight text-white flex items-center gap-2">
                YogaSutra
                <span className="text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded bg-violet-500/15 text-violet-300 border border-violet-500/25">
                  MAS v0.2
                </span>
              </div>
              <div className="text-[11px] text-slate-400 hidden sm:block">
                Multimodal Biomechanics &amp; Voice Studio
              </div>
            </div>
          </Link>

          <nav className="flex items-center gap-1 p-1 rounded-xl bg-white/[0.03] border border-white/[0.06]">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive =
                item.href === "/"
                  ? pathname === "/"
                  : pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                    isActive
                      ? "bg-violet-500/20 text-white border border-violet-400/30 shadow-sm"
                      : "text-slate-400 hover:text-slate-200 hover:bg-white/[0.04] border border-transparent"
                  }`}
                >
                  <Icon className="h-3.5 w-3.5" />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Live Provider Status Pills */}
        <div className="flex items-center gap-2">
          <div className="hidden lg:flex items-center gap-1.5">
            {PROVIDERS.map((p) => {
              const online = status?.providers?.[p.key] ?? false;
              return (
                <div
                  key={p.key}
                  title={
                    online
                      ? `${p.label} configured & ready`
                      : `${p.label} not configured`
                  }
                  className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono border transition-colors ${
                    online
                      ? "bg-white/[0.04] border-white/[0.1] text-slate-200"
                      : "bg-white/[0.015] border-white/[0.05] text-slate-500"
                  }`}
                >
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${
                      online
                        ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]"
                        : "bg-slate-600"
                    }`}
                  />
                  <span className={online ? p.accent : "text-slate-500"}>
                    {p.label}
                  </span>
                </div>
              );
            })}
          </div>

          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono border ${
              apiReachable === true
                ? "bg-emerald-500/10 border-emerald-500/25 text-emerald-300"
                : apiReachable === false
                  ? "bg-rose-500/10 border-rose-500/25 text-rose-300"
                  : "bg-white/[0.04] border-white/[0.08] text-slate-400"
            }`}
          >
            {apiReachable === false ? (
              <WifiOff className="h-3 w-3" />
            ) : (
              <Wifi className="h-3 w-3" />
            )}
            <span className="hidden sm:inline">
              {apiReachable === true
                ? "API Live"
                : apiReachable === false
                  ? "API Offline"
                  : "Checking"}
            </span>
          </div>

          <button
            type="button"
            onClick={() => void loadModels()}
            disabled={refreshing}
            title="Refresh provider status"
            className="p-1.5 rounded-lg border border-white/[0.08] bg-white/[0.03] text-slate-400 hover:text-white hover:border-white/[0.16] transition-colors disabled:opacity-50"
          >
            <RefreshCw
              className={`h-3.5 w-3.5 ${refreshing ? "animate-spin" : ""}`}
            />
          </button>
        </div>
      </div>
    </header>
  );
}
