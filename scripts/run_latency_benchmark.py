"""
scripts/run_latency_benchmark.py — Initial Latency Benchmark Runner

Fulfills Phase 2 Milestone: 'Initial latency benchmark available' (PLAN.md §35 & §43).
Measures:
1. Google Gemini Flash Lite latency
2. Ollama Cloud (gpt-oss:120b & gemma4:31b) latency
3. NVIDIA NIM (meta/llama-3.2-11b-vision-instruct) latency
4. ElevenLabs Turbo v2.5 voice synthesis latency
5. Response & Audio Cache latency (sub-millisecond)

Saves results to `data/evaluation/initial_latency_benchmark.json`.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.table import Table

from app.audio.elevenlabs import elevenlabs_client
from app.models.gateway import model_gateway
from app.models.cache import response_cache

console = Console()
logging.basicConfig(level=logging.WARNING)

_EVAL_DIR = Path("data/evaluation")
_EVAL_DIR.mkdir(parents=True, exist_ok=True)


def run_benchmark() -> dict[str, Any]:
    console.print("\n[bold magenta]🧘 YogaSutra — Multi-Provider Latency Benchmark[/bold magenta]\n")
    results = []

    prompt = "Suggest three gentle morning yoga poses for relaxation."

    # 1. Google Gemini
    if model_gateway.google_api_key:
        try:
            t0 = time.monotonic()
            resp = model_gateway.generate(prompt, provider="gemini", model="gemini-3.5-flash-lite", use_cache=False)
            results.append({
                "provider": "Google Gemini",
                "model": "gemini-3.5-flash-lite",
                "task": "Text Generation",
                "latency_ms": resp.latency_ms,
                "status": "PASS",
            })
        except Exception as e:
            results.append({
                "provider": "Google Gemini",
                "model": "gemini-3.5-flash-lite",
                "task": "Text Generation",
                "latency_ms": 0.0,
                "status": f"FAIL ({e})",
            })

    # 2. Ollama Cloud Gemma 4 (Fast)
    if model_gateway.ollama_api_key:
        try:
            resp = model_gateway.generate(prompt, provider="ollama", model="gemma4:31b", use_cache=False)
            results.append({
                "provider": "Ollama Cloud",
                "model": "gemma4:31b",
                "task": "Fast Routing",
                "latency_ms": resp.latency_ms,
                "status": "PASS",
            })
        except Exception as e:
            results.append({
                "provider": "Ollama Cloud",
                "model": "gemma4:31b",
                "task": "Fast Routing",
                "latency_ms": 0.0,
                "status": f"FAIL ({e})",
            })

    # 3. Ollama Cloud GPT-OSS 120B (Reasoning)
    if model_gateway.ollama_api_key:
        try:
            resp = model_gateway.generate(prompt, provider="ollama", model="gpt-oss:120b", use_cache=False)
            results.append({
                "provider": "Ollama Cloud",
                "model": "gpt-oss:120b",
                "task": "Deep Reasoning",
                "latency_ms": resp.latency_ms,
                "status": "PASS",
            })
        except Exception as e:
            results.append({
                "provider": "Ollama Cloud",
                "model": "gpt-oss:120b",
                "task": "Deep Reasoning",
                "latency_ms": 0.0,
                "status": f"FAIL ({e})",
            })

    # 4. NVIDIA NIM
    if model_gateway.nvidia_api_key:
        try:
            resp = model_gateway.generate(prompt, provider="nvidia", model="meta/llama-3.2-11b-vision-instruct", use_cache=False)
            results.append({
                "provider": "NVIDIA NIM",
                "model": "meta/llama-3.2-11b-vision-instruct",
                "task": "Vision & Instruction",
                "latency_ms": resp.latency_ms,
                "status": "PASS",
            })
        except Exception as e:
            results.append({
                "provider": "NVIDIA NIM",
                "model": "meta/llama-3.2-11b-vision-instruct",
                "task": "Vision & Instruction",
                "latency_ms": 0.0,
                "status": f"FAIL ({e})",
            })

    # 5. ElevenLabs Voice Synthesis
    if elevenlabs_client.is_configured:
        try:
            t0 = time.monotonic()
            cue_path = elevenlabs_client.synthesize("Inhale deeply and lengthen your spine.")
            ms = round((time.monotonic() - t0) * 1000, 2)
            results.append({
                "provider": "ElevenLabs",
                "model": "eleven_turbo_v2_5",
                "task": "Spoken Cue Voice Synthesis",
                "latency_ms": ms,
                "status": "PASS" if cue_path else "CACHED / FALLBACK",
            })
        except Exception as e:
            results.append({
                "provider": "ElevenLabs",
                "model": "eleven_turbo_v2_5",
                "task": "Spoken Cue Voice Synthesis",
                "latency_ms": 0.0,
                "status": f"FAIL ({e})",
            })

    # 6. In-Memory Cache
    t0 = time.monotonic()
    response_cache.set("bench_k", "bench_v")
    _ = response_cache.get("bench_k")
    cache_ms = round((time.monotonic() - t0) * 1000, 4)
    results.append({
        "provider": "YogaSutra In-Memory Cache",
        "model": "SHA-256 LRU",
        "task": "Zero-Cost Deduplication",
        "latency_ms": cache_ms,
        "status": "PASS",
    })

    # Print Table
    table = Table(title="📊 Initial Multi-Model & Speech Latency Benchmark")
    table.add_column("Provider", style="bold cyan")
    table.add_column("Model / Tier", style="white")
    table.add_column("Role / Task", style="magenta")
    table.add_column("Latency (ms)", justify="right", style="green")
    table.add_column("Status", style="yellow")

    for r in results:
        table.add_row(
            r["provider"],
            r["model"],
            r["task"],
            f"{r['latency_ms']:.1f}" if r["latency_ms"] > 0 else "-",
            r["status"],
        )

    console.print(table)
    console.print()

    out_file = _EVAL_DIR / "initial_latency_benchmark.json"
    out_file.write_text(json.dumps({"timestamp": time.time(), "benchmarks": results}, indent=2), encoding="utf-8")
    console.print(f"[dim]Benchmark results saved to: {out_file}[/dim]\n")
    return {"benchmarks": results}


if __name__ == "__main__":
    run_benchmark()
