"""
main.py — YogaSutra CLI Entry Point

Usage:
    python3 main.py
    python3 main.py --input "I am a beginner and have 20 minutes in the morning. I want a relaxing yoga practice."
    python3 main.py --input "..." --persona P01

Prints the final response and execution trace summary to stdout using Rich.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings

# Suppress framework noise, experimental feature warnings, and client notices
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("google.adk").setLevel(logging.ERROR)
try:
    from google.genai.models import AsyncModels, Models
    AsyncModels._logged_afc_warning = True
    Models._logged_afc_warning = True
except Exception:
    pass

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import print as rprint

console = Console()

# Default demo scenario (from Phase 1 plan)
DEFAULT_INPUT = (
    "I am a beginner and have 20 minutes in the morning. "
    "I want a relaxing yoga practice."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="yogasutra",
        description="YogaSutra — AI-powered personalised yoga practice planner",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=DEFAULT_INPUT,
        help="User's yoga request (default: Phase 1 demo scenario)",
    )
    parser.add_argument(
        "--persona", "-p",
        type=str,
        default="P00",
        help="Persona ID for tracing (default: P00)",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=2,
        help="Workflow phase to execute: 1 (Phase 1 Minimal PoC) or 2 (Phase 2 Full MAS with Safety & Reflection)",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Skip printing the execution trace summary",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Use simple spinner instead of live pipeline graph",
    )
    return parser.parse_args()


def print_trace_summary(result) -> None:
    """Print a Rich-formatted execution trace summary."""
    console.print()
    console.print(Rule("📊 Execution Trace", style="dim cyan"))

    # Agents
    agents_str = " → ".join(result.trace.agents_called)
    console.print(f"[bold]Agents called:[/bold] {agents_str}")

    # Retrieved docs
    if result.trace.retrieved_documents:
        console.print()
        console.print(f"[bold]Retrieved Knowledge Chunks ({len(result.trace.retrieved_documents)}):[/bold]")
        docs_table = Table(box=None, padding=(0, 2))
        docs_table.add_column("Chunk ID", style="cyan")
        docs_table.add_column("Title", style="white")
        docs_table.add_column("Topic", style="magenta")
        for doc in result.trace.retrieved_documents:
            docs_table.add_row(doc.get("chunk_id", ""), doc.get("title", ""), doc.get("topic", ""))
        console.print(docs_table)

    # Safety Validation Summary
    if result.trace.validation_results:
        val = result.trace.validation_results
        val_status = val.get("status", "UNKNOWN")
        status_color = "green" if val_status == "PASS" else "red"
        console.print()
        console.print(f"[bold]🛡️ Safety Audit:[/bold] [{status_color}]{val_status}[/{status_color}] — {val.get('summary', '')}")
        violations = val.get("violations", [])
        for v in violations:
            console.print(f"   ⚠️ [{v.get('severity', 'WARN')}] {v.get('pose', '')}: {v.get('recommendation', '')}")

    # Critic & Reflection Summary
    if result.trace.critic_result:
        crit = result.trace.critic_result
        crit_decision = crit.get("decision", "UNKNOWN")
        decision_color = "green" if crit_decision == "PASS" else "yellow"
        console.print()
        console.print(f"[bold]🔍 Critic Reflection:[/bold] [{decision_color}]{crit_decision}[/{decision_color}] (Score: {crit.get('score', '?')}/10)")
        if crit.get("critique"):
            console.print(f"   [dim]{crit.get('critique')}[/dim]")

    # State changes
    if result.trace.state_changes:
        console.print()
        console.print("[bold]State Transitions (Blackboard):[/bold]")
        for change in result.trace.state_changes:
            console.print(f"   • [dim]{change.get('key')}[/dim] written by [cyan]{change.get('agent')}[/cyan]")

    # Timing & run metadata
    console.print()
    console.print(f"[dim]Run ID:[/dim] [bold]{result.run_id}[/bold]")
    console.print(f"[dim]Session ID:[/dim] {result.session_id}")
    console.print(f"[dim]Phase:[/dim] Phase {result.phase}")
    if result.phase == 2:
        console.print(f"[dim]Revisions:[/dim] {result.trace.revision_count}")
    console.print(f"[dim]Total latency:[/dim] {result.trace.total_latency_ms:.0f} ms")
    if result.trace_path:
        console.print(f"[dim]Trace saved to:[/dim] {result.trace_path}")


def main() -> int:
    args = parse_args()

    phase_label = "Phase 1 Minimal Proof of Concept" if args.phase == 1 else "Phase 2 Working Multi-Agent Prototype"
    console.print()
    console.print(Panel.fit(
        f"[bold magenta]🧘 YogaSutra[/bold magenta] — Multi-Agent AI Yoga Planner\n"
        f"[dim]{phase_label} | Google ADK[/dim]",
        border_style="magenta",
    ))
    console.print()
    console.print(f"[bold]User input:[/bold] {args.input}")
    console.print()

    # Import here so dotenv is loaded before google-adk initialises
    from app.runner import run_yogasutra  # noqa: PLC0415

    if args.no_graph:
        # ── Fallback: simple spinner ────────────────────────────────────────
        with console.status("[bold green]Running YogaSutra agents...[/bold green]", spinner="dots"):
            result = run_yogasutra(
                user_input=args.input,
                persona_id=args.persona,
                phase=args.phase,
            )
    else:
        # ── Live pipeline architecture graph ────────────────────────────────
        from app.ui.live_graph import PipelineGraph  # noqa: PLC0415

        graph = PipelineGraph(console=console, user_input=args.input, phase=args.phase)

        with graph.live():
            result = run_yogasutra(
                user_input=args.input,
                persona_id=args.persona,
                phase=args.phase,
                on_agent_start=graph.activate,
                on_agent_done=graph.complete,
                on_agent_error=graph.error,
            )

    # Print final response
    console.print(Rule("🧘 Practice Plan", style="magenta"))
    console.print(Markdown(result.final_response))

    # Print trace summary
    if not args.no_trace:
        print_trace_summary(result)

    console.print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
