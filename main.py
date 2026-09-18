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
logging.getLogger("google.adk").setLevel(logging.ERROR)

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
        "--no-trace",
        action="store_true",
        help="Skip printing the execution trace summary",
    )
    return parser.parse_args()


def print_trace_summary(result) -> None:
    """Print a Rich-formatted execution trace summary."""
    console.print()
    console.print(Rule("📊 Execution Trace", style="dim cyan"))

    # Agents
    agents_table = Table(title="Agents Called", show_header=True, header_style="bold cyan")
    agents_table.add_column("Agent", style="cyan")
    agents_table.add_column("Latency (ms)", justify="right")
    agents_table.add_column("Tokens (in/out)", justify="right")

    for agent in result.trace.agents_called:
        latency = result.trace.agent_latency_ms.get(agent, 0.0)
        token_data = result.trace.token_usage.get(agent, {})
        token_str = f"{token_data.get('input_tokens', '?')}/{token_data.get('output_tokens', '?')}"
        agents_table.add_row(agent, f"{latency:.1f}", token_str)

    console.print(agents_table)

    # Retrieved documents
    if result.trace.retrieved_documents:
        docs_table = Table(title="Retrieved Evidence", show_header=True, header_style="bold green")
        docs_table.add_column("Chunk ID", style="green")
        docs_table.add_column("Title")
        docs_table.add_column("Topic")
        for doc in result.trace.retrieved_documents:
            docs_table.add_row(doc.get("chunk_id", ""), doc.get("title", ""), doc.get("topic", ""))
        console.print(docs_table)

    # State changes
    if result.trace.state_changes:
        console.print()
        console.print("[bold]State fields populated:[/bold]")
        for change in result.trace.state_changes:
            console.print(f"  ✓ [cyan]{change['field']}[/cyan] — {change['summary']}")

    # Summary
    console.print()
    console.print(f"[dim]Run ID:[/dim] [bold]{result.run_id}[/bold]")
    console.print(f"[dim]Session ID:[/dim] {result.session_id}")
    console.print(f"[dim]Total latency:[/dim] {result.trace.total_latency_ms:.0f} ms")
    if result.trace_path:
        console.print(f"[dim]Trace saved to:[/dim] {result.trace_path}")


def main() -> int:
    args = parse_args()

    console.print()
    console.print(Panel.fit(
        "[bold magenta]🧘 YogaSutra[/bold magenta] — Multi-Agent AI Yoga Planner\n"
        "[dim]Phase 1 Proof of Concept | Google ADK[/dim]",
        border_style="magenta",
    ))
    console.print()
    console.print(f"[bold]User input:[/bold] {args.input}")
    console.print()

    with console.status("[bold green]Running YogaSutra agents...[/bold green]", spinner="dots"):
        # Import here so dotenv is loaded before google-adk initialises
        from app.runner import run_yogasutra  # noqa: PLC0415
        result = run_yogasutra(user_input=args.input, persona_id=args.persona)

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
