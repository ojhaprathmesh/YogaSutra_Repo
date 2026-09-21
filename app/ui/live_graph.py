"""
app/ui/live_graph.py — Live Terminal Architecture Visualizer

Renders the full YogaSutra multi-agent pipeline in the terminal, matching
the .mmd architecture diagram exactly:

  ┌─ User Request ──────────────────────────────┐
  └─────────────────────┬───────────────────────┘
                        ▼
  ┌─ 🎯  Root Agent ────────────────────────────┐
  └─────────────────────┬───────────────────────┘
          ╔═════════════╩═════════════╗
          ▼                           ▼
  ┌─ 👤 Profile ─┐           ┌─ 📚 Knowledge ─┐
          ╚═════════════╦═════════════╝
                        ▼
  ┌─ 📁 Planner ── 🔒 Safety ── 🔍 Critic ── ✨ Response ─┐
                        ▼
  ┌─ 💻 Terminal Output ───────────────────────┐

  ─────────────────────────────────────────────────────────
  ┌─ 📋 BB ──────────┐  ┌─ 👥 AGENTS ──────┐  ┌─ 🔧 TOOLS ──┐
  │  S_PROF  ✓       │  │  A1 Profile  ✓   │  │  T_RAG  ✓   │
  │  S_EVID  ✓       │  │  A2 Knowledge✓   │  │  T_DUR  ✓   │
  │  ...             │  │  ...             │  │  ...         │
  └──────────────────┘  └──────────────────┘  └─────────────┘
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Generator

from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text


# ---------------------------------------------------------------------------
# Geometry: two-column canvas for the pipeline section
# ---------------------------------------------------------------------------

PANEL_W   = 48     # width of each agent panel in the pipeline
COL_PAD   = 2      # padding each side in 2-column Columns
CELL_W    = PANEL_W + 2 * COL_PAD   # 52
CANVAS_W  = 2 * CELL_W              # 104

L_CENTER  = CELL_W // 2             # 26 — centre of left cell
R_CENTER  = CELL_W + CELL_W // 2   # 78 — centre of right cell
CENTER    = CANVAS_W // 2           # 52 — canvas midpoint / single-panel centre

# Geometry: three-column canvas for the subgraph section
# Table.grid never wraps columns — use it instead of Columns
# 3 panels × 30 width + 2 × 1 padding = 92 chars → fits cleanly in 100-char terminals
TRIPLE_W   = 30    # width of each subgraph panel
TRIPLE_PAD = 1     # padding between columns in Table.grid

# ---------------------------------------------------------------------------
# Pre-computed connector strings (exactly CANVAS_W = 104 chars wide)
# Align.center() of a 104-char string always lands at the same terminal
# position as Align.center() of the 104-char Columns — alignment guaranteed.
# ---------------------------------------------------------------------------

def _conn(*rows: str) -> Text:
    return Text("\n".join(rows), style=Style(color="grey50"), no_wrap=True)


FORK = _conn(
    " " * L_CENTER
    + "╔" + "═" * (CENTER - L_CENTER - 1)
    + "╩" + "═" * (R_CENTER - CENTER - 1)
    + "╗" + " " * (CANVAS_W - R_CENTER - 1),

    " " * L_CENTER + "║"
    + " " * (R_CENTER - L_CENTER - 1)
    + "║" + " " * (CANVAS_W - R_CENTER - 1),

    " " * L_CENTER + "▼"
    + " " * (R_CENTER - L_CENTER - 1)
    + "▼" + " " * (CANVAS_W - R_CENTER - 1),
)

MERGE = _conn(
    " " * L_CENTER
    + "╚" + "═" * (CENTER - L_CENTER - 1)
    + "╦" + "═" * (R_CENTER - CENTER - 1)
    + "╝" + " " * (CANVAS_W - R_CENTER - 1),

    " " * CENTER + "║" + " " * (CANVAS_W - CENTER - 1),
    " " * CENTER + "▼" + " " * (CANVAS_W - CENTER - 1),
)

VERT = _conn(
    " " * CENTER + "│" + " " * (CANVAS_W - CENTER - 1),
    " " * CENTER + "▼" + " " * (CANVAS_W - CENTER - 1),
)

# ---------------------------------------------------------------------------
# Node / graph definitions (names match ADK agent names exactly)
# All emojis use standard single-codepoint wide glyphs (ea_width='W')
# without variation selector-16 (\ufe0f) to ensure pixel-perfect panel
# border alignment across all terminal emulators (macOS Terminal, iTerm, etc.)
# ---------------------------------------------------------------------------

PIPELINE_NODES_PHASE1: list[str] = [
    "RootAgent",
    "ProfileAgent",
    "KnowledgeAgent",
    "PlannerAgent",
    "ResponseAgent",
]

PIPELINE_NODES: list[str] = [
    "RootAgent",
    "ProfileAgent",
    "KnowledgeAgent",
    "PlannerAgent",
    "SafetyAgent",
    "CriticAgent",
    "ResponseAgent",
]

NODE_LABELS: dict[str, str] = {
    "RootAgent":      "🎯  Root Agent",
    "ProfileAgent":   "👤  Profile Agent",
    "KnowledgeAgent": "📚  Knowledge Agent",
    "PlannerAgent":   "📁  Planner Agent",
    "SafetyAgent":    "🔒  Safety Agent",
    "CriticAgent":    "🔍  Critic Agent",
    "ResponseAgent":  "✨  Response Agent",
}

NODE_DESC: dict[str, str] = {
    "RootAgent":      "Orchestrates the full pipeline",
    "ProfileAgent":   "Extracts user profile & constraints",
    "KnowledgeAgent": "RAG retrieval from yoga corpus",
    "PlannerAgent":   "Generates pose sequence & plan",
    "SafetyAgent":    "Deterministic safety audit",
    "CriticAgent":    "Pedagogical reflection & revision",
    "ResponseAgent":  "Synthesises final human response",
}

# BB subgraph: session-state field → agent that writes it
BB_NODES: list[tuple[str, str, str]] = [
    ("S_PROF", "user_profile",       "ProfileAgent"),
    ("S_EVID", "retrieved_evidence", "KnowledgeAgent"),
    ("S_PLAN", "candidate_plan",     "PlannerAgent"),
    ("S_VAL",  "validation_results", "SafetyAgent"),
    ("S_CRIT", "critique",           "CriticAgent"),
    ("S_RESP", "final_response",     "ResponseAgent"),
]

# AGENTS subgraph: MMD node ID → agent name
AGENT_NODES: list[tuple[str, str]] = [
    ("A1", "ProfileAgent"),
    ("A2", "KnowledgeAgent"),
    ("A3", "PlannerAgent"),
    ("A4", "SafetyAgent"),
    ("A5", "CriticAgent"),
    ("A6", "ResponseAgent"),
]

# TOOLS subgraph: MMD node ID → display label (≤18 chars) → calling agent (None = corpus)
TOOL_NODES: list[tuple[str, str, str | None]] = [
    ("T_RAG",   "retrieve_yoga()",  "KnowledgeAgent"),
    ("T_DUR",   "calc_duration()",  "PlannerAgent"),
    ("T_SAFE",  "validate_safe()",  "SafetyAgent"),
    ("CORPUS",  "289-Chunk Corpus", None),
]

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


# ---------------------------------------------------------------------------
# PipelineGraph
# ---------------------------------------------------------------------------


class PipelineGraph:
    """
    Live terminal visualizer for the YogaSutra multi-agent pipeline.

    Node states (per agent):
        idle    — not yet reached (grey)
        active  — currently executing (bright yellow + animated spinner)
        done    — completed (bright green)
        error   — failed (bright red)
    """

    def __init__(
        self,
        console: Console | None = None,
        user_input: str = "",
        phase: int = 2,
    ) -> None:
        self._console    = console or Console()
        self._user_input = user_input
        self._phase      = phase
        self._nodes      = PIPELINE_NODES_PHASE1 if phase == 1 else PIPELINE_NODES
        self._state: dict[str, str] = {n: "idle" for n in self._nodes}
        self._messages: dict[str, str] = {}
        self._spinner_idx: int = 0
        self._live: Live | None = None
        self._stop_event = threading.Event()
        self._tick_thread: threading.Thread | None = None
        self._start_time: float = time.monotonic()
        self._agent_start: dict[str, float] = {}
        self._agent_done_time: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def activate(self, name: str, message: str = "") -> None:
        """Mark an agent as currently executing."""
        if name in self._state:
            self._state[name] = "active"
            self._messages[name] = message
            self._agent_start.setdefault(name, time.monotonic())
            self._refresh()

    def complete(self, name: str, message: str = "") -> None:
        """Mark an agent as successfully completed and freeze its timer."""
        if name in self._state:
            self._state[name] = "done"
            if name not in self._agent_start:
                self._agent_start[name] = time.monotonic()
            if name not in self._agent_done_time:
                self._agent_done_time[name] = time.monotonic()
            if message:
                self._messages[name] = message
            self._refresh()

    def error(self, name: str, message: str = "") -> None:
        """Mark an agent as failed and freeze its timer."""
        if name in self._state:
            self._state[name] = "error"
            if name not in self._agent_start:
                self._agent_start[name] = time.monotonic()
            if name not in self._agent_done_time:
                self._agent_done_time[name] = time.monotonic()
            if message:
                self._messages[name] = message
            self._refresh()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def live(self) -> Generator["PipelineGraph", None, None]:
        self._start_time = time.monotonic()
        with Live(
            self._render(),
            console=self._console,
            refresh_per_second=10,
            transient=False,
        ) as live_ctx:
            self._live = live_ctx
            self._stop_event.clear()
            self._tick_thread = threading.Thread(target=self._tick, daemon=True)
            self._tick_thread.start()
            try:
                yield self
            finally:
                self._stop_event.set()
                if self._tick_thread:
                    self._tick_thread.join(timeout=1.0)
                self._refresh()
                self._live = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        while not self._stop_event.is_set():
            self._spinner_idx = (self._spinner_idx + 1) % len(SPINNER_FRAMES)
            self._refresh()
            time.sleep(0.1)

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def _elapsed(self) -> str:
        return f"{time.monotonic() - self._start_time:.1f}s"

    def _agent_elapsed(self, name: str) -> str:
        start = self._agent_start.get(name)
        if not start:
            return ""
        end = self._agent_done_time.get(name, time.monotonic())
        return f"{end - start:.1f}s"

    def _state_icon_color(self, state: str) -> tuple[str, str]:
        """Return (icon, rich_color) for a given node state."""
        if state == "done":
            return "✓", "bright_green"
        if state == "active":
            return SPINNER_FRAMES[self._spinner_idx], "bright_yellow"
        if state == "error":
            return "✗", "bright_red"
        return "○", "grey50"

    # ------------------------------------------------------------------
    # Pipeline node renderer (used in the top agent-flow section)
    # ------------------------------------------------------------------

    def _node(self, name: str) -> Panel:
        state  = self._state[name]
        label  = NODE_LABELS[name]
        desc   = NODE_DESC[name]
        msg    = self._messages.get(name, "")

        if state == "idle":
            icon, border, title_sty = "○", "grey50", Style(color="grey50")
            body = Text(f"  {desc}", style=Style(color="grey50"))

        elif state == "active":
            spin = SPINNER_FRAMES[self._spinner_idx]
            icon, border = spin, "bright_yellow"
            title_sty = Style(color="bright_yellow", bold=True)
            elapsed   = self._agent_elapsed(name)
            body      = Text()
            body.append(f"  {desc}\n", style=Style(color="grey70"))
            body.append("  ● Running... ", style=Style(color="bright_yellow"))
            body.append(elapsed, style=Style(color="yellow", dim=True))
            if msg:
                body.append(f"\n  {msg}", style=Style(color="grey70", italic=True))

        elif state == "done":
            icon, border = "✓", "green"
            title_sty = Style(color="bright_green", bold=True)
            elapsed   = self._agent_elapsed(name)
            body      = Text()
            body.append(f"  {desc}\n", style=Style(color="grey70"))
            body.append("  ✔ Completed ", style=Style(color="bright_green"))
            body.append(elapsed, style=Style(color="green", dim=True))
            if msg:
                body.append(f"\n  {msg}", style=Style(color="grey70", italic=True))

        else:  # error
            icon, border = "✗", "red"
            title_sty = Style(color="bright_red", bold=True)
            body      = Text()
            body.append(f"  {desc}\n", style=Style(color="grey70"))
            body.append("  ✘ Failed", style=Style(color="bright_red"))
            if msg:
                body.append(f"\n  {msg}", style=Style(color="red", italic=True))

        title = Text()
        title.append(f" {icon} ", style=title_sty)
        title.append(label, style=title_sty)

        return Panel(body, title=title, title_align="left",
                     border_style=border, width=PANEL_W, padding=(0, 1))

    # ------------------------------------------------------------------
    # Subgraph panels (BB / AGENTS / TOOLS — match .mmd structure)
    # ------------------------------------------------------------------

    def _subgraph_bb(self) -> Panel:
        """📋 BB — Shared Session State subgraph."""
        # Content width = TRIPLE_W - 2 (borders) = 30 chars
        body = Text(no_wrap=True)
        body.append(" Shared Session State\n",
                    style=Style(color="grey50", italic=True))
        body.append(" ─" * 14 + "\n", style=Style(color="grey50", dim=True))

        for node_id, field, agent in BB_NODES:
            st = self._state.get(agent, "idle")
            icon, color = self._state_icon_color(st)
            # Max line: 1+1+1+6+1+18 = 28 chars ≤ 30 ✓
            trunc = field[:18] + ("…" if len(field) > 18 else "")
            body.append(f" {icon} ", style=Style(color=color, bold=(st == "done")))
            body.append(f"{node_id:<6} ", style=Style(color="grey70", bold=True))
            body.append(trunc, style=Style(color=color))
            body.append("\n")

        title = Text(" 📋 BB", style=Style(color="bright_cyan", bold=True))
        return Panel(body, title=title, title_align="left",
                     border_style="cyan", width=TRIPLE_W, padding=(0, 0))

    def _subgraph_agents(self) -> Panel:
        """👥 AGENTS — Specialized LLM Agents subgraph."""
        # Content width = TRIPLE_W - 2 (borders) = 30 chars
        body = Text(no_wrap=True)
        body.append(" Specialized LLM Agents\n",
                    style=Style(color="grey50", italic=True))
        body.append(" ─" * 14 + "\n", style=Style(color="grey50", dim=True))

        for node_id, agent in AGENT_NODES:
            st = self._state.get(agent, "idle")
            icon, color = self._state_icon_color(st)
            # Max line: 1+1+1+3+1+14 = 21 chars ≤ 30 ✓
            body.append(f" {icon} ", style=Style(color=color, bold=(st == "done")))
            body.append(f"{node_id:<3} ", style=Style(color="grey70", bold=True))
            body.append(agent, style=Style(color=color))
            body.append("\n")

        title = Text(" 👥 AGENTS", style=Style(color="bright_yellow", bold=True))
        return Panel(body, title=title, title_align="left",
                     border_style="yellow", width=TRIPLE_W, padding=(0, 0))

    def _subgraph_tools(self) -> Panel:
        """🔧 TOOLS — Deterministic Tools & Corpus subgraph."""
        # Content width = TRIPLE_W - 2 (borders) = 30 chars
        body = Text(no_wrap=True)
        body.append(" Deterministic Tools\n",
                    style=Style(color="grey50", italic=True))
        body.append(" ─" * 14 + "\n", style=Style(color="grey50", dim=True))

        for node_id, label, agent in TOOL_NODES:
            # Max line: 1+1+1+6+1+18 = 28 chars ≤ 30 ✓
            trunc = label[:18] + ("…" if len(label) > 18 else "")
            if agent is None:
                # Corpus — always visible, magenta
                body.append(" 📦 ", style=Style(color="bright_magenta"))
                body.append(f"{node_id:<6} ", style=Style(color="grey70", bold=True))
                body.append(trunc, style=Style(color="bright_magenta"))
            else:
                st = self._state.get(agent, "idle")
                icon, color = self._state_icon_color(st)
                body.append(f" {icon} ", style=Style(color=color, bold=(st == "done")))
                body.append(f"{node_id:<6} ", style=Style(color="grey70", bold=True))
                body.append(trunc, style=Style(color=color))
            body.append("\n")

        # Pad to match BB (8 rows: subtitle+sep+6 fields+blank) and AGENTS (same).
        # TOOLS has 6 rows (subtitle+sep+4 tools) → add 2 blank lines.
        body.append("\n\n")

        title = Text(" 🔧 TOOLS", style=Style(color="bright_magenta", bold=True))
        return Panel(body, title=title, title_align="left",
                     border_style="magenta", width=TRIPLE_W, padding=(0, 0))

    # ------------------------------------------------------------------
    # Top-level render
    # ------------------------------------------------------------------

    def _render(self):
        elapsed   = self._elapsed()
        done_n    = sum(1 for s in self._state.values() if s == "done")
        active_ns = [n for n, s in self._state.items() if s == "active"]
        total     = len(self._nodes)

        rows: list = []

        # ── Header ──────────────────────────────────────────────────────
        hdr = Text()
        hdr.append("  🧘 YogaSutra", style=Style(color="bright_magenta", bold=True))
        subtitle = " — Phase 1 Minimal Proof of Concept" if self._phase == 1 else " — Multi-Agent Pipeline Architecture"
        hdr.append(subtitle, style=Style(color="grey70"))
        hdr.append(f"   ⏱ {elapsed}", style=Style(color="grey50", italic=True))
        rows += [Text(), hdr, Rule(style="dim magenta"), Text()]

        # ── User Request (static context node) ──────────────────────────
        short = (self._user_input[:48] + "…") if len(self._user_input) > 48 else self._user_input
        user_panel = Panel(
            Text(f"  {short}", style=Style(color="white")),
            title=Text(" 💬 User Request", style=Style(color="white", bold=True)),
            title_align="left", border_style="grey50",
            width=PANEL_W, padding=(0, 1),
        )
        rows += [Align.center(user_panel), Align.center(VERT)]

        # ── Root Agent ──────────────────────────────────────────────────
        rows += [Align.center(self._node("RootAgent")), Text()]

        # ── Fork → Profile + Knowledge (parallel) ───────────────────────
        rows.append(Align.center(FORK))
        rows.append(Text())
        rows.append(Align.center(
            Columns(
                [self._node("ProfileAgent"), self._node("KnowledgeAgent")],
                padding=(0, COL_PAD), equal=True, expand=False,
            )
        ))
        rows.append(Text())

        # ── Merge ────────────────────────────────────────────────────────
        rows.append(Align.center(MERGE))
        rows.append(Text())

        # ── Sequential agents ────────────────────────────────────────────
        # Phase 1: Planner -> Response; Phase 2: Planner -> Safety -> Critic -> Response
        seq_agents = ["PlannerAgent", "ResponseAgent"] if self._phase == 1 else ["PlannerAgent", "SafetyAgent", "CriticAgent", "ResponseAgent"]
        for name in seq_agents:
            rows.append(Align.center(self._node(name)))
            if name != "ResponseAgent":
                rows += [Text(), Align.center(VERT), Text()]

        # ── Terminal Output (static result node) ─────────────────────────
        out_done  = self._state.get("ResponseAgent") == "done"
        out_color = "bright_green" if out_done else "grey50"
        out_icon  = "✓" if out_done else "○"
        out_panel = Panel(
            Text("  Trace JSON + Rich terminal display",
                 style=Style(color=out_color)),
            title=Text(f" {out_icon} 💻  Terminal Output",
                       style=Style(color=out_color, bold=out_done)),
            title_align="left", border_style=out_color,
            width=PANEL_W, padding=(0, 1),
        )
        rows += [Text(), Align.center(VERT), Text(), Align.center(out_panel)]

        # ── Subgraph section (BB / AGENTS / TOOLS — Phase 2 only) ───────
        if self._phase != 1:
            rows += [Text(), Rule(style="dim cyan"), Text()]

            sg_label = Text(
                "  Subgraph Architecture  ·  mirrors docs/diagrams/multi-agent_and_blackboard_architecture.mmd",
                style=Style(color="grey50", italic=True),
            )
            rows += [Align.center(sg_label), Text()]

            sg_tbl = Table.grid(padding=(0, TRIPLE_PAD))
            sg_tbl.add_column(width=TRIPLE_W)
            sg_tbl.add_column(width=TRIPLE_W)
            sg_tbl.add_column(width=TRIPLE_W)
            sg_tbl.add_row(
                self._subgraph_bb(),
                self._subgraph_agents(),
                self._subgraph_tools(),
            )
            rows.append(Align.center(sg_tbl))

        # ── Status bar ───────────────────────────────────────────────────
        rows += [Text(), Rule(style="dim magenta")]
        status = Text()
        if active_ns:
            status.append("  ● ", style=Style(color="bright_yellow"))
            status.append(f"Active: {', '.join(active_ns)}",
                          style=Style(color="yellow"))
        elif done_n == total:
            status.append("  ✔ ", style=Style(color="bright_green"))
            status.append("Pipeline complete",
                          style=Style(color="bright_green", bold=True))
        else:
            status.append("  ○ ", style=Style(color="grey50"))
            status.append("Initializing…", style=Style(color="grey50"))
        status.append(f"   {done_n}/{total} agents done",
                      style=Style(color="grey50", italic=True))
        rows += [status, Text()]

        return Group(*rows)
