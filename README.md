# YogaSutra 🧘

**Transparent Multi-Agent AI System for Personalised Yoga Practice**

> Phase 1 Proof of Concept — Google ADK Multi-Agent Architecture

---

## What Is This?

YogaSutra is a multi-agent AI system that generates personalised yoga practice plans using the [Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/). It demonstrates a clean, auditable multi-agent pipeline where:

- Every agent's decision is traceable
- Every piece of evidence is sourced
- Every step of the reasoning process is visible

### Demo Scenario (Phase 1)

> *"I am a beginner and have 20 minutes in the morning. I want a relaxing yoga practice."*

The system processes this through four agents and returns a structured, evidence-grounded practice plan.

---

## Architecture

```
User Input
    ↓
Root Agent         → classifies request, orchestrates pipeline
    ↓
Profile Agent      → extracts: experience=beginner, duration=20min, goal=relaxation
    ↓
Knowledge Agent    → retrieves relevant evidence chunks (in-memory RAG)
    ↓
Practice Planner   → generates structured plan grounded in evidence
    ↓
Final Response     → formatted plan + execution trace
```

### Agents

| Agent | Role | ADK Pattern |
|-------|------|-------------|
| `RootAgent` | Orchestrator | `LlmAgent` with `sub_agents` |
| `ProfileAgent` | Profile extraction | `LlmAgent` with `output_key` |
| `KnowledgeAgent` | RAG retrieval | `LlmAgent` (ReAct) with tool |
| `PlannerAgent` | Plan generation | `LlmAgent` with CoT prompting |

---

## Quick Start

### Prerequisites

- Python 3.11+
- A valid `GOOGLE_API_KEY` with Gemini access ([get one here](https://aistudio.google.com/apikey))

### Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd yogasutra

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Run the Demo

```bash
python3 main.py
```

With a custom input:

```bash
python3 main.py --input "I am intermediate level and have 45 minutes. I want to work on flexibility."
```

Options:
- `--input / -i`: Your yoga request (default: Phase 1 demo scenario)
- `--persona / -p`: Persona ID for tracing (default: `P00`)
- `--no-trace`: Skip printing the execution trace summary

### Run Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
yogasutra/
├── app/
│   ├── agents/
│   │   ├── root/agent.py        # Root orchestrator
│   │   ├── profile/agent.py     # Profile extraction agent
│   │   ├── knowledge/agent.py   # RAG knowledge agent
│   │   └── planner/agent.py     # Practice planner agent
│   ├── tools/
│   │   ├── retrieval.py         # retrieve_yoga_knowledge()
│   │   ├── profile.py           # get_user_profile() / store_user_profile()
│   │   └── duration.py          # calculate_practice_duration()
│   ├── tracing/
│   │   └── trace.py             # ExecutionTrace collector
│   ├── state.py                 # Pydantic session state models
│   └── runner.py                # InMemoryRunner setup
├── data/
│   ├── knowledge/corpus.json    # 20 yoga knowledge chunks (Phase 1 RAG)
│   └── traces/                  # JSON trace files (auto-generated)
├── tests/
│   ├── test_tools.py            # Deterministic tool tests
│   └── test_state.py            # Pydantic model tests
├── main.py                      # CLI entry point
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Session State (Blackboard)

All agents communicate via a shared ADK session state (`YogaSessionState`):

| Field | Populated by | Type |
|-------|-------------|------|
| `user_input` | Runner | `str` |
| `user_profile` | ProfileAgent | `UserProfile` |
| `retrieved_evidence` | KnowledgeAgent | `list[EvidenceChunk]` |
| `candidate_plan` | PlannerAgent | `PracticePlan` |
| `final_response` | RootAgent | `str` |
| `execution_trace` | Runner | `list[TraceStep]` |

---

## Execution Trace

Every run produces a JSON trace file at `data/traces/<run_id>.json` matching PLAN.md §27:

```json
{
  "run_id": "run_abc12345",
  "persona_id": "P00",
  "session_id": "session_...",
  "input": "I am a beginner...",
  "agents_called": ["RootAgent", "ProfileAgent", "KnowledgeAgent", "PlannerAgent"],
  "tools_called": [...],
  "retrieved_documents": [...],
  "agent_latency_ms": {"ProfileAgent": 412.3, ...},
  "total_latency_ms": 4821.0,
  "token_usage": {...},
  "final_output": "..."
}
```

---

## What Phase 1 Does NOT Include

Per PLAN.md "Do Not Build Yet":
- ❌ Safety & Constraint Agent
- ❌ Critic / Reflection Agent  
- ❌ PostgreSQL / pgvector (static in-memory corpus only)
- ❌ MCP server
- ❌ Persistent memory / database
- ❌ Tree of Thought candidate planning
- ❌ Frontend (Next.js)
- ❌ Architecture benchmarking
- ❌ Production deployment

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | *(required)* | Gemini API key |
| `YOGASUTRA_MODEL` | `gemini-2.0-flash` | Model for all agents |
| `YOGASUTRA_TRACE_DIR` | `data/traces` | Where to save trace JSON files |
| `YOGASUTRA_LOG_LEVEL` | `INFO` | Log level |
