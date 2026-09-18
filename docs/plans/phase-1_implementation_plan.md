# YogaSutra — Phase 1 Implementation Plan

## Goal

Build a **Minimal Google ADK Proof of Concept** that demonstrates the proposed multi-agent architecture is technically feasible.

The demo scenario is:
> "I am a beginner and have 20 minutes in the morning. I want a relaxing yoga practice."

The system will show: request → profile extraction → RAG retrieval → planning → state transfer → final response → basic execution trace.

---

## Proposed Changes

### Repository Scaffolding

Create the full project layout as specified in Section 39 of PLAN.md (Phase 1 relevant directories only).

#### [NEW] `yoga-sutra/pyproject.toml`
Project metadata, dependencies, and tooling (uv, ruff, pytest).

#### [NEW] `yoga-sutra/README.md`
Quick-start and architecture overview.

#### [NEW] `yoga-sutra/.env.example`
Environment variable template (GOOGLE_API_KEY, DB connection strings).

#### [NEW] `yoga-sutra/.gitignore`
Standard Python + ADK gitignore.

---

### App Package

#### [NEW] `yoga-sutra/app/__init__.py`

#### [NEW] `yoga-sutra/app/state.py`
Pydantic models for the shared **ADK Session State** (blackboard):
```
YogaSessionState:
  - user_input: str
  - user_profile: UserProfile | None
  - retrieved_evidence: list[EvidenceChunk]
  - candidate_plan: PracticePlan | None
  - validation_results: dict
  - critique: str | None
  - revision_count: int
  - final_response: str | None
  - execution_trace: list[TraceStep]
```

---

### Agents

#### [NEW] `yoga-sutra/app/agents/root/agent.py`
**Root Manager Agent** — classifies the request and orchestrates the sequential Phase 1 workflow:
`Profile → Knowledge → Planner → Final Response`

Uses ADK `LlmAgent` with `sub_agents` delegation.

#### [NEW] `yoga-sutra/app/agents/profile/agent.py`
**Profile Agent** — extracts structured user profile from natural language:
- experience_level, available_duration, preferred_time, goal, constraints

Writes extracted profile to session state.

#### [NEW] `yoga-sutra/app/agents/knowledge/agent.py`
**Knowledge/RAG Agent** — uses `retrieve_yoga_knowledge()` tool to fetch relevant evidence chunks, then grounds the response with source metadata.

ReAct-style: the agent calls the tool, reads results, then decides what evidence to include.

#### [NEW] `yoga-sutra/app/agents/planner/agent.py`
**Practice Planner Agent** — generates a structured practice plan from the user profile + retrieved evidence using CoT-style structured prompting.

Output schema: `{ goal, duration, sequence: [...], intensity, rationale, evidence_sources }`.

---

### Tools

#### [NEW] `yoga-sutra/app/tools/retrieval.py`
`retrieve_yoga_knowledge(query: str, top_k: int) -> list[EvidenceChunk]`

**Phase 1**: uses a **static in-memory corpus** (no database required yet). Contains ~20 yoga knowledge chunks covering beginner asanas, pranayama, relaxation sequences, and basic philosophy.

Each chunk has metadata: `{ chunk_id, title, topic, section, source_type }`.

#### [NEW] `yoga-sutra/app/tools/profile.py`
`get_user_profile(session_id: str) -> UserProfile | None`

Returns any previously stored profile (in-memory dict for Phase 1).

#### [NEW] `yoga-sutra/app/tools/duration.py`
`calculate_practice_duration(sequence: list[PoseItem]) -> int`

Deterministic tool: sums up pose durations and returns total minutes.

---

### Tracing

#### [NEW] `yoga-sutra/app/tracing/trace.py`
`TraceStep` dataclass and `ExecutionTrace` collector. Writes structured JSON traces to `data/traces/`.

Output matches the per-run trace schema from Section 27 of PLAN.md:
- run_id, agents_called, tools_called, retrieved_documents, agent_latency_ms, total_latency_ms, token_usage, final_output

---

### Runner / Entry Point

#### [NEW] `yoga-sutra/app/runner.py`
ADK `InMemoryRunner` setup — creates a session, runs the root agent with a user message, collects the structured trace, and prints the final response + execution trace to stdout.

#### [NEW] `yoga-sutra/main.py`
CLI entry point. Accepts `--input` flag for the user query. Runs the demo scenario by default.

---

### Static Knowledge Corpus (Phase 1)

#### [NEW] `yoga-sutra/data/knowledge/corpus.json`
~20 curated yoga knowledge chunks for Phase 1 RAG:
- beginner relaxation poses (Child's Pose, Legs-up-the-Wall, Savasana, etc.)
- 20-minute morning sequence structure
- pranayama basics (Nadi Shodhana, Box Breathing)
- alignment cues and contraindications for beginners
- philosophical grounding (yoga nidra, relaxation intention)

Each chunk has the metadata schema from Section 17 of PLAN.md.

---

### Tests

#### [NEW] `yoga-sutra/tests/test_tools.py`
Deterministic unit tests:
- `retrieve_yoga_knowledge` returns correct number of chunks
- `calculate_practice_duration` sums correctly
- `get_user_profile` returns None for unknown sessions

#### [NEW] `yoga-sutra/tests/test_state.py`
Pydantic model validation tests.

---

## Phase 1 Workflow (Sequential)

```
User Input
    ↓
Root Agent (classifies: Practice Mode)
    ↓
Profile Agent (extracts: beginner, 20min, morning, relaxation) → writes to state
    ↓
Knowledge Agent (calls retrieve_yoga_knowledge, selects evidence) → writes to state
    ↓
Practice Planner Agent (generates structured plan from profile + evidence) → writes to state
    ↓
Final Response (structured plan + execution trace printed to stdout)
```

---

## What is NOT built in Phase 1

Per PLAN.md Section "Do Not Build Yet":
- Safety & Constraint Agent
- Critic / Reflection Agent
- PostgreSQL / pgvector (static corpus only)
- MCP server
- Persistent memory / database
- ToT candidate planning
- Frontend (Next.js)
- Architecture benchmarking
- Production deployment

---

## Verification Plan

### Automated Tests
```bash
cd yoga-sutra
uv run pytest tests/ -v
```

### Manual Demo Run
```bash
uv run python main.py --input "I am a beginner and have 20 minutes in the morning. I want a relaxing yoga practice."
```

Expected output:
1. Profile extraction printed (beginner, 20min, morning, relaxation)
2. Knowledge retrieval: 3–5 evidence chunks with source metadata
3. Structured practice plan (sequence of poses with durations)
4. Execution trace showing each agent, latency, and tool calls

### Trace Validation
- `data/traces/run_001.json` is written with the correct schema
- All required fields present (run_id, agents_called, tools_called, retrieved_documents, agent_latency_ms, total_latency_ms)

---

## Open Questions

> [!NOTE]
> **Google API Key**: A valid `GOOGLE_API_KEY` with Gemini access is required. Please ensure it is available in the environment or `.env` file before running.

> [!NOTE]
> **ADK Version**: The plan targets the latest stable `google-adk` package via `uv`. If a specific version is required, please specify.

> [!NOTE]
> **Model**: Phase 1 will use `gemini-2.0-flash` for all agents (cost-efficient for proof of concept). Per PLAN.md Section 31, per-agent model tuning is a Phase 2B experiment.
