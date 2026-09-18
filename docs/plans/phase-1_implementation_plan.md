# YogaSutra — Phase 1 Implementation Plan

## Status: ✅ COMPLETE

> **Prototype is live.** All files implemented, 44/44 tests passing.
> Repository: [github.com/ojhaprathmesh/YogaSutra_Repo](https://github.com/ojhaprathmesh/YogaSutra_Repo)

---

## Goal

Build a **Minimal Google ADK Proof of Concept** that demonstrates the proposed multi-agent architecture is technically feasible.

The demo scenario is:
> "I am a beginner and have 20 minutes in the morning. I want a relaxing yoga practice."

The system shows: request → profile extraction → RAG retrieval → planning → state transfer → final response → basic execution trace.

---

## Phase 1 Definition of Done (PLAN.md §35)

| Item | Status |
|------|--------|
| Problem statement approved | ✅ PLAN.md |
| PEAS completed | ✅ PLAN.md §10 |
| Environment analysis completed | ✅ PLAN.md §11 |
| Agent responsibilities defined | ✅ PLAN.md §9 |
| Architecture diagrams completed | ✅ PLAN.md §13–14 |
| RAG strategy defined | ✅ PLAN.md §16–17 |
| Reasoning strategy defined | ✅ PLAN.md §21 |
| Communication strategy defined | ✅ PLAN.md §15 |
| Tech stack justified | ✅ PLAN.md §25 |
| Timeline completed | ✅ PLAN.md §36 |
| Team responsibility matrix completed | ✅ PLAN.md §37 |
| **Minimal ADK prototype demonstrated** | ✅ **Implemented** |

---

## Implemented Changes

### Repository Scaffolding ✅

Created the full project layout per PLAN.md §39.

#### [DONE] `pyproject.toml`
Project metadata, dependencies: `google-adk>=1.0.0`, `pydantic>=2.0.0`, `python-dotenv`, `rich`. Dev deps: `pytest`, `pytest-asyncio`, `ruff`.

#### [DONE] `README.md`
Quick-start, architecture overview, agent table, session state blackboard, trace schema, environment variables.

#### [DONE] `.env.example`
Environment variable template (`GOOGLE_API_KEY`, `YOGASUTRA_MODEL`, `YOGASUTRA_TRACE_DIR`).

#### [DONE] `.gitignore`
Python + ADK gitignore. `.env` excluded; `data/traces/*.json` excluded.

---

### App Package ✅

#### [DONE] `app/__init__.py`

#### [DONE] `app/state.py`
Pydantic models for the shared **ADK Session State** (blackboard):

```
YogaSessionState:
  - run_id: str            (auto-generated, format: run_<hex8>)
  - session_id: str
  - persona_id: str
  - user_input: str
  - request_mode: RequestMode
  - user_profile: UserProfile | None
  - retrieved_evidence: list[EvidenceChunk]
  - candidate_plan: PracticePlan | None
  - validation_results: dict     (Phase 2+)
  - critique: str | None         (Phase 2+)
  - revision_count: int          (Phase 2+)
  - final_response: str | None
  - agents_called: list[str]
  - execution_trace: list[TraceStep]
  - total_latency_ms: float
  - start_time: str
```

Supporting models: `UserProfile`, `EvidenceChunk`, `PoseItem`, `PracticePlan`, `TraceStep`.

Enumerations: `ExperienceLevel`, `PracticeGoal`, `RequestMode`.

---

### Agents ✅

#### [DONE] `app/agents/root/agent.py`
**Root Manager Agent** — classifies the request and orchestrates the sequential Phase 1 workflow:
`Profile → Knowledge → Planner → Final Response`

Uses ADK `LlmAgent` with `sub_agents=[profile_agent, knowledge_agent, planner_agent]`.
Produces a beautifully formatted final response after the pipeline completes.

#### [DONE] `app/agents/profile/agent.py`
**Profile Agent** — extracts structured user profile from natural language using a constrained JSON-output prompt.
Uses ADK `output_key="user_profile"` to write directly to session state.

Extracted fields: `experience_level`, `available_duration_minutes`, `preferred_time`, `goal`, `constraints`, `preferences`.

#### [DONE] `app/agents/knowledge/agent.py`
**Knowledge/RAG Agent** — ReAct-style: calls `retrieve_yoga_knowledge()`, reviews results, selects 3–5 most relevant chunks.
Uses ADK `output_key="retrieved_evidence"`. Grounds response with chunk metadata.

#### [DONE] `app/agents/planner/agent.py`
**Practice Planner Agent** — generates a structured practice plan using CoT-style prompting.
Tools: `calculate_practice_duration` (to verify sequence totals). Uses `output_key="candidate_plan"`.

Output schema: `{ goal, duration_minutes, intensity, sequence: [...], rationale, evidence_sources, expected_outcome }`.

---

### Tools ✅

#### [DONE] `app/tools/retrieval.py`
`retrieve_yoga_knowledge(query: str, top_k: int = 5) -> list[dict]`

**Phase 1**: TF-overlap keyword scoring over the static in-memory corpus loaded from `data/knowledge/corpus.json`. Returns chunks sorted by normalized relevance score (0–1). No vector database required.

Each returned chunk has the metadata schema from PLAN.md §17:
`{ chunk_id, document_id, title, section, topic, content, source_type, license, relevance_score }`.

#### [DONE] `app/tools/profile.py`
`get_user_profile(session_id: str) -> dict | None`
`store_user_profile(session_id: str, profile: dict) -> None`

In-memory dict store for Phase 1. Returns `None` for unknown sessions.

#### [DONE] `app/tools/duration.py`
`calculate_practice_duration(sequence: list[dict]) -> int`

Pure-Python deterministic tool. Sums `duration_minutes` across the pose sequence. Handles missing/invalid values gracefully.

---

### Tracing ✅

#### [DONE] `app/tracing/trace.py`
`ExecutionTrace` dataclass and collector. Writes structured JSON traces to `data/traces/`.

Output matches the per-run trace schema from PLAN.md §27:

```json
{
  "run_id": "run_abc12345",
  "persona_id": "P00",
  "session_id": "session_...",
  "input": "...",
  "agents_called": [],
  "tools_called": [],
  "retrieved_documents": [],
  "state_changes": [],
  "agent_latency_ms": {},
  "total_latency_ms": 0,
  "token_usage": {},
  "validation_results": {},
  "critic_result": {},
  "revision_count": 0,
  "final_output": "...",
  "evaluation": {}
}
```

---

### Runner / Entry Point ✅

#### [DONE] `app/runner.py`
ADK `InMemoryRunner` setup — creates a session with initial blackboard state, runs the root agent with a user message, iterates events to capture the final response and token usage, collects and saves the structured trace.

Returns `YogaSutraResult(run_id, session_id, final_response, session_state, trace, trace_path)`.

#### [DONE] `main.py`
CLI entry point using `rich` for formatted output.

```
Flags: --input / -i   User's yoga request (default: demo scenario)
       --persona / -p  Persona ID for tracing (default: P00)
       --no-trace      Skip printing the execution trace summary
```

Displays: spinner while agents run → formatted practice plan → execution trace table (agents, latencies, retrieved evidence, state changes, run ID, trace path).

---

### Static Knowledge Corpus ✅

#### [DONE] `data/knowledge/corpus.json`
20 curated yoga knowledge chunks:
- Beginner relaxation poses: Child's Pose, Knees-to-Chest, Supine Spinal Twist, Happy Baby, Reclined Bound Angle
- Restorative: Legs-up-the-Wall, Savasana
- Warm-up: Cat-Cow, Seated Neck/Shoulder Rolls
- Pranayama: Nadi Shodhana (Alternate Nostril), Box Breathing (Sama Vritti), Fundamentals
- Alignment cues and contraindications for beginners
- Philosophical grounding: Yoga Nidra, Eight Limbs (Patanjali)
- Practice guidance: Morning timing, Sustainable habit-building

Each chunk follows the metadata schema from PLAN.md §17: `{ document_id, chunk_id, title, section, topic, source_type, license, content }`.

---

### Tests ✅

#### [DONE] `tests/test_tools.py`
20 deterministic unit tests:
- `retrieve_yoga_knowledge`: top_k limit, required fields, normalised scores, relevance ordering, query routing (pranayama, beginner), empty query returns `[]`
- `calculate_practice_duration`: correct sum, 20-min sequence, empty → 0, missing key graceful, invalid type graceful
- `get_user_profile` / `store_user_profile`: unknown session → `None`, round-trip, overwrite, session isolation

#### [DONE] `tests/test_state.py`
24 Pydantic model validation tests:
- `ExperienceLevel`, `PracticeGoal`, `RequestMode` enum values
- `UserProfile`: defaults, custom, enum coercion, invalid values raise `ValidationError`
- `EvidenceChunk`: required fields, defaults, missing field raises
- `PoseItem` and `PracticePlan`: construction, defaults
- `TraceStep`: timestamp auto-set, tools_called default empty
- `YogaSessionState`: auto-generated `run_id`, unique IDs per instance, defaults, field population, `start_time` set

---

## Phase 1 Workflow (Sequential)

```
User Input
    ↓
Root Agent (classifies: Practice Mode)
    ↓
Profile Agent (extracts: beginner, 20min, morning, relaxation) → writes user_profile to state
    ↓
Knowledge Agent (calls retrieve_yoga_knowledge, selects evidence) → writes retrieved_evidence to state
    ↓
Practice Planner Agent (generates structured plan from profile + evidence) → writes candidate_plan to state
    ↓
Root Agent (formats final human-readable response from candidate_plan)
    ↓
Final Response + Execution Trace (printed to stdout, trace saved to data/traces/<run_id>.json)
```

---

## What is NOT built in Phase 1

Per PLAN.md §35 Phase 1 scope boundary — deferred to Phase 2 or Phase 3:

**Phase 2:**
- Safety & Constraint Agent
- Critic / Reflection Agent
- PostgreSQL / pgvector (static in-memory corpus only in Phase 1)
- MCP server integration
- Persistent memory / database
- Model gateway abstraction (multi-provider)
- Working frontend (Next.js)

**Phase 3 (PLAN.md §31 — Multi-Model Intelligence & Dynamic Model Routing):**
- Model Capability Registry
- Dynamic model routing / scoring engine (PLAN.md §31.1–31.4)
- Model cascading / escalation (PLAN.md §31.8)
- Parallel model execution (PLAN.md §31.6)
- Multi-model routing experiments A/B/C/D (PLAN.md §31.9)
- Latency budget enforcement (PLAN.md §31.5)

**Phase 3 (PLAN.md §32 — Containerized Runtime):**
- Docker Compose production-like runtime
- Concurrent service execution, connection pooling, RAG caching
- OpenTelemetry collector + observability dashboard

**Phase 3 (PLAN.md §34 — Updated Evaluation Strategy):**
- 40+ persona-linked evaluation runs
- Architecture comparison (sequential vs broadcast vs blackboard vs hierarchical)
- Model-routing experiment matrix (Single / Static / Dynamic / Cascade)
- p95 latency benchmarking against the 30-second ceiling (PLAN.md §31.5)

**Never built:**
- ToT (Tree of Thought) candidate planning
- Production deployment

---

## Verification Results ✅

### Automated Tests
```bash
python3 -m pytest tests/ -v
# Result: 44 passed, 0 warnings in 0.08s
```

### Import Check
```bash
python3 -c "from app.agents.root.agent import root_agent; print(root_agent.sub_agents)"
# Result: ['ProfileAgent', 'KnowledgeAgent', 'PlannerAgent']
```

### Manual Demo Run
```bash
# Requires GOOGLE_API_KEY in .env
python3 main.py --input "I am a beginner and have 20 minutes in the morning. I want a relaxing yoga practice."
```

Expected output:
1. Profile extraction: beginner, 20 min, morning, relaxation
2. Knowledge retrieval: 3–5 evidence chunks with source metadata
3. Structured practice plan (sequence of poses with durations totalling 20 min)
4. Execution trace table (agents called, retrieved evidence, state changes, total latency, run ID)

### Trace Validation
- `data/traces/<run_id>.json` written with the PLAN.md §27 schema
- All required fields present: `run_id`, `agents_called`, `tools_called`, `retrieved_documents`, `agent_latency_ms`, `total_latency_ms`, `token_usage`, `final_output`

---

## Open Questions (Resolved)

> [!NOTE]
> **Google API Key**: A valid `GOOGLE_API_KEY` with Gemini access is required. Get yours at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Add to `.env` (already in `.gitignore`).

> [!NOTE]
> **ADK Version**: Implemented with `google-adk==2.9.1` (installed via `pip3`). `uv` is not required — `pip3` works directly with Python 3.12.

> [!NOTE]
> **Model**: Phase 1 uses `gemini-3.5-flash` for all agents.
> Per PLAN.md §31 (*Multi-Model Intelligence & Dynamic Model Routing*), multi-provider model routing with a Model Capability Registry and routing score engine is a **Phase 3** feature. Phase 1 intentionally uses a single provider.

> [!NOTE]
> **Package conflict resolved**: An empty `app/state/` directory was shadowing `app/state.py`. The directory has been removed.
