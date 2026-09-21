# YogaSutra — Phase 2 Implementation Plan

## Status: ✅ IMPLEMENTED & VERIFIED

> **Phase 2 Working Multi-Agent Prototype is operational.**
> 6 specialized agents active, 287-pose enriched knowledge corpus, deterministic safety validation, critic reflection, 54/54 tests passing.

---

## Goal

Build the complete **Phase 2 Multi-Agent Architecture** for YogaSutra as defined in PLAN.md §2154–2250:
1. Fix Google GenAI AFC advisory warnings and reduce execution latency by adopting `gemini-3.5-flash-lite`.
2. Ingest and normalize raw datasets (`data/source/final_asan1_1.csv` and `data/source/yoga-api.json`) into an authentic 287-chunk knowledge corpus with standardized contraindication arrays.
3. Implement the **Safety & Constraint Agent** with deterministic contraindication and duration checking.
4. Implement the **Critic / Reflection Agent** for pedagogical audit and quality scoring.
5. Upgrade the **Root Manager Agent** and **ADK Runner** to orchestrate the complete Phase 2 workflow:
   `Profile → Knowledge → Planner → Safety → Critic → Response`
6. Record full execution trace schemas matching PLAN.md §27.

---

## Phase 2 Definition of Done (PLAN.md §2164–2250)

| Component | Target in PLAN.md | Status |
|---|---|---|
| **Root Manager Agent** | Request classification & end-to-end orchestration | ✅ `app/agents/root/agent.py` |
| **Profile Agent** | Structured profile extraction into Blackboard state | ✅ `app/agents/profile/agent.py` |
| **Knowledge / RAG Agent** | Evidence retrieval from 287-chunk corpus | ✅ `app/agents/knowledge/agent.py` |
| **Practice Planner Agent** | Structured practice planning with duration verification | ✅ `app/agents/planner/agent.py` |
| **Safety & Constraint Agent** | Deterministic contraindication & duration auditing | ✅ `app/agents/safety/agent.py` |
| **Critic / Reflection Agent** | Quality scoring & reflection decision (`PASS` / `REVISE`) | ✅ `app/agents/critic/agent.py` |
| **Enriched Knowledge Base** | 287 authentic yoga chunks with normalized contraindications | ✅ `data/knowledge/corpus.json` |
| **Ingestion Pipeline** | Automated parser & normalizer for source datasets | ✅ `scripts/ingest_corpus.py` |
| **Execution Tracing** | Full JSON trace with validation and critic results | ✅ `data/traces/<run_id>.json` |
| **Automated Tests** | 54 passing unit tests covering state, tools, safety, reflection | ✅ `tests/` (54 passed in 0.79s) |

---

## Architecture Workflow

```
                  User Request
                       │
                       ▼
                 Profile Agent
       (extracts: level, duration, constraints)
                       │
                       ▼
             Knowledge / RAG Agent
    (retrieves from 287-pose enriched corpus)
                       │
                       ▼
             Practice Planner Agent
     (designs sequence using duration tool)
                       │
                       ▼
           Safety & Constraint Agent
  (deterministic contraindication & timing audit)
                       │
                       ▼
            Critic & Reflection Agent
       (scores plan, evaluates suitability)
                       │
                       ▼
                Response Agent
 (synthesizes final plan, safety clearance & citations)
                       │
                       ▼
       Final Markdown Output + Execution Trace
```

---

## Verification Results ✅

### Automated Tests
```bash
python3 -m pytest tests/ -v
# Result: 54 passed, 3 warnings in 0.79s
```

### Reference Scenario 1 (Healthy Beginner Relaxation)
```bash
python3 main.py --input "I am a beginner and have 20 minutes in the morning. I want a relaxing yoga practice."
```
- **Execution Time**: ~16.4 seconds (vs 149 seconds previously)
- **Safety Audit**: `PASS` (All poses beginner-friendly, total duration matches 20m)
- **Critic Reflection**: `PASS` (Score: 10/10)

### Reference Scenario 2 (Injury / Contraindication Detection)
```bash
python3 main.py --input "I am a beginner with lower back pain and 15 minutes. I want an energizing practice."
```
- **Safety Audit**: Successfully flagged standard poses contraindicated for lower back pain.
- **Critic Reflection**: Highlighted the contraindications and recommended back-safe alternatives (Bridge Pose, Apanasana).
- **Final Output**: Rendered with back-safe modifications and explicit Safety Clearance notice.
