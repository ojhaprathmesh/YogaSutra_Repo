# YogaSutra --- Agentic AI Project Plan

> **YogaSutra: A Transparent Multi-Agent AI System for Personalized Yoga
> Practice and Knowledge Assistance**
>
> A Google ADK-based agentic system combining multi-agent orchestration,
> grounded retrieval, planning, memory, safety validation, structured
> reasoning traces, and measurable system-level evaluation.

------------------------------------------------------------------------

## 1. Project Overview

YogaSutra is a multi-agent AI application for **personalized yoga
practice planning and yoga knowledge assistance**.

The project is intentionally designed as an **Agentic AI systems
project**, not as a conventional chatbot. The application domain is
Yoga, while the technical focus is the design and evaluation of a system
capable of:

-   decomposing user goals;
-   coordinating specialized agents;
-   retrieving grounded knowledge;
-   using tools;
-   maintaining session and persistent context;
-   generating and evaluating multi-step practice plans;
-   validating constraints and safety rules;
-   reflecting on and revising its own outputs;
-   communicating through explicit workflow/state mechanisms;
-   exposing a structured execution trace;
-   measuring latency, token usage, quality, grounding, and agent-level
    bottlenecks.

The implementation framework will be **Google Agent Development Kit
(ADK)**. Concepts introduced through the course's CrewAI lectures will
be adapted at the architectural level rather than implemented using
CrewAI.

The project therefore separates:

1.  **Application layer** --- personalized yoga practice and knowledge
    assistance.
2.  **Agentic systems layer** --- orchestration, communication,
    reasoning, memory, RAG, tools, safety, evaluation, and
    observability.

------------------------------------------------------------------------

## 2. Project Motivation

A conventional LLM can generate a plausible yoga response from a prompt,
but that alone does not demonstrate a meaningful agentic system.

YogaSutra addresses a more structured problem:

> Given a user's goal, experience, available time, preferences,
> constraints, prior context, and a curated knowledge base, how can a
> multi-agent system retrieve relevant evidence, construct a suitable
> practice plan, validate it, critique it, revise it when necessary, and
> provide a transparent record of how the final result was produced?

This makes the project suitable for studying the transition from:

**LLM → Tool-using Agent → Multi-Agent System → Observable Agentic
Workflow**

------------------------------------------------------------------------

## 3. Project Goals

### Primary Goal

Design and implement a **transparent, personalized, grounded, and
measurable multi-agent Yoga assistant using Google ADK**.

### Secondary Goals

-   Demonstrate hierarchical multi-agent orchestration.
-   Demonstrate sequential and workflow-based execution.
-   Demonstrate event-driven behavior.
-   Demonstrate multiple agent communication patterns.
-   Integrate Agentic RAG.
-   Demonstrate ReAct-style tool-oriented reasoning.
-   Demonstrate structured planning and reflection.
-   Introduce ToT-style candidate exploration as an advanced feature.
-   Maintain session and persistent user context.
-   Integrate MCP meaningfully.
-   Implement safety and constraint guardrails.
-   Record per-run traces and agent-level metrics.
-   Evaluate the system using at least 40 structured runs.
-   Compare architecture/model configurations quantitatively.
-   Deploy a functional application.

------------------------------------------------------------------------

# 4. Scope

## 4.1 In Scope

### Practice Mode

The user provides information such as:

-   objective;
-   experience level;
-   available duration;
-   preferred practice style;
-   constraints;
-   previous-session context.

The system produces a structured practice plan.

### Knowledge Mode

The user asks questions about yoga concepts, practices, terminology,
philosophy, or related curated knowledge.

The system retrieves relevant sources and generates an evidence-grounded
response.

### Personalization

The system uses:

-   current session state;
-   previous interactions;
-   user preferences;
-   feedback;
-   historical practice context.

### Agentic Workflow

The system can:

-   determine which agents are required;
-   call tools;
-   retrieve evidence;
-   generate plans;
-   validate outputs;
-   critique outputs;
-   revise outputs;
-   record the execution trace.

### Evaluation

The project will measure:

-   task success;
-   grounding;
-   hallucination;
-   constraint satisfaction;
-   personalization;
-   safety compliance;
-   latency;
-   token usage;
-   tool usage;
-   message complexity;
-   redundancy;
-   agent-level contribution;
-   architecture-level performance.

### Multimodal Extension

Phase 3 may include:

-   voice input;
-   image input;
-   pose landmark analysis;
-   multimodal explanations.

Multimodal functionality is an extension and must not compromise the
core agentic system.

------------------------------------------------------------------------

## 4.2 Out of Scope

The system will **not** be presented as:

-   a medical diagnosis system;
-   a treatment recommendation system;
-   a substitute for a qualified yoga instructor or healthcare
    professional;
-   a system capable of clinically diagnosing injury from an image;
-   a fully autonomous real-time physical coach.

Model training or fine-tuning is also not a core project requirement.
The project evaluates **agent architecture and orchestration**, rather
than training a foundation model.

------------------------------------------------------------------------

# 5. Core Agent Architecture

The system will use a small number of specialized agents with clearly
separated responsibilities.

  -----------------------------------------------------------------------
  Agent                   Responsibility          Primary Concepts
  ----------------------- ----------------------- -----------------------
  **Root Manager Agent**  Understand request,     Hierarchical
                          select workflow,        orchestration, planning
                          coordinate specialists  

  **Profile Agent**       Extract user goals,     Perception, memory,
                          preferences,            structured extraction
                          constraints and context 

  **Knowledge/RAG Agent** Retrieve and ground     RAG, ReAct, tools
                          information from the    
                          curated knowledge base  

  **Practice Planner      Generate structured     Goal-based planning,
  Agent**                 practice plans          CoT-style planning

  **Safety & Constraint   Validate explicit       Guardrails, KRR,
  Agent**                 constraints and project deterministic
                          safety rules            validation

  **Critic/Reflection     Evaluate the generated  Reflection,
  Agent**                 plan and request        self-critique,
                          revision when needed    evaluation
  -----------------------------------------------------------------------

A later Phase 3 may add:

  -----------------------------------------------------------------------
  Agent                               Responsibility
  ----------------------------------- -----------------------------------
  **Pose Understanding Agent**        Analyze image-derived pose
                                      landmarks/features and provide
                                      non-clinical practice feedback

  -----------------------------------------------------------------------

The project deliberately avoids excessive agent decomposition. Each
agent must have a measurable responsibility and a clear reason for
existing.

------------------------------------------------------------------------

# 6. System Architecture

YogaSutra combines several architecture patterns at different levels
rather than treating them as mutually exclusive.

``` mermaid
flowchart TB
    U["User"] --> UI["YogaSutra Web Interface"]
    UI --> RM["Root Manager Agent"]

    RM --> PA["Profile Agent"]
    RM --> KA["Knowledge / RAG Agent"]
    RM --> PP["Practice Planner Agent"]

    PA --> SS[("Shared Session State")]
    KA --> SS
    PP --> SS

    SS --> SV["Safety & Constraint Agent"]
    SV --> CR["Critic / Reflection Agent"]

    CR -->|PASS| FR["Final Response"]
    CR -->|REVISION REQUIRED| PP

    FR --> UI
    UI --> FB["User Feedback"]

    FB --> EV["Session / Feedback Event"]
    EV --> MEM[("Persistent User Memory")]
    MEM --> PA

    KA --> KB[("PostgreSQL + pgvector")]
    KA --> MCP["MCP Tool Server"]
    PP --> TOOLS["Planning / Validation Tools"]
    SV --> RULES[("Safety & Constraint Rules")]

    RM -.-> TRACE["Execution Trace / Observability"]
    PA -.-> TRACE
    KA -.-> TRACE
    PP -.-> TRACE
    SV -.-> TRACE
    CR -.-> TRACE
```

------------------------------------------------------------------------

# 7. Why This Is Agentic

YogaSutra is designed around a **goal-driven, sequential, partially
observable and dynamic environment**.

### Environment Characteristics

  -----------------------------------------------------------------------
  Property                            YogaSutra Interpretation
  ----------------------------------- -----------------------------------
  Observability                       **Partially observable** --- the
                                      system does not know the complete
                                      user state

  Determinism                         **Uncertain / stochastic** --- user
                                      responses and preferences are not
                                      fully predictable

  Time structure                      **Sequential** --- previous
                                      sessions can influence future
                                      recommendations

  Dynamics                            **Dynamic** --- user goals,
                                      preferences and context can change

  State space                         Large and partly continuous

  Agents                              Multiple specialized agents

  Interaction                         Human-agent and agent-agent

  Goal structure                      Explicit user goals + system
                                      constraints
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 8. PEAS Specification

  -----------------------------------------------------------------------
  Component                           YogaSutra Definition
  ----------------------------------- -----------------------------------
  **Performance**                     Goal alignment, personalization,
                                      grounding, constraint satisfaction,
                                      safety compliance, coherence,
                                      latency, user feedback

  **Environment**                     User profile, goals, preferences,
                                      available time, prior sessions,
                                      feedback, yoga knowledge corpus

  **Actuators**                       Ask questions, retrieve knowledge,
                                      generate plans, validate plans,
                                      revise outputs, record feedback

  **Sensors / Percepts**              Text, voice, profile data, session
                                      history, feedback, image input,
                                      retrieved documents
  -----------------------------------------------------------------------

The PEAS model provides the formal problem definition for the agent
system and establishes the basis for subsequent evaluation.

------------------------------------------------------------------------

# 9. Agent Communication Architecture

The project will explicitly study the communication mechanisms discussed
in the course.

## 9.1 Sequential Communication

Used for dependencies where the next agent requires the previous agent's
output.

``` mermaid
flowchart LR
    A["Profile Agent"] --> B["Knowledge Agent"]
    B --> C["Planner Agent"]
    C --> D["Safety Agent"]
    D --> E["Critic Agent"]
```

Typical use:

-   profile extraction must happen before planning;
-   retrieved evidence must be available before evidence-grounded
    planning;
-   safety validation must happen after plan generation.

------------------------------------------------------------------------

## 9.2 Broadcast / Parallel Communication

Independent agents can receive the same request and produce independent
assessments.

``` mermaid
flowchart TD
    R["Root Manager"] --> P["Profile Agent"]
    R --> K["Knowledge Agent"]
    R --> C["Constraint Agent"]

    P --> S["Synthesis / Planner"]
    K --> S
    C --> S
```

This pattern will be evaluated against sequential execution where
parallelism is safe.

------------------------------------------------------------------------

## 9.3 Blackboard Communication

Agents communicate indirectly through shared state.

``` mermaid
flowchart TB
    B[("Shared Blackboard / ADK Session State")]

    P["Profile Agent"] <--> B
    K["Knowledge Agent"] <--> B
    PL["Planner Agent"] <--> B
    SV["Safety Agent"] <--> B
    CR["Critic Agent"] <--> B
```

Example state:

``` text
user_profile
goal
constraints
retrieved_evidence
candidate_plan
validation_results
critique
revision_count
final_response
```

The shared state provides a concrete implementation of the blackboard
concept while keeping agent responsibilities isolated.

------------------------------------------------------------------------

## 9.4 Hierarchical Communication

The Root Manager acts as the coordination layer.

``` mermaid
flowchart TD
    R["Root Manager Agent"]

    R --> P["Profile Agent"]
    R --> K["Knowledge / RAG Agent"]
    R --> PL["Practice Planner"]
    R --> SV["Safety & Constraint"]
    R --> CR["Critic / Reflection"]
```

The manager does not perform every specialist task itself. Its purpose
is to coordinate the workflow.

------------------------------------------------------------------------

# 10. Workflow Architecture

The main practice-generation workflow is:

``` mermaid
flowchart TD
    A["User Request"] --> B["Perception & Request Classification"]
    B --> C["Profile / Context Retrieval"]
    C --> D["Goal & Constraint Extraction"]
    D --> E["Knowledge Retrieval"]
    E --> F["Practice Plan Generation"]
    F --> G["Safety & Constraint Validation"]
    G --> H["Critic / Reflection"]

    H -->|Pass| I["Final Response"]
    H -->|Revision Required| F

    I --> J["User Feedback"]
    J --> K["Memory Update"]
    K --> C
```

The workflow deliberately contains a feedback loop rather than a single
prompt-response interaction.

------------------------------------------------------------------------

# 11. Event-Driven Architecture

Event-driven behavior will be introduced primarily around session
completion, feedback and exceptional conditions.

``` mermaid
flowchart TD
    A["Session Completed"] --> E1["SessionCompleted Event"]

    E1 --> F["Feedback Processing"]
    E1 --> M["Memory Update"]
    E1 --> EV["Evaluation Pipeline"]

    F --> P["User Preference Update"]
    M --> PM[("Persistent Memory")]
    EV --> TS[("Evaluation Store")]

    S["Safety Concern Detected"] --> E2["SafetyConcern Event"]
    E2 --> STOP["Stop / Request Clarification"]
```

This allows the system to react to events without forcing every
operation into the main synchronous workflow.

------------------------------------------------------------------------

# 12. Google ADK Implementation Strategy

Google ADK is the implementation framework.

CrewAI concepts from lectures are treated as **architectural patterns**,
not framework dependencies.

## ADK Responsibilities

Google ADK will provide the foundation for:

-   agents;
-   sub-agents;
-   workflow orchestration;
-   sessions;
-   state;
-   tools;
-   callbacks;
-   evaluation;
-   observability;
-   deployment.

The project will use ADK-native mechanisms wherever practical instead of
recreating framework functionality manually.

------------------------------------------------------------------------

# 13. ADK Execution Model

The core application lifecycle is:

``` mermaid
sequenceDiagram
    participant UI as User Interface
    participant R as Root Agent
    participant A as ADK Runner
    participant S as Session
    participant AG as Specialist Agents
    participant T as Tools
    participant O as Observability

    UI->>A: User request
    A->>S: Load/create session
    A->>R: Execute root agent
    R->>AG: Delegate / orchestrate
    AG->>T: Tool call
    T-->>AG: Tool result
    AG-->>R: Agent result
    R-->>A: Final workflow result
    A-->>S: Update state
    A-->>O: Emit execution events
    A-->>UI: Final response + structured trace
```

The application will maintain clear boundaries between:

-   agent logic;
-   workflow logic;
-   state;
-   tools;
-   external integrations;
-   observability.

------------------------------------------------------------------------

# 14. Reasoning Strategy

Different reasoning approaches will be applied where they provide a
concrete benefit.

  -----------------------------------------------------------------------
  Component               Strategy                Reason
  ----------------------- ----------------------- -----------------------
  Root Manager            Goal decomposition      Determine required
                                                  workflow

  Profile Agent           Structured extraction   Convert natural
                                                  language into state

  Knowledge Agent         **ReAct-style tool      Retrieve evidence
                          use**                   dynamically

  Planner                 Structured CoT-style    Break goal into plan
                          planning                components

  Critic                  Reflection              Evaluate and improve
                                                  output

  Advanced Planner        **ToT-style             Compare candidate plans
                          exploration**           
  -----------------------------------------------------------------------

Raw private chain-of-thought will **not** be displayed to users.

Instead, YogaSutra will expose a structured **decision and execution
trace**.

------------------------------------------------------------------------

# 15. Transparent Execution Trace

The user-facing system will show an interpretable trace such as:

``` text
RUN #027

Request
  → 30-minute beginner relaxation practice

Profile
  → Beginner
  → 30 minutes
  → Morning
  → Relaxation goal

Knowledge Retrieval
  → 5 documents retrieved
  → 4 evidence chunks selected

Planning
  → Candidate plan generated

Validation
  → Duration: PASS
  → Constraint check: PASS
  → Evidence coverage: PASS

Reflection
  → Initial plan reviewed
  → 1 revision requested

Final
  → Revised plan returned

Metrics
  → Total latency
  → Agent latencies
  → Tool calls
  → Token usage
  → Retrieved sources
```

The trace is an explanation of **what the system did and why a decision
was accepted/rejected**, not an exposure of hidden model reasoning.

------------------------------------------------------------------------

# 16. Agentic RAG Architecture

RAG is a mandatory project component.

The knowledge pipeline will be:

``` mermaid
flowchart LR
    D["Curated Yoga Documents"] --> L["Document Loader"]
    L --> C["Chunking"]
    C --> M["Metadata"]
    M --> E["Embeddings"]
    E --> V[("PostgreSQL + pgvector")]

    Q["User Query"] --> R["Retriever"]
    V --> R
    R --> RR["Ranking / Re-ranking"]
    RR --> X["Evidence Context"]
    X --> KA["Knowledge Agent"]
    KA --> ANS["Grounded Response + Sources"]
```

## Knowledge Corpus

The corpus may contain appropriately sourced and licensed material
covering:

-   yoga terminology;
-   asanas;
-   pranayama;
-   meditation;
-   philosophy;
-   practice descriptions;
-   structured metadata;
-   project-specific safety/constraint information.

Every knowledge-grounded answer should retain source metadata.

------------------------------------------------------------------------

# 17. RAG Metadata Model

Each knowledge chunk should retain metadata similar to:

``` json
{
  "document_id": "doc_001",
  "title": "Example Source",
  "section": "Example Section",
  "topic": "pranayama",
  "chunk_id": "doc_001_023",
  "source_type": "reference",
  "license": "..."
}
```

The final system should be able to answer:

> Which evidence was used to generate this answer?

------------------------------------------------------------------------

# 18. Tools

Agents will use narrowly scoped tools rather than arbitrary unrestricted
access.

Initial tools:

``` text
retrieve_yoga_knowledge()
get_user_profile()
get_session_history()
calculate_practice_duration()
validate_constraints()
record_feedback()
get_source_metadata()
```

Possible Phase 3 tools:

``` text
analyze_pose_landmarks()
generate_plan_candidates()
evaluate_candidate_plan()
```

Each tool should have:

-   a clear input schema;
-   a clear output schema;
-   validation;
-   logging;
-   an owner agent;
-   measurable latency.

------------------------------------------------------------------------

# 19. MCP Strategy

MCP will be included as a meaningful tool-access mechanism.

The project will expose selected capabilities through an MCP server
rather than adding MCP solely as a technology checkbox.

Potential MCP tools:

``` text
search_yoga_knowledge
get_asana_metadata
get_source_metadata
get_practice_constraints
```

Architecture:

``` mermaid
flowchart LR
    A["Knowledge Agent"] --> M["MCP Client"]
    M --> S["YogaSutra MCP Server"]

    S --> K[("Knowledge Store")]
    S --> C[("Constraint Store")]
```

MCP will be evaluated in terms of:

-   interoperability;
-   implementation complexity;
-   latency;
-   observability;
-   usefulness compared with direct function tools.

------------------------------------------------------------------------

# 20. Memory Architecture

Memory is separated into two levels.

## Session Memory

Short-lived context required during a conversation/workflow.

``` text
current_goal
current_constraints
retrieved_evidence
candidate_plan
validation_results
critique
revision_count
```

## Persistent User Memory

Longer-term information useful for future personalization.

``` text
experience_level
preferences
historical_goals
practice_history
feedback_patterns
preferred_duration
```

Architecture:

``` mermaid
flowchart TB
    U["User Interaction"] --> S["ADK Session State"]
    S --> W["Agent Workflow"]
    W --> F["Feedback"]

    F --> PM["Memory Processing"]
    PM --> P[("Persistent User Memory")]

    P --> C["Relevant Context Retrieval"]
    C --> S
```

Persistent memory must store useful user context rather than raw
unrestricted conversation history.

------------------------------------------------------------------------

# 21. Safety and Guardrails

YogaSutra will implement safety as a **constraint-validation layer**,
not as a medical diagnosis system.

## Validation order

``` mermaid
flowchart TD
    A["Generated Plan"] --> B["Deterministic Rule Checks"]
    B -->|Fail| X["Reject / Revise"]
    B -->|Pass| C["LLM-based Consistency Review"]
    C -->|Concern| X
    C -->|Pass| D["Critic"]
    D -->|Pass| E["Final Response"]
    D -->|Revision| F["Planner"]
    F --> B
```

Deterministic validation should be preferred whenever a requirement can
be represented as a rule.

Examples:

-   requested duration must be satisfied;
-   mandatory user constraints must not be ignored;
-   unavailable information must not be invented;
-   unsupported claims must not be presented as sourced facts.

When information is insufficient, the system should ask for
clarification or communicate uncertainty rather than fabricate
certainty.

------------------------------------------------------------------------

# 22. Planning and Utility Model

The planner will generate structured candidate plans rather than
free-form prose.

A candidate plan can be represented as:

``` text
Goal
Duration
Sequence
Intensity
Rationale
Evidence
Constraints
Expected outcome
```

For Phase 3 candidate selection, a utility function can be used:

$$
U(plan) =
w_1 GoalAlignment +
w_2 ConstraintSatisfaction +
w_3 EvidenceGrounding +
w_4 PreferenceFit +
w_5 SequenceCoherence -
w_6 Complexity
$$

This is a **software evaluation/selection metric for the project**, not
a clinically validated measure.

------------------------------------------------------------------------

# 23. Reflection Loop

The planner and critic form an iterative workflow.

``` mermaid
flowchart LR
    P["Planner"] --> C["Critic"]
    C -->|PASS| F["Final Plan"]
    C -->|REVISION_REQUIRED| R["Revision Feedback"]
    R --> P
```

The critic evaluates:

-   goal alignment;
-   constraint satisfaction;
-   evidence grounding;
-   sequence coherence;
-   duration;
-   personalization;
-   safety-rule compliance.

The system records:

-   number of revisions;
-   reasons for revision;
-   whether revision improved evaluation metrics.

------------------------------------------------------------------------

# 24. ToT-Style Candidate Planning

ToT-style exploration is a Phase 3 feature.

``` mermaid
flowchart TD
    G["User Goal"] --> P["Planner"]

    P --> A["Candidate A"]
    P --> B["Candidate B"]
    P --> C["Candidate C"]

    A --> E["Candidate Evaluator"]
    B --> E
    C --> E

    E --> S["Selected Candidate"]
    S --> V["Safety Validation"]
```

The project will compare:

-   single-plan generation;
-   candidate generation + evaluation.

The objective is to determine whether additional search improves quality
enough to justify additional latency and token cost.

------------------------------------------------------------------------

# 25. Evaluation Strategy

Evaluation is a first-class subsystem rather than a final-stage
checklist.

The project guidelines require **at least 40 runs**, with traceability
to users/personas and agent-level analysis.

YogaSutra will target **44+ structured evaluation runs**.

## Proposed Dataset

  Category                             Runs
  -------------------------------- --------
  Beginner practice planning              8
  Intermediate practice planning          6
  Flexibility goals                       5
  Relaxation goals                        5
  Time-constrained requests               4
  Conflicting constraints                 4
  Knowledge/RAG questions                 4
  Out-of-corpus questions                 2
  Safety-sensitive cases                  2
  Follow-up/personalization               4
  **Total**                          **44**

------------------------------------------------------------------------

# 26. Persona Design

Every evaluation run will be associated with a synthetic target persona.

Example:

``` json
{
  "persona_id": "P01",
  "experience": "beginner",
  "primary_goal": "relaxation",
  "preferred_duration": 20,
  "preferred_time": "morning",
  "preferences": [],
  "constraints": []
}
```

Additional personas will cover different:

-   experience levels;
-   goals;
-   durations;
-   preferences;
-   constraints;
-   knowledge interests.

This allows evaluation of personalization rather than merely generic
answer quality.

------------------------------------------------------------------------

# 27. Per-Run Trace Schema

Every run should produce a machine-readable trace.

``` json
{
  "run_id": "run_027",
  "persona_id": "P01",
  "session_id": "session_027",
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

This allows the project to answer:

-   Which run produced this result?
-   Which agents participated?
-   Which tools were called?
-   What evidence was retrieved?
-   Which agent caused the delay?
-   Which agent produced an incorrect intermediate result?
-   How many revisions occurred?
-   How did the architecture affect final quality?

------------------------------------------------------------------------

# 28. Evaluation Metrics

## Quality

-   Task success
-   Goal alignment
-   Plan coherence
-   Personalization
-   Constraint satisfaction

## Grounding

-   Retrieval relevance
-   Source coverage
-   Citation correctness
-   Hallucination rate

## Agent Performance

-   Per-agent latency
-   Per-agent token usage
-   Tool-call accuracy
-   Error rate
-   Revision frequency

## Multi-Agent Performance

-   Total latency
-   Message count
-   Redundancy
-   Coordination overhead
-   Final quality

## Safety

-   Rule compliance
-   Unsafe-output detection
-   Clarification rate
-   Escalation rate

## System

-   End-to-end latency
-   Cost/token consumption
-   Failure rate
-   Availability during evaluation

------------------------------------------------------------------------

# 29. LLM-as-Judge

Where suitable, an LLM judge may evaluate generated outputs on
dimensions such as:

``` text
Goal Alignment
Grounding
Coherence
Personalization
Constraint Satisfaction
Instruction Following
```

The judge itself must be treated as an evaluation component rather than
an unquestionable ground truth.

For important metrics, human review or deterministic checks should
complement LLM-based evaluation.

------------------------------------------------------------------------

# 30. Architecture Comparison Experiment

One of the central experiments will compare communication/orchestration
patterns.

### Configuration A --- Sequential

``` mermaid
flowchart LR
    A["Profile"] --> B["Knowledge"] --> C["Planner"] --> D["Safety"] --> E["Critic"]
```

### Configuration B --- Parallel/Broadcast

``` mermaid
flowchart TD
    R["Root"] --> A["Profile"]
    R --> B["Knowledge"]
    R --> C["Constraints"]

    A --> S["Planner"]
    B --> S
    C --> S
```

### Configuration C --- Blackboard

``` mermaid
flowchart TB
    S[("Shared State")]
    A["Profile"] <--> S
    B["Knowledge"] <--> S
    C["Planner"] <--> S
    D["Safety"] <--> S
    E["Critic"] <--> S
```

### Configuration D --- Hierarchical

``` mermaid
flowchart TD
    R["Manager"] --> A["Profile"]
    R --> B["Knowledge"]
    R --> C["Planner"]
    R --> D["Safety"]
    R --> E["Critic"]
```

The comparison will focus on measured performance rather than assuming
one architecture is inherently superior.

------------------------------------------------------------------------

# 31. LLM / Model Comparison

Different agents may not require identical models.

The project may evaluate model configurations based on:

-   quality;
-   latency;
-   token usage;
-   tool-use reliability;
-   grounding;
-   planning quality.

Potential initial configuration:

  Agent             Candidate Model
  ----------------- -----------------------------------------
  Root Manager      Gemini Flash
  Profile Agent     Gemini Flash-Lite
  Knowledge Agent   Gemini Flash-Lite / Flash
  Planner           Gemini Flash
  Safety Agent      Gemini Flash-Lite + deterministic rules
  Critic            Gemini Flash

The final assignment will be based on measured results.

------------------------------------------------------------------------

# 32. Observability Architecture

Observability must work at both the **MAS level** and the **individual
agent level**.

``` mermaid
flowchart TB
    R["Run"] --> O["Observability Layer"]

    O --> A["Agent Traces"]
    O --> T["Tool Traces"]
    O --> S["State Transitions"]
    O --> L["Latency Metrics"]
    O --> K["Token Metrics"]
    O --> Q["Quality Metrics"]

    A --> D["Dashboard / Analysis"]
    T --> D
    S --> D
    L --> D
    K --> D
    Q --> D
```

Potential implementation:

-   structured JSON logs during development;
-   OpenTelemetry instrumentation;
-   trace/span identifiers;
-   Google Cloud Trace for deployed observability;
-   evaluation results stored separately for analysis.

------------------------------------------------------------------------

# 33. Bottleneck Analysis

The system should identify bottlenecks per run.

Example analytical output:

``` text
RUN #032

Total latency: 8.4 s
Total tokens: 6,420
Agents: 6
Tool calls: 4

Agent             Latency      Tokens
------------------------------------------------
Root Manager       1.2 s         850
Profile Agent      0.8 s         620
Knowledge Agent    2.7 s       1,940   <-- bottleneck
Planner            1.9 s       1,540
Safety             0.7 s         530
Critic             1.1 s         940
```

The dashboard should allow both:

-   **MAS-level view**
-   **individual-agent view**

as required by the project guidelines.

------------------------------------------------------------------------

# 34. Phase Plan

The university evaluation is divided into three project evaluation
phases:

-   **Phase 1 --- 10%**
-   **Phase 2 --- 30%**
-   **Phase 3 --- 40%**
-   Quiz --- 20%

The following engineering phases map onto those evaluations.

------------------------------------------------------------------------

## Phase 0 --- Research, Scope and Architecture Freeze

### Target

Before Phase 1 submission.

### Objectives

Freeze:

-   problem statement;
-   scope;
-   PEAS;
-   environment;
-   architecture;
-   agent responsibilities;
-   communication strategy;
-   RAG strategy;
-   reasoning strategy;
-   technology stack;
-   evaluation methodology;
-   project timeline.

### Deliverables

-   `PLAN.md`
-   one-page project charter/synopsis
-   architecture diagram
-   PEAS
-   environment analysis
-   agent responsibility matrix
-   initial methodology
-   project timeline
-   feasibility analysis
-   risk register

### Implementation

Only lightweight validation of technical feasibility.

------------------------------------------------------------------------

# Phase 1 --- Minimal Google ADK Proof of Concept

### Target

September --- Phase 1 evaluation window.

### Goal

Demonstrate that the proposed architecture is technically feasible.

### Implement

-   Google ADK setup
-   Gemini integration
-   Root Agent
-   Profile Agent
-   Knowledge/RAG Agent
-   Planner Agent
-   ADK Runner
-   Session
-   Session State
-   basic tools
-   basic sequential workflow
-   basic structured execution trace

### Minimal workflow

``` mermaid
flowchart LR
    U["User"] --> R["Root Agent"]
    R --> P["Profile Agent"]
    P --> K["RAG Agent"]
    K --> PL["Planner"]
    PL --> F["Final Response"]
```

### Phase 1 Demo Scenario

Example:

> "I am a beginner and have 20 minutes in the morning. I want a relaxing
> yoga practice."

The demonstration should show:

1.  request;
2.  profile extraction;
3.  RAG retrieval;
4.  planning;
5.  state transfer;
6.  final response;
7.  basic execution trace.

### Do Not Build Yet

-   pose recognition;
-   complex persistent memory;
-   full MCP;
-   ToT;
-   architecture benchmarking;
-   production deployment;
-   advanced observability dashboard.

The objective is **proof of architecture and understanding**, not
feature completeness.

------------------------------------------------------------------------

# Phase 2 --- Working Multi-Agent Prototype

### Target

September → October, culminating in the October Phase 2 evaluation.

### Goal

Build the complete core agentic workflow.

### Implement

#### Agents

-   Root Manager
-   Profile
-   Knowledge/RAG
-   Planner
-   Safety
-   Critic

#### Orchestration

-   sequential workflow;
-   hierarchical manager;
-   parallel/broadcast experiment;
-   shared-state/blackboard workflow.

#### Tools

-   retrieval;
-   profile;
-   history;
-   validation;
-   duration calculation;
-   feedback.

#### Memory

-   session state;
-   persistent user context.

#### Safety

-   deterministic validation;
-   LLM consistency validation;
-   clarification flow.

#### Reflection

-   planner → critic → revision.

#### Observability

-   structured traces;
-   per-agent timing;
-   tool calls;
-   state transitions.

#### MCP

-   one meaningful MCP server/tool integration.

### Phase 2 Prototype

``` mermaid
flowchart TD
    U["User"] --> R["Root Manager"]

    R --> P["Profile"]
    R --> K["Knowledge / RAG"]
    R --> PL["Planner"]

    P --> S[("Shared State")]
    K --> S
    PL --> S

    S --> SV["Safety"]
    SV --> CR["Critic"]

    CR -->|Pass| F["Final Response"]
    CR -->|Revise| PL

    F --> U
```

### Phase 2 Deliverables

-   updated charter;
-   revised timeline;
-   working prototype;
-   source repository;
-   progress report;
-   demonstration;
-   technical documentation;
-   initial evaluation results.

------------------------------------------------------------------------

# Phase 2B --- Evaluation and Architecture Experiments

### Target

Immediately after the core prototype is stable.

### Experiments

#### Experiment 1 --- Communication Pattern

Compare:

-   sequential;
-   broadcast/parallel;
-   blackboard;
-   hierarchical.

#### Experiment 2 --- Model Assignment

Compare candidate model configurations.

#### Experiment 3 --- RAG

Compare:

-   no RAG;
-   RAG;
-   RAG + re-ranking.

#### Experiment 4 --- Reflection

Compare:

-   planner only;
-   planner + critic;
-   planner + critic + revision.

### Dataset

At least 40 runs; target 44+.

### Outputs

-   metrics;
-   traces;
-   bottleneck analysis;
-   architecture comparison;
-   error analysis.

------------------------------------------------------------------------

# Phase 3 --- Final Agentic System

### Target

Late October → final evaluation.

### Goal

Turn the core prototype into the final demonstrable system and validate
the agentic design experimentally.

### Add

#### Advanced Planning

-   ToT-style candidate exploration;
-   utility-based candidate evaluation.

#### Event-driven Behavior

-   session-completed event;
-   feedback event;
-   memory-update workflow;
-   safety escalation events.

#### Multimodal

-   voice input;
-   image input;
-   optional pose landmark analysis.

#### Advanced Memory

-   preference extraction;
-   historical feedback;
-   personalization.

#### Observability

-   trace dashboard;
-   MAS-level analysis;
-   agent-level analysis;
-   latency bottleneck visualization.

#### Evaluation

-   40+ completed runs;
-   architecture comparison;
-   model comparison;
-   grounding analysis;
-   reflection analysis;
-   safety analysis.

#### Deployment

-   production-like deployment;
-   frontend;
-   backend/ADK service;
-   observability;
-   reproducible configuration.

------------------------------------------------------------------------

# 35. Final System Architecture

``` mermaid
flowchart TB
    U["User"] --> UI["Next.js Interface"]

    UI --> R["Root Manager"]

    subgraph MAS["YogaSutra Multi-Agent System"]
        R --> P["Profile Agent"]
        R --> K["Knowledge / RAG Agent"]
        R --> PL["Practice Planner"]
        R --> SV["Safety & Constraint"]
        R --> CR["Critic / Reflection"]

        P <--> ST[("ADK Session State")]
        K <--> ST
        PL <--> ST
        SV <--> ST
        CR <--> ST

        CR -->|Pass| OUT["Final Response"]
        CR -->|Revision| PL
    end

    K --> PG[("PostgreSQL + pgvector")]
    K --> MCP["MCP Server"]
    SV --> RULES[("Constraint / Safety Rules")]

    OUT --> UI
    UI --> FB["Feedback Event"]
    FB --> MEM[("Persistent User Memory")]
    MEM --> P

    subgraph OBS["Observability & Evaluation"]
        TRACE["Execution Traces"]
        MET["Metrics"]
        EVAL["Evaluation Engine"]
        DASH["Analysis Dashboard"]
    end

    R -.-> TRACE
    P -.-> TRACE
    K -.-> TRACE
    PL -.-> TRACE
    SV -.-> TRACE
    CR -.-> TRACE

    TRACE --> MET
    MET --> EVAL
    EVAL --> DASH
```

------------------------------------------------------------------------

# 36. Phase Timeline

  -----------------------------------------------------------------------
  Period                              Milestone
  ----------------------------------- -----------------------------------
  **Now → Sep 21**                    Problem definition, charter,
                                      architecture and Phase 1 submission

  **Sep 22--27**                      ADK foundation, Runner, Sessions,
                                      State

  **Sep 28--Oct 4**                   Root + Profile + RAG

  **Oct 5--11**                       Planner + tools + sequential
                                      workflow

  **Oct 12--18**                      Safety + Critic + reflection

  **Oct 19--25**                      Phase 2 prototype + documentation

  **Oct 26--Nov 2**                   Communication architecture
                                      experiments

  **Nov 3--9**                        40+ runs + evaluation

  **Nov 10--16**                      ToT + event-driven memory +
                                      multimodal

  **Nov 17 onward**                   Observability + deployment + final
                                      polish

  **Final week**                      Final report + PPT +
                                      demonstration + viva
  -----------------------------------------------------------------------

Dates are a working plan and should be adjusted if faculty deadlines
differ.

------------------------------------------------------------------------

# 37. Team Responsibility Matrix

The project should use ownership without creating silos.

  Area                   Primary Owner   Secondary Owner
  ---------------------- --------------- -----------------
  ADK architecture       Member A        Member B
  Agent orchestration    Member A        Member C
  RAG / knowledge base   Member B        Member A
  Memory / state         Member B        Member C
  Safety / validation    Member C        Member A
  Frontend               Member C        Member B
  Observability          Member A        Member C
  Evaluation             Member B        Member A
  Documentation          All             ---
  Final integration      All             ---

Every team member should understand:

-   the complete architecture;
-   the purpose of every agent;
-   the main workflows;
-   the evaluation methodology;
-   the major design decisions.

------------------------------------------------------------------------

# 38. Technology Stack

## Backend / Agent Runtime

-   Python 3.11+
-   Google ADK
-   Gemini
-   FastAPI
-   Pydantic
-   `uv`

## Knowledge

-   PostgreSQL
-   pgvector
-   embeddings
-   RAG pipeline

## Frontend

-   Next.js
-   TypeScript
-   Tailwind CSS

## Agent Protocol / Integration

-   MCP
-   optional A2A in Phase 3 only if justified

## Multimodal

-   Gemini multimodal capabilities
-   MediaPipe for pose landmarks where appropriate

## Observability

-   structured JSON logging
-   OpenTelemetry
-   Google Cloud Trace

## Testing

-   pytest
-   deterministic tool tests
-   agent evaluation datasets
-   end-to-end workflow tests

## Code Quality

-   Ruff
-   type checking
-   CI

------------------------------------------------------------------------

# 39. Repository Structure

``` text
yoga-sutra/
│
├── app/
│   ├── agents/
│   │   ├── root/
│   │   ├── profile/
│   │   ├── knowledge/
│   │   ├── planner/
│   │   ├── safety/
│   │   └── critic/
│   │
│   ├── workflows/
│   │   ├── sequential.py
│   │   ├── hierarchical.py
│   │   ├── broadcast.py
│   │   ├── blackboard.py
│   │   └── event_driven.py
│   │
│   ├── tools/
│   ├── mcp/
│   ├── rag/
│   ├── memory/
│   ├── state/
│   ├── callbacks/
│   ├── tracing/
│   └── evaluation/
│
├── frontend/
│
├── data/
│   ├── knowledge/
│   ├── personas/
│   ├── evaluation/
│   └── traces/
│
├── experiments/
│   ├── architecture/
│   ├── models/
│   ├── prompts/
│   └── communication/
│
├── tests/
│
├── docs/
│   ├── architecture/
│   ├── methodology/
│   ├── experiments/
│   ├── evaluation/
│   └── report/
│
├── PLAN.md
├── README.md
├── pyproject.toml
└── uv.lock
```

------------------------------------------------------------------------

# 40. Course Concept → Implementation Mapping

  Course Concept              YogaSutra Implementation
  --------------------------- -----------------------------------------------------
  Intelligent Agents          Specialized ADK agents
  PEAS                        Formal YogaSutra environment specification
  Sensors / Percepts          Text, voice, profile, feedback, images
  Actuators                   Questions, retrieval, plans, validation, adaptation
  Goal-based Agent            Practice Planner
  Utility-based Agent         Candidate plan scoring
  Model-based Agent           User profile + persistent context
  CoT                         Structured planner reasoning
  ReAct                       Knowledge/tool agent
  ToT                         Candidate plan exploration
  Self-consistency            Optional candidate verification
  Reflection                  Critic → revision
  Planning                    Practice sequence construction
  Tool Use                    Retrieval, validation, profile and analytics tools
  RAG                         Yoga Knowledge Agent
  Agentic RAG                 Retrieval + reasoning + tool loop
  Multi-Agent Workflow        Specialized agents
  Hierarchical Architecture   Root Manager
  Sequential Architecture     Ordered workflow
  Broadcast                   Parallel specialists
  Blackboard                  Shared ADK state
  Event Driven                Feedback/session events
  Memory                      Session + persistent memory
  KRR                         Rules + structured knowledge + LLM
  Guardrails                  Input/tool/output validation
  Safety                      Constraint validator
  MCP                         Standardized tool access
  A2A                         Optional Phase 3 extension
  Observability               Traces and metrics
  LLM-as-Judge                Automated quality evaluation
  Human-in-the-loop           Clarification/escalation
  Multimodal AI               Voice/image/pose extension
  Deployment                  Cloud-hosted application

------------------------------------------------------------------------

# 41. Design Decisions

## DD-01 --- Google ADK over CrewAI

**Decision:** Use Google ADK.

**Reason:** The project will implement the framework covered in the
Google ADK portion of the course while adapting general multi-agent
concepts from CrewAI lectures.

CrewAI concepts remain useful for understanding:

-   role specialization;
-   task decomposition;
-   sequential workflows;
-   hierarchical management;
-   parallel/broadcast communication;
-   blackboard communication;
-   memory;
-   observability.

They do not require CrewAI as a runtime dependency.

------------------------------------------------------------------------

## DD-02 --- Six Core Agents

**Decision:** Use a small number of specialized agents.

**Reason:** Agent proliferation increases communication overhead,
latency, debugging complexity and redundant reasoning. Every agent must
have a distinct responsibility.

------------------------------------------------------------------------

## DD-03 --- PostgreSQL + pgvector

**Decision:** Use PostgreSQL with pgvector for the initial knowledge
system.

**Reason:** It provides relational application data and vector retrieval
without introducing an unnecessary second database platform.

------------------------------------------------------------------------

## DD-04 --- Structured Trace Instead of Raw CoT

**Decision:** Expose execution/decision traces rather than hidden model
chain-of-thought.

**Reason:** Users need actionable transparency: what agents ran, what
evidence was retrieved, which constraints were checked, what changed,
and why a result was accepted or revised.

------------------------------------------------------------------------

## DD-05 --- Deterministic Safety Before LLM Judgment

**Decision:** Use deterministic checks whenever possible.

**Reason:** Explicit constraints are better validated by deterministic
logic than by relying entirely on probabilistic generation.

------------------------------------------------------------------------

## DD-06 --- Architecture Must Be Measured

**Decision:** Treat orchestration architecture as an experimental
variable.

**Reason:** Sequential, hierarchical, broadcast and blackboard
architectures have different latency, communication and coordination
characteristics. The project should measure their effects rather than
assume a preferred architecture.

------------------------------------------------------------------------

## DD-07 --- Multimodality Is Phase 3

**Decision:** Delay multimodal functionality until the core agentic
system works.

**Reason:** Multimodal features are valuable, but they should not
consume time needed to establish the required multi-agent architecture,
RAG, communication, observability and evaluation.

------------------------------------------------------------------------

## DD-08 --- No Medical Positioning

**Decision:** Position YogaSutra as a yoga practice and knowledge
assistant.

**Reason:** The system is not clinically validated and should not make
medical diagnosis or treatment claims.

------------------------------------------------------------------------

# 42. Risk Register

  -----------------------------------------------------------------------
  Risk                    Impact                  Mitigation
  ----------------------- ----------------------- -----------------------
  Agent proliferation     High                    Limit core agents to
                                                  meaningful
                                                  responsibilities

  Excessive latency       High                    Parallelize independent
                                                  tasks; measure
                                                  per-agent latency

  Hallucination           High                    RAG + source tracking +
                                                  critic

  Poor retrieval          High                    Metadata, ranking,
                                                  evaluation dataset

  Unsafe recommendation   High                    Deterministic
                                                  constraints +
                                                  validation +
                                                  clarification

  Redundant agent work    Medium                  Explicit state
                                                  contracts and
                                                  architecture comparison

  Excessive token usage   Medium                  Model selection and
                                                  structured prompts

  MCP complexity          Medium                  One meaningful MCP
                                                  integration first

  Multimodal scope creep  High                    Phase 3 only

  Evaluation subjectivity Medium                  Deterministic metrics +
                                                  LLM judge + human
                                                  review

  Framework/API changes   Medium                  Pin dependencies and
                                                  document ADK version

  Deployment issues       Medium                  Deploy a minimal
                                                  version before final
                                                  polish

  Insufficient evaluation High                    Prepare persona/test
  runs                                            dataset early

  Team knowledge silos    High                    Cross-review and
                                                  individual architecture
                                                  walkthroughs
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 43. Definition of Done

## Phase 1

-   [ ] Problem statement approved
-   [ ] PEAS completed
-   [ ] Environment analysis completed
-   [ ] Agent responsibilities defined
-   [ ] Architecture diagrams completed
-   [ ] RAG strategy defined
-   [ ] Reasoning strategy defined
-   [ ] Communication strategy defined
-   [ ] Tech stack justified
-   [ ] Timeline completed
-   [ ] Team responsibility matrix completed
-   [ ] Minimal ADK prototype demonstrated

## Phase 2

-   [ ] Six core agents operational
-   [ ] Sequential workflow operational
-   [ ] Hierarchical workflow operational
-   [ ] Shared state operational
-   [ ] RAG operational
-   [ ] Tools operational
-   [ ] Safety validation operational
-   [ ] Critic/reflection operational
-   [ ] Persistent memory operational
-   [ ] MCP integration operational
-   [ ] Structured traces operational
-   [ ] Working frontend
-   [ ] Initial evaluation suite

## Phase 3

-   [ ] 40+ evaluation runs completed
-   [ ] Persona-linked evaluation completed
-   [ ] Architecture comparison completed
-   [ ] Agent bottleneck analysis completed
-   [ ] Model comparison completed
-   [ ] Reflection experiment completed
-   [ ] Advanced planning evaluated
-   [ ] Event-driven memory operational
-   [ ] Multimodal extension implemented where feasible
-   [ ] Observability dashboard operational
-   [ ] Deployment completed
-   [ ] Final report completed
-   [ ] Final presentation completed
-   [ ] Viva preparation completed

------------------------------------------------------------------------

# 44. Final Evaluation Matrix

The final report should explicitly map project evidence to the course
requirements.

  -----------------------------------------------------------------------
  Requirement                         Evidence
  ----------------------------------- -----------------------------------
  Problem Definition                  Problem statement + PEAS

  Objectives / Outcomes               Project goals

  Methodology                         Workflow + architecture

  Feasibility                         Technology/resource analysis

  Reasoning Strategy                  ReAct / structured planning /
                                      reflection

  RAG                                 Knowledge agent + retrieval
                                      pipeline

  Tools                               Tool registry + traces

  Messaging                           Sequential / broadcast / blackboard
                                      / hierarchy

  MCP                                 MCP server integration

  Observability                       Run traces

  Minimum Runs                        44+ evaluation cases

  Persona Linkage                     Synthetic persona IDs

  Latency                             Per-agent and total latency

  Accuracy / Quality                  Evaluation metrics

  Bottleneck Analysis                 Agent-level trace analysis

  Architecture Comparison             Sequential vs broadcast vs
                                      blackboard vs hierarchical

  Technical Depth                     ADK + state + workflows + tools +
                                      memory

  Innovation                          Transparent multi-agent Yoga
                                      application

  Application                         Functional personalized Yoga
                                      assistant

  Individual Contribution             Responsibility + Git history +
                                      documentation
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 45. Final Project Positioning

YogaSutra should ultimately be presented as:

> **A measurable multi-agent systems project implemented with Google
> ADK, using Yoga as the application domain.**

The project is not evaluated only by whether it produces a good yoga
answer.

It is evaluated by whether the system can demonstrate:

``` mermaid
flowchart LR
    G["Goal"] --> D["Decomposition"]
    D --> C["Coordination"]
    C --> R["Retrieval"]
    R --> P["Planning"]
    P --> V["Validation"]
    V --> X["Reflection"]
    X --> A["Action / Response"]
    A --> F["Feedback"]
    F --> M["Memory"]
    M --> G

    C --> O["Observability"]
    R --> O
    P --> O
    V --> O
    X --> O
    O --> E["Evaluation"]
```

The final system should therefore answer two questions simultaneously:

### Application question

**Can YogaSutra provide useful, grounded and personalized yoga
practice/knowledge assistance?**

### Agentic AI question

**Can we demonstrate, measure and explain how multi-agent orchestration,
communication, retrieval, planning, memory, reflection and tool use
affect the system's behavior?**

That second question is the core of the project.

------------------------------------------------------------------------

## 46. Guiding Principle

> **Build the smallest system that demonstrates the largest number of
> course concepts, then measure whether each added capability actually
> improves the system.**

This principle should govern the entire implementation.

Do not add an agent, protocol, model, database, workflow or multimodal
capability merely because it appears in the syllabus.

Every component should have:

1.  a defined responsibility;
2.  a reason for existing;
3.  an implementation;
4.  an observable trace;
5.  an evaluation metric;
6.  a documented design decision.

That is the standard YogaSutra should follow from Phase 1 through the
final demonstration.
