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

---------------------------------------------------------------------## 3. Project Goals

### Primary Goal

Design and implement a **transparent, personalized, grounded, hands-free speech-enabled, and measurable multi-agent Yoga assistant using Google ADK**, with an extensible **real-time computer vision posture correction module**.

### Secondary Goals

-   Demonstrate hierarchical multi-agent orchestration.
-   Demonstrate sequential and workflow-based execution.
-   Demonstrate event-driven behavior (including posture deviation and voice events).
-   Demonstrate multiple agent communication patterns.
-   Integrate Agentic RAG for both practice plans and asana biomechanical alignment standards.
-   Demonstrate ReAct-style tool-oriented reasoning.
-   Demonstrate structured planning and reflection.
-   Introduce ToT-style candidate exploration as an advanced feature.
-   Maintain session and persistent user context (including posture history).
-   Integrate MCP meaningfully.
-   Implement safety and constraint guardrails (physiological & biomechanical).
-   Implement bidirectional speech communication (hands-free STT & TTS).
-   Provide a plug-and-play real-time computer vision module (OpenCV + MediaPipe / YOLO-Pose) for posture feedback.
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

### Speech Communication Mode (Hands-Free Yoga Interface)

Yoga practice is inherently physical and hands-free. The system supports:

-   bidirectional voice conversation (Speech-to-Text via Whisper / Web Speech API);
-   expressive, pace-adjusted spoken responses (Text-to-Speech via Google Cloud TTS / ElevenLabs / local synthesis);
-   hands-free session control ("Next pose", "Pause", "Explain alignment");
-   real-time audio cues spoken during active practice execution.

### Modular Real-Time Video Posture Correction (CV Extension)

A decoupled, plug-and-play computer vision pipeline that can be activated during practice, engineered for high-efficiency edge execution and resource-constrained environments (sub-100MB active memory footprint, 30 FPS throughput):

-   **High-Efficiency Ingestion Clamping**: Real-time OpenCV frame capture clamped immediately to 480p (640×480) at capture boundary, reducing per-frame memory bandwidth by 85% (from >6MB down to <900KB per frame);
-   **Zero-Lag Bounded Buffering**: Bounded ring buffer `Queue(maxsize=1)` dropping stale frames to prevent buffer bloat and guarantee zero cumulative frame latency;
-   **Lightweight 3D Pose Extraction**: MediaPipe Pose Lite (`model_complexity=0`, `enable_segmentation=False`) extracting 33 3D skeletal landmarks under <60MB active RAM with sub-12ms inference;
-   **Zero-Hallucination Biomechanical Engine**: 100% deterministic pure-Python 3D vector geometry ($\arccos$ dot product) computing exact joint angles (knee flexion, spinal verticality, shoulder/hip symmetry) with zero LLM hallucination;
-   **Event-Driven Closed Feedback Loop**: Emits `PostureDeviationEvent` into the Phase 3A Event Bus when joint angles deviate from classical `corpus.json` asana criteria for >1.5s;
-   **Real-Time Telemetry HUD**: Live on-screen display of FPS (25–30 FPS), inference latency (<12ms), and active memory consumption (<100MB).

### Personalization

The system uses:

-   current session state;
-   previous interactions;
-   user preferences;
-   feedback;
-   historical practice context and posture progress metrics.

### Agentic Workflow

The system can:

-   determine which agents are required;
-   call tools (including speech synthesis and pose biomechanics calculators);
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
-   latency (text, speech STT/TTS, and posture event turnaround);
-   token usage;
-   tool usage;
-   message complexity;
-   redundancy;
-   agent-level contribution;
-   architecture-level performance.

------------------------------------------------------------------------

## 4.2 Out of Scope

The system will **not** be presented as:

-   a medical diagnosis system;
-   a clinical treatment recommendation system;
-   a substitute for a licensed physical therapist, orthopedist, or healthcare professional;
-   a system capable of diagnosing internal injuries or musculoskeletal pathology;
-   a guarantee against injury if user ignores safety disclaimers.

The real-time posture correction module is strictly an **instructional biomechanical alignment aid** based on standard geometric yoga heuristics and curated reference angles. Foundation model training is out of scope; foundation models are accessed via APIs and local lightweight vision/audio models.

------------------------------------------------------------------------

# 5. Core Agent Architecture

The system uses specialized agents with clearly separated responsibilities, structured to allow modular multimodal additions.

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
                          (including asana
                          biomechanical rules)

  **Practice Planner      Generate structured     Goal-based planning,
  Agent**                 practice plans          CoT-style planning

  **Safety & Constraint   Validate explicit       Guardrails, KRR,
  Agent**                 constraints, medical    deterministic
                          contraindications, and  validation
                          project safety rules

  **Critic/Reflection     Evaluate the generated  Reflection,
  Agent**                 plan and request        self-critique,
                          revision when needed    evaluation

  **Speech / Audio Agent** Ingest user voice       Audio perception, STT,
                          transcripts, manage     conversational pacing,
                          conversational pacing,  TTS actuation
                          and synthesize spoken
                          cues & audio responses

  **Pose Understanding &  Analyze extracted       Computer vision,
  Correction Agent**      skeletal landmarks and  biomechanics,
  *(Modular CV Plug-in)*  joint angles, compare   event-driven corrective
                          with RAG asana criteria, feedback
                          and formulate concise
                          real-time adjustments
  -----------------------------------------------------------------------

The core architecture operates with 6 core text/planning agents + 1 Speech Agent, while the **Pose Understanding & Correction Agent** is engineered as a plug-and-play module that attaches via the shared blackboard and event bus.

------------------------------------------------------------------------

# 6. System Architecture

YogaSutra combines hierarchical orchestration, blackboard shared state, and decoupled real-time event-driven streaming.

``` mermaid
---
config:
  layout: elk
---
flowchart TB
    subgraph UI_Layer["User Interaction & Multimodal Client (Next.js)"]
        U["User"]
        MIC["Microphone / Speech Input"]
        CAM["Webcam / Video Stream"]
        SPEAKER["Audio Output / Spoken Cues"]
        SCREEN["Live Practice Canvas & Observability HUD"]

        U -->|Voice Utterance| MIC
        U -->|Text / Goal Prompt| SCREEN
        U -->|Active Asana Practice| CAM
        SPEAKER -->|Hands-Free Voice Guidance| U
        SCREEN -->|Visual Skeleton & Biomechanical Feedback| U
    end

    subgraph Multimodal_Pipeline["Multimodal Perception & Biomechanics Engine"]
        MIC --> STT["STT Engine (Whisper / WebSpeech)"]
        CAM --> CV["OpenCV Ingestion (480p Clamped, Queue maxsize=1)"]
        CV --> MP["MediaPipe Pose Lite (33 3D Landmarks)"]
        MP --> GEOM["Pure-Python 3D Vector Geometry Engine"]
        GEOM -->|Skeleton Overlay & Angle HUD| SCREEN
    end

    subgraph Multi_Agent_Core["YogaSutra Multi-Agent Core (Google ADK)"]
        RM["Root Manager Agent"]
        PA["Profile Agent"]
        KA["Knowledge / RAG Agent"]
        PP["Practice Planner Agent"]
        SV["Safety & Constraint Agent"]
        CR["Critic / Reflection Agent"]
        SA["Speech / Audio Agent"]
        POA["Pose Understanding & Correction Agent"]

        SS[("ADK Session State\n(Shared Blackboard)")]
        BUS{{"Reactive Event Bus"}}
    end

    subgraph Knowledge_and_Tools["Knowledge, Constraints & External Integration"]
        KB[("PostgreSQL + pgvector\n(Asanas & Biomechanics)")]
        MCP["YogaSutra MCP Server"]
        RULES[("Safety Rules & Contraindications")]
        MEM[("Persistent Long-Term Memory")]
    end

    subgraph Observability_Layer["Observability, Telemetry & Evaluation"]
        TRACE["Execution Traces (Agent & Tool Spans)"]
        MET["Telemetry Metrics (Latency, WER, CV FPS, RAM)"]
        EVAL["Scientific Evaluation Engine (40+ Persona Runs)"]
        DASH["MAS Trace Analytics Dashboard"]
    end

    %% Multimodal Perception to Agents & Bus
    STT -->|Transcribed Text| RM
    SCREEN -->|Goal & Session Configuration| RM
    GEOM -->|PostureDeviationEvent| BUS
    BUS -->|Real-Time Alignment Alerts| POA
    BUS -->|VoiceCommandEvent| RM

    %% Agent State & Blackboard Access
    RM <--> SS
    PA <--> SS
    KA <--> SS
    PP <--> SS
    SV <--> SS
    CR <--> SS
    SA <--> SS
    POA <--> SS

    %% Planning, Reflection & Corrective Loops
    POA -->|Immediate Corrective Cue| SA
    CR -->|Pass: Approved Plan| SA
    CR -->|Pass: Render Plan| SCREEN
    CR -->|Revision Required| PP

    %% Audio Dispatch
    SA -->|Synthesized TTS Audio Stream| SPEAKER

    %% Knowledge Retrieval & Tools
    KA --> KB
    KA --> MCP
    SV --> RULES
    MEM --> PA

    %% Closed-Loop Feedback
    SCREEN -->|Post-Practice Feedback & Rating| FB["Session Feedback Event"]
    FB --> MEM

    %% Observability Streams
    Multi_Agent_Core -.-> TRACE
    Multimodal_Pipeline -.-> TRACE
    TRACE --> MET
    MET --> EVAL
    EVAL --> DASH
```

------------------------------------------------------------------------

# 7. Why This Is Agentic

YogaSutra is designed around a **goal-driven, sequential, partially
observable and dynamic environment**.

### Environment Characteristics

  ---------------------------------------------------------------------
  Property                           YogaSutra Interpretation
  ---------------------------------- ----------------------------------
  Observability                      **Partially observable** --- the
                                     system does not know the complete
                                     user state

  Determinism                        **Uncertain / stochastic** ---
                                     user responses and preferences are
                                     not fully predictable

  Time structure                     **Sequential** --- previous
                                     sessions can influence future
                                     recommendations

  Dynamics                           **Dynamic** --- user goals,
                                     preferences and context can change

  State space                        Large and partly continuous

  Agents                             Multiple specialized agents

  Interaction                        Human-agent and agent-agent

  Goal structure                     Explicit user goals + system
                                     constraints
  ---------------------------------------------------------------------

------------------------------------------------------------------------

# 8. PEAS Specification

  ---------------------------------------------------------------------
  Component                          YogaSutra Definition
  ---------------------------------- ----------------------------------
  **Performance**                    Goal alignment, personalization,
                                     grounding, constraint
                                     satisfaction, safety compliance,
                                     coherence, text latency, STT/TTS
                                     speech latency, posture deviation
                                     detection precision, real-time cue
                                     turnaround (<2.0s), user feedback

  **Environment**                    User profile, goals, constraints,
                                     prior sessions, feedback history,
                                     curated yoga knowledge &
                                     biomechanical criteria corpus,
                                     physical mat environment (audio &
                                     video stream)

  **Actuators**                      Ask clarifying questions, spoken
                                     voice prompts (TTS), real-time verbal
                                     corrective cues, UI visual skeleton
                                     overlay & angle HUD badges, generate
                                     plans, validate & revise plans,
                                     persist user memory & trace metrics

  **Sensors / Percepts**             User text prompts, live microphone
                                     audio stream (voice utterances),
                                     live webcam video frames, extracted
                                     skeletal landmarks (MediaPipe 33 3D /
                                     YOLO-Pose 17), computed joint
                                     angles, session history, retrieved
                                     evidence chunks
  ---------------------------------------------------------------------

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
---
config:
  layout: elk
---
flowchart TB
    B[("Shared Blackboard / ADK Session State")]

    P["Profile Agent"] <--> B
    K["Knowledge Agent"] <--> B
    PL["Planner Agent"] <--> B
    SV["Safety Agent"] <--> B
    CR["Critic Agent"] <--> B
    SA["Speech / Audio Agent"] <--> B
    POA["Pose Understanding Agent (CV Plug-in)"] <--> B
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

# Multimodal Extensions (Blackboard Channels)
audio_context:
  input_audio_url
  voice_transcript
  speech_mode_active
  output_audio_url

pose_tracking_state:
  current_asana_name
  target_hold_duration_sec
  current_hold_duration_sec
  joint_angles: {knee, hip, shoulder, spine}
  active_deviations: [{joint, delta_deg, severity, cue}]
  overall_alignment_score
  last_cue_timestamp
```

The shared state provides a concrete implementation of the blackboard
concept while keeping agent responsibilities isolated.

------------------------------------------------------------------------

## 9.4 Hierarchical Communication

The Root Manager acts as the coordination layer.

``` mermaid
---
config:
  layout: elk
---
flowchart TD
    R["Root Manager Agent"]

    R --> P["Profile Agent"]
    R --> K["Knowledge / RAG Agent"]
    R --> PL["Practice Planner"]
    R --> SV["Safety & Constraint"]
    R --> CR["Critic / Reflection"]
    R --> SA["Speech / Audio Agent"]
```

The manager does not perform every specialist task itself. Its purpose
is to coordinate the workflow.

------------------------------------------------------------------------

# 10. Workflow Architecture

## 10.1 Practice Plan Generation Workflow

The main practice-generation workflow is:

``` mermaid
flowchart TD
    A["User Request\n(Voice or Text)"] --> B["Perception & Request Classification\n(Speech / Root Agent)"]
    B --> C["Profile / Context Retrieval"]
    C --> D["Goal & Constraint Extraction"]
    D --> E["Knowledge Retrieval (RAG)"]
    E --> F["Practice Plan Generation"]
    F --> G["Safety & Constraint Validation"]
    G --> H["Critic / Reflection"]

    H -->|Pass| I["Final Response Synthesis\n(Text Trace + Spoken Audio TTS)"]
    H -->|Revision Required| F

    I --> J["User Feedback\n(Voice or Click)"]
    J --> K["Memory Update"]
    K --> C
```

The workflow deliberately contains a feedback loop rather than a single
prompt-response interaction.

------------------------------------------------------------------------

## 10.2 Live Multimodal Practice & Posture Guidance Workflow (Decoupled Loop)

When the user enters **Practice Mode**, YogaSutra runs a decoupled, dual-speed workflow:
- A high-frequency CV loop (30 FPS) processes video frames locally;
- An event-driven multi-agent loop triggers only when biomechanical deviations or voice queries occur.

``` mermaid
sequenceDiagram
    autonumber
    actor User as User (On Yoga Mat)
    participant Cam as Webcam & OpenCV
    participant CV as Pose Engine (MediaPipe/YOLO)
    participant State as ADK Blackboard State
    participant PoseAgent as Pose Understanding Agent
    participant RAG as Knowledge / Biomechanics RAG
    participant SpeechAgent as Speech / Audio Agent
    participant UI as Web HUD & Speakers

    User->>Cam: Live Practice Video Feed
    Cam->>CV: Video Frames (30 FPS)
    CV->>CV: Extract 33 Landmarks & Compute Joint Angles
    CV->>State: Update current_joint_angles & alignment_score

    alt Posture Deviation Detected (> threshold for > 1.5s)
        CV->>State: Emit PostureDeviationEvent(joint="left_knee", angle=68, target=90)
        State->>PoseAgent: Trigger Event
        PoseAgent->>RAG: Retrieve Asana Alignment Evidence (Virabhadrasana II knee safety)
        RAG-->>PoseAgent: Grounded biomechanical adjustment rule
        PoseAgent->>PoseAgent: Formulate concise physical cue
        PoseAgent->>SpeechAgent: Dispatch Spoken Cue ("Gently press your left knee forward over your ankle")
        SpeechAgent->>UI: Play Audio TTS Cue (< 1.5s latency)
        CV->>UI: Render Skeleton Wireframe & Angle HUD Badge (Green/Red)
        UI-->>User: Spoken voice cue + Visual screen feedback
    end

    opt User Voice Query during Pose
        User->>SpeechAgent: "How long do I hold this?"
        SpeechAgent->>State: Read current_hold_duration_sec & target
        SpeechAgent->>UI: Spoken Answer ("15 seconds remaining. Breathe deeply.")
        UI-->>User: Audio Response
    end
```

------------------------------------------------------------------------

# 11. Event-Driven Architecture

Event-driven behavior governs session completion, feedback, live posture deviations, and voice commands without blocking the main workflow.

``` mermaid
---
config:
  layout: dagre
---
flowchart BT
    subgraph Session_Lifecycle_Events["Session Lifecycle & Memory Events"]
        F["Feedback Processing"]
        E1["SessionCompleted Event"]
        M["Memory Update"]
        EV["Evaluation Pipeline"]
        P["User Preference Update"]
        PM[("Persistent Memory")]
        TS[("Evaluation Store")]
    end

    subgraph RealTime_Posture_Events["Real-Time Posture & Multimodal Events"]
        V2["Angle Calculation"]
        V1["Webcam Frame Ingested"]
        E2["PostureDeviation Event"]
        POA["Pose Understanding Agent"]
        SA["Speech / Audio Agent (TTS Cue)"]
        E3["PoseHoldCompleted Event"]
        ADVANCE["Transition to Next Asana Sequence"]
        E4["VoiceCommand Event"]
        AUDIO["User Voice Utterance"]
        DISPATCH["Hands-Free Voice Dispatcher"]
    end

    subgraph Safety_Events["Safety Escalation Events"]
        E5["SafetyConcern Event"]
        S["Severe Strain / Contraindication Detected"]
        STOP["Audio Alert: Stop & Rest / Clarify"]
    end

    E1 --> F & M & EV
    F --> P
    M --> PM
    EV --> TS
    V1 --> V2
    V2 -->|Deviation > Threshold| E2
    E2 --> POA
    POA --> SA
    V2 -->|Hold Target Reached| E3
    E3 --> ADVANCE
    AUDIO --> E4
    E4 --> DISPATCH
    S --> E5
    E5 --> STOP
```

This allows the system to react to events without forcing high-frequency frame operations into synchronous LLM workflows.

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

Knowledge Retrieval & Chunk-Level Claims
  → 5 documents retrieved
  → 4 evidence chunks selected
  → Verified Chunk Claims:
      • "Hold Balasana for 3 minutes"
        → File: restorative_yoga.md | Chunk: chunk_res_012 (L34-45)
      • "Stack front knee directly above heel at 90°"
        → File: virabhadrasana_2.md | Chunk: chunk_v2_004 (L42-58)
      • "Inhale to lengthen spine before folding"
        → File: paschimottanasana.md | Chunk: chunk_pas_007 (L18-29)

Planning
  → Candidate plan generated with claim-to-chunk bindings

Validation
  → Duration: PASS
  → Constraint check: PASS
  → Evidence coverage & Claim Verification: PASS

Reflection
  → Initial plan reviewed against chunk claims
  → 1 revision requested

Final
  → Revised plan returned with auditable chunk citations

Metrics
  → Total latency
  → Agent latencies
  → Tool calls
  → Token usage
  → Retrieved sources & Chunk claims
```

The trace is an explanation of **what the system did and why a decision
was accepted/rejected**, with verifiable claims traceable down to the specific file chunk.

------------------------------------------------------------------------

# 16. Agentic RAG Architecture

RAG is a mandatory project component.

The knowledge pipeline will be:

``` mermaid
---
config:
  layout: dagre
---
flowchart TB
    D["Curated Yoga Documents"] --> L["Document Loader"]
    L --> C["Chunking & Line Mapping"]
    C --> M["Provenance Metadata\n(File, Chunk ID, Lines, Hash)"]
    M --> E["Embeddings"]
    E --> V[("PostgreSQL + pgvector")]

    Q["User Query"] --> R["Retriever"]
    V --> R
    R --> RR["Ranking / Re-ranking"]
    RR --> X["Evidence Context + Chunk Claims"]
    X --> KA["Knowledge Agent"]
    KA --> ANS["Grounded Response +\nAuditable Chunk Claims"]
```

## Knowledge Corpus

The corpus contains appropriately sourced and licensed material
covering:

-   yoga terminology;
-   asanas and biomechanical alignment standards;
-   pranayama;
-   meditation;
-   philosophy;
-   practice descriptions;
-   structured metadata and file chunk hashes;
-   project-specific safety and contraindication information.

Every knowledge-grounded answer must retain chunk-level source metadata and explicit claim attribution.

------------------------------------------------------------------------

# 17. Chunk-Level Knowledge Provenance & Claim Attribution Model

To satisfy rigorous academic grounding and safety standards, YogaSutra avoids generic document-level references. Every factual statement, asana instruction, contraindication rule, or real-time posture cue is bound to an **atomic claim linked directly to a specific chunk of a specific file**.

### Provenance Schema

Each knowledge chunk in the vector store and execution trace contains:

``` json
{
  "document_id": "doc_asana_014",
  "source_file": "data/knowledge/asanas/virabhadrasana_ii.md",
  "chunk_id": "chunk_v2_004",
  "section_heading": "Biomechanical Alignment & Knee Safety",
  "line_range": "L42-L58",
  "content_hash": "sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
  "extracted_claims": [
    "The front knee must remain stacked vertically over the heel without collapsing inward (valgus)",
    "The front thigh should approach parallel to the floor at an angle of 90 degrees",
    "Hyperextension of the back knee must be avoided by engaging the quadriceps"
  ],
  "verbatim_quote": "Bend your front knee until the thigh is parallel with the mat, ensuring the knee is directly over the ankle...",
  "source_type": "curated_reference",
  "license": "CC-BY-SA-4.0"
}
```

### Claim Attribution Workflow

1. **Retrieval**: The Knowledge Agent retrieves chunks matching the user's goals or asana requirements.
2. **Claim Extraction**: The Knowledge Agent extracts atomic claims from retrieved chunks and includes both the claim text and the chunk reference in its response.
3. **Plan Binding**: The Planner Agent explicitly annotates each step in the practice plan with its supporting `chunk_id`, `source_file`, and `line_range`.
4. **Validation & Critique**: The Safety and Critic Agents verify that each generated instruction is strictly entailed by the referenced chunk, rejecting or revising any unsourced assertions.

The system is able to answer at any time:

> **"Where did this instruction originate?"**
> $\rightarrow$ *Originates from `data/knowledge/asanas/virabhadrasana_ii.md`, Chunk `chunk_v2_004` (Lines 42–58).*

------------------------------------------------------------------------

# 18. Tools

Agents will use narrowly scoped tools rather than arbitrary unrestricted
access.

### Core Planning & Knowledge Tools:

``` text
retrieve_yoga_knowledge(query, top_k) -> list[ChunkWithClaims]
verify_chunk_claims(plan_step_text, chunk_ids) -> claim_attribution_report
get_chunk_provenance(chunk_id) -> source_file, line_range, content_hash, quote
get_user_profile()
get_session_history()
calculate_practice_duration()
validate_constraints()
record_feedback()
```

### Speech & Audio Tools (Active Scope):

``` text
transcribe_speech_audio(audio_bytes, language) -> transcript, confidence
synthesize_speech_feedback(text_cue, tone_style, speaking_rate) -> audio_url
dispatch_audio_guidance(cue_text, priority) -> bool
```

### Modular Video & Posture Analysis Tools (CV Extension Plug-in):

``` text
extract_pose_landmarks(frame_image) -> landmarks_3d, visibility_scores
compute_joint_biomechanics(landmarks_3d) -> joint_angles_dict
compare_pose_against_standard(asana_id, joint_angles) -> deviation_report
generate_posture_cue(deviation_report, persona_level) -> concise_verbal_cue
```

### Advanced Planning Tools (Phase 3):

``` text
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

# Voice & Practice Session Context
audio_context:
  voice_transcript_history
  active_speech_mode
  last_audio_cue_id

# Real-Time Practice Pose State
pose_tracking_session:
  active_asana
  pose_start_timestamp
  accumulated_hold_seconds
  joint_angles_live
  deviations_logged: list[dict]
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

# Biomechanical & Multimodal Progress Memory
posture_biomechanics_profile:
  flexibility_baseline: {hip, hamstring, shoulder}
  recurring_alignment_tendencies: ["knee_valgus_inward", "rounded_lumbar"]
  average_hold_stability_score
  preferred_audio_guidance_density: "minimal" | "detailed"
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
---
config:
  layout: elk
---
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
---
config:
  layout: elk
---
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

Every run produces a machine-readable execution trace capturing full provenance down to the exact file chunk.

``` json
{
  "run_id": "run_027",
  "persona_id": "P01",
  "session_id": "session_027",
  "input": "I am a beginner and have 20 minutes in the morning. I want a relaxing practice.",
  "input_modality": "voice",
  "agents_called": ["root", "profile", "knowledge", "planner", "safety", "critic", "speech"],
  "tools_called": ["retrieve_yoga_knowledge", "verify_chunk_claims", "validate_constraints", "synthesize_speech_feedback"],
  "retrieved_evidence_chunks": [
    {
      "chunk_id": "chunk_v2_004",
      "source_file": "data/knowledge/asanas/virabhadrasana_ii.md",
      "section": "Biomechanical Alignment & Knee Safety",
      "line_range": "L42-L58",
      "content_hash": "sha256:7f83b165..."
    }
  ],
  "chunk_provenance_claims": [
    {
      "claim_statement": "Front knee must be stacked vertically over the heel at 90 degrees",
      "source_file": "data/knowledge/asanas/virabhadrasana_ii.md",
      "chunk_id": "chunk_v2_004",
      "line_range": "L42-L58",
      "verification_status": "VERIFIED"
    }
  ],
  "state_changes": [],
  "agent_latency_ms": {},
  "total_latency_ms": 0,
  "token_usage": {},
  "validation_results": {
    "deterministic_status": "PASS",
    "contraindications_checked": ["knee_strain", "hypertension"]
  },
  "critic_result": {
    "decision": "PASS",
    "score": 9,
    "critique": "Plan respects duration and beginner constraints."
  },
  "revision_count": 0,
  "final_output": "...",
  "audio_cue_urls": ["/static/audio/run_027_guidance.mp3"],
  "hitl_evaluation": {
    "expert_id": "evaluator_expert_01",
    "safety_score": 5,
    "chunk_fidelity_score": 5,
    "sequencing_score": 4,
    "audio_clarity_score": 5,
    "notes": "Verified against virabhadrasana_ii.md chunk. Excellent anatomical safety."
  }
}
```

This allows the project to answer:

-   Which run produced this result?
-   Which agents participated?
-   Which tools were called?
-   **Which exact file chunk and line range originated each claim?**
-   Which agent caused latency bottlenecks?
-   How did human expert judges rate physical safety and instructional usability?

------------------------------------------------------------------------

# 28. Evaluation Metrics

## Human-in-the-Loop Domain Expert Assessment (Primary)

-   **Biomechanical Safety Score (1–5)**: Anatomical risk, contraindication avoidance, joint strain protection.
-   **Chunk-Level Claim Fidelity (1–5)**: Verification that plan assertions match referenced file chunk text.
-   **Sequencing & Physiological Pacing (1–5)**: Warm-up $\rightarrow$ peak $\rightarrow$ counter-pose $\rightarrow$ cool-down coherence.
-   **Auditory Guidance & Usability (1–5)**: Hands-free instructional usability on a physical yoga mat.
-   **Inter-Rater Reliability (IRR)**: Cohen's Kappa ($\kappa$) / Fleiss' Kappa across independent evaluators.

## Grounding & Provenance

-   **Chunk Retrieval Precision & Recall**: Relevance of retrieved chunks against curated gold standards.
-   **Claim-to-Chunk Attribution Rate**: Percentage of generated instructions with verified chunk citations.
-   **Hallucination Rate**: Percentage of unsourced or contradictory claims detected.

## Agent Performance & Efficiency

-   Per-agent latency (including STT, RAG, Planner, Safety, Critic, TTS).
-   Per-agent token usage and tool invocation accuracy.
-   Revision frequency and convergence rate.

## System & Multimodal Operations

-   End-to-end latency (voice-in to voice-out and text-in to text-out).
-   Real-time posture correction event latency ($< 2.0\text{s}$).
-   Failure rate and pipeline availability.

------------------------------------------------------------------------

# 29. Human-in-the-Loop (HITL) Evaluation & Domain Expert Judge

Yoga is an applied physical discipline directly involving the human body, joints, muscles, and autonomic nervous system. Recommending improper asanas, reckless breath retentions, or ungrounded physical alignments can cause acute musculoskeletal injury (e.g., knee ligament strain, cervical disc compression) or adverse cardiovascular events.

### Rejection of Pure Agent-as-Judge

Relying exclusively on an autonomous LLM or Agent-as-Judge presents a severe flaw: **hallucinated plausibility**. LLMs can generate eloquent, persuasive reasoning that appears authoritative while containing dangerous physical advice or subtle biomechanical errors. An LLM evaluator frequently rates such outputs highly because of stylistic fluency.

Therefore, YogaSutra **rejects pure Agent-as-Judge as the final arbiter** and establishes a **Human-in-the-Loop (HITL) Evaluation System**.

### Two-Tier Evaluation Architecture

``` mermaid
---
config:
  layout: elk
---
flowchart TD
    RUN["YogaSutra Multi-Agent Run Output\n(Plan + Chunk Claims + Audio Cues)"] --> T1["Tier 1: Automated Advisory Pre-Screen\n(Deterministic Rules + Critic Reflection)"]

    T1 -->|Pre-Screen Flag: Violation| REV["Agent Self-Revision Loop"]
    REV --> RUN

    T1 -->|Pre-Screen: Cleared| TRACE["Per-Run Structured Execution Trace\n(Logged to Evaluation Store)"]

    TRACE --> T2["Tier 2: Human-in-the-Loop (HITL) Expert Panel\n(Certified Yoga Instructors & Domain Evaluators)"]

    subgraph HITL_Rubric["Human Evaluation Rubric (1–5 Likert)"]
        R1["1. Biomechanical Safety\n(Anatomical risk & contraindications)"]
        R2["2. Chunk Claim Fidelity\n(Audit claim against source file chunk)"]
        R3["3. Physiological Sequencing\n(Warm-up, peak pose, cool-down flow)"]
        R4["4. Hands-Free Usability\n(Auditory pacing & voice cue clarity)"]
    end

    T2 --> R1
    T2 --> R2
    T2 --> R3
    T2 --> R4

    R1 --> FINAL["Final Benchmark Scorecard & Inter-Rater Reliability (Kappa)"]
    R2 --> FINAL
    R3 --> FINAL
    R4 --> FINAL
```

### Standardized Human Expert Rubric

Human judges evaluate runs across 4 dimensions:

| Dimension | Score Range | Description | Critical Failure Rule |
| :--- | :--- | :--- | :--- |
| **1. Biomechanical Safety** | 1 (Dangerous) – 5 (Flawless) | Evaluates joint alignment, spinal compression risk, and strict adherence to stated user medical contraindications. | **Score $\le 2$ results in immediate run failure regardless of other scores.** |
| **2. Chunk-to-Claim Fidelity** | 1 (Hallucinated) – 5 (Fully Grounded) | Evaluator audits generated instructions against the exact referenced file chunk (`source_file` and `line_range`). | Any fabricated medical or anatomical claim receives a score of 1. |
| **3. Sequencing & Pacing** | 1 (Disjointed) – 5 (Optimal Flow) | Evaluates energetic progression: gentle mobilization $\rightarrow$ standing/strengthening $\rightarrow$ counter-poses $\rightarrow$ restorative rest. | Omission of warm-up or counter-pose caps score at 3. |
| **4. Auditory Usability** | 1 (Confusing) – 5 (Intuitive) | Evaluates whether spoken audio cues are concise, non-distracting, and actionable while eyes are off the screen. | Verbose or delayed cues receive $\le 2$. |

### Human-in-the-Loop Inspection Interface

The evaluation tool provides domain evaluators with:
1. **Interactive Plan Viewer**: Inspecting sequence, timing, and posture cues.
2. **Chunk Claim Auditor**: Clickable link to the exact source markdown file and line range for instantaneous verification of every claim.
3. **Audio Playback**: Listening to synthesized voice cues to evaluate pacing.
4. **Scoring & Annotation Form**: Recording 1–5 scores, qualitative clinical notes, and approval/rejection flags.

Inter-rater agreement is calculated using **Cohen's Kappa ($\kappa$)** (for pairs) and **Fleiss' Kappa** (for multi-evaluator panels), establishing an objective, scientifically defensible evaluation standard.

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
---
config:
  layout: elk
---
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
---
config:
  layout: elk
---
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

# 31. Multi-Model Intelligence & Dynamic Model Routing

YogaSutra will **not be tied to a single LLM provider**.

The final system will use a **multi-model architecture** in which
different tasks can be routed to different model families according to
the requirements of the task.

Candidate providers/model families include:

-   OpenAI GPT
-   Anthropic Claude
-   Google Gemini
-   DeepSeek
-   Qwen
-   Kimi / Moonshot
-   other compatible models added through the model registry

The exact model IDs will be selected and benchmarked during
implementation because model availability, pricing, latency and
capabilities change over time.

The architecture therefore treats **models as interchangeable execution
resources**, rather than embedding one model directly into every agent.

## 31.1 Master Agent / Model Router

The Root Manager will be extended with a **Model Routing Layer**.

The routing process is:

``` mermaid
flowchart TD
    U["User Request"] --> M["Master Agent / Root Manager"]

    M --> T["Task Classification"]
    T --> R["Requirement Extraction"]

    R --> REQ["Task Requirements"]

    REQ --> REG[("Model Capability Registry")]
    REG --> C["Candidate Models"]

    C --> SC["Routing / Scoring Engine"]
    REQ --> SC

    SC --> SEL["Selected Model + Agent"]

    SEL --> A["Specialized Agent"]
    A --> O["Output"]

    O --> V["Validation / Critic"]

    V -->|Accept| F["Final Response"]
    V -->|Retry / Escalate| SC
```

The Master Agent decides:

1.  **which specialist agent should run;**
2.  **whether the task requires a model call at all;**
3.  **which model family should execute the task;**
4.  **what reasoning effort is appropriate;**
5.  **whether a faster/cheaper model is sufficient;**
6.  **whether a stronger model is required;**
7.  **whether the result should be escalated to another model.**

The router should not simply choose the "most powerful" model. It should
optimize for **task quality subject to latency and resource
constraints**.

------------------------------------------------------------------------

## 31.2 Model Capability Registry

Every available model will have a machine-readable capability profile.

Example:

``` json
{
  "model_id": "provider/model-name",
  "provider": "example-provider",
  "capabilities": {
    "reasoning": 4,
    "tool_use": true,
    "structured_output": true,
    "vision": true,
    "long_context": true
  },
  "context_window": 1000000,
  "max_output_tokens": 128000,
  "latency": {
    "p50_ms": 1200,
    "p95_ms": 3000
  },
  "cost": {
    "input_per_million": 0.0,
    "output_per_million": 0.0
  },
  "reliability": 0.98
}
```

The registry will be periodically refreshed using provider metadata and
internal benchmark measurements.

------------------------------------------------------------------------

## 31.3 Routing Parameters

Model selection will consider multiple parameters rather than a single
intelligence score.

### Task Requirements

-   required reasoning depth;
-   expected output complexity;
-   context size;
-   structured-output requirement;
-   tool-calling requirement;
-   multimodal requirement;
-   retrieval-grounding requirement;
-   response length;
-   precision requirement.

### Model Characteristics

-   reasoning capability;
-   context window;
-   tool-call reliability;
-   structured-output reliability;
-   multimodal support;
-   measured latency;
-   throughput;
-   token cost;
-   historical task accuracy;
-   historical failure rate;
-   provider availability.

### Runtime Constraints

-   current queue/load;
-   remaining latency budget;
-   retry budget;
-   API availability;
-   token budget.

------------------------------------------------------------------------

## 31.4 Routing Score

A normalized routing score can be used to select among eligible models:

$$
Score(m,t) =
w_q Q(m,t)
+ w_r R(m,t)
+ w_c C(m,t)
+ w_o O(m,t)
+ w_v V(m,t)
- w_l L(m)
- w_k K(m)
$$

Where:

-   $Q$ = expected task quality;
-   $R$ = reasoning suitability;
-   $C$ = context suitability;
-   $O$ = output/tool capability;
-   $V$ = historical reliability;
-   $L$ = measured latency;
-   $K$ = estimated token/API cost.

The weights are configurable by task class.

This is a **software routing objective**, not a claim that one model is
universally superior.

------------------------------------------------------------------------

## 31.5 Latency Budget

The final system will target:

> **End-to-end latency ≤ 30 seconds**

30 seconds is treated as a generous hard ceiling rather than the
performance target.

The engineering objective is:

> **Minimize latency while preserving task quality and grounding.**

The system should maintain an internal latency budget, for example:

``` text
Total budget:                 30 s
Target operating range:       < 10–15 s
Routing / orchestration:       < 1 s
RAG retrieval:                 < 1.5 s
Primary model calls:           parallel where possible
Validation / critic:           < 4 s
Final synthesis:               < 3 s
Reserved retry budget:         remaining time
```

These numbers are initial engineering targets and will be replaced by
measured benchmarks.

The router should avoid unnecessary sequential model calls.

------------------------------------------------------------------------

## 31.6 Parallel Model Execution

Independent tasks should execute concurrently.

``` mermaid
flowchart TD
    R["Master Agent"] --> P["Profile Task"]
    R --> K["Knowledge Task"]
    R --> C["Constraint Task"]

    P --> PM["Selected Model"]
    K --> KM["Selected Model"]
    C --> CM["Selected Model"]

    PM --> S["Planner / Synthesizer"]
    KM --> S
    CM --> S

    S --> V["Validation"]
    V --> F["Final Response"]
```

This allows the system to use different models for different subtasks
without multiplying latency unnecessarily.

------------------------------------------------------------------------

## 31.7 Model Specialization

The final mapping will be **learned from benchmarks**, not hard-coded
permanently.

An initial hypothesis is:

  -----------------------------------------------------------------------
  Task Type                           Candidate Model Classes
  ----------------------------------- -----------------------------------
  Fast profile extraction             Low-latency / low-cost model

  Simple classification               Fast model

  RAG query rewriting                 Fast model

  Knowledge synthesis                 Strong tool/RAG-capable model

  Complex practice planning           Strong reasoning model

  Safety validation                   Fast model + deterministic rules,
                                      escalated when needed

  Critic / reflection                 Strong reasoning model when
                                      complexity warrants it

  Long-context synthesis              Long-context model

  Vision / pose analysis              Strong multimodal model

  Final response formatting           Fast structured-output model
  -----------------------------------------------------------------------

The project will benchmark candidates from GPT, Claude, Gemini,
DeepSeek, Qwen and Kimi/Moonshot where API access and project resources
permit.

------------------------------------------------------------------------

## 31.8 Model Cascading

The system will support escalation.

``` mermaid
---
config:
  layout: elk
---
flowchart TB
    A["Task"] --> F["Fast / Low-Cost Model"]
    F --> V["Quality Check"]

    V -->|Sufficient| O["Accept"]
    V -->|Insufficient| S["Stronger Model"]

    S --> V2["Quality Check"]
    V2 -->|Pass| O
    V2 -->|Fail| H["Human / Clarification / Safe Fallback"]
```

This allows routine tasks to remain fast while difficult tasks receive
additional inference capacity.

------------------------------------------------------------------------

## 31.9 Model Routing Experiments

The project will compare:

### Experiment A --- Single Model

Every agent uses the same model.

### Experiment B --- Static Multi-Model

Each agent has a fixed model selected before execution.

### Experiment C --- Dynamic Routing

The Master Agent chooses a model per task.

### Experiment D --- Dynamic Routing + Cascade

The system starts with a fast model and escalates only when quality or
complexity requires it.

Metrics:

-   task quality;
-   grounding;
-   constraint satisfaction;
-   latency;
-   p50/p95 latency;
-   token usage;
-   estimated API cost;
-   model-call count;
-   retry count;
-   escalation rate;
-   failure rate.

The final project should report whether dynamic routing actually
provides a useful quality/latency trade-off.

------------------------------------------------------------------------

# 32. Containerized Runtime & Performance Architecture

Docker is a **runtime and performance component**, not merely a
packaging format.

The final application will use Docker Compose to run the major project
services consistently and concurrently.

## 32.1 Runtime Architecture

``` mermaid
---
config:
  layout: elk
---
flowchart TB
    subgraph DOCKER["Docker Compose Runtime"]
        API["FastAPI + ADK Container"]
        ROUTER["Model Router"]
        G["Model Gateway"]
        RAG["RAG Service"]
        MCP["MCP Server"]
        PG[("PostgreSQL + pgvector")]
        MEM[("Persistent Memory")]
        CACHE["Cache"]
        OBS["OpenTelemetry Collector"]
        UI["Next.js Frontend Container"]
    end

    UI --> API
    API --> ROUTER & RAG & MCP & MEM & OBS
    ROUTER --> G
    G --> GPT["OpenAI"] & CL["Anthropic"] & GE["Google Gemini"] & DS["DeepSeek"] & QW["Qwen"] & KM["Kimi / Moonshot"]
    RAG --> PG & CACHE
    OBS --> TRACE["Trace / Metrics Backend"]
```

External model providers remain external APIs; Docker controls the
application's **routing, orchestration, retrieval, caching, MCP,
observability and frontend/backend runtime**.

------------------------------------------------------------------------

## 32.2 Model Gateway

A unified model gateway will sit between the agents and external
providers.

Conceptually:

``` text
Agent
  ↓
Model Router
  ↓
Model Gateway
  ├── OpenAI adapter
  ├── Anthropic adapter
  ├── Gemini adapter
  ├── DeepSeek adapter
  ├── Qwen adapter
  └── Kimi adapter
```

A gateway abstraction prevents provider-specific logic from leaking into
every agent.

The implementation may use a compatible model gateway such as LiteLLM or
a small project-specific adapter layer, depending on the final
dependency decision.

------------------------------------------------------------------------

## 32.3 Docker Services

The initial production-like Compose environment should contain:

``` text
frontend
api
model-router
mcp-server
rag
postgres
cache
otel-collector
```

A local inference service may be added only where it provides a
measurable benefit.

Potential optional service:

``` text
local-model-runtime
```

for compatible open-weight models when local inference is useful for
low-latency or privacy-sensitive lightweight tasks.

------------------------------------------------------------------------

## 32.4 Docker for Performance

Docker will be used to make the runtime **efficient and reproducible**,
not simply portable.

Performance techniques include:

-   concurrent service execution;
-   persistent database connections;
-   connection pooling;
-   RAG result caching;
-   model response caching where safe;
-   asynchronous FastAPI execution;
-   parallel independent agent calls;
-   HTTP connection reuse;
-   warm service processes;
-   bounded retries;
-   timeout budgets;
-   lightweight containers;
-   preloaded RAG/index resources;
-   centralized model gateway;
-   OpenTelemetry timing;
-   health checks.

The system should not create a new container or process for every agent
invocation.

------------------------------------------------------------------------

## 32.5 Latency-Aware Execution

Every model/tool invocation receives a timeout and budget.

``` mermaid
---
config:
  layout: elk
---
flowchart TD
    A["Request"] --> B["Global 30s Deadline"]
    B --> C["Router"]

    C --> D["Parallel Independent Calls"]
    D --> E["RAG"]
    D --> F["Profile"]
    D --> G["Constraints"]

    E --> H["Planner"]
    F --> H
    G --> H

    H --> I["Critic / Validator"]

    I -->|Within Budget| J["Final Response"]
    I -->|Budget Risk| K["Fast Fallback / Reduced Workflow"]
    I -->|Quality Failure| L["Escalation if Budget Allows"]

    K --> J
    L --> J
```

The system should prefer **degrading gracefully** over exceeding the
latency ceiling.

Examples:

-   skip an unnecessary secondary critic if the answer already passed
    deterministic validation;
-   use a faster model when the task does not justify stronger
    reasoning;
-   avoid repeated retrieval when cached evidence is sufficient;
-   parallelize independent retrieval and profile extraction;
-   stop retries when the remaining latency budget is too small.

------------------------------------------------------------------------

## 32.6 Performance Metrics

The observability system will track:

``` text
End-to-end latency
p50 latency
p95 latency
p99 latency
Model inference latency
RAG latency
Database latency
MCP latency
Queue/wait latency
Routing latency
Number of model calls
Number of tool calls
Token usage
Cache hit rate
Retry count
Escalation count
```

The primary performance target is:

> **p95 end-to-end latency should remain comfortably below the 30-second
> ceiling for normal supported requests.**

The exact target will be finalized after baseline measurements.

------------------------------------------------------------------------

## 32.7 Local vs External Models

The architecture supports both:

### External APIs

-   GPT
-   Claude
-   Gemini
-   DeepSeek
-   Qwen
-   Kimi

### Optional local/open-weight models

A local model can be deployed in Docker when it is useful for:

-   low-complexity classification;
-   privacy-sensitive preprocessing;
-   embeddings/reranking;
-   fallback inference;
-   experimentation.

The project should **not** force every model to run locally. The
objective is to optimize the complete system.

------------------------------------------------------------------------

## 32.8 Container-Level Resource Controls

Docker Compose should define resource-aware configuration for services.

Examples:

``` text
CPU limits
Memory limits
Health checks
Restart policies
Environment-based configuration
Connection pool limits
Service dependencies
Startup ordering
```

For model-serving containers, GPU support can be enabled only if local
hardware and model choice justify it.

------------------------------------------------------------------------

# 33. Updated Observability Architecture

The observability layer must capture both **agent behavior and
model-routing behavior**.

``` mermaid
---
config:
  layout: elk
---
flowchart TB
    R["Request"] --> M["Master Agent"]

    M --> RT["Router Decision"]
    RT --> MD["Selected Model"]

    MD --> AG["Agent Execution"]
    AG --> TOOL["Tool / MCP Calls"]

    AG --> ST["State Changes"]
    AG --> LAT["Latency"]
    AG --> TOK["Token Usage"]

    RT --> ROUTE["Routing Metadata"]
    ROUTE --> O["Observability"]

    TOOL --> O
    ST --> O
    LAT --> O
    TOK --> O

    O --> TR["Distributed Trace"]
    O --> MET["Metrics"]
    O --> EV["Evaluation Store"]

    TR --> DB["Analysis Dashboard"]
    MET --> DB
    EV --> DB
```

Each model invocation should record:

``` text
run_id
agent_id
task_id
provider
model_id
routing_reason
required_capabilities
estimated_latency
actual_latency
input_tokens
output_tokens
cache_status
retry_count
quality_result
```

This makes the routing system itself experimentally observable.

------------------------------------------------------------------------

# 34. Updated Evaluation Strategy

The original 44-run evaluation suite remains mandatory, but the final
project will add **model-routing experiments** to it.

For each persona/task, the system can be evaluated under:

``` text
Single Model
Static Multi-Model
Dynamic Router
Dynamic Router + Cascade
```

This produces a matrix such as:

``` mermaid
flowchart LR
    P["44 Persona Runs"]

    P --> A["Single Model"]
    P --> B["Static Multi-Model"]
    P --> C["Dynamic Routing"]
    P --> D["Dynamic Routing + Cascade"]

    A --> E["Metrics"]
    B --> E
    C --> E
    D --> E

    E --> F["Quality / Latency / Cost / Reliability"]
```

This turns model routing into a measurable contribution rather than a
collection of API integrations.

------------------------------------------------------------------------

# 35. Updated Definition of Done

## Phase 1

-   [x] Problem statement approved
-   [x] PEAS completed
-   [x] Environment analysis completed
-   [x] Agent responsibilities defined
-   [x] Architecture diagrams completed
-   [x] RAG strategy defined
-   [x] Reasoning strategy defined
-   [x] Communication strategy defined
-   [x] Tech stack justified
-   [x] Timeline completed
-   [x] Team responsibility matrix completed
-   [x] Minimal ADK prototype demonstrated

## Phase 2

-   [x] Six core agents operational
-   [x] Speech / Audio Agent operational (bidirectional STT/TTS hands-free loop)
-   [x] Sequential workflow operational
-   [x] Hierarchical workflow operational
-   [x] Shared state operational (including audio & pose event blackboard channels)
-   [x] RAG operational with atomic claim-to-chunk provenance (`source_file`, `chunk_id`, `line_range`)
-   [x] Tools operational (including speech transcription, synthesis, and chunk claim verification)
-   [x] Safety validation operational
-   [x] Critic/reflection operational (verifying plan claims against referenced chunk evidence)
-   [x] Persistent memory operational
-   [x] MCP integration operational
-   [x] Structured traces operational with chunk provenance claims
-   [x] Working frontend with speech interaction
-   [x] Model gateway abstraction operational
-   [x] At least three model providers integrated
-   [x] Initial latency benchmark available

## Phase 3A — Advanced Agent Intelligence & Event-Driven Architecture

-   [x] Tree-of-Thoughts (ToT) multi-candidate sequence exploration operational in PlannerAgent
-   [x] Weighted utility function candidate scoring operational (duration, safety, transition, grounding)
-   [x] Asynchronous Pub/Sub Event Bus (`app/workflows/event_driven.py`) operational
-   [x] Event schemas defined and typed (`SessionCompletedEvent`, `FeedbackReceivedEvent`, `SafetyEscalationEvent`, `PostureDeviationEvent`)
-   [x] Advanced longitudinal memory (`app/memory/longitudinal.py`) operational with multi-session progression tracking
-   [x] Automated unit tests for ToT planning and Event Bus passing

## Phase 3B — Modular Real-Time Computer Vision Posture Engine

-   [x] Video frame ingestion worker via OpenCV operational with 480p capture clamp and zero-lag bounded buffer (`app/vision/detector.py`)
-   [x] MediaPipe 3D Pose Lite (`model_complexity=0`, `enable_segmentation=False`) integrated under <60MB RAM footprint
-   [x] Zero-hallucination pure-Python deterministic 3D vector geometry calculator operational (<2ms latency, `app/vision/biomechanics.py`)
-   [x] Knee flexion, elbow flexion, spinal verticality, and shoulder/hip symmetry angle verification
-   [x] Classical asana reference criteria matrix established (`app/vision/criteria.py`)
-   [x] Pose Understanding & Correction Agent (`app/agents/pose/agent.py`) operational with 1.5s temporal debounce
-   [x] Closed-loop corrective cues dispatched to SpeechAgent via `PostureDeviationEvent`
-   [x] Standalone interactive verification CLI runner (`scripts/run_pose_detector.py`) with telemetry HUD (FPS, latency, RAM)
-   [x] Unit tests for 3D biomechanics calculations, zero-hallucination assertion, and PoseAgent passing

## Phase 3C — Fullstack Live Practice Studio UI & Observability Dashboard

-   [ ] HTML5 / WebRTC Video Canvas with real-time 33-point skeletal wireframe overlay
-   [ ] Live visual HUD with green/yellow/red joint angle indicator rings
-   [ ] Synchronous voice audio player for ElevenLabs / WebSpeech corrective cues
-   [ ] MAS Observability & Trace Analytics Dashboard (`app/ui/dashboard.py`) operational
-   [ ] Per-agent latency waterfalls, token usage by provider, and communication pattern visualizer
-   [ ] Real-time CV FPS counter and speech synthesis turnaround monitor

## Phase 4 — Scientific Evaluation & Benchmark Experiments (40+ Runs)

-   [ ] 40+ diverse evaluation runs executed across personas P01–P05 and edge cases
-   [ ] 4-way MAS architecture comparison benchmark (Sequential vs Hierarchical vs Broadcast vs Blackboard)
-   [ ] Multi-model provider comparison (Google Gemini vs Ollama Cloud 120B vs NVIDIA NIM)
-   [ ] Human-in-the-Loop (HITL) domain expert panel evaluation completed across all 40+ runs
-   [ ] Inter-rater reliability (Cohen's / Fleiss' Kappa) measured on expert rubric scores
-   [ ] CriticAgent reflection loop ablation study completed
-   [ ] End-to-end latency benchmark with p95 measured against 30-second ceiling

## Phase 5 — Enterprise Containerization & Production Docker Packaging

-   [ ] Production multi-stage backend Dockerfile (`Dockerfile.backend`) with non-root security user & minimal slim runtime
-   [ ] Production frontend Dockerfile (`Dockerfile.frontend`) with multi-stage build optimization
-   [ ] Orchestrated `docker-compose.yml` and `docker-compose.prod.yml` with isolated networks (`api`, `frontend`, `reverse-proxy`)
-   [ ] Enterprise resource quotas, health checks, and restart policies configured for client host environments
-   [ ] Persistent volume orchestration for practice memory (`data/memory`), corpus (`data/corpus.json`), and model caches
-   [ ] Reverse proxy configuration (Nginx) with WebSocket / WebRTC support, CORS, and TLS termination
-   [ ] Production environment template (`.env.production.example`) with secret masking and strict validation
-   [ ] Automated container smoke tests and CI verification script (`scripts/docker_smoke_test.sh`)
-   [ ] Enterprise client deployment guide & runbook (`docs/DEPLOYMENT_GUIDE.md`)

## Phase 6 — Client Handover, Final Technical Report & Defense Deliverables

-   [ ] Comprehensive IEEE-style academic final report & enterprise technical whitepaper completed (`docs/FINAL_REPORT.md`)
-   [ ] Turnkey Client Handover Package with SLA verification and architectural runbook
-   [ ] Executive presentation slide deck and live interactive demonstration script completed
-   [ ] Viva defense & technical review preparation guide with design rationale completed

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

Build the complete core agentic workflow with bidirectional hands-free speech communication.

### Implement

#### Agents

-   Root Manager
-   Profile
-   Knowledge/RAG
-   Planner
-   Safety
-   Critic
-   **Speech / Audio Agent** (bidirectional STT transcription & TTS spoken synthesis)

#### Orchestration

-   sequential workflow;
-   hierarchical manager;
-   parallel/broadcast experiment;
-   shared-state/blackboard workflow (including audio channels).

#### Tools

-   retrieval;
-   profile;
-   history;
-   validation;
-   duration calculation;
-   speech transcription & synthesis;
-   feedback.

#### Memory

-   session state (including voice conversation state);
-   persistent user context.

#### Safety

-   deterministic validation;
-   LLM consistency validation;
-   clarification flow.

#### Reflection

-   planner → critic → revision.

#### Observability

-   structured traces;
-   per-agent timing (including STT/TTS latency);
-   tool calls;
-   state transitions.

#### MCP

-   one meaningful MCP server/tool integration.

### Phase 2 Prototype

``` mermaid
---
config:
  layout: elk
---
flowchart TD
    U["User\n(Voice or Text)"] --> SA["Speech / Audio Agent"]
    U --> R["Root Manager"]
    SA --> R

    R --> P["Profile"]
    R --> K["Knowledge / RAG"]
    R --> PL["Planner"]

    P --> S[("Shared State\n(Blackboard)")]
    K --> S
    PL --> S

    S --> SV["Safety"]
    SV --> CR["Critic"]

    CR -->|Pass| F["Final Response\n(Text Trace + Spoken Voice)"]
    CR -->|Revise| PL

    F --> U
```

### Phase 2 Deliverables

-   updated charter;
-   revised timeline;
-   working prototype with speech communication;
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

# Phase 3A --- Advanced Agent Intelligence & Event-Driven Architecture

### Dependency Order
**Prerequisites**: Phase 2 (Completed ✅)
**Rationale**: Before real-time video posture correction or complex UI dashboards can be built, the system requires the underlying asynchronous event backbone (`event_driven.py`), event schemas, and search-based planning reasoning (ToT + utility ranking).

### Deliverables
1. **Tree-of-Thoughts (ToT) Multi-Candidate Sequence Exploration**:
   - `PlannerAgent` explores multiple candidate sequence paths (branching based on user energy, prop availability, and intensity).
   - Candidate utility function scoring:
     $$\text{Utility} = w_1 \cdot \text{DurationCompliance} + w_2 \cdot \text{SafetyScore} + w_3 \cdot \text{TransitionSmoothness} + w_4 \cdot \text{EvidenceGrounding}$$
2. **Asynchronous Event-Driven Bus (`app/workflows/event_driven.py`)**:
   - Pub/Sub event dispatcher operating over the shared blackboard.
   - Core typed events: `SessionCompletedEvent`, `FeedbackReceivedEvent`, `SafetyEscalationEvent`, `PostureDeviationEvent`.
3. **Advanced Longitudinal Memory (`app/memory/longitudinal.py`)**:
   - Multi-session progression tracking (stamina curve, pose hold duration history).
   - Adaptive difficulty scaling and preference persistence.

------------------------------------------------------------------------

# Phase 3B --- Modular Real-Time Computer Vision Posture Engine

### Dependency Order
**Prerequisites**: Phase 3A (Event-Driven Bus & `PostureDeviationEvent` schema)
**Rationale**: Posture analysis runs as a decoupled high-speed worker that emits deviation events into the Phase 3A Event Bus. Building the CV pipeline before the event layer would result in tight coupling or blocking the main agent loop.

### High-Efficiency Edge & Client Hardware Optimization
High-resolution camera feeds (e.g. 1080p/4K webcams) stream uncompressed video frames at >6MB per frame, which can quickly saturate system bus bandwidth and induce thermal and memory throttling on edge workstations, laptops, or resource-constrained client machines (e.g. ≤8GB RAM environments). To guarantee enterprise-grade stability, sub-15ms processing turnaround, and an ultra-lean (<100MB active memory) footprint, Phase 3B enforces 5 production-grade streamlining optimizations:
1. **Immediate Ingestion Clamping**: Clamps video capture at the driver/capture boundary to 480p (640×480), reducing per-frame RAM allocation from 6.2MB to <900KB (an 85% bandwidth reduction).
2. **Bounded Buffer Ring (`Queue(maxsize=1)`)**: Drops stale frames whenever processing lags, guaranteeing zero cumulative latency lag and preventing memory buffer bloat.
3. **MediaPipe Pose Lite Runtime (`model_complexity=0`, `enable_segmentation=False`)**: Disables background segmentation (which consumes over 60% of CV memory) to achieve an ultra-compact ~6MB model footprint running in 8–12ms on modern CPUs.
4. **Adaptive Decoupled Inference Rate**: Decouples smooth 30 FPS display rendering from 12–15 FPS skeletal inference with lightweight exponential smoothing, cutting CPU compute in half during static yoga holds.
5. **Zero-Allocation In-Place Mathematics**: Pure-Python 3D vector geometry without heap churn, executing angle calculations in <2ms.

### Zero-Hallucination Guarantee
To eliminate any risk of LLM hallucinations during live biomechanical analysis:
- **No LLM in the measurement loop**: Joint angles are calculated deterministically via 3D vector geometry:
  $$\theta = \arccos\left(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}\right)$$
- **Deterministic Asana Reference Matrix (`app/vision/criteria.py`)**: Exact target intervals derived from classical `corpus.json` literature (e.g. Virabhadrasana II front knee $90^\circ \pm 5^\circ$; Tadasana spine deviation $<5^\circ$).
- **Temporal Debounce Filter**: Deviances must persist $>1.5\text{s}$ before triggering alerts, eliminating sensor noise and twitch alarms.
- **Direct Cue Mapping**: Corrective feedback is mapped directly from validated biomechanical rules, not generated stochastically.

### Deliverables
1. **Video Ingestion & 3D Pose Extraction (`app/vision/detector.py`)**:
   - OpenCV video frame ingestion worker (webcam / video stream / test image).
   - Ingestion clamping to 640×480 with zero-lag bounded buffer queue.
   - MediaPipe Pose Lite extracting 33 3D skeletal landmarks (<60MB active RAM).
2. **Deterministic Biomechanical Joint Calculator (`app/vision/biomechanics.py`)**:
   - Fast pure-Python trigonometric 3D angle calculations (<2ms turnaround):
     - Knee flexion angle (left and right)
     - Elbow flexion angle (left and right)
     - Spinal verticality and lateral torso alignment
     - Shoulder-hip horizontal level and hip squareness
3. **Classical Asana Criteria Standards (`app/vision/criteria.py`)**:
   - Grounded angle thresholds for core asanas (Virabhadrasana II, Tadasana, Vrikshasana, Utkatasana, Bhujangasana).
4. **Pose Understanding & Correction Agent (`app/agents/pose/agent.py`)**:
   - Compares live joint angles against criteria table with 1.5s temporal debounce.
   - Generates high-priority corrective voice cues dispatched to `SpeechAgent` via `PostureDeviationEvent`.
5. **Standalone Interactive Verification CLI (`scripts/run_pose_detector.py`)**:
   - Supports `--webcam 0`, `--image <path>`, and `--video <path>`.
   - On-screen telemetry HUD displaying live FPS, inference latency (ms), and active RAM consumption (MB).

------------------------------------------------------------------------

# Phase 3C --- Fullstack Live Practice Studio UI & Observability Dashboard

### Dependency Order
**Prerequisites**: Phase 3A (Event Bus) & Phase 3B (CV Engine & Joint Landmarks)
**Rationale**: The interactive UI requires both the live skeletal landmark stream from Phase 3B to draw the wireframe canvas and the event bus from Phase 3A to display traces and real-time audio playback.

### Deliverables
1. **Live Practice Studio Canvas**:
   - HTML5 / WebRTC Video Canvas with real-time 33-point skeletal wireframe overlay.
   - Interactive visual HUD: Green/Yellow/Red joint angle indicator rings.
   - Synchronous audio player for ElevenLabs / WebSpeech corrective cues.
2. **MAS Observability & Trace Analytics Dashboard (`app/ui/dashboard.py`)**:
   - Visual execution trace breakdown: per-agent latency waterfalls, token usage by provider.
   - Multi-agent communication pattern visualizer (Sequential vs. Hierarchical vs. Broadcast).
   - Real-time CV FPS counter and speech synthesis turnaround latency monitor.

------------------------------------------------------------------------

# Phase 4 --- Scientific Evaluation & Benchmark Experiments (40+ Runs)

### Dependency Order
**Prerequisites**: Phase 3C (Complete operational system across text, speech, vision, and UI)
**Rationale**: Empirical benchmarking and expert evaluations require a fully integrated, operational system. Testing an incomplete system produces invalid evaluation data.

### Deliverables
1. **40+ Diverse Evaluation Runs (`data/evaluation/runs/`)**:
   - Executed across synthetic personas P01–P05 and edge cases (injury constraints, morning/evening, time constraints, out-of-corpus queries).
2. **Four-Way Architectural Comparison Benchmark**:
   - Compare Sequential vs. Hierarchical vs. Broadcast vs. Blackboard on identical test requests.
   - Measure task completion latency, token overhead, and revision counts.
3. **Multi-Model Provider Comparison**:
   - Compare Google Gemini vs. Ollama Cloud (GPT-OSS 120B) vs. NVIDIA NIM across reasoning quality and response latency.
4. **Human-in-the-Loop (HITL) Domain Expert Evaluation**:
   - Certified yoga instructor panel grading plans across 4 standardized dimensions: Biomechanical Safety, Classical Authenticity, Sequencing Flow, and Tone.
   - Inter-rater reliability measurement using Cohen's and Fleiss' Kappa.
5. **CriticAgent Reflection Loop Ablation Study**:
   - Quantify safety violation reduction and grounding improvement with reflection loop vs. single-pass baseline.

------------------------------------------------------------------------

# Phase 5 --- Enterprise Containerization & Production Docker Packaging

### Dependency Order
**Prerequisites**: Phase 3C (Fullstack Live Practice Studio UI) & Phase 4 (Empirical Evaluation Benchmarks)
**Rationale**: Containerizing the verified multi-agent system, vision pipeline, and frontend into a production-grade container orchestration stack enables zero-friction, repeatable commercial deployment for client handover and enterprise cloud hosting.

### Deliverables
1. **Production-Grade Multi-Stage Dockerfiles**:
   - `Dockerfile.backend`: Multi-stage build on Debian slim/Alpine with Python 3.12, separating compilation dependencies from runtime, running under a dedicated non-root `appuser`, minimizing attack surface and image size.
   - `Dockerfile.frontend`: Multi-stage build compiling static/standalone Next.js/Vite bundles served via a hardened, ultra-lightweight Alpine runtime.
2. **Enterprise Container Orchestration (`docker-compose.yml` & `docker-compose.prod.yml`)**:
   - Multi-container architecture:
     - `api`: FastAPI MAS core, computer vision, and speech endpoints with Uvicorn/Gunicorn workers.
     - `frontend`: Live Practice Studio UI.
     - `reverse-proxy`: Hardened Nginx container handling SSL/TLS termination, HTTP/2, WebSocket / WebRTC streaming, and CORS enforcement.
     - `eval-runner`: On-demand container for isolated CI/CD regression testing and performance benchmarking.
   - Network segmentation (isolated backend network vs public-facing DMZ network).
   - Strict resource governance (`deploy.resources.limits` for CPU and memory ceilings) ensuring predictable performance across host environments.
   - Standardized container health checks (`HEALTHCHECK`) with automated restart policies.
3. **Data Persistence & Caching Volumes**:
   - Dedicated named volumes for persistent user practice memory (`data/memory`), classical knowledge corpus (`data/corpus.json`), and model weight/audio caches (`data/cache`).
4. **Security, Secrets & Configuration Management**:
   - Production environment template (`.env.production.example`) with strict runtime validation and zero secret leakage.
   - Support for Docker secrets and environment-driven configuration across dev, staging, and production tiers.
5. **Cross-Platform & Hardware Acceleration Support**:
   - Multi-architecture container builds supporting both `linux/amd64` and `linux/arm64` (AWS Graviton, Apple Silicon, and Intel/AMD servers).
   - Optional GPU pass-through flags for NVIDIA Container Toolkit acceleration where available.
6. **Automated Smoke Test & Build Pipeline (`scripts/docker_smoke_test.sh`)**:
   - Automated script building images, spinning up the compose stack, validating `/health` and WebSocket handshakes, and asserting end-to-end responsiveness.
7. **Client Handover & Deployment Runbook (`docs/DEPLOYMENT_GUIDE.md`)**:
   - Turnkey, professional operations manual covering single-command deployment (`docker compose up -d`), environment provisioning, backup/restore procedures, and cloud migration (AWS ECS, GCP Cloud Run, Azure Container Instances).

------------------------------------------------------------------------

# Phase 6 --- Client Handover, Final Technical Report & Defense Deliverables

### Dependency Order
**Prerequisites**: Phase 4 (Evaluation Benchmarks) & Phase 5 (Enterprise Containerization)
**Rationale**: Final technical reports, client handover packages, and presentation decks must report verified empirical metrics and containerized deployment artifacts.

### Deliverables
1. **Turnkey Client Handover Package & SLA Verification**:
   - Client-ready deliverables bundle, SLA compliance audit (latency, uptime, safety guarantees), and administrative operations manual.
2. **Comprehensive Final Technical Project Report (`docs/FINAL_REPORT.md`)**:
   - IEEE / Academic MAS standard: Problem Statement, PEAS Framework, Multi-Agent Architecture, RAG Grounding, Safety Proofs, Empirical Evaluation Results, and Architectural Trade-off Analysis.
3. **Executive Presentation & Viva Defense Package**:
   - Professional slide deck outline, interactive live demonstration script, and comprehensive Q&A preparation guide addressing multi-agent architectural rationale.

------------------------------------------------------------------------

# 35. System Architectural Synthesis & Integration Matrix

The comprehensive visual architecture of YogaSutra is unified in **[6. System Architecture](#6-system-architecture)**. This section synthesizes the cross-subsystem contracts, interaction protocols, and runtime SLAs governing all system components.

### 35.1 Multi-Tier Architectural Topology

The complete production system integrates seven distinct operational tiers into a cohesive, decoupled runtime:

1. **Multimodal Edge Client Tier (`frontend`)**: Next.js / React application rendering the live HTML5 WebRTC/Canvas skeleton overlay, bidirectional Web Speech / ElevenLabs audio player, session planner interface, and real-time posture telemetry HUD.
2. **Multi-Agent Reasoning Core (`api`)**: Google ADK-powered MAS orchestrating Root Manager, Profile, Knowledge/RAG, Practice Planner, Safety & Constraint, Critic/Reflection, and Speech agents over a shared Blackboard state.
3. **High-Speed Biomechanics Engine (`app/vision`)**: Pure-Python 3D vector geometry pipeline consuming 480p clamped video frames, extracting 33 MediaPipe Pose landmarks in <12ms, and verifying physical joint angles against classical criteria matrices with zero LLM hallucination.
4. **Reactive Event Bus (`app/events`)**: Asynchronous, non-blocking event-driven bus decoupling high-frequency video frames and voice inputs from synchronous agent planning loops (`PostureDeviationEvent`, `VoiceCommandEvent`, `SessionFeedbackEvent`).
5. **Knowledge, Tools & Persistence (`data`)**: PostgreSQL + pgvector vector store, Model Context Protocol (MCP) tool server, auditable classical chunk registry (`corpus.json`), and long-term user practice memory (`data/memory`).
6. **Observability & Evaluation Layer (`app/observability`)**: OpenTelemetry distributed tracing, real-time latency/WER/FPS metrics collector, and interactive Streamlit analytics dashboard (`app/ui/dashboard.py`).
7. **Container Orchestration Runtime (`docker-compose.yml`)**: Production multi-container stack with network isolation, healthchecks, volume persistence, and strict CPU/memory limits.

### 35.2 Subsystem Integration & Protocol Matrix

| Source Subsystem | Target Subsystem | Interaction Pattern | Transport / Protocol | Message / Payload Schema | SLA / Target Latency | Failure & Recovery Policy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Webcam Driver** | **Frame Ingestion Worker** | Streaming Ingestion | OpenCV VideoCapture | Clamped 480p RGB Frame Buffer | < 5 ms | Drop stale frame via `Queue(maxsize=1)` |
| **Biomechanics Engine** | **Pose Agent** | Event-Driven Publish | In-Process Event Bus | `PostureDeviationEvent` (joint, angle, target, severity) | < 2 ms | 1.5s temporal debounce filter |
| **Pose Agent** | **Speech Agent** | Asynchronous Dispatch | Event Bus / In-Memory Queue | Corrective Audio Cue String | < 10 ms | Priority override: Interrupt routine cues |
| **Speech Agent** | **Client Audio Player** | Streaming Audio | WebSocket / Audio Chunk Stream | PCM / MP3 Chunk Stream (ElevenLabs / TTS) | < 1,500 ms TTFB | Fallback to Web Speech API locally |
| **Client UI** | **Root Manager** | Request / Response | REST (HTTP/2 JSON) | `UserGoalRequest` (prompt, constraints, time) | < 200 ms | Return 422 with input validation message |
| **Planner Agent** | **Safety / Critic Agents** | Hierarchical / Sequential | Shared Blackboard (`SessionState`) | `PracticePlan` schema (asanas, duration, cues) | < 3,000 ms | Reflection loop triggers up to 2 revisions |
| **Knowledge Agent** | **MCP / pgvector** | Tool Invocation | MCP JSON-RPC / SQL | `RetrievalQuery` -> `ChunkEvidence` with line provenance | < 400 ms | Fallback to cached local `corpus.json` |
| **Client Session UI** | **Long-Term Memory** | Post-Session Push | Event Bus -> File Store | `SessionFeedbackEvent` (rating, fatigue, notes) | Asynchronous | Write to disk journal; retry on lock error |
| **All MAS Agents** | **Telemetry Collector** | Background Tracing | OpenTelemetry / In-Memory Queue | `TraceSpan` (agent, tool, latency_ms, tokens, status) | Zero overhead (< 1 ms) | Drop telemetry if buffer reaches 90% |

### 35.3 System Invariants

To guarantee production reliability and safety across all tiers:
- **Zero-Hallucination Invariant**: Joint angle calculations and posture deviation flags are 100% computed via deterministic 3D trigonometry. LLMs never perform visual landmark estimation.
- **Strict Safety Gate Invariant**: No practice plan is ever presented to the user or dispatched to the audio engine without passing explicit constraint checks and Critic validation.
- **Bounded Latency Invariant**: Any camera frame arriving while the biomechanics engine is processing is discarded immediately (`Queue(maxsize=1)`), guaranteeing the user never experiences cumulative posture tracking delay.

------------------------------------------------------------------------

# 36. Phase Timeline

  ---------------------------------------------------------------------
  Period                             Milestone
  ---------------------------------- ----------------------------------
  **Now → Sep 21**                   Problem definition, charter,
                                     architecture and Phase 1
                                     submission

  **Sep 22--27**                     ADK foundation, Runner, Sessions,
                                     State

  **Sep 28--Oct 4**                  Root + Profile + RAG

  **Oct 5--11**                      Planner + tools + sequential
                                     workflow

  **Oct 12--18**                     Safety + Critic + reflection

  **Oct 19--25**                     Phase 2 prototype + documentation

  **Oct 26--Nov 2**                  Communication architecture
                                     experiments

  **Nov 3--9**                       40+ runs + evaluation

  **Nov 10--16**                     ToT + event-driven memory +
                                     multimodal

  **Nov 17 onward**                  Observability + deployment + final
                                     polish

  **Final week**                     Final report + PPT +
                                     demonstration + viva
  ---------------------------------------------------------------------

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
│   │   ├── critic/
│   │   ├── speech/                 # Speech & Audio Agent (STT/TTS conversational loop)
│   │   └── pose/                   # Modular CV: Pose Understanding & Correction Agent
│   │
│   ├── audio/                      # Speech streaming, Whisper/WebSpeech STT & TTS synthesizer
│   ├── vision/                     # Modular CV: OpenCV capture, MediaPipe 3D / YOLO-Pose, joint angles
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
├── frontend/                       # Next.js UI with Live Practice Studio (HUD & Voice widget)
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
  Chunk-Level Grounding       Atomic claim-to-chunk file attribution model
  Multi-Agent Workflow        Specialized agents
  Hierarchical Architecture   Root Manager
  Sequential Architecture     Ordered workflow
  Broadcast                   Parallel specialists
  Blackboard                  Shared ADK state
  Event Driven                Feedback/session and real-time posture events
  Memory                      Session + persistent memory
  KRR                         Rules + structured knowledge + LLM
  Guardrails                  Input/tool/output validation
  Safety                      Constraint validator & biomechanical safety
  MCP                         Standardized tool access
  A2A                         Optional Phase 3 extension
  Observability               Traces and metrics
  Human-in-the-Loop Judge     Domain expert (certified instructor) benchmark evaluation
  Automated Pre-Screen        Deterministic constraint verification & self-critique
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

-   task decomposition;
-   sequential workflows;
-   hierarchical management;
-   parallel/broadcast communication;
-   blackboard communication;
-   reflection and memory.

They do not require CrewAI as a runtime dependency.

------------------------------------------------------------------------

## DD-02 --- Small Specialist Agents over Mega-Agents

**Decision:** Maintain a small team of single-purpose agents.

**Reason:** Decomposing the system into dozens of micro-agents creates
excessive latency and token overhead. Combining everything into a single
agent eliminates the architectural comparison value of the project.

------------------------------------------------------------------------

## DD-03 --- Dedicated Vector DB for Knowledge

**Decision:** Use PostgreSQL + pgvector for the curated knowledge base.

**Reason:** The project requires evidence grounding, source tracking,
and metadata filtering. A proper vector store enables true Agentic RAG
and keeps evidence distinct from session memory.

------------------------------------------------------------------------

## DD-04 --- Shared Blackboard for Session Context

**Decision:** Use ADK session state as a blackboard for multi-agent
coordination.

**Reason:** Passing complete conversational history between agents
wastes tokens and creates confusion. A structured blackboard provides a
clean contract between agents.

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

## DD-07 --- Multimodal Phasing & Decoupled Architecture

**Decision:** Promote bidirectional Speech Communication to Phase 2 core scope; engineer Real-Time Video Posture Correction as a decoupled plug-and-play module for Phase 3.

**Reason:** Yoga practice is inherently physical and hands-free; voice communication (STT/TTS) is natural and immediately elevates the human-agent interaction without altering the core planning orchestration. The high-speed computer vision pipeline (OpenCV + MediaPipe/YOLO) is isolated behind asynchronous event triggers (`PostureDeviationEvent`), ensuring the core multi-agent reasoning, RAG, and evaluation remain intact while enabling a plug-and-play demonstration.

------------------------------------------------------------------------

## DD-08 --- No Medical Positioning

**Decision:** Position YogaSutra as a yoga practice and knowledge
assistant.

**Reason:** The system is not clinically validated and should not make
medical diagnosis or treatment claims.

------------------------------------------------------------------------

## DD-09 --- Human-in-the-Loop Evaluation for Physical Health

**Decision:** Employ certified yoga instructors and domain human evaluators as the primary benchmark judges rather than relying on an autonomous LLM/Agent-as-Judge.

**Reason:** Yoga instructions directly affect human bodily posture, joint mechanics, spinal alignment, and cardiovascular safety. LLM evaluators suffer from "hallucinated plausibility," frequently grading dangerous sequencing or improper joint alignments as acceptable if presented with fluent prose. A rigorous Human-in-the-Loop (HITL) panel with a standardized 4-dimensional rubric and inter-rater reliability metrics is scientifically and ethically mandatory when evaluating physical human movement applications.

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

  Hallucination           High                    RAG + chunk-level claim
                                                  attribution + critic

  Poor retrieval          High                    Metadata, ranking,
                                                  chunk line mapping,
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

  Multimodal scope creep  High                    Decoupled architecture:
                                                  speech uses standard
                                                  STT/TTS; CV runs as an
                                                  isolated async event worker

  Evaluation subjectivity Medium                  Standardized 4-dimension
  & LLM judge bias                                human expert rubric +
                                                  inter-rater reliability
                                                  (Cohen's Kappa) +
                                                  chunk-level claim audits

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

-   [x] Problem statement approved
-   [x] PEAS completed
-   [x] Environment analysis completed
-   [x] Agent responsibilities defined
-   [x] Architecture diagrams completed
-   [x] RAG strategy defined
-   [x] Reasoning strategy defined
-   [x] Communication strategy defined
-   [x] Tech stack justified
-   [x] Timeline completed
-   [x] Team responsibility matrix completed
-   [x] Minimal ADK prototype demonstrated

## Phase 2

-   [x] Six core agents operational
-   [x] Speech / Audio Agent operational (bidirectional STT/TTS hands-free loop)
-   [x] Sequential workflow operational
-   [x] Hierarchical workflow operational
-   [x] Shared state operational (including audio & pose blackboard channels)
-   [x] RAG operational with atomic claim-to-chunk provenance (`source_file`, `chunk_id`, `line_range`)
-   [x] Tools operational (including speech transcription, synthesis, and chunk claim verification)
-   [x] Safety validation operational
-   [x] Critic/reflection operational (verifying plan claims against referenced chunk evidence)
-   [x] Persistent memory operational
-   [x] MCP integration operational
-   [x] Structured traces operational with chunk provenance claims
-   [x] Working frontend with speech interaction
-   [x] Initial evaluation suite

## Phase 3A — Advanced Agent Intelligence & Event-Driven Architecture

-   [x] Tree-of-Thoughts (ToT) multi-candidate sequence exploration operational in PlannerAgent
-   [x] Weighted utility function candidate scoring operational
-   [x] Asynchronous Pub/Sub Event Bus (`app/workflows/event_driven.py`) operational
-   [x] Typed event schemas (`SessionCompletedEvent`, `FeedbackReceivedEvent`, `SafetyEscalationEvent`, `PostureDeviationEvent`)
-   [x] Advanced longitudinal memory (`app/memory/longitudinal.py`) operational
-   [x] Unit tests for ToT planning and Event Bus passing

## Phase 3B — Modular Real-Time Computer Vision Posture Engine

-   [x] Video frame ingestion worker via OpenCV operational with 480p capture clamp and zero-lag bounded buffer (`app/vision/detector.py`)
-   [x] MediaPipe 3D Pose Lite (`model_complexity=0`, `enable_segmentation=False`) integrated under <60MB RAM footprint
-   [x] Zero-hallucination pure-Python deterministic 3D vector geometry calculator operational (<2ms latency, `app/vision/biomechanics.py`)
-   [x] Classical asana reference criteria matrix established (`app/vision/criteria.py`)
-   [x] Pose Understanding & Correction Agent (`app/agents/pose/agent.py`) operational with 1.5s temporal debounce
-   [x] Closed-loop corrective cues dispatched to SpeechAgent via `PostureDeviationEvent`
-   [x] Standalone interactive verification CLI runner (`scripts/run_pose_detector.py`) with telemetry HUD (FPS, latency, RAM)
-   [x] Unit tests for 3D biomechanics calculations, zero-hallucination assertion, and PoseAgent passing

## Phase 3C — Fullstack Live Practice Studio UI & Observability Dashboard

-   [ ] HTML5 / WebRTC Video Canvas with real-time 33-point skeletal wireframe overlay
-   [ ] Live visual HUD with green/yellow/red joint angle indicator rings
-   [ ] Synchronous voice audio player for ElevenLabs / WebSpeech corrective cues
-   [ ] MAS Observability & Trace Analytics Dashboard (`app/ui/dashboard.py`) operational
-   [ ] Per-agent latency waterfalls, token usage by provider, and communication visualizer
-   [ ] Real-time CV FPS counter and speech synthesis turnaround monitor

## Phase 4 — Scientific Evaluation & Benchmark Experiments (40+ Runs)

-   [ ] 40+ diverse evaluation runs executed across personas P01–P05 and edge cases
-   [ ] 4-way MAS architecture comparison benchmark (Sequential vs Hierarchical vs Broadcast vs Blackboard)
-   [ ] Multi-model provider comparison (Google Gemini vs Ollama Cloud 120B vs NVIDIA NIM)
-   [ ] Human-in-the-Loop (HITL) domain expert panel evaluation completed across all 40+ runs
-   [ ] Inter-rater reliability (Cohen's / Fleiss' Kappa) measured on expert rubric scores
-   [ ] CriticAgent reflection loop ablation study completed
-   [ ] End-to-end latency benchmark with p95 measured against 30-second ceiling

## Phase 5 — Enterprise Containerization & Production Docker Packaging

-   [ ] Production multi-stage backend Dockerfile (`Dockerfile.backend`) with non-root security user & minimal slim runtime
-   [ ] Production frontend Dockerfile (`Dockerfile.frontend`) with multi-stage build optimization
-   [ ] Orchestrated `docker-compose.yml` and `docker-compose.prod.yml` with isolated networks (`api`, `frontend`, `reverse-proxy`)
-   [ ] Enterprise resource quotas, health checks, and restart policies configured for client host environments
-   [ ] Persistent volume orchestration for practice memory (`data/memory`), corpus (`data/corpus.json`), and model caches
-   [ ] Reverse proxy configuration (Nginx) with WebSocket / WebRTC support, CORS, and TLS termination
-   [ ] Production environment template (`.env.production.example`) with secret masking and strict validation
-   [ ] Automated container smoke tests and CI verification script (`scripts/docker_smoke_test.sh`)
-   [ ] Enterprise client deployment guide & runbook (`docs/DEPLOYMENT_GUIDE.md`)

## Phase 6 — Client Handover, Final Technical Report & Defense Deliverables

-   [ ] Comprehensive IEEE-style academic final report & enterprise technical whitepaper completed (`docs/FINAL_REPORT.md`)
-   [ ] Turnkey Client Handover Package with SLA verification and architectural runbook
-   [ ] Executive presentation slide deck and live interactive demonstration script completed
-   [ ] Viva defense & technical review preparation guide with design rationale completed

------------------------------------------------------------------------

# 44. Final Evaluation Matrix

The final report should explicitly map project evidence to the course
requirements.

  ---------------------------------------------------------------------
  Requirement                        Evidence
  ---------------------------------- ----------------------------------
  Problem Definition                 Problem statement + PEAS

  Objectives / Outcomes              Project goals

  Methodology                        Workflow + architecture

  Feasibility                        Technology/resource analysis

  Reasoning Strategy                 ReAct / structured planning /
                                     reflection

  RAG                                Knowledge agent + retrieval
                                     pipeline

  Tools                              Tool registry + traces

  Messaging                          Sequential / broadcast /
                                     blackboard / hierarchy

  MCP                                MCP server integration

  Observability                      Run traces

  Minimum Runs                       44+ evaluation cases

  Persona Linkage                    Synthetic persona IDs

  Latency                            Per-agent and total latency

  Accuracy / Quality                 Evaluation metrics

  Bottleneck Analysis                Agent-level trace analysis

  Architecture Comparison            Sequential vs broadcast vs
                                     blackboard vs hierarchical

  Technical Depth                    ADK + state + workflows + tools +
                                     memory

  Innovation                         Transparent multi-agent Yoga
                                     application

  Application                        Functional personalized Yoga
                                     assistant

  Individual Contribution            Responsibility + Git history +
                                     documentation
  ---------------------------------------------------------------------

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
