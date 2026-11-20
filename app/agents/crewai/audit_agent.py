"""
app/agents/crewai/audit_agent.py — CrewAI Independent Audit & Critic Agent

Implements Requirement 7 (Dual-Framework Interoperability: Google ADK + CrewAI):
- Defines a specialist Yoga Safety & Pedagogical Auditor using CrewAI's
  `Agent`, `Task`, and `Crew` abstractions.
- Automatically detects whether the `crewai` package is installed in the
  active Python runtime. When `crewai` is unavailable (e.g. Python 3.14) or
  when running in deterministic/offline mode (`use_fallback=True`), executes
  through faithful in-process CrewAI-compatible abstractions (`FallbackAgent`,
  `FallbackTask`, `FallbackCrew`) backed by YogaSutra's deterministic safety
  and pedagogical rules.
- Exposes an A2A handler (`handle_a2a_message`) so Google ADK agents can
  delegate cross-framework plan audits via `A2ABridge`.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.a2a.bridge import A2ABridge, get_a2a_bridge
from app.a2a.protocol import A2AAgentCard, A2AMessage, FrameworkType, Performative
from app.tools.safety import validate_safety_and_constraints

logger = logging.getLogger("yogasutra.crewai.audit_agent")

# ---------------------------------------------------------------------------
# Optional CrewAI Runtime Import with Resilient In-Process Fallback
# ---------------------------------------------------------------------------

try:
    from crewai import Agent as _CrewAIAgent
    from crewai import Crew as _CrewAICrew
    from crewai import Process as _CrewAIProcess
    from crewai import Task as _CrewAITask

    CREWAI_AVAILABLE = True
except ImportError:  # pragma: no cover - expected on Python 3.14 without crewai
    _CrewAIAgent = None  # type: ignore[assignment]
    _CrewAICrew = None  # type: ignore[assignment]
    _CrewAIProcess = None  # type: ignore[assignment]
    _CrewAITask = None  # type: ignore[assignment]
    CREWAI_AVAILABLE = False


class FallbackProcess(StrEnum):
    """Mirrors `crewai.Process` for in-process fallback execution."""

    sequential = "sequential"
    hierarchical = "hierarchical"


@dataclass
class FallbackAgent:
    """In-process structural equivalent of `crewai.Agent`."""

    role: str
    goal: str
    backstory: str
    verbose: bool = False
    allow_delegation: bool = False
    tools: list[Callable[..., Any]] = field(default_factory=list)


@dataclass
class FallbackTask:
    """In-process structural equivalent of `crewai.Task`."""

    description: str
    expected_output: str
    agent: FallbackAgent
    executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None


@dataclass
class FallbackCrewOutput:
    """In-process structural equivalent of `crewai.CrewOutput`."""

    raw: str
    json_dict: dict[str, Any]
    tasks_output: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FallbackCrew:
    """In-process structural equivalent of `crewai.Crew`."""

    agents: list[FallbackAgent]
    tasks: list[FallbackTask]
    process: FallbackProcess = FallbackProcess.sequential
    verbose: bool = False

    def kickoff(self, inputs: dict[str, Any] | None = None) -> FallbackCrewOutput:
        """Execute each Crew task sequentially and return a structured CrewOutput."""
        context = dict(inputs or {})
        tasks_output: list[dict[str, Any]] = []
        last_result: dict[str, Any] = {}

        for task in self.tasks:
            if task.executor is not None:
                last_result = task.executor(context)
                context["previous_task_output"] = last_result
            else:
                last_result = {
                    "agent_role": task.agent.role,
                    "status": "completed",
                }
            tasks_output.append(last_result)

        return FallbackCrewOutput(
            raw=json.dumps(last_result, ensure_ascii=False),
            json_dict=last_result,
            tasks_output=tasks_output,
        )


# ---------------------------------------------------------------------------
# Deterministic Pedagogical & Safety Audit Logic
# ---------------------------------------------------------------------------

_WARMUP_KEYWORDS = (
    "warm",
    "cat-cow",
    "cat cow",
    "marjaryasana",
    "bitilasana",
    "sukhasana",
    "easy pose",
    "child",
    "balasana",
    "tadasana",
    "mountain",
    "breath",
    "pranayama",
    "gentle",
    "neck roll",
    "shoulder",
)

_COOLDOWN_KEYWORDS = (
    "savasana",
    "corpse",
    "cool",
    "relaxation",
    "rest",
    "balasana",
    "child",
    "supine",
    "supta",
    "meditation",
    "pranayama",
)


def _coerce_dict(val: Any) -> dict[str, Any]:
    """Best-effort conversion of a dict or JSON string into a dict."""
    if isinstance(val, dict):
        return val
    if hasattr(val, "model_dump"):
        return val.model_dump()
    if isinstance(val, str):
        cleaned = val.strip()
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
    return {}


def _run_deterministic_yoga_audit(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Independent CrewAI specialist audit logic.

    Combines deterministic safety validation (`validate_safety_and_constraints`)
    with pedagogical quality checks (warm-up, cool-down/Savasana, constraint
    modification coverage, and sequence completeness).
    """
    candidate_plan = _coerce_dict(inputs.get("candidate_plan"))
    user_profile = _coerce_dict(inputs.get("user_profile"))
    adk_validation = _coerce_dict(inputs.get("validation_results"))

    # 1. Run independent deterministic safety & constraint validation
    safety_report = validate_safety_and_constraints(candidate_plan, user_profile)
    violations: list[dict[str, Any]] = list(safety_report.get("violations", []))

    sequence = candidate_plan.get("sequence") or []
    findings: list[str] = []
    recommendations: list[str] = []

    # 2. Pedagogical structure checks
    has_warmup = False
    has_cooldown = False
    has_modifications = False

    if sequence:
        first_pose_text = (
            f"{sequence[0].get('name', '')} {sequence[0].get('sanskrit_name', '')} "
            f"{sequence[0].get('instructions', '')}"
        ).lower()
        last_pose_text = (
            f"{sequence[-1].get('name', '')} {sequence[-1].get('sanskrit_name', '')} "
            f"{sequence[-1].get('instructions', '')}"
        ).lower()

        has_warmup = any(k in first_pose_text for k in _WARMUP_KEYWORDS)
        has_cooldown = any(k in last_pose_text for k in _COOLDOWN_KEYWORDS)
        has_modifications = any(
            bool(str(p.get("modifications", "")).strip()) for p in sequence if isinstance(p, dict)
        )
    else:
        findings.append("Candidate plan contains an empty pose sequence.")
        recommendations.append("Provide at least 3 sequenced poses including warm-up and cool-down.")

    if sequence and not has_warmup:
        findings.append("Sequence lacks a gentle opening warm-up or centering pose.")
        recommendations.append(
            "Begin with a gentle centering or warm-up pose (e.g., Sukhasana or Cat-Cow)."
        )

    if sequence and not has_cooldown:
        findings.append("Sequence lacks a restorative cool-down or Savasana closing.")
        recommendations.append(
            "Conclude the practice with Savasana (Corpse Pose) or Balasana for nervous system integration."
        )

    raw_constraints = user_profile.get("constraints") or []
    if isinstance(raw_constraints, str):
        raw_constraints = [c.strip() for c in raw_constraints.split(",") if c.strip()]
    if raw_constraints and sequence and not has_modifications:
        findings.append(
            f"User has explicit constraints ({', '.join(raw_constraints)}) but poses lack explicit modification cues."
        )
        recommendations.append(
            "Add prop or alignment modifications tailored to each user constraint."
        )

    for v in violations:
        findings.append(f"[{v.get('severity', 'MEDIUM')}] {v.get('type')}: {v.get('recommendation')}")
        if v.get("recommendation"):
            recommendations.append(v["recommendation"])

    # 3. Compute audit score (1.0 - 10.0) & verdict
    high_violations = [v for v in violations if v.get("severity") == "HIGH"]
    med_violations = [v for v in violations if v.get("severity") == "MEDIUM"]

    score = 10.0
    score -= 3.5 * len(high_violations)
    score -= 1.5 * len(med_violations)
    if not sequence:
        score -= 5.0
    if sequence and not has_warmup:
        score -= 0.75
    if sequence and not has_cooldown:
        score -= 0.75
    if raw_constraints and sequence and not has_modifications:
        score -= 0.75
    score = round(max(1.0, min(10.0, score)), 2)

    if high_violations or not sequence:
        verdict = "REJECT"
        decision = "REVISION_REQUIRED"
    elif med_violations or score < 7.5:
        verdict = "REVISE"
        decision = "REVISION_REQUIRED"
    else:
        verdict = "APPROVE"
        decision = "PASS"

    # Cross-check agreement with ADK SafetyAgent if provided
    adk_status = adk_validation.get("status")
    cross_framework_agreement = (
        True if not adk_status else ((adk_status == "PASS") == (verdict == "APPROVE"))
    )

    return {
        "auditor": "CrewAIAuditAgent",
        "framework": FrameworkType.CREWAI.value,
        "verdict": verdict,
        "decision": decision,
        "score": score,
        "is_safe": safety_report.get("is_safe", True) and bool(sequence),
        "violations": violations,
        "pedagogical_checks": {
            "has_warmup": has_warmup,
            "has_cooldown": has_cooldown,
            "has_constraint_modifications": has_modifications if raw_constraints else True,
            "pose_count": len(sequence),
        },
        "duration_check": safety_report.get("duration_check", {}),
        "level_check": safety_report.get("level_check", {}),
        "findings": findings,
        "recommendations": recommendations,
        "cross_framework_agreement": cross_framework_agreement,
        "summary": (
            f"CrewAI Audit {verdict} (score={score}/10.0): "
            f"{len(violations)} safety/constraint violations, {len(findings)} total findings."
        ),
    }


# ---------------------------------------------------------------------------
# CrewAI Specialist Agent Wrapper
# ---------------------------------------------------------------------------


class CrewAIAuditAgent:
    """
    CrewAI-based Yoga Safety & Pedagogical Audit Specialist.

    Builds a CrewAI `Agent`, `Task`, and `Crew` to independently audit
    practice plans generated by Google ADK agents. Seamlessly uses the
    in-process CrewAI-compatible bridge when `crewai` is not installed
    (e.g., Python 3.14) or when `use_fallback=True`.
    """

    AGENT_NAME = "CrewAIAuditAgent"

    def __init__(
        self,
        agent_name: str = AGENT_NAME,
        use_fallback: bool | None = None,
        llm: Any | None = None,
    ) -> None:
        self.agent_name = agent_name
        # Default to fallback if crewai is not installed or no explicit LLM is configured
        self.use_fallback = (not CREWAI_AVAILABLE) if use_fallback is None else use_fallback
        self.llm = llm
        self._crew_spec = self._build_crew_abstractions()

    @property
    def runtime_mode(self) -> str:
        """Report whether native CrewAI or the in-process bridge is active."""
        if CREWAI_AVAILABLE and not self.use_fallback:
            return "crewai_native"
        return "crewai_in_process_bridge"

    def _build_crew_abstractions(self) -> dict[str, Any]:
        """Construct CrewAI Agent, Task, and Crew objects (native or fallback)."""
        role = "Independent Yoga Safety & Pedagogical Auditor"
        goal = (
            "Independently verify candidate yoga practice plans for contraindications, "
            "duration accuracy, experience-level suitability, and pedagogical pacing."
        )
        backstory = (
            "You are a senior yoga therapist and independent quality auditor operating "
            "inside a CrewAI crew. You receive practice plans from Google ADK agents "
            "via the A2A protocol and return structured audit verdicts."
        )
        task_description = (
            "Audit the provided candidate_plan against user_profile and optional "
            "ADK validation_results. Check contraindications, duration compliance, "
            "warm-up/Savasana pacing, and modification cues."
        )
        expected_output = (
            "JSON object with keys: verdict (APPROVE|REVISE|REJECT), decision "
            "(PASS|REVISION_REQUIRED), score (1-10), is_safe, violations, findings, "
            "recommendations, and summary."
        )

        if CREWAI_AVAILABLE and not self.use_fallback and _CrewAIAgent is not None:
            agent_kwargs: dict[str, Any] = {
                "role": role,
                "goal": goal,
                "backstory": backstory,
                "verbose": False,
                "allow_delegation": False,
            }
            if self.llm is not None:
                agent_kwargs["llm"] = self.llm
            agent = _CrewAIAgent(**agent_kwargs)
            task = _CrewAITask(
                description=task_description,
                expected_output=expected_output,
                agent=agent,
            )
            crew = _CrewAICrew(
                agents=[agent],
                tasks=[task],
                process=_CrewAIProcess.sequential,
                verbose=False,
            )
            return {"agent": agent, "task": task, "crew": crew}

        agent = FallbackAgent(
            role=role,
            goal=goal,
            backstory=backstory,
            verbose=False,
            allow_delegation=False,
            tools=[validate_safety_and_constraints],
        )
        task = FallbackTask(
            description=task_description,
            expected_output=expected_output,
            agent=agent,
            executor=_run_deterministic_yoga_audit,
        )
        crew = FallbackCrew(
            agents=[agent],
            tasks=[task],
            process=FallbackProcess.sequential,
            verbose=False,
        )
        return {"agent": agent, "task": task, "crew": crew}

    @property
    def agent(self) -> Any:
        return self._crew_spec["agent"]

    @property
    def task(self) -> Any:
        return self._crew_spec["task"]

    @property
    def crew(self) -> Any:
        return self._crew_spec["crew"]

    def get_agent_card(self) -> A2AAgentCard:
        """Return the A2A capability advertisement for this CrewAI specialist."""
        return A2AAgentCard(
            agent_name=self.agent_name,
            framework=FrameworkType.CREWAI,
            role="Independent Yoga Safety & Pedagogical Auditor",
            description=(
                "CrewAI specialist crew performing independent safety, contraindication, "
                "and pedagogical audits on ADK-generated yoga practice plans."
            ),
            capabilities=[
                "safety_audit",
                "contraindication_check",
                "pedagogical_critique",
                "cross_framework_verification",
            ],
            supported_performatives=[
                Performative.AUDIT,
                Performative.CRITIQUE,
                Performative.REQUEST,
                Performative.QUERY,
            ],
        )

    def audit_plan(
        self,
        candidate_plan: dict[str, Any] | str,
        user_profile: dict[str, Any] | str,
        validation_results: dict[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        """Execute the CrewAI audit crew and return the structured audit dict."""
        inputs = {
            "candidate_plan": candidate_plan,
            "user_profile": user_profile,
            "validation_results": validation_results or {},
        }

        if CREWAI_AVAILABLE and not self.use_fallback:
            try:
                crew_output = self.crew.kickoff(inputs=inputs)
                raw_text = getattr(crew_output, "raw", str(crew_output))
                parsed = _coerce_dict(raw_text)
                if parsed and "verdict" in parsed:
                    parsed["runtime_mode"] = self.runtime_mode
                    parsed["framework"] = FrameworkType.CREWAI.value
                    parsed["auditor"] = self.agent_name
                    return parsed
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[CrewAIAuditAgent] Native CrewAI kickoff failed (%s); falling back to in-process bridge.",
                    exc,
                )

        # Deterministic in-process Crew kickoff
        fallback_crew = (
            self.crew
            if isinstance(self.crew, FallbackCrew)
            else FallbackCrew(
                agents=[
                    FallbackAgent(
                        role="Independent Yoga Safety & Pedagogical Auditor",
                        goal="Audit yoga practice plan",
                        backstory="CrewAI fallback auditor",
                        tools=[validate_safety_and_constraints],
                    )
                ],
                tasks=[
                    FallbackTask(
                        description="Audit candidate plan",
                        expected_output="Structured audit JSON",
                        agent=FallbackAgent(
                            role="Independent Yoga Safety & Pedagogical Auditor",
                            goal="Audit yoga practice plan",
                            backstory="CrewAI fallback auditor",
                        ),
                        executor=_run_deterministic_yoga_audit,
                    )
                ],
            )
        )
        output = fallback_crew.kickoff(inputs=inputs)
        result = dict(output.json_dict)
        result["auditor"] = self.agent_name
        result["runtime_mode"] = self.runtime_mode
        result["crew_role"] = getattr(self.agent, "role", "Independent Yoga Safety & Pedagogical Auditor")
        return result

    def handle_a2a_message(self, message: A2AMessage) -> A2AMessage:
        """
        A2A protocol message handler for cross-framework requests from ADK.

        Accepts AUDIT / CRITIQUE / REQUEST / QUERY performatives and returns
        a correlated CRITIQUE or INFORM reply envelope.
        """
        payload = message.payload or {}
        audit_result = self.audit_plan(
            candidate_plan=payload.get("candidate_plan", {}),
            user_profile=payload.get("user_profile", {}),
            validation_results=payload.get("validation_results"),
        )

        reply_performative = (
            Performative.CRITIQUE
            if message.performative in (Performative.AUDIT, Performative.CRITIQUE, "AUDIT", "CRITIQUE")
            else Performative.INFORM
        )

        return message.create_reply(
            performative=reply_performative,
            payload=audit_result,
            sender_agent=self.agent_name,
            sender_framework=FrameworkType.CREWAI,
            metadata={
                "runtime_mode": self.runtime_mode,
                "crewai_available": CREWAI_AVAILABLE,
            },
        )


# Semantic alias matching brief specification
CrewAICriticAgent = CrewAIAuditAgent


def register_crewai_audit_agent(
    bridge: A2ABridge | None = None,
    agent_name: str = CrewAIAuditAgent.AGENT_NAME,
    use_fallback: bool | None = None,
) -> CrewAIAuditAgent:
    """Instantiate and register a CrewAIAuditAgent on the given (or global) A2ABridge."""
    target_bridge = bridge if bridge is not None else get_a2a_bridge()
    specialist = CrewAIAuditAgent(agent_name=agent_name, use_fallback=use_fallback)
    target_bridge.register_agent(specialist.get_agent_card(), specialist.handle_a2a_message)
    return specialist
