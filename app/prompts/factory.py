"""
app/prompts/factory.py — Prompt Architecture & Content Variance Factory

Satisfies PLAN.md §8 requirement:
  "Try to use multiple prompts with variances in Prompts. Variances can be
   induced from changes in architecture (REACT PROMPT VS FEW SHOT) OR
   CONTENT or persona or variances in content."

Three architectural prompt strategies are available per agent:

  REACT     — Explicit Thought → Action → Observation cycle forcing the
               model to reason over tool calls step-by-step before
               committing to output. Highest interpretability.

  FEW_SHOT  — In-context exemplars of successful input→output pairs
               grounded in the yoga domain. Improves structured output
               reliability and reduces hallucinated pose names.

  COT       — Chain-of-Thought instruction that prompts the model to
               narrate its reasoning steps inside <thinking> tags before
               emitting the final JSON. Balances latency vs transparency.

Five persona content variances map to PLAN.md §26 canonical personas:

  P01 — Beginner with physical constraints (back pain, pregnancy)
  P02 — Athletic intermediate (strength / endurance focus)
  P03 — Geriatric / senior (osteoarthritis, reduced ROM)
  P04 — Mental wellness / anxiety (slow breath, Yoga Nidra emphasis)
  P05 — Advanced Ashtanga practitioner (vinyasa, arm balances)

Usage::

    from app.prompts.factory import PromptStrategy, build_agent_instruction

    instruction = build_agent_instruction(
        agent="planner",
        strategy=PromptStrategy.REACT,
        persona_id="P02",
    )

    # Then create an ADK LlmAgent with this instruction:
    agent = LlmAgent(name="PlannerAgent", model=model, instruction=instruction, ...)
"""
from __future__ import annotations

import logging
from enum import Enum
from string import Template

logger = logging.getLogger("yogasutra.prompts.factory")


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------

class PromptStrategy(str, Enum):
    REACT     = "react"
    FEW_SHOT  = "few_shot"
    COT       = "cot"
    DEFAULT   = "default"   # falls back to the original production prompt


# ---------------------------------------------------------------------------
# Persona context snippets
# ---------------------------------------------------------------------------

PERSONA_CONTEXTS: dict[str, str] = {
    "P00": "",   # generic / no persona
    "P01": (
        "The user is a **beginner** with reported physical constraints such as lower back "
        "pain or pregnancy. Prioritise gentle, restorative poses. Avoid deep spinal twists, "
        "inversions, and poses requiring significant core loading. Always offer props."
    ),
    "P02": (
        "The user is an **intermediate** practitioner with an athletic background focused on "
        "strength and endurance. Include standing balances, Warriors, and Chaturanga sequences. "
        "Mention breath-to-movement ratios and peak-pose progressions."
    ),
    "P03": (
        "The user is a **senior / geriatric** practitioner managing osteoarthritis and reduced "
        "range of motion. All poses must be chair-supported or wall-supported alternatives. "
        "Duration per pose should be short (1–2 min). Prioritise joint mobility and circulation."
    ),
    "P04": (
        "The user seeks **mental wellness and anxiety relief**. Prioritise Yoga Nidra, slow "
        "diaphragmatic breathing (4-7-8 or box breath), Legs-Up-the-Wall, and long Savasana. "
        "Tone must be calming, non-directive, and emotionally supportive."
    ),
    "P05": (
        "The user is an **advanced Ashtanga practitioner** comfortable with vinyasa, arm "
        "balances (Bakasana, Titibasana), and deep backbends. Include Surya Namaskara A/B "
        "as warm-up. Reference Mysore-style self-practice cadence."
    ),
}


# ---------------------------------------------------------------------------
# Base agent instructions (DEFAULT / production versions)
# These mirror the actual agent instructions and serve as the fallback.
# ---------------------------------------------------------------------------

_DEFAULT_PLANNER = """\
You are the Practice Planner Agent for YogaSutra, an expert AI yoga instructor.

You will receive:
- A user profile (from session state: user_profile)
- Retrieved knowledge chunks (from session state: retrieved_evidence)
- The original user request

Your job is to generate a structured, personalised yoga practice plan.

IMPORTANT RULES:
1. The total duration of all poses MUST match the user's available_duration_minutes exactly.
2. The sequence must start gently (breathing/warm-up) and end with Savasana.
3. Reference specific chunk_ids from the retrieved evidence to ground each pose.
4. Match the user's experience level.
5. You may call `calculate_practice_duration` to verify sequence totals.

Return ONLY a single JSON object with the PracticePlan schema. No markdown, no extra text.\
"""

_DEFAULT_CRITIC = """\
You are the Critic & Reflection Agent for YogaSutra, an expert yoga pedagogue.

Review the candidate plan against the user profile and safety results.
Return ONLY a JSON object with keys: decision, score, critique, recommended_modifications.\
"""

_DEFAULT_KNOWLEDGE = """\
You are the Knowledge Retrieval Agent for YogaSutra.

Call `retrieve_yoga_knowledge` with targeted queries to fetch the 4–6 most
relevant evidence chunks. Return ONLY a JSON array of chunk objects.\
"""

_DEFAULT_PROFILE = """\
You are the Profile Extraction Agent for YogaSutra.

Extract: experience_level, available_duration_minutes, preferred_time, goal,
constraints, preferences. Return ONLY valid JSON matching the UserProfile schema.\
"""


# ---------------------------------------------------------------------------
# ReAct prompt templates
# ---------------------------------------------------------------------------
# Template variables: $persona_context, $base_task

_REACT_TEMPLATE = Template("""\
You are a YogaSutra specialist agent operating in **ReAct mode** (Reason + Act).

$persona_context

═══ TASK ═══
$base_task

═══ REACT CYCLE ═══
Before writing your final answer, work through the following cycle explicitly:

Thought: What do I know so far? What information is missing? What tool should I call next?
Action: Call the appropriate tool with specific arguments.
Observation: What did the tool return? Is it valid? Does it satisfy the task?
... (repeat Thought / Action / Observation as needed) ...
Thought: I now have all the information needed. Here is my final reasoning.
Final Answer: <your structured JSON output>

IMPORTANT:
- Use at least ONE Thought/Action/Observation cycle if tools are available.
- Do NOT skip the ReAct cycle and go straight to Final Answer.
- Final Answer must be ONLY the JSON output specified in the task; no extra prose.\
""")


# ---------------------------------------------------------------------------
# Few-Shot prompt templates
# ---------------------------------------------------------------------------

_FEW_SHOT_PLANNER_EXAMPLES = """\
=== EXAMPLE 1 ===
Input: Beginner, 20 min morning, relaxation goal.
Output:
{
  "goal": "Morning relaxation",
  "duration_minutes": 20,
  "intensity": "gentle",
  "sequence": [
    {"name": "Child's Pose", "sanskrit_name": "Balasana", "duration_minutes": 3,
     "instructions": "Kneel, sink hips toward heels, arms extended forward.",
     "modifications": "Place a blanket under hips if knees are sensitive.",
     "evidence_chunk_ids": ["doc_002_004"]},
    {"name": "Cat-Cow", "sanskrit_name": "Marjaryasana-Bitilasana", "duration_minutes": 3, ...},
    {"name": "Seated Forward Fold", "sanskrit_name": "Paschimottanasana", "duration_minutes": 4, ...},
    {"name": "Supine Twist", "sanskrit_name": "Supta Matsyendrasana", "duration_minutes": 5, ...},
    {"name": "Savasana", "sanskrit_name": "Savasana", "duration_minutes": 5, ...}
  ],
  "rationale": "Gentle spinal mobilisation followed by hip openers for morning stiffness.",
  "evidence_sources": ["doc_002_004"],
  "expected_outcome": "Calm nervous system, reduced morning stiffness."
}

=== EXAMPLE 2 ===
Input: Intermediate, 30 min, strength goal, no constraints.
Output: (a 30-minute intermediate strength sequence using Warriors, Plank, Boat Pose)

"""

_FEW_SHOT_TEMPLATE = Template("""\
You are a YogaSutra specialist agent using **Few-Shot In-Context Learning**.

$persona_context

═══ EXEMPLARS ═══
Study these reference examples before generating your response:

$examples

═══ TASK ═══
$base_task

Now produce your output in exactly the same JSON structure as shown in the exemplars.
Return ONLY the JSON object — no extra text, no markdown fences.\
""")


# ---------------------------------------------------------------------------
# Chain-of-Thought prompt template
# ---------------------------------------------------------------------------

_COT_TEMPLATE = Template("""\
You are a YogaSutra specialist agent using **Chain-of-Thought reasoning**.

$persona_context

═══ TASK ═══
$base_task

═══ INSTRUCTIONS ═══
Before producing the final JSON output, write your step-by-step reasoning
inside <thinking> tags. Structure your reasoning as:

<thinking>
Step 1: What is the user's primary goal and constraints?
Step 2: Which knowledge chunks are most relevant to those constraints?
Step 3: How should I structure the pose sequence to meet duration, safety, and goal?
Step 4: Does my sequence cover ALL stated concerns? If not, adjust.
Step 5: Verify: does the total duration add up correctly?
</thinking>

Then output ONLY the final JSON object after the closing </thinking> tag.
Do NOT include any other text outside the <thinking> block and the JSON.\
""")


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_agent_instruction(
    agent: str,
    strategy: PromptStrategy = PromptStrategy.DEFAULT,
    persona_id: str = "P00",
) -> str:
    """
    Build a fully-parameterised agent instruction string.

    Args:
        agent:      Agent key — one of ``"planner"``, ``"critic"``,
                    ``"knowledge"``, ``"profile"``.
        strategy:   Prompt architecture variant (ReAct / Few-Shot / CoT / Default).
        persona_id: Canonical persona identifier (P00–P05). Used to inject
                    persona-specific context into the prompt.

    Returns:
        A string suitable for use as the ``instruction`` parameter of an
        ADK :class:`LlmAgent`.
    """
    # Resolve base task description
    base_map: dict[str, str] = {
        "planner":   _DEFAULT_PLANNER,
        "critic":    _DEFAULT_CRITIC,
        "knowledge": _DEFAULT_KNOWLEDGE,
        "profile":   _DEFAULT_PROFILE,
    }
    base_task = base_map.get(agent.lower(), _DEFAULT_PLANNER)
    persona_ctx = PERSONA_CONTEXTS.get(persona_id, "")

    # Optionally prepend persona context header
    persona_block = (
        f"═══ PERSONA CONTEXT ═══\n{persona_ctx}\n"
        if persona_ctx else ""
    )

    if strategy == PromptStrategy.DEFAULT:
        result = (persona_block + "\n" + base_task).strip()

    elif strategy == PromptStrategy.REACT:
        result = _REACT_TEMPLATE.substitute(
            persona_context=persona_block,
            base_task=base_task,
        )

    elif strategy == PromptStrategy.FEW_SHOT:
        examples = _FEW_SHOT_PLANNER_EXAMPLES if agent.lower() == "planner" else ""
        result = _FEW_SHOT_TEMPLATE.substitute(
            persona_context=persona_block,
            examples=examples,
            base_task=base_task,
        )

    elif strategy == PromptStrategy.COT:
        result = _COT_TEMPLATE.substitute(
            persona_context=persona_block,
            base_task=base_task,
        )

    else:
        result = base_task

    logger.debug(
        "[PromptFactory] Built instruction: agent=%s strategy=%s persona=%s length=%d",
        agent, strategy.value, persona_id, len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Convenience: build a strategy-variant LlmAgent
# ---------------------------------------------------------------------------

def build_variant_agents(
    persona_id: str = "P00",
) -> dict[str, str]:
    """
    Return a dict mapping strategy name → planner instruction string for
    all three architectural variants. Useful for benchmark / evaluation runs.

    Example::

        variants = build_variant_agents(persona_id="P02")
        for strategy, instruction in variants.items():
            agent = LlmAgent(name="PlannerAgent", instruction=instruction, ...)
    """
    return {
        s.value: build_agent_instruction("planner", s, persona_id)
        for s in [PromptStrategy.REACT, PromptStrategy.FEW_SHOT, PromptStrategy.COT]
    }

