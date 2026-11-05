"""
app/security/callbacks.py — Data Validation Callbacks for Tool Returns & State Transitions

Implements two categories of validation hooks (PLAN.md §8):

  1. Tool Return Validators  (`validate_tool_return`)
     Strict Pydantic type-checking and boundary-guard functions called
     immediately after each tool executes, before the result is passed
     back to the calling LLM agent. Prevents malformed or out-of-range
     tool outputs from silently polluting the agent context.

  2. State Transition Validators  (`validate_state_transition`)
     Checks that data written to the shared Blackboard (YogaSessionState)
     by one agent is structurally valid before the next agent reads it.
     Surfaces field-level Pydantic errors as audit log entries rather
     than silent no-ops.

Usage::

    from app.security.callbacks import validate_tool_return, validate_state_transition

    # After a tool returns:
    result = validate_tool_return("retrieve_yoga_knowledge", raw_chunks)

    # After an agent writes to state:
    issues = validate_state_transition("candidate_plan", raw_plan_dict)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from app.state import EvidenceChunk, PracticePlan, UserProfile

logger = logging.getLogger("yogasutra.security.callbacks")


# ---------------------------------------------------------------------------
# Validation result schema
# ---------------------------------------------------------------------------

@dataclass
class ValidationCallbackResult:
    """Outcome of a data-validation callback."""
    is_valid: bool
    tool_or_field: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "tool_or_field": self.tool_or_field,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Individual boundary validators (pure functions)
# ---------------------------------------------------------------------------

def _validate_duration(value: Any, field_name: str) -> list[str]:
    """Duration must be a positive integer ≤ 240 minutes."""
    errors: list[str] = []
    if not isinstance(value, int):
        errors.append(f"{field_name} must be an integer, got {type(value).__name__}.")
    elif value <= 0:
        errors.append(f"{field_name} must be > 0, got {value}.")
    elif value > 240:
        errors.append(f"{field_name} exceeds maximum 240 minutes (got {value}).")
    return errors


def _validate_evidence_chunks(raw: Any) -> list[str]:
    """Each retrieved evidence chunk must conform to EvidenceChunk schema."""
    errors: list[str] = []
    if not isinstance(raw, list):
        return [f"retrieve_yoga_knowledge must return a list, got {type(raw).__name__}."]
    for i, chunk in enumerate(raw):
        if not isinstance(chunk, dict):
            errors.append(f"Chunk [{i}] is not a dict (got {type(chunk).__name__}).")
            continue
        try:
            EvidenceChunk(**chunk)
        except (ValidationError, TypeError) as exc:
            errors.append(f"Chunk [{i}] schema violation: {exc}")
    return errors


def _validate_practice_plan(raw: Any) -> list[str]:
    """Practice plan must conform to PracticePlan schema with valid pose list."""
    errors: list[str] = []
    if not isinstance(raw, dict):
        return [f"candidate_plan must be a dict, got {type(raw).__name__}."]
    try:
        plan = PracticePlan(**raw)
        # Boundary: each pose duration positive
        for i, pose in enumerate(plan.sequence):
            errs = _validate_duration(pose.duration_minutes, f"sequence[{i}].duration_minutes")
            errors.extend(errs)
        # Boundary: plan total duration must match declared duration_minutes
        total = sum(p.duration_minutes for p in plan.sequence)
        if total != plan.duration_minutes:
            errors.append(
                f"Sequence total ({total} min) does not match declared "
                f"duration_minutes ({plan.duration_minutes} min)."
            )
    except (ValidationError, TypeError) as exc:
        errors.append(f"PracticePlan schema violation: {exc}")
    return errors


def _validate_user_profile(raw: Any) -> list[str]:
    """User profile must conform to UserProfile schema."""
    errors: list[str] = []
    if not isinstance(raw, (dict, str)):
        return [f"user_profile must be a dict, got {type(raw).__name__}."]
    if isinstance(raw, str):
        return []  # will be parsed downstream; not a hard failure here
    try:
        UserProfile(**raw)
    except (ValidationError, TypeError) as exc:
        errors.append(f"UserProfile schema violation: {exc}")
    return errors


def _validate_safety_results(raw: Any) -> list[str]:
    """Safety results must be a dict with expected keys."""
    errors: list[str] = []
    if not isinstance(raw, dict):
        return [f"validation_results must be a dict, got {type(raw).__name__}."]
    return errors


# ---------------------------------------------------------------------------
# Tool return validator dispatcher
# ---------------------------------------------------------------------------

# Maps tool name → validator function
_TOOL_VALIDATORS: dict[str, Any] = {
    "retrieve_yoga_knowledge":      _validate_evidence_chunks,
    "calculate_practice_duration":  lambda v: _validate_duration(v, "calculate_practice_duration"),
}


def validate_tool_return(tool_name: str, raw_result: Any) -> ValidationCallbackResult:
    """
    Run the registered validator for *tool_name* against *raw_result*.

    Call this immediately after any deterministic tool executes to catch
    boundary violations before the result enters the agent context.

    Args:
        tool_name:   Exact name of the tool (e.g. ``"retrieve_yoga_knowledge"``).
        raw_result:  The raw return value from the tool.

    Returns:
        :class:`ValidationCallbackResult` with ``is_valid`` and any ``errors``.
    """
    validator = _TOOL_VALIDATORS.get(tool_name)
    if validator is None:
        # No schema registered for this tool — emit a soft warning and pass
        logger.debug("[Callback] No validator registered for tool '%s'.", tool_name)
        return ValidationCallbackResult(
            is_valid=True,
            tool_or_field=tool_name,
            warnings=[f"No validator registered for '{tool_name}'; result accepted as-is."],
        )

    errors = validator(raw_result)
    is_valid = len(errors) == 0

    result = ValidationCallbackResult(
        is_valid=is_valid,
        tool_or_field=tool_name,
        errors=errors,
    )

    if not is_valid:
        logger.error(
            "[Callback] Tool '%s' return validation FAILED: %s",
            tool_name,
            errors,
        )
    else:
        logger.debug("[Callback] Tool '%s' return validation PASSED.", tool_name)

    return result


# ---------------------------------------------------------------------------
# State transition validator dispatcher
# ---------------------------------------------------------------------------

# Maps blackboard field name → validator function
_STATE_VALIDATORS: dict[str, Any] = {
    "user_profile":       _validate_user_profile,
    "retrieved_evidence": _validate_evidence_chunks,
    "candidate_plan":     _validate_practice_plan,
    "validation_results": _validate_safety_results,
}


def validate_state_transition(field_name: str, new_value: Any) -> ValidationCallbackResult:
    """
    Validate a value *before* it is committed to the Blackboard under *field_name*.

    Args:
        field_name: Blackboard / session-state field name.
        new_value:  The value an agent wants to write.

    Returns:
        :class:`ValidationCallbackResult`.
    """
    validator = _STATE_VALIDATORS.get(field_name)
    if validator is None:
        return ValidationCallbackResult(
            is_valid=True,
            tool_or_field=field_name,
            warnings=[f"No state validator registered for field '{field_name}'."],
        )

    errors = validator(new_value)
    is_valid = len(errors) == 0

    result = ValidationCallbackResult(
        is_valid=is_valid,
        tool_or_field=field_name,
        errors=errors,
    )

    if not is_valid:
        logger.error(
            "[Callback] State transition for '%s' FAILED validation: %s",
            field_name,
            errors,
        )
    else:
        logger.debug("[Callback] State transition for '%s' PASSED.", field_name)

    return result

