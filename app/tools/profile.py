"""
app/tools/profile.py — User Profile Store Tool

Phase 1: in-memory dict-based profile storage (no database).
Profiles are keyed by session_id and persist for the lifetime of the process.

Tool contract:
    get_user_profile(session_id: str) -> dict | None
    store_user_profile(session_id: str, profile: dict) -> None
    get_user_history(user_id: str) -> dict
"""
from __future__ import annotations

from typing import Any

from app.memory.longitudinal import build_longitudinal_context

# In-memory store: session_id -> profile dict
_PROFILE_STORE: dict[str, dict[str, Any]] = {}


def get_user_profile(session_id: str) -> dict[str, Any] | None:
    """
    Retrieve the stored user profile for the given session.

    Args:
        session_id: Unique session identifier.

    Returns:
        Profile dict (conforming to UserProfile schema) if found, else None.
    """
    return _PROFILE_STORE.get(session_id)


def store_user_profile(session_id: str, profile: dict[str, Any]) -> None:
    """
    Persist a user profile dict for the given session.

    Args:
        session_id: Unique session identifier.
        profile:    Dict conforming to the UserProfile Pydantic schema.
    """
    _PROFILE_STORE[session_id] = profile


def get_user_history(user_id: str) -> dict[str, Any]:
    """
    Retrieve a user's longitudinal practice history, active contraindications,
    injury history, streak, and suggested experience level override.

    Args:
        user_id: Unique practitioner or persona identifier (e.g. persona_id).

    Returns:
        Summary dict of the user's longitudinal memory context.
    """
    return build_longitudinal_context(user_id)
