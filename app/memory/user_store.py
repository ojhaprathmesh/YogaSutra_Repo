"""
app/memory/user_store.py — Persistent User Profile & Practice History Store

Persists user context, physical limitations, injury history, and past sessions
to local JSON files in `data/personas/`. Zero cloud database cost, full durability.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("yogasutra.memory.user_store")

_PERSONAS_DIR = Path("data/personas")


@dataclass
class UserContext:
    """Persistent profile for a yoga practitioner."""
    user_id: str
    experience_level: str = "beginner"
    default_duration_minutes: int = 20
    goals: list[str] = field(default_factory=lambda: ["relaxation"])
    constraints: list[str] = field(default_factory=list)
    preferred_time_of_day: str = "morning"
    completed_sessions_count: int = 0
    past_session_ids: list[str] = field(default_factory=list)


class UserStore:
    """Manages persistent user context files."""

    def __init__(self, personas_dir: Path | None = None) -> None:
        self.personas_dir = personas_dir or _PERSONAS_DIR
        self.personas_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, user_id: str) -> Path:
        safe_id = "".join(c for c in user_id if c.isalnum() or c in ("-", "_"))
        return self.personas_dir / f"{safe_id}.json"

    def get_user(self, user_id: str) -> UserContext:
        """Fetch persistent user profile or create default if new."""
        path = self._get_path(user_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return UserContext(**data)
            except Exception as e:
                logger.warning(f"[UserStore] Failed reading {path.name}: {e}")

        # Default new user
        new_user = UserContext(user_id=user_id)
        self.save_user(new_user)
        return new_user

    def save_user(self, user: UserContext) -> None:
        """Save user profile to disk."""
        path = self._get_path(user.user_id)
        path.write_text(json.dumps(asdict(user), indent=2), encoding="utf-8")
        logger.info(f"[UserStore] Saved profile for user: {user.user_id}")

    def record_completed_session(self, user_id: str, session_id: str) -> None:
        """Increment practice count and append session ID."""
        u = self.get_user(user_id)
        u.completed_sessions_count += 1
        if session_id not in u.past_session_ids:
            u.past_session_ids.append(session_id)
        self.save_user(u)


# Global singleton user store
user_store = UserStore()
