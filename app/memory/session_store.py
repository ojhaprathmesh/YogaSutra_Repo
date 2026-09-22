"""
app/memory/session_store.py — Persistent Session Store

Maintains active and historical YogaSutra session state on disk or in memory.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("yogasutra.memory.session_store")

_SESSIONS_DIR = Path("data/sessions")


class SessionStore:
    """Manages session states across multi-turn interactions."""

    def __init__(self, sessions_dir: Path | None = None) -> None:
        self.sessions_dir = sessions_dir or _SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, dict[str, Any]] = {}

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Fetch session state by ID."""
        if session_id in self._memory_cache:
            return self._memory_cache[session_id]

        path = self.sessions_dir / f"{session_id}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._memory_cache[session_id] = data
                return data
            except Exception as e:
                logger.warning(f"[SessionStore] Failed reading session {session_id}: {e}")

        return {}

    def save_session(self, session_id: str, state: dict[str, Any]) -> None:
        """Save session state to memory and disk."""
        self._memory_cache[session_id] = state
        path = self.sessions_dir / f"{session_id}.json"
        try:
            path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[SessionStore] Could not persist session {session_id} to disk: {e}")


# Global singleton session store
session_store = SessionStore()
