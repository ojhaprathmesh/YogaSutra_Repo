"""
tests/test_memory.py — Unit tests for Persistent UserStore and SessionStore
"""
from app.memory.user_store import UserStore
from app.memory.session_store import SessionStore


class TestUserStore:
    def test_create_and_get_user(self, tmp_path):
        store = UserStore(personas_dir=tmp_path)
        u = store.get_user("test_user_42")
        assert u.user_id == "test_user_42"
        assert u.experience_level == "beginner"

        # Update constraints and save
        u.constraints.append("lower back stiffness")
        store.save_user(u)

        # Reload from disk
        u_loaded = store.get_user("test_user_42")
        assert "lower back stiffness" in u_loaded.constraints

    def test_record_completed_session(self, tmp_path):
        store = UserStore(personas_dir=tmp_path)
        store.record_completed_session("user_test", "sess_001")
        u = store.get_user("user_test")
        assert u.completed_sessions_count == 1
        assert "sess_001" in u.past_session_ids


class TestSessionStore:
    def test_session_save_and_retrieve(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path)
        state = {"user_input": "morning yoga", "plan": ["Sukhasana"]}
        store.save_session("session_123", state)

        loaded = store.get_session("session_123")
        assert loaded["user_input"] == "morning yoga"
        assert loaded["plan"] == ["Sukhasana"]
