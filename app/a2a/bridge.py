"""
app/a2a/bridge.py — A2A Dispatcher & Cross-Framework Bridge (ADK <-> CrewAI)

Provides an in-process Agent-to-Agent (A2A) message bus and dispatcher that:
1. Registers agents from heterogeneous frameworks (Google ADK, CrewAI, etc.)
   with discoverable A2AAgentCards.
2. Routes validated A2AMessage envelopes between agents synchronously or
   asynchronously.
3. Maintains a bounded audit trail of cross-framework conversations.
4. Persists exchange results directly into the shared YogaSutra Blackboard
   and/or ADK session state dictionary.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from typing import Any

from app.a2a.protocol import A2AAgentCard, A2AMessage, FrameworkType, Performative
from app.workflows.blackboard import Blackboard

logger = logging.getLogger("yogasutra.a2a.bridge")

A2AHandler = Callable[[A2AMessage], Any]


class A2ABridge:
    """
    Central A2A Dispatcher connecting Google ADK and CrewAI agents.

    Supports agent registration, capability discovery, synchronous and
    asynchronous envelope dispatch, conversation history tracking, and
    automatic synchronization with the shared Blackboard.
    """

    def __init__(self, max_history: int = 500) -> None:
        self._cards: dict[str, A2AAgentCard] = {}
        self._handlers: dict[str, A2AHandler] = {}
        self._history: list[A2AMessage] = []
        self._max_history = max_history

    # -----------------------------------------------------------------------
    # Agent Registration & Discovery
    # -----------------------------------------------------------------------

    def register_agent(self, card: A2AAgentCard, handler: A2AHandler) -> None:
        """Register an agent capability card and its message handler."""
        self._cards[card.agent_name] = card
        self._handlers[card.agent_name] = handler
        logger.info(
            "[A2ABridge] Registered agent '%s' (framework=%s)",
            card.agent_name,
            card.framework,
        )

    def unregister_agent(self, agent_name: str) -> None:
        """Remove an agent from the bridge registry."""
        self._cards.pop(agent_name, None)
        self._handlers.pop(agent_name, None)

    def has_agent(self, agent_name: str) -> bool:
        """Check whether an agent is registered on the bridge."""
        return agent_name in self._handlers

    def get_agent_card(self, agent_name: str) -> A2AAgentCard | None:
        """Return the capability card for a registered agent."""
        return self._cards.get(agent_name)

    def list_agents(self, framework: FrameworkType | str | None = None) -> list[A2AAgentCard]:
        """List registered agent cards, optionally filtered by framework."""
        cards = list(self._cards.values())
        if framework is not None:
            target = framework.value if isinstance(framework, FrameworkType) else str(framework)
            cards = [c for c in cards if c.framework == target]
        return cards

    # -----------------------------------------------------------------------
    # History & Recording
    # -----------------------------------------------------------------------

    def _record(self, message: A2AMessage) -> None:
        self._history.append(message)
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def get_history(self, limit: int = 50) -> list[A2AMessage]:
        """Return the most recent A2A messages routed through the bridge."""
        return list(self._history[-limit:])

    def get_conversation(self, conversation_id: str) -> list[A2AMessage]:
        """Return all A2A messages belonging to a specific conversation_id."""
        return [m for m in self._history if m.conversation_id == conversation_id]

    def clear_history(self) -> None:
        """Clear recorded A2A message history."""
        self._history.clear()

    def clear_all(self) -> None:
        """Clear all registered agents and message history (for test isolation)."""
        self._cards.clear()
        self._handlers.clear()
        self._history.clear()

    # -----------------------------------------------------------------------
    # Blackboard & Session State Sync
    # -----------------------------------------------------------------------

    def _sync_state(
        self,
        request: A2AMessage,
        response: A2AMessage,
        blackboard: Blackboard | None = None,
        session_state: dict[str, Any] | None = None,
        state_channel: str | None = None,
    ) -> None:
        """Persist A2A envelopes and response payload into Blackboard / session_state."""
        serialized_pair = [request.to_dict(), response.to_dict()]

        if blackboard is not None:
            existing_log = blackboard.read("a2a_messages")
            updated_log = list(existing_log) if isinstance(existing_log, list) else []
            updated_log.extend(serialized_pair)
            blackboard.write(
                channel="a2a_messages",
                writer=f"A2ABridge:{response.sender_agent}",
                value=updated_log,
            )
            if state_channel:
                blackboard.write(
                    channel=state_channel,
                    writer=response.sender_agent,
                    value=response.payload,
                )

        if session_state is not None:
            existing_msgs = session_state.get("a2a_messages")
            if not isinstance(existing_msgs, list):
                existing_msgs = []
            existing_msgs.extend(serialized_pair)
            session_state["a2a_messages"] = existing_msgs
            if state_channel:
                session_state[state_channel] = response.payload

    def _normalize_response(self, request: A2AMessage, raw_result: Any) -> A2AMessage:
        """Convert a handler's return value into a valid A2AMessage reply."""
        if isinstance(raw_result, A2AMessage):
            return raw_result
        if isinstance(raw_result, dict):
            return request.create_reply(
                performative=Performative.INFORM,
                payload=raw_result,
            )
        return request.create_reply(
            performative=Performative.INFORM,
            payload={"result": raw_result},
        )

    # -----------------------------------------------------------------------
    # Dispatch (Sync & Async)
    # -----------------------------------------------------------------------

    def dispatch(
        self,
        message: A2AMessage,
        blackboard: Blackboard | None = None,
        session_state: dict[str, Any] | None = None,
        state_channel: str | None = None,
        raise_on_missing: bool = False,
    ) -> A2AMessage:
        """
        Synchronously route an A2AMessage to its registered recipient.

        If the recipient is unregistered, returns an ERROR envelope unless
        `raise_on_missing=True`.
        """
        self._record(message)
        handler = self._handlers.get(message.recipient_agent)

        if handler is None:
            err_msg = f"Recipient agent '{message.recipient_agent}' is not registered on A2ABridge."
            if raise_on_missing:
                raise KeyError(err_msg)
            logger.warning("[A2ABridge] %s", err_msg)
            error_reply = message.create_reply(
                performative=Performative.ERROR,
                payload={"error": err_msg, "recipient": message.recipient_agent},
                sender_agent="A2ABridge",
                sender_framework=FrameworkType.SYSTEM,
            )
            self._record(error_reply)
            self._sync_state(message, error_reply, blackboard, session_state, state_channel)
            return error_reply

        try:
            if inspect.iscoroutinefunction(handler):
                try:
                    asyncio.get_running_loop()
                    # Running inside an active event loop; execute in a worker thread
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        raw_result = pool.submit(asyncio.run, handler(message)).result()
                except RuntimeError:
                    raw_result = asyncio.run(handler(message))
            else:
                raw_result = handler(message)

            reply = self._normalize_response(message, raw_result)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[A2ABridge] Handler error for '%s': %s",
                message.recipient_agent,
                exc,
            )
            reply = message.create_reply(
                performative=Performative.ERROR,
                payload={"error": str(exc), "agent": message.recipient_agent},
            )

        self._record(reply)
        self._sync_state(message, reply, blackboard, session_state, state_channel)
        return reply

    async def adispatch(
        self,
        message: A2AMessage,
        blackboard: Blackboard | None = None,
        session_state: dict[str, Any] | None = None,
        state_channel: str | None = None,
        raise_on_missing: bool = False,
    ) -> A2AMessage:
        """Asynchronously route an A2AMessage to its registered recipient."""
        self._record(message)
        handler = self._handlers.get(message.recipient_agent)

        if handler is None:
            err_msg = f"Recipient agent '{message.recipient_agent}' is not registered on A2ABridge."
            if raise_on_missing:
                raise KeyError(err_msg)
            error_reply = message.create_reply(
                performative=Performative.ERROR,
                payload={"error": err_msg, "recipient": message.recipient_agent},
                sender_agent="A2ABridge",
                sender_framework=FrameworkType.SYSTEM,
            )
            self._record(error_reply)
            self._sync_state(message, error_reply, blackboard, session_state, state_channel)
            return error_reply

        try:
            if inspect.iscoroutinefunction(handler):
                raw_result = await handler(message)
            else:
                raw_result = handler(message)
            reply = self._normalize_response(message, raw_result)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[A2ABridge] Async handler error for '%s': %s",
                message.recipient_agent,
                exc,
            )
            reply = message.create_reply(
                performative=Performative.ERROR,
                payload={"error": str(exc), "agent": message.recipient_agent},
            )

        self._record(reply)
        self._sync_state(message, reply, blackboard, session_state, state_channel)
        return reply

    def send_and_update_blackboard(
        self,
        sender_agent: str,
        sender_framework: FrameworkType,
        recipient_agent: str,
        recipient_framework: FrameworkType,
        performative: Performative,
        payload: dict[str, Any],
        blackboard: Blackboard | None = None,
        session_state: dict[str, Any] | None = None,
        state_channel: str | None = None,
        conversation_id: str | None = None,
    ) -> tuple[A2AMessage, A2AMessage]:
        """
        Convenience helper that builds an A2AMessage, dispatches it, updates
        the Blackboard/session_state, and returns (request, response).
        """
        kwargs: dict[str, Any] = {
            "sender_agent": sender_agent,
            "sender_framework": sender_framework,
            "recipient_agent": recipient_agent,
            "recipient_framework": recipient_framework,
            "performative": performative,
            "payload": payload,
        }
        if conversation_id:
            kwargs["conversation_id"] = conversation_id

        request = A2AMessage(**kwargs)
        response = self.dispatch(
            request,
            blackboard=blackboard,
            session_state=session_state,
            state_channel=state_channel,
        )
        return request, response


# Alias matching brief terminology
A2ADispatcher = A2ABridge


# ---------------------------------------------------------------------------
# Global Singleton Bridge
# ---------------------------------------------------------------------------

_GLOBAL_A2A_BRIDGE: A2ABridge | None = None


def get_a2a_bridge() -> A2ABridge:
    """Retrieve or initialize the global A2ABridge singleton."""
    global _GLOBAL_A2A_BRIDGE
    if _GLOBAL_A2A_BRIDGE is None:
        _GLOBAL_A2A_BRIDGE = A2ABridge()
    return _GLOBAL_A2A_BRIDGE


def reset_a2a_bridge() -> A2ABridge:
    """Reset the global A2ABridge singleton (useful for test isolation)."""
    global _GLOBAL_A2A_BRIDGE
    _GLOBAL_A2A_BRIDGE = A2ABridge()
    return _GLOBAL_A2A_BRIDGE
