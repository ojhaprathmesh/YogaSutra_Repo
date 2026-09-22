"""
app/workflows/broadcast.py — Parallel / Broadcast Specialist Workflow

Executes independent specialist tasks concurrently:
- Concurrent Knowledge Retrieval and Safety Constraint analysis
- Aggregates multi-agent specialist findings onto the shared Blackboard.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from app.workflows.blackboard import Blackboard

logger = logging.getLogger("yogasutra.workflows.broadcast")


class BroadcastWorkflow:
    """Coordinates parallel broadcast specialist queries."""

    async def broadcast_specialists(
        self,
        tasks: dict[str, Callable[[], Any]],
        blackboard: Blackboard,
    ) -> dict[str, Any]:
        """
        Executes named specialist tasks concurrently and records results onto blackboard.
        """
        results: dict[str, Any] = {}

        async def _run_task(name: str, fn: Callable[[], Any]) -> tuple[str, Any]:
            if asyncio.iscoroutinefunction(fn):
                res = await fn()
            else:
                loop = asyncio.get_running_loop()
                res = await loop.run_in_executor(None, fn)
            return name, res

        coros = [_run_task(name, fn) for name, fn in tasks.items()]
        completed = await asyncio.gather(*coros, return_exceptions=True)

        for item in completed:
            if isinstance(item, Exception):
                logger.error(f"[Broadcast] Task failed with exception: {item}")
                continue
            name, res = item
            results[name] = res
            blackboard.write(channel=name, writer="BroadcastWorkflow", value=res)

        return results


# Global singleton broadcast workflow
broadcast_workflow = BroadcastWorkflow()
