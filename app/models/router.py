"""
app/models/router.py — Intelligent Context & Complexity Model Router with Automatic Failover

Routes agent tasks to optimal models based on:
1. Task Complexity (FAST vs. REASONING vs. MULTIMODAL)
2. Context Size / Prompt Length
3. Provider Health & Free-Tier Quota Availability
4. Automatic Seamless Failover across Google Gemini, Ollama Cloud, and NVIDIA NIM.
"""
from __future__ import annotations

import enum
import logging
from typing import Any

from app.models.gateway import ModelGateway, ModelResponse, model_gateway

logger = logging.getLogger("yogasutra.router")


class TaskComplexity(str, enum.Enum):
    """Task complexity tier for routing."""
    FAST = "fast"                 # Profile extraction, speech cue generation, summary
    REASONING = "reasoning"       # Plan sequencing, safety contraindication audit, critic reflection
    MULTIMODAL = "multimodal"     # Vision & pose alignment verification


class ModelRouter:
    """Intelligent Router selecting the optimal free model with automatic failover."""

    def __init__(self, gateway: ModelGateway | None = None) -> None:
        self.gateway = gateway or model_gateway

        # Routing preference chains (ordered by preference)
        self.routing_chains: dict[TaskComplexity, list[dict[str, str]]] = {
            TaskComplexity.FAST: [
                {"provider": "gemini", "model": "gemini-3.5-flash-lite"},
                {"provider": "ollama", "model": "gemma4:31b"},
                {"provider": "nvidia", "model": "meta/llama-3.2-11b-vision-instruct"},
            ],
            TaskComplexity.REASONING: [
                {"provider": "ollama", "model": "gpt-oss:120b"},
                {"provider": "gemini", "model": "gemini-3.5-flash-lite"},
                {"provider": "ollama", "model": "nemotron-3-nano:30b"},
                {"provider": "nvidia", "model": "meta/llama-3.2-11b-vision-instruct"},
            ],
            TaskComplexity.MULTIMODAL: [
                {"provider": "nvidia", "model": "meta/llama-3.2-11b-vision-instruct"},
                {"provider": "gemini", "model": "gemini-3.5-flash-lite"},
            ],
        }

    def route_and_generate(
        self,
        prompt: str,
        system_instruction: str = "",
        complexity: TaskComplexity = TaskComplexity.FAST,
        preferred_provider: str | None = None,
        max_tokens: int = 2048,
    ) -> ModelResponse:
        """
        Selects model according to complexity/context, with automatic fallback on failure.
        """
        # Dynamic complexity upgrade for long prompts (> 2000 chars)
        if len(prompt) > 2000 and complexity == TaskComplexity.FAST:
            complexity = TaskComplexity.REASONING

        chain = list(self.routing_chains.get(complexity, self.routing_chains[TaskComplexity.FAST]))

        # If user/caller explicitly prefers a provider, move it to the front of the chain
        if preferred_provider:
            p = preferred_provider.lower()
            chain = sorted(chain, key=lambda c: 0 if c["provider"] == p else 1)

        errors: list[str] = []
        for candidate in chain:
            prov = candidate["provider"]
            mod = candidate["model"]
            try:
                logger.info(f"[Router] Routing {complexity.value} task to {prov}:{mod}")
                resp = self.gateway.generate(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    model=mod,
                    provider=prov,
                    max_tokens=max_tokens,
                )
                return resp
            except Exception as e:
                err_msg = f"{prov}:{mod} failed: {e}"
                logger.warning(f"[Router] Failover triggered. {err_msg}")
                errors.append(err_msg)
                continue

        raise RuntimeError(f"All model providers failed in router chain: {'; '.join(errors)}")


# Global singleton router
model_router = ModelRouter()
