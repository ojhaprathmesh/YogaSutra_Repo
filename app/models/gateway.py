"""
app/models/gateway.py — Unified Multi-Provider Model Gateway

Provides seamless, standardized access across:
1. Google Gemini (AI Studio free tier: gemini-3.5-flash-lite, gemini-3.1-flash-lite)
2. Ollama Cloud API (gpt-oss:120b, gemma4:31b, nemotron-3-nano:30b)
3. NVIDIA NIM API (meta/llama-3.2-11b-vision-instruct, etc.)

Normalizes outputs, token metrics, and latency across providers with zero external dependencies.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from app.models.cache import response_cache
from app.models.rate_limiter import rate_limiter

logger = logging.getLogger("yogasutra.gateway")


@dataclass
class ModelResponse:
    """Standardized response from any model provider."""
    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    thinking: str | None = None
    cached: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


class ModelGateway:
    """Unified Gateway coordinating LLM requests across free-tier providers."""

    def __init__(self) -> None:
        self.google_api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        self.nvidia_api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        self.ollama_api_key = os.environ.get("OLLAMA_API_KEY", "").strip()

    def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        model: str | None = None,
        provider: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        use_cache: bool = True,
    ) -> ModelResponse:
        """
        Generate completion using the specified or auto-selected provider.
        """
        # Determine provider and model defaults
        if not provider:
            if model and ("gemini" in model.lower()):
                provider = "gemini"
            elif model and ("gpt-oss" in model.lower() or "gemma" in model.lower() or "nemotron" in model.lower()):
                provider = "ollama"
            elif model and ("llama" in model.lower() or "mistral" in model.lower() or "nvidia" in model.lower()):
                provider = "nvidia"
            else:
                provider = "gemini"

        if not model:
            if provider == "gemini":
                model = os.environ.get("YOGASUTRA_MODEL", "gemini-3.5-flash-lite")
            elif provider == "ollama":
                model = "gpt-oss:120b"
            elif provider == "nvidia":
                model = "meta/llama-3.2-11b-vision-instruct"

        # Check cache
        cache_key = {"provider": provider, "model": model, "prompt": prompt, "system": system_instruction}
        if use_cache:
            cached_val = response_cache.get(cache_key)
            if cached_val:
                cached_val.cached = True
                logger.info(f"[Gateway] Cache HIT for {provider}:{model}")
                return cached_val

        # Execute generation with provider-specific handler wrapped by rate limiter
        t0 = time.monotonic()
        if provider == "ollama":
            resp = rate_limiter.execute_with_retry(
                lambda: self._call_ollama(prompt, system_instruction, model, temperature, max_tokens),
                provider="ollama",
            )
        elif provider == "nvidia":
            resp = rate_limiter.execute_with_retry(
                lambda: self._call_nvidia(prompt, system_instruction, model, temperature, max_tokens),
                provider="nvidia",
            )
        else:
            resp = rate_limiter.execute_with_retry(
                lambda: self._call_gemini(prompt, system_instruction, model, temperature, max_tokens),
                provider="gemini",
            )

        resp.latency_ms = round((time.monotonic() - t0) * 1000, 2)
        if use_cache:
            response_cache.set(cache_key, resp)
        return resp

    # -----------------------------------------------------------------------
    # Provider Implementations (Zero external library dependencies)
    # -----------------------------------------------------------------------

    def _call_gemini(
        self, prompt: str, system: str, model: str, temperature: float, max_tokens: int
    ) -> ModelResponse:
        """Call Google Gemini generateContent REST API."""
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is not configured")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.google_api_key}"
        contents: list[dict[str, Any]] = []
        if system:
            contents.append({"role": "user", "parts": [{"text": f"[System Instructions]: {system}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I will follow these instructions."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        content = ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                content = parts[0].get("text", "")

        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)

        return ModelResponse(
            content=content,
            model=model,
            provider="gemini",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            raw=data,
        )

    def _call_ollama(
        self, prompt: str, system: str, model: str, temperature: float, max_tokens: int
    ) -> ModelResponse:
        """Call Ollama Cloud generate API."""
        if not self.ollama_api_key:
            raise ValueError("OLLAMA_API_KEY is not configured")

        url = "https://ollama.com/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.ollama_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))

        content = data.get("response", "")
        thinking = data.get("thinking")
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return ModelResponse(
            content=content,
            model=model,
            provider="ollama",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            thinking=thinking,
            raw=data,
        )

    def _call_nvidia(
        self, prompt: str, system: str, model: str, temperature: float, max_tokens: int
    ) -> ModelResponse:
        """Call NVIDIA NIM chat completions API."""
        if not self.nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY is not configured")

        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.nvidia_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        content = ""
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return ModelResponse(
            content=content,
            model=model,
            provider="nvidia",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            raw=data,
        )


# Global singleton gateway
model_gateway = ModelGateway()
