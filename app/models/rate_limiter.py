"""
app/models/rate_limiter.py — Adaptive Token-Bucket Rate Limiter & Exponential Backoff

Protects free-tier API quotas (Google Gemini, NVIDIA NIM, Ollama Cloud) by:
1. Enforcing safe request rates (e.g. Gemini <= 12 RPM, well below the 15 RPM cap).
2. Spacing out consecutive multi-agent calls to prevent burst exhaustion.
3. Automatically intercepting 429 (Resource Exhausted) and 503 (Unavailable) errors
   with jittered exponential backoff retries.
4. Tracking wait delays for execution trace observability.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

logger = logging.getLogger("yogasutra.rate_limiter")

T = TypeVar("T")


@dataclass
class ProviderQuota:
    """Configuration for a specific provider's rate limits."""
    name: str
    max_rpm: float = 12.0  # requests per minute
    burst_capacity: float = 3.0  # max tokens in bucket
    min_interval_sec: float = 1.5  # minimum spacing between requests
    tokens: float = field(default=3.0, init=False)
    last_update: float = field(default_factory=time.monotonic, init=False)
    last_request_time: float = field(default=0.0, init=False)
    total_wait_sec: float = field(default=0.0, init=False)

    def refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_update
        self.last_update = now
        # Add tokens based on RPM
        refill_rate = self.max_rpm / 60.0  # tokens per second
        self.tokens = min(self.burst_capacity, self.tokens + elapsed * refill_rate)


class TokenBucketRateLimiter:
    """
    Thread-safe & async-safe Token Bucket Rate Limiter supporting multiple providers.
    """

    def __init__(self) -> None:
        self._quotas: dict[str, ProviderQuota] = {
            "gemini": ProviderQuota(name="gemini", max_rpm=12.0, burst_capacity=2.0, min_interval_sec=1.5),
            "ollama": ProviderQuota(name="ollama", max_rpm=20.0, burst_capacity=4.0, min_interval_sec=0.5),
            "nvidia": ProviderQuota(name="nvidia", max_rpm=15.0, burst_capacity=3.0, min_interval_sec=1.0),
        }
        self._lock = asyncio.Lock()

    def get_quota(self, provider: str) -> ProviderQuota:
        p = provider.lower()
        if p not in self._quotas:
            self._quotas[p] = ProviderQuota(name=p, max_rpm=15.0, burst_capacity=3.0, min_interval_sec=1.0)
        return self._quotas[p]

    def acquire_sync(self, provider: str = "gemini") -> float:
        """
        Synchronously wait until a token is available and min_interval is respected.
        Returns the duration waited in seconds.
        """
        quota = self.get_quota(provider)
        quota.refill()

        now = time.monotonic()
        wait_needed = 0.0

        # Enforce min_interval spacing
        since_last = now - quota.last_request_time
        if since_last < quota.min_interval_sec:
            wait_needed = max(wait_needed, quota.min_interval_sec - since_last)

        # Enforce token availability
        if quota.tokens < 1.0:
            tokens_needed = 1.0 - quota.tokens
            refill_rate = quota.max_rpm / 60.0
            time_for_tokens = tokens_needed / refill_rate
            wait_needed = max(wait_needed, time_for_tokens)

        if wait_needed > 0.05:
            logger.info(
                f"[RateLimiter] Throttling {provider} for {wait_needed:.2f}s "
                f"(remaining tokens: {quota.tokens:.2f})"
            )
            time.sleep(wait_needed)

        # Deduct token
        quota.refill()
        quota.tokens = max(0.0, quota.tokens - 1.0)
        quota.last_request_time = time.monotonic()
        quota.total_wait_sec += wait_needed
        return wait_needed

    async def acquire_async(self, provider: str = "gemini") -> float:
        """
        Asynchronously wait until a token is available.
        Returns the duration waited in seconds.
        """
        async with self._lock:
            quota = self.get_quota(provider)
            quota.refill()

            now = time.monotonic()
            wait_needed = 0.0

            since_last = now - quota.last_request_time
            if since_last < quota.min_interval_sec:
                wait_needed = max(wait_needed, quota.min_interval_sec - since_last)

            if quota.tokens < 1.0:
                tokens_needed = 1.0 - quota.tokens
                refill_rate = quota.max_rpm / 60.0
                time_for_tokens = tokens_needed / refill_rate
                wait_needed = max(wait_needed, time_for_tokens)

            if wait_needed > 0.05:
                logger.info(
                    f"[RateLimiter] Async throttling {provider} for {wait_needed:.2f}s "
                    f"(remaining tokens: {quota.tokens:.2f})"
                )
                await asyncio.sleep(wait_needed)

            quota.refill()
            quota.tokens = max(0.0, quota.tokens - 1.0)
            quota.last_request_time = time.monotonic()
            quota.total_wait_sec += wait_needed
            return wait_needed

    def execute_with_retry(
        self,
        func: Callable[[], T],
        provider: str = "gemini",
        max_retries: int = 3,
        initial_backoff: float = 2.0,
    ) -> T:
        """
        Execute a function with rate limiting and exponential backoff on 429/503.
        """
        retries = 0
        backoff = initial_backoff

        while True:
            self.acquire_sync(provider)
            try:
                return func()
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = (
                    "429" in err_str
                    or "resource_exhausted" in err_str
                    or "quota" in err_str
                    or "rate limit" in err_str
                    or "too many requests" in err_str
                )
                is_unavailable = "503" in err_str or "unavailable" in err_str or "high demand" in err_str

                if (is_rate_limit or is_unavailable) and retries < max_retries:
                    retries += 1
                    # Jittered backoff: backoff +/- 25%
                    jitter = random.uniform(0.75, 1.25)
                    sleep_time = backoff * jitter
                    logger.warning(
                        f"[RateLimiter] Caught {type(e).__name__} ({e}). "
                        f"Retrying in {sleep_time:.2f}s (attempt {retries}/{max_retries})..."
                    )
                    time.sleep(sleep_time)
                    backoff *= 2.0
                else:
                    raise


# Global singleton instance
rate_limiter = TokenBucketRateLimiter()
