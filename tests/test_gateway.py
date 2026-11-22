"""
tests/test_gateway.py — Unit tests for ModelGateway, ModelRouter, and Cache.
"""
import os

from app.models.cache import ResponseCache
from app.models.gateway import DEFAULT_NVIDIA_MODEL, ModelGateway, ModelResponse
from app.models.rate_limiter import TokenBucketRateLimiter
from app.models.router import ModelRouter, TaskComplexity


class TestResponseCache:
    def test_cache_set_and_get(self):
        cache = ResponseCache(max_size=10, default_ttl_sec=60)
        cache.set("key1", "val1")
        assert cache.get("key1") == "val1"
        assert cache.hits == 1

    def test_cache_miss(self):
        cache = ResponseCache()
        assert cache.get("nonexistent") is None
        assert cache.misses == 1

    def test_cache_eviction(self):
        cache = ResponseCache(max_size=2)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")  # should evict k1
        assert cache.get("k1") is None
        assert cache.get("k2") == "v2"
        assert cache.get("k3") == "v3"


class TestRateLimiter:
    def test_token_bucket_initial_tokens(self):
        limiter = TokenBucketRateLimiter()
        quota = limiter.get_quota("gemini")
        assert quota.max_rpm == 12.0
        assert quota.burst_capacity == 2.0

    def test_acquire_sync_non_blocking_initial(self):
        limiter = TokenBucketRateLimiter()
        # Fast provider with 0 min interval for test
        quota = limiter.get_quota("test_provider")
        quota.min_interval_sec = 0.0
        quota.tokens = 2.0
        wait_time = limiter.acquire_sync("test_provider")
        assert wait_time < 0.1

    def test_execute_with_retry_success(self):
        limiter = TokenBucketRateLimiter()
        calls = 0

        def flaky_func():
            nonlocal calls
            calls += 1
            if calls < 2:
                raise RuntimeError("503 Service Unavailable")
            return "success"

        result = limiter.execute_with_retry(
            flaky_func, provider="ollama", max_retries=2, initial_backoff=0.01
        )
        assert result == "success"
        assert calls == 2


class TestModelRouter:
    def test_router_initialization(self):
        router = ModelRouter()
        assert TaskComplexity.FAST in router.routing_chains
        assert TaskComplexity.REASONING in router.routing_chains
        assert len(router.routing_chains[TaskComplexity.REASONING]) >= 3

    def test_router_mock_fallback(self):
        class MockGateway(ModelGateway):
            def generate(self, prompt, system_instruction="", model=None, provider=None, **_kwargs):
                if provider == "gemini":
                    raise RuntimeError("Gemini 429 quota exhausted")
                return ModelResponse(
                    content="Fallback response from " + str(provider),
                    model=model or "mock",
                    provider=provider or "mock",
                )

        mock_gw = MockGateway()
        router = ModelRouter(gateway=mock_gw)
        resp = router.route_and_generate(
            prompt="Plan a morning session",
            complexity=TaskComplexity.FAST,
            preferred_provider="gemini",
        )
        # Should have fallen over to ollama or nvidia
        assert resp.provider != "gemini"
        assert "Fallback response" in resp.content

    def test_router_uses_default_nvidia_model(self):
        router = ModelRouter()
        for complexity in (TaskComplexity.FAST, TaskComplexity.REASONING, TaskComplexity.MULTIMODAL):
            nvidia_entries = [c for c in router.routing_chains[complexity] if c["provider"] == "nvidia"]
            assert len(nvidia_entries) == 1
            assert nvidia_entries[0]["model"] == DEFAULT_NVIDIA_MODEL

    def test_router_fallback_to_nvidia_uses_default_model(self):
        called: list[tuple[str | None, str | None]] = []

        class MockGateway(ModelGateway):
            def generate(self, prompt, system_instruction="", model=None, provider=None, **_kwargs):
                called.append((provider, model))
                if provider != "nvidia":
                    raise RuntimeError(f"{provider} unavailable")
                return ModelResponse(content="ok", model=model or "", provider=provider or "")

        router = ModelRouter(gateway=MockGateway())
        resp = router.route_and_generate(prompt="Test fallback", complexity=TaskComplexity.FAST)
        assert resp.provider == "nvidia"
        assert resp.model == DEFAULT_NVIDIA_MODEL
        assert ("nvidia", DEFAULT_NVIDIA_MODEL) in called


class TestModelGatewayDefaultsAndRouting:
    def test_default_nvidia_model_constant(self):
        expected = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3.5-bolt-e4b")
        assert DEFAULT_NVIDIA_MODEL == expected

    def test_generate_autodetects_nvidia_and_ollama_nemotron(self, monkeypatch):
        gw = ModelGateway()
        calls: list[tuple[str, str]] = []

        def fake_nvidia(prompt, system, model, temperature, max_tokens):
            calls.append(("nvidia", model))
            return ModelResponse(content="nim", model=model, provider="nvidia")

        def fake_ollama(prompt, system, model, temperature, max_tokens):
            calls.append(("ollama", model))
            return ModelResponse(content="ollama", model=model, provider="ollama")

        monkeypatch.setattr(gw, "_call_nvidia", fake_nvidia)
        monkeypatch.setattr(gw, "_call_ollama", fake_ollama)

        resp_nim = gw.generate("hello", model="nvidia/nemotron-3.5-bolt-e4b", use_cache=False)
        assert resp_nim.provider == "nvidia"
        assert resp_nim.model == "nvidia/nemotron-3.5-bolt-e4b"

        resp_ollama = gw.generate("hello", model="nemotron-3-nano:30b", use_cache=False)
        assert resp_ollama.provider == "ollama"
        assert resp_ollama.model == "nemotron-3-nano:30b"

        resp_default = gw.generate("hello", provider="nvidia", use_cache=False)
        assert resp_default.provider == "nvidia"
        assert resp_default.model == DEFAULT_NVIDIA_MODEL

        assert calls == [
            ("nvidia", "nvidia/nemotron-3.5-bolt-e4b"),
            ("ollama", "nemotron-3-nano:30b"),
            ("nvidia", DEFAULT_NVIDIA_MODEL),
        ]
