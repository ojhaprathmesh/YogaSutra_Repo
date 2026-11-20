"""
tests/test_dashboard.py — Unit and Integration tests for Observability Dashboard (Phase 3C)
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.server import app
from app.ui.dashboard import (
    collect_dashboard_stats,
    record_guardrail_audit,
    render_dashboard_html,
)

client = TestClient(app)


def test_collect_dashboard_stats_structure():
    """Verify that dashboard stats collection produces the required schemas."""
    stats = collect_dashboard_stats()

    # Summary section
    assert "summary" in stats
    summary = stats["summary"]
    assert "total_runs" in summary
    assert "avg_latency_ms" in summary
    assert "total_tokens" in summary
    assert summary["actual_total_cost_usd"] == 0.00
    assert "commercial_savings_usd" in summary
    assert "safety_clearance_rate_pct" in summary

    # Latency Waterfall section
    assert "latency_waterfall" in stats
    waterfall = stats["latency_waterfall"]
    assert "total_pipeline_ms" in waterfall
    assert "stages" in waterfall
    assert len(waterfall["stages"]) >= 5
    for stage in waterfall["stages"]:
        assert "agent" in stage
        assert "duration_ms" in stage
        assert "start_offset_ms" in stage
        assert "pct_of_total" in stage

    # Cost and Token breakdown across Gemini, Ollama, NVIDIA
    assert "cost_and_tokens" in stats
    cost_tokens = stats["cost_and_tokens"]
    assert "providers" in cost_tokens
    providers = cost_tokens["providers"]
    assert "gemini" in providers
    assert "ollama" in providers
    assert "nvidia" in providers
    for p_key in ("gemini", "ollama", "nvidia"):
        p_data = providers[p_key]
        assert p_data["actual_cost_usd"] == 0.00
        assert p_data["total_tokens"] >= 0
        assert p_data["commercial_equiv_usd"] >= 0.0

    # Prompt Strategy Comparisons (Default, ReAct, Few-Shot, CoT)
    assert "prompt_strategies" in stats
    strategies = stats["prompt_strategies"]
    strategy_names = [s["strategy"] for s in strategies]
    assert "default" in strategy_names
    assert "react" in strategy_names
    assert "few_shot" in strategy_names
    assert "cot" in strategy_names

    # Guardrail Security & Event Bus Stream
    assert "guardrail_security" in stats
    guard = stats["guardrail_security"]
    assert "total_blocked" in guard
    assert "violation_breakdown" in guard
    assert "prompt_injection" in guard["violation_breakdown"]

    assert "event_bus_stream" in stats
    assert "events" in stats["event_bus_stream"]


def test_record_guardrail_audit():
    """Verify that guardrail violations are recorded into the live audit log."""
    record_guardrail_audit(
        input_text="System test injection query ignore previous",
        violations=["prompt_injection"],
        risk_score=0.99,
        status="BLOCKED",
    )
    stats = collect_dashboard_stats()
    audit_log = stats["guardrail_security"]["recent_audit_log"]
    assert len(audit_log) > 0
    assert any("System test injection" in item["input_snippet"] for item in audit_log)


def test_render_dashboard_html():
    """Verify that HTML rendering produces complete, valid HTML with expected components."""
    html = render_dashboard_html()
    assert "<!DOCTYPE html>" in html
    assert "YogaSutra Observability" in html
    assert "waterfallContainer" in html
    assert "providersGrid" in html
    assert "matrixBody" in html
    assert "auditLogContainer" in html
    assert "eventBusLog" in html


def test_api_dashboard_stats_endpoint():
    """Verify GET /api/dashboard/stats returns valid JSON payload."""
    resp = client.get("/api/dashboard/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "latency_waterfall" in data
    assert "cost_and_tokens" in data
    assert "prompt_strategies" in data
    assert "guardrail_security" in data
    assert "event_bus_stream" in data


def test_api_dashboard_html_endpoint():
    """Verify GET /dashboard serves HTML dashboard page."""
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "YogaSutra Observability" in resp.text
    assert "waterfall-track" in resp.text


def test_api_static_index_enhancements():
    """Verify GET / serves enhanced Studio with guardrail badge, prompt strategy selector, and biomechanics HUD."""
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    # Security Safety Badge / Indicator
    assert "securityBadge" in html
    assert "guardrailBar" in html

    # Prompt Strategy Selector
    assert "strategyChips" in html
    assert "data-strategy=\"react\"" in html
    assert "data-strategy=\"few_shot\"" in html
    assert "data-strategy=\"cot\"" in html

    # Real-Time Biomechanics HUD Canvas (Phase 3B)
    assert "skeletonCanvas" in html
    assert "simulateKneeDeviation" in html
    assert "resetToOptimal" in html
