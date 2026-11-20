"""
tests/test_benchmark_metrics.py — Unit tests for Performance, Token, Cost, & Safety Benchmark Metrics
"""
from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.benchmark_metrics import (
    CANONICAL_PERSONAS,
    ArchitecturalPattern,
    BenchmarkAggregator,
    BenchmarkRunResult,
    PromptStrategy,
    calculate_adherence_and_safety,
    calculate_cost_estimation,
    calculate_percentiles,
    calculate_token_consumption,
)
from scripts.run_comprehensive_benchmark import run_benchmark


class TestLatencyPercentiles:
    def test_empty_list_returns_zeros(self):
        res = calculate_percentiles([])
        assert res["p50"] == 0.0
        assert res["p90"] == 0.0
        assert res["p95"] == 0.0
        assert res["min"] == 0.0
        assert res["max"] == 0.0
        assert res["mean"] == 0.0
        assert res["count"] == 0

    def test_single_element(self):
        res = calculate_percentiles([120.0])
        assert res["p50"] == 120.0
        assert res["p90"] == 120.0
        assert res["p95"] == 120.0
        assert res["min"] == 120.0
        assert res["max"] == 120.0
        assert res["mean"] == 120.0
        assert res["count"] == 1

    def test_multi_element_distribution(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        res = calculate_percentiles(values)
        assert res["min"] == 10.0
        assert res["max"] == 100.0
        assert res["p50"] == 55.0
        assert res["p90"] == 91.0
        assert res["p95"] == 95.5
        assert res["mean"] == 55.0
        assert res["count"] == 10


class TestTokenConsumption:
    def test_token_summation(self):
        records = [
            {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            {"input_tokens": 200, "output_tokens": 80, "total_tokens": 280},
        ]
        res = calculate_token_consumption(records)
        assert res["input_tokens"] == 300
        assert res["output_tokens"] == 130
        assert res["total_tokens"] == 430

    def test_empty_records(self):
        res = calculate_token_consumption([])
        assert res["input_tokens"] == 0
        assert res["output_tokens"] == 0
        assert res["total_tokens"] == 0


class TestCostEstimation:
    def test_free_tier_actual_cost_zero(self):
        res = calculate_cost_estimation(
            model_or_provider="gemini-3.5-flash-lite",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        assert res["actual_cost_usd"] == 0.0
        # Commercial equivalent: 1M * 0.075 + 1M * 0.300 = $0.375
        assert res["commercial_input_usd"] == 0.075
        assert res["commercial_output_usd"] == 0.300
        assert res["commercial_total_usd"] == 0.375
        assert res["savings_usd"] == 0.375

    def test_ollama_commercial_rate_card(self):
        res = calculate_cost_estimation(
            model_or_provider="gpt-oss:120b",
            input_tokens=500_000,
            output_tokens=250_000,
        )
        assert res["actual_cost_usd"] == 0.0
        # Commercial input: 0.5 * 0.60 = 0.30, output: 0.25 * 1.80 = 0.45, total = 0.75
        assert round(res["commercial_input_usd"], 2) == 0.30
        assert round(res["commercial_output_usd"], 2) == 0.45
        assert round(res["commercial_total_usd"], 2) == 0.75

    def test_audio_cost_calculation(self):
        res = calculate_cost_estimation(
            model_or_provider="eleven_turbo_v2_5",
            input_tokens=0,
            output_tokens=0,
            audio_chars=2000,
        )
        # 2k chars * $0.150 / 1k chars = $0.30
        assert round(res["commercial_audio_usd"], 2) == 0.30


class TestSafetyAndAdherenceMetrics:
    def test_adherence_and_safety_rates(self):
        runs = [
            {"constraint_adhered": True, "safety_cleared": True, "contraindication_count": 0, "revisions_count": 0},
            {"constraint_adhered": True, "safety_cleared": True, "contraindication_count": 0, "revisions_count": 1},
            {"constraint_adhered": False, "safety_cleared": True, "contraindication_count": 1, "revisions_count": 0},
            {"constraint_adhered": True, "safety_cleared": False, "contraindication_count": 2, "revisions_count": 1},
        ]
        metrics = calculate_adherence_and_safety(runs)
        assert metrics["total_runs"] == 4
        assert metrics["constraint_adherence_rate_pct"] == 75.0
        assert metrics["safety_clearance_rate_pct"] == 75.0
        assert metrics["contraindication_count"] == 3
        assert metrics["reflection_revision_rate_pct"] == 50.0

    def test_empty_runs_safety(self):
        metrics = calculate_adherence_and_safety([])
        assert metrics["total_runs"] == 0
        assert metrics["constraint_adherence_rate_pct"] == 0.0
        assert metrics["safety_clearance_rate_pct"] == 0.0


class TestCanonicalPersonas:
    def test_canonical_personas_complete(self):
        expected_ids = {"P01", "P02", "P03", "P04", "P05"}
        assert set(CANONICAL_PERSONAS.keys()) == expected_ids

        # P01 beginner with back pain
        p01 = CANONICAL_PERSONAS["P01"]
        assert p01.experience_level == "beginner"
        assert any("back" in c for c in p01.constraints)

        # P03 senior with osteoarthritis
        p03 = CANONICAL_PERSONAS["P03"]
        assert p03.experience_level == "senior"
        assert any("osteoarthritis" in c for c in p03.constraints)


class TestBenchmarkAggregatorAndExecution:
    def test_aggregator_summarization(self, tmp_path: Path):
        agg = BenchmarkAggregator()
        r1 = BenchmarkRunResult(
            run_id="run_1",
            persona_id="P01",
            persona_name="Beginner",
            prompt_strategy="react",
            architectural_pattern="sequential",
            latency_ms=150.0,
            input_tokens=800,
            output_tokens=400,
            total_tokens=1200,
            actual_cost_usd=0.0,
            commercial_cost_usd=0.0003,
            safety_cleared=True,
            constraint_adhered=True,
            revisions_count=1,
            agents_called=["PlannerAgent"],
        )
        r2 = BenchmarkRunResult(
            run_id="run_2",
            persona_id="P02",
            persona_name="Athletic",
            prompt_strategy="cot",
            architectural_pattern="hierarchical",
            latency_ms=90.0,
            input_tokens=600,
            output_tokens=300,
            total_tokens=900,
            actual_cost_usd=0.0,
            commercial_cost_usd=0.0002,
            safety_cleared=True,
            constraint_adhered=True,
            revisions_count=0,
            agents_called=["PlannerAgent"],
        )
        agg.add_run(r1)
        agg.add_run(r2)

        summary = agg.summarize()
        assert summary.total_runs == 2
        assert summary.tokens["total_tokens"] == 2100
        assert summary.latency["p50"] == 120.0
        assert "P01" in summary.by_persona
        assert "P02" in summary.by_persona
        assert "react" in summary.by_prompt_strategy
        assert "sequential" in summary.by_architecture_pattern

        md = agg.generate_markdown_report()
        assert "# YogaSutra: Comprehensive Multi-Run Performance & Cost Benchmark Report" in md
        assert "Sequential" in md
        assert "Hierarchical" in md

    def test_run_benchmark_execution(self, tmp_path: Path):
        # Execute focused subset to test end-to-end integration
        agg = run_benchmark(
            personas=["P01", "P02"],
            strategies=[PromptStrategy.DEFAULT, PromptStrategy.COT],
            patterns=[ArchitecturalPattern.SEQUENTIAL, ArchitecturalPattern.HIERARCHICAL],
            output_dir=tmp_path,
        )
        assert len(agg.runs) == 8  # 2 * 2 * 2 = 8 runs

        json_file = tmp_path / "comprehensive_benchmark.json"
        md_file = tmp_path / "BENCHMARK_REPORT.md"

        assert json_file.exists()
        assert md_file.exists()

        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert "summary" in data
        assert "runs" in data
        assert len(data["runs"]) == 8
        assert data["summary"]["total_runs"] == 8
