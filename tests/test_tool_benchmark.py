import asyncio

from evaluation.tool_benchmark import run_tool_benchmark


def test_tool_benchmark_measures_success_security_idempotency_and_fallback(tmp_path):
    report = asyncio.run(run_tool_benchmark(tmp_path / "campus.db"))

    assert report["case_count"] >= 8
    assert report["metrics"]["tool_correctness"] == 1.0
    assert report["metrics"]["trusted_context_security"] == 1.0
    assert report["metrics"]["idempotency_correctness"] == 1.0
    assert report["metrics"]["fallback_correctness"] == 1.0
    assert report["latency_ms"]["p50"] >= 0.0
    assert report["latency_ms"]["p95"] >= report["latency_ms"]["p50"]
    assert all(case["passed"] for case in report["cases"])


def test_tool_benchmark_records_real_manager_stats(tmp_path):
    report = asyncio.run(run_tool_benchmark(tmp_path / "campus.db"))

    assert report["tool_stats"]["query_campus_card"]["executed"] >= 1
    assert report["tool_stats"]["create_ticket"]["executed"] >= 2
    assert report["tool_stats"]["get_ticket"]["rejected"] >= 1
