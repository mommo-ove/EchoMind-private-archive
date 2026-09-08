import json

from evaluation.retrieval_ab import (
    compare_rankings,
    compare_variants,
    cosine_rank,
    merge_unique_rankings,
    rewrite_succeeded,
    latency_summary,
    categorize_retrieval_case,
    write_report,
)


def test_compare_rankings_uses_same_cases_and_reports_metric_deltas():
    cases = [
        {
            "case_id": "network-401",
            "question": "campus network authentication failed with 401",
            "reference_relevance": {"gold": 3.0},
        },
        {
            "case_id": "card-charge",
            "question": "campus card charged twice",
            "reference_relevance": {"card": 3.0},
        },
    ]
    baseline = {
        "network-401": ["noise", "gold"],
        "card-charge": ["noise-1", "noise-2"],
    }
    candidate = {
        "network-401": ["gold", "noise"],
        "card-charge": ["card", "noise-1"],
    }

    report = compare_rankings(
        cases,
        baseline,
        candidate,
        top_k=2,
        baseline_name="vector_top_k",
        candidate_name="vector_plus_cross_encoder",
    )

    assert report["case_count"] == 2
    assert report["top_k"] == 2
    assert report["baseline"]["retrieval_hit_rate"] == 0.5
    assert report["candidate"]["retrieval_hit_rate"] == 1.0
    assert report["delta"]["retrieval_hit_rate"] == 0.5
    assert report["candidate"]["retrieval_mrr"] == 1.0
    assert report["delta"]["retrieval_ndcg_at_k"] > 0
    assert report["cases"][0]["baseline_ids"] == ["noise", "gold"]
    assert report["cases"][0]["candidate_ids"] == ["gold", "noise"]


def test_write_report_persists_reproducible_json(tmp_path):
    report = {
        "experiment": "retrieval_ab",
        "case_count": 1,
        "baseline": {"retrieval_mrr": 0.5},
        "candidate": {"retrieval_mrr": 1.0},
        "delta": {"retrieval_mrr": 0.5},
    }

    output = write_report(report, tmp_path / "retrieval-ab.json")

    assert output == tmp_path / "retrieval-ab.json"
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_compare_variants_reports_each_pipeline_against_one_fixed_baseline():
    cases = [
        {
            "case_id": "network-401",
            "question": "campus network 401",
            "reference_relevance": {"gold": 3.0},
        }
    ]
    rankings = {
        "default_vector": {"network-401": ["noise"]},
        "bge_vector": {"network-401": ["gold"]},
        "bge_rerank": {"network-401": ["gold"]},
    }

    report = compare_variants(
        cases,
        rankings,
        top_k=1,
        baseline_name="default_vector",
    )

    assert report["variant_order"] == [
        "default_vector",
        "bge_vector",
        "bge_rerank",
    ]
    assert report["variants"]["default_vector"]["retrieval_hit_rate"] == 0.0
    assert report["variants"]["bge_vector"]["retrieval_hit_rate"] == 1.0
    assert report["delta_vs_baseline"]["bge_vector"]["retrieval_hit_rate"] == 1.0
    assert report["cases"][0]["rankings"]["bge_rerank"] == ["gold"]


def test_cosine_rank_orders_document_ids_by_similarity():
    ranked = cosine_rank(
        [1.0, 0.0],
        {
            "network": [0.9, 0.1],
            "card": [0.0, 1.0],
            "mixed": [0.5, 0.5],
        },
    )

    assert ranked == ["network", "mixed", "card"]


def test_merge_unique_rankings_preserves_first_seen_recall_order():
    merged = merge_unique_rankings(
        [
            ["network", "dns", "card"],
            ["dns", "maintenance", "network"],
        ]
    )

    assert merged == ["network", "dns", "card", "maintenance"]


def test_rewrite_succeeded_requires_a_new_non_original_query():
    assert rewrite_succeeded("network error", ["network error"]) is False
    assert rewrite_succeeded(
        "network error",
        ["network error", "campus authentication failure"],
    ) is True


def test_latency_summary_reports_mean_p50_and_p95():
    summary = latency_summary([10.0, 20.0, 30.0, 40.0, 100.0])

    assert summary == {
        "count": 5,
        "mean_ms": 40.0,
        "p50_ms": 30.0,
        "p95_ms": 88.0,
        "max_ms": 100.0,
    }


def test_retrieval_case_categories_separate_tool_and_composite_queries():
    assert categorize_retrieval_case({"expected_tools": []}) == "knowledge_only"
    assert categorize_retrieval_case({
        "expected_tools": ["query_network_status"]
    }) == "tool_assisted"
    assert categorize_retrieval_case({
        "expected_tools": ["query_network_status", "query_campus_card"]
    }) == "composite"


def test_retrieval_case_category_prefers_explicit_benchmark_category():
    assert categorize_retrieval_case({
        "category": "single_document",
        "expected_tools": [],
    }) == "single_document"
    assert categorize_retrieval_case({
        "category": "composite",
        "expected_tools": [],
    }) == "composite"
