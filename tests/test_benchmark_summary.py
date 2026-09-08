import json
from pathlib import Path

from evaluation.benchmark_summary import build_summary, render_markdown


def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_summary_combines_measured_routing_and_retrieval_without_inventing_generation(tmp_path):
    results = tmp_path / "results"
    _write(results / "intent_robustness_latest.json", {
        "dataset": {"total": 1100, "validation": 1000, "test": 100},
        "classification_quality": {
            "deepseek_llm_only": {"macro_f1": 0.986105, "exact_match": 0.98},
            "bge_plus_pattern": {"macro_f1": 0.908882, "exact_match": 0.91},
        },
        "fault_ablation": {
            "failure_rate_100": {
                "local_fallback": {
                    "response_availability": 1.0,
                    "routing_exact_match_mean": 0.91,
                    "failed_request_recovery_rate": 0.91,
                }
            }
        },
    })
    _write(results / "expanded_bge_baseline_latest.json", {
        "dataset_case_count": 140,
        "answerable_case_count": 120,
        "unanswerable_case_count": 20,
        "corpus_document_count": 20,
        "embedding_model": "BAAI/bge-small-zh-v1.5",
        "variants": {
            "bge_vector_top_3": {
                "metrics": {
                    "retrieval_hit_rate": 0.958333,
                    "retrieval_recall_at_k": 0.930556,
                    "retrieval_mrr": 0.876389,
                    "retrieval_ndcg_at_k": 0.875826,
                },
                "chroma_query_latency_ms": {"p95_ms": 33.768},
            }
        },
        "query_rewrite": {"status": "not_run", "reason": "API unavailable"},
    })
    _write(results / "end_to_end_run_status.json", {
        "status": "run_incomplete",
        "failure": {
            "reason": "DeepSeek thinking/text compatibility prevented a complete report.",
            "ragas_reason": "Optional ragas package is not installed.",
        },
    })

    summary = build_summary(results)

    assert summary["layers"]["routing"]["status"] == "measured"
    assert summary["layers"]["retrieval"]["top_3"]["hit_rate"] == 0.958333
    assert summary["layers"]["execution"]["status"] == "not_run"
    assert "thinking/text" in summary["layers"]["execution"]["reason"]
    assert summary["layers"]["generation"]["status"] == "not_run"
    assert "ragas" in summary["layers"]["generation"]["reason"]
    assert summary["layers"]["generation"]["metrics"] == {}
    assert summary["coverage"]["measured_layers"] == 2


def test_summary_uses_real_end_to_end_report_when_present(tmp_path):
    results = tmp_path / "results"
    _write(results / "intent_robustness_latest.json", {
        "dataset": {"total": 1, "validation": 0, "test": 1},
        "classification_quality": {
            "deepseek_llm_only": {"macro_f1": 1.0, "exact_match": 1.0},
            "bge_plus_pattern": {"macro_f1": 1.0, "exact_match": 1.0},
        },
        "fault_ablation": {"failure_rate_100": {"local_fallback": {
            "response_availability": 1.0, "routing_exact_match_mean": 1.0,
            "failed_request_recovery_rate": 1.0,
        }}},
    })
    _write(results / "expanded_bge_baseline_latest.json", {
        "dataset_case_count": 1, "answerable_case_count": 1,
        "unanswerable_case_count": 0, "corpus_document_count": 1,
        "embedding_model": "bge", "variants": {"bge_vector_top_3": {
            "metrics": {"retrieval_hit_rate": 1.0, "retrieval_recall_at_k": 1.0,
                        "retrieval_mrr": 1.0, "retrieval_ndcg_at_k": 1.0},
            "chroma_query_latency_ms": {"p95_ms": 1.0},
        }},
    })
    _write(results / "campus_end_to_end_compact_latest.json", {
        "total": 10,
        "metrics": {
            "route_correctness": 0.9,
            "tool_correctness": 0.8,
            "ticket_correctness": 1.0,
            "knowledge_correctness": 0.9,
            "task_completion": 0.8,
            "ragas_case_count": 6,
            "ragas_pass_rate": 0.75,
        },
        "avg_scores": {
            "faithfulness": 0.88,
            "answer_relevancy": 0.84,
        },
    })
    _write(results / "campus_end_to_end_compact_grounded_latest.json", {
        "total": 10,
        "metrics": {
            "route_correctness": 1.0,
            "tool_correctness": 0.95,
            "ticket_correctness": 1.0,
            "knowledge_correctness": 1.0,
            "task_completion": 0.9,
            "ragas_case_count": 6,
            "ragas_pass_rate": 0.8,
        },
        "avg_scores": {
            "faithfulness": 0.9,
            "answer_relevancy": 0.86,
        },
    })

    summary = build_summary(results)

    assert summary["layers"]["execution"]["status"] == "measured"
    assert summary["layers"]["execution"]["tool_correctness"] == 0.95
    assert summary["layers"]["generation"]["status"] == "measured"
    assert summary["layers"]["generation"]["metrics"]["faithfulness"] == 0.9
    assert summary["layers"]["execution"]["scope"] == "compact_representative_grounded"
    assert summary["coverage"]["measured_layers"] == 4


def test_summary_adds_real_tool_component_without_claiming_end_to_end_success(tmp_path):
    results = tmp_path / "results"
    _write(results / "intent_robustness_latest.json", {
        "dataset": {"total": 1, "validation": 0, "test": 1},
        "classification_quality": {
            "deepseek_llm_only": {"macro_f1": 1.0, "exact_match": 1.0},
            "bge_plus_pattern": {"macro_f1": 1.0, "exact_match": 1.0},
        },
        "fault_ablation": {"failure_rate_100": {"local_fallback": {
            "response_availability": 1.0, "routing_exact_match_mean": 1.0,
            "failed_request_recovery_rate": 1.0,
        }}},
    })
    _write(results / "expanded_bge_baseline_latest.json", {
        "dataset_case_count": 1, "answerable_case_count": 1,
        "unanswerable_case_count": 0, "corpus_document_count": 1,
        "embedding_model": "bge", "variants": {"bge_vector_top_3": {
            "metrics": {"retrieval_hit_rate": 1.0, "retrieval_recall_at_k": 1.0,
                        "retrieval_mrr": 1.0, "retrieval_ndcg_at_k": 1.0},
            "chroma_query_latency_ms": {"p95_ms": 1.0},
        }},
    })
    _write(results / "tool_benchmark_latest.json", {
        "evidence": "real_tool_manager_sqlite",
        "case_count": 9,
        "metrics": {
            "tool_correctness": 1.0,
            "trusted_context_security": 1.0,
            "idempotency_correctness": 1.0,
            "fallback_correctness": 1.0,
        },
        "latency_ms": {"p95": 6.39},
    })

    summary = build_summary(results)

    assert summary["layers"]["tool_execution"]["status"] == "measured"
    assert summary["layers"]["tool_execution"]["case_count"] == 9
    assert summary["layers"]["execution"]["status"] == "not_run"
    assert summary["coverage"]["measured_layers"] == 3
    markdown = render_markdown(summary)
    assert "Tool执行 | 已实测" in markdown
    assert "端到端Agent链路 | 未运行" in markdown


def test_markdown_labels_unrun_metrics_instead_of_presenting_fake_scores(tmp_path):
    results = tmp_path / "results"
    _write(results / "intent_robustness_latest.json", {
        "dataset": {"total": 1, "validation": 0, "test": 1},
        "classification_quality": {
            "deepseek_llm_only": {"macro_f1": 1.0, "exact_match": 1.0},
            "bge_plus_pattern": {"macro_f1": 1.0, "exact_match": 1.0},
        },
        "fault_ablation": {"failure_rate_100": {"local_fallback": {
            "response_availability": 1.0, "routing_exact_match_mean": 1.0,
            "failed_request_recovery_rate": 1.0,
        }}},
    })
    _write(results / "expanded_bge_baseline_latest.json", {
        "dataset_case_count": 1, "answerable_case_count": 1,
        "unanswerable_case_count": 0, "corpus_document_count": 1,
        "embedding_model": "bge", "variants": {"bge_vector_top_3": {
            "metrics": {"retrieval_hit_rate": 1.0, "retrieval_recall_at_k": 1.0,
                        "retrieval_mrr": 1.0, "retrieval_ndcg_at_k": 1.0},
            "chroma_query_latency_ms": {"p95_ms": 1.0},
        }},
    })

    markdown = render_markdown(build_summary(results))

    assert "端到端Agent链路 | 未运行" in markdown
    assert "RAGAS生成质量 | 未运行" in markdown
    assert "Faithfulness：100.00%" not in markdown


def test_generated_markdown_is_utf8_chinese(tmp_path):
    results = tmp_path / "results"
    _write(results / "intent_robustness_latest.json", {
        "dataset": {"total": 1, "validation": 0, "test": 1},
        "classification_quality": {
            "deepseek_llm_only": {"macro_f1": 1.0, "exact_match": 1.0},
            "bge_plus_pattern": {"macro_f1": 1.0, "exact_match": 1.0},
        },
        "fault_ablation": {"failure_rate_100": {"local_fallback": {
            "response_availability": 1.0, "routing_exact_match_mean": 1.0,
            "failed_request_recovery_rate": 1.0,
        }}},
    })
    _write(results / "expanded_bge_baseline_latest.json", {
        "dataset_case_count": 1, "answerable_case_count": 1,
        "unanswerable_case_count": 0, "corpus_document_count": 1,
        "embedding_model": "bge", "variants": {"bge_vector_top_3": {
            "metrics": {"retrieval_hit_rate": 1.0, "retrieval_recall_at_k": 1.0,
                        "retrieval_mrr": 1.0, "retrieval_ndcg_at_k": 1.0},
            "chroma_query_latency_ms": {"p95_ms": 1.0},
        }},
    })

    payload = render_markdown(build_summary(results)).encode("utf-8")

    assert payload.decode("utf-8").startswith("# EchoMind 统一评测总报告")


def test_summary_includes_fair_three_way_fusion_ab_with_prompt_boundary(tmp_path):
    results = tmp_path / "results"
    _write(results / "intent_robustness_latest.json", {
        "dataset": {"total": 1100, "validation": 1000, "test": 100},
        "classification_quality": {
            "deepseek_llm_only": {"macro_f1": 0.986105, "exact_match": 0.98},
            "bge_plus_pattern": {"macro_f1": 0.908882, "exact_match": 0.91},
        },
        "fault_ablation": {"failure_rate_100": {"local_fallback": {
            "response_availability": 1.0, "routing_exact_match_mean": 0.91,
            "failed_request_recovery_rate": 0.91,
        }}},
    })
    _write(results / "expanded_bge_baseline_latest.json", {
        "dataset_case_count": 1, "answerable_case_count": 1,
        "unanswerable_case_count": 0, "corpus_document_count": 1,
        "embedding_model": "bge", "variants": {"bge_vector_top_3": {
            "metrics": {"retrieval_hit_rate": 1.0, "retrieval_recall_at_k": 1.0,
                        "retrieval_mrr": 1.0, "retrieval_ndcg_at_k": 1.0},
            "chroma_query_latency_ms": {"p95_ms": 1.0},
        }},
    })
    _write(results / "intent_fusion_ab_latest.json", {
        "selection_case_ids": ["v1", "v2"],
        "test_case_ids": ["t1", "t2"],
        "variants": {
            "deepseek_only": {
                "config": {"mode": "direct multi-label model output"},
                "metrics": {"macro_f1": 0.932351, "exact_match": 0.92},
            },
            "fixed_70_20_10": {
                "config": {"llm_weight": 0.7, "embedding_weight": 0.2, "pattern_weight": 0.1},
                "metrics": {"macro_f1": 0.924714, "exact_match": 0.92},
            },
            "grid_search_best_true_three_way": {
                "config": {"llm_weight": 0.8, "embedding_weight": 0.05, "pattern_weight": 0.15},
                "metrics": {"macro_f1": 0.948413, "exact_match": 0.94},
                "delta_vs_deepseek": {"macro_f1": 0.016062, "exact_match": 0.02},
            },
        },
        "error_analysis": {"grid_search_best_true_three_way": {
            "rescued_case_ids": ["a", "b", "c", "d"],
            "regressed_case_ids": ["e", "f"],
        }},
        "methodology": {"llm_model": "deepseek-v4-flash"},
    })

    summary = build_summary(results)
    markdown = render_markdown(summary)

    assert summary["layers"]["routing"]["fusion_ab"]["selected_weights"] == {
        "llm": 0.8, "embedding": 0.05, "pattern": 0.15,
    }
    assert summary["layers"]["routing"]["fusion_ab"]["macro_f1_delta"] == 0.016062
    assert "同提示词公平A/B" in markdown
    assert "94.84%" in markdown
    assert "不能与前述98.61%直接比较" in markdown
