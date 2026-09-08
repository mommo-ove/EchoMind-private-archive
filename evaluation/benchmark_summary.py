"""Aggregate evidence-backed evaluation layers without manufacturing missing scores."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _unrun(reason: str, *, framework: str) -> dict[str, Any]:
    return {
        "status": "not_run",
        "evidence": "framework_only",
        "framework": framework,
        "reason": reason,
        "metrics": {},
    }


def build_summary(results_dir: Path) -> dict[str, Any]:
    routing_source = results_dir / "intent_robustness_latest.json"
    fusion_source = results_dir / "intent_fusion_ab_latest.json"
    retrieval_source = results_dir / "expanded_bge_baseline_latest.json"
    retrieval_variants_source = results_dir / "retrieval_variants_latest.json"
    full_end_to_end_source = results_dir / "campus_end_to_end_latest.json"
    grounded_end_to_end_source = (
        results_dir / "campus_end_to_end_compact_grounded_latest.json"
    )
    compact_end_to_end_source = results_dir / "campus_end_to_end_compact_latest.json"
    end_to_end_source = (
        full_end_to_end_source
        if full_end_to_end_source.exists()
        else (
            grounded_end_to_end_source
            if grounded_end_to_end_source.exists()
            else compact_end_to_end_source
        )
    )
    end_to_end_scope = (
        "full"
        if end_to_end_source == full_end_to_end_source
        else (
            "compact_representative_grounded"
            if end_to_end_source == grounded_end_to_end_source
            else "compact_representative"
        )
    )
    tool_source = results_dir / "tool_benchmark_latest.json"
    end_to_end_status_source = results_dir / "end_to_end_run_status.json"
    routing = _read(routing_source)
    fusion = _read(fusion_source)
    retrieval = _read(retrieval_source)
    retrieval_variants = _read(retrieval_variants_source)
    end_to_end = _read(end_to_end_source)
    tool_report = _read(tool_source)
    end_to_end_status = _read(end_to_end_status_source)
    if routing is None:
        raise FileNotFoundError(routing_source)
    if retrieval is None:
        raise FileNotFoundError(retrieval_source)

    llm_quality = routing["classification_quality"]["deepseek_llm_only"]
    local_quality = routing["classification_quality"]["bge_plus_pattern"]
    outage = routing["fault_ablation"]["failure_rate_100"]["local_fallback"]
    retrieval_top_3 = retrieval["variants"]["bge_vector_top_3"]
    retrieval_metrics = retrieval_top_3["metrics"]

    layers: dict[str, dict[str, Any]] = {
        "routing": {
            "status": "measured",
            "evidence": "held_out_dataset",
            "source": str(routing_source),
            "dataset": routing["dataset"],
            "llm_macro_f1": llm_quality["macro_f1"],
            "llm_exact_match": llm_quality["exact_match"],
            "local_fallback_macro_f1": local_quality["macro_f1"],
            "full_outage_availability": outage["response_availability"],
            "full_outage_exact_match": outage["routing_exact_match_mean"],
            "full_outage_recovery_rate": outage["failed_request_recovery_rate"],
        },
        "retrieval": {
            "status": "measured",
            "evidence": "http_chroma_benchmark",
            "source": str(retrieval_source),
            "dataset_case_count": retrieval["dataset_case_count"],
            "answerable_case_count": retrieval["answerable_case_count"],
            "unanswerable_case_count": retrieval["unanswerable_case_count"],
            "corpus_document_count": retrieval["corpus_document_count"],
            "embedding_model": retrieval["embedding_model"],
            "top_3": {
                "hit_rate": retrieval_metrics["retrieval_hit_rate"],
                "recall_at_k": retrieval_metrics["retrieval_recall_at_k"],
                "mrr": retrieval_metrics["retrieval_mrr"],
                "ndcg_at_k": retrieval_metrics["retrieval_ndcg_at_k"],
                "p95_ms": retrieval_top_3["chroma_query_latency_ms"]["p95_ms"],
            },
            "query_rewrite": retrieval.get("query_rewrite", {"status": "unknown"}),
        },
    }
    if retrieval_variants is not None:
        layers["retrieval"]["small_sample_ablation"] = {
            "status": "measured_small_sample",
            "source": str(retrieval_variants_source),
            "case_count": retrieval_variants.get("case_count"),
            "variants": retrieval_variants.get("variants", {}),
            "rewrite_valid": bool(
                retrieval_variants.get("configuration", {}).get("rewrite_variant_valid", False)
            ),
            "warning": "Small-sample diagnostic only; does not replace the 140-case baseline.",
        }
    if fusion is not None:
        direct = fusion["variants"]["deepseek_only"]
        fixed = fusion["variants"]["fixed_70_20_10"]
        selected = fusion["variants"]["grid_search_best_true_three_way"]
        config = selected["config"]
        errors = fusion.get("error_analysis", {}).get("grid_search_best_true_three_way", {})
        layers["routing"]["fusion_ab"] = {
            "status": "measured",
            "source": str(fusion_source),
            "selection_case_count": len(fusion.get("selection_case_ids", [])),
            "test_case_count": len(fusion.get("test_case_ids", [])),
            "llm_model": fusion.get("methodology", {}).get("llm_model"),
            "deepseek_only": direct["metrics"],
            "fixed_70_20_10": fixed["metrics"],
            "selected_three_way": selected["metrics"],
            "selected_weights": {
                "llm": config["llm_weight"],
                "embedding": config["embedding_weight"],
                "pattern": config["pattern_weight"],
            },
            "selected_thresholds": config.get("thresholds", {}),
            "macro_f1_delta": selected["delta_vs_deepseek"]["macro_f1"],
            "exact_match_delta": selected["delta_vs_deepseek"]["exact_match"],
            "rescued_case_count": len(errors.get("rescued_case_ids", [])),
            "regressed_case_count": len(errors.get("regressed_case_ids", [])),
            "comparison_boundary": (
                "Uses a domain-confidence prompt shared by every A/B variant; "
                "not directly comparable with the earlier robustness checkpoint prompt."
            ),
        }
    if tool_report is not None:
        layers["tool_execution"] = {
            "status": "measured",
            "evidence": tool_report.get("evidence", "real_tool_component"),
            "source": str(tool_source),
            "case_count": tool_report.get("case_count"),
            "metrics": tool_report.get("metrics", {}),
            "latency_ms": tool_report.get("latency_ms", {}),
            "boundary": "Component benchmark; does not claim full Agent end-to-end success.",
        }

    if end_to_end is None:
        failure = (end_to_end_status or {}).get("failure", {})
        layers["execution"] = _unrun(
            failure.get("reason")
            or "Docker/API was unavailable during aggregation; unit tests are not reported as accuracy.",
            framework="EndToEndEvaluator deterministic route/tool/ticket/knowledge checks",
        )
        layers["generation"] = _unrun(
            failure.get("ragas_reason")
            or "No persisted real RAGAS/LLM-as-Judge baseline is available.",
            framework="RAGAS faithfulness/relevancy/context precision/context recall/correctness",
        )
    else:
        metrics = end_to_end.get("metrics", {})
        layers["execution"] = {
            "status": "measured",
            "evidence": "real_chat_pipeline",
            "source": str(end_to_end_source),
            "scope": end_to_end_scope,
            "case_count": end_to_end.get("total"),
            "route_correctness": metrics.get("route_correctness"),
            "tool_correctness": metrics.get("tool_correctness"),
            "ticket_correctness": metrics.get("ticket_correctness"),
            "knowledge_correctness": metrics.get("knowledge_correctness"),
            "task_completion": metrics.get("task_completion"),
        }
        avg_scores = end_to_end.get("avg_scores", {})
        ragas_metrics = {
            key: (metrics.get(key) if metrics.get(key) is not None else avg_scores.get(key))
            for key in (
                "faithfulness", "answer_relevancy", "context_precision",
                "context_recall", "answer_correctness", "ragas_pass_rate",
            )
            if metrics.get(key) is not None or avg_scores.get(key) is not None
        }
        if ragas_metrics:
            layers["generation"] = {
                "status": "measured",
                "evidence": "ragas_real_answers",
                "source": str(end_to_end_source),
                "case_count": metrics.get("ragas_case_count"),
                "metrics": ragas_metrics,
            }
        else:
            layers["generation"] = _unrun(
                "End-to-end report contains no completed RAGAS cases.",
                framework="RAGAS faithfulness/relevancy/context precision/context recall/correctness",
            )

    measured = sum(layer["status"] == "measured" for layer in layers.values())
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_policy": {
            "measured": "metric produced by a persisted evaluation run",
            "verified": "automated behavior tests passed; not an accuracy metric",
            "not_run": "framework exists but no real persisted run is available",
        },
        "coverage": {
            "total_layers": len(layers),
            "measured_layers": measured,
            "pending_layers": len(layers) - measured,
        },
        "layers": layers,
    }


def _pct(value: Any) -> str:
    return "未运行" if value is None else f"{float(value):.2%}"


def render_markdown(summary: Mapping[str, Any]) -> str:
    routing = summary["layers"]["routing"]
    retrieval = summary["layers"]["retrieval"]
    execution = summary["layers"]["execution"]
    generation = summary["layers"]["generation"]
    tool_execution = summary["layers"].get("tool_execution")
    top_3 = retrieval["top_3"]
    lines = [
        "# EchoMind 统一评测总报告",
        "",
        "## 总览",
        "",
        "| 评测层 | 状态 | 核心结果 | 证据 |",
        "|---|---|---|---|",
        f"| 意图路由 | 已实测 | Macro-F1 {_pct(routing['llm_macro_f1'])}，严格命中 {_pct(routing['llm_exact_match'])} | 独立测试集 |",
        f"| RAG检索 | 已实测 | Hit@3 {_pct(top_3['hit_rate'])}，Recall@3 {_pct(top_3['recall_at_k'])} | HTTP＋ChromaDB |",
        (f"| Tool执行 | 已实测 | 正确率 {_pct(tool_execution['metrics'].get('tool_correctness'))}，安全校验 {_pct(tool_execution['metrics'].get('trusted_context_security'))} | ToolManager＋SQLite |"
         if tool_execution else "| Tool执行 | 未运行 | 暂无组件基线 | - |"),
        (f"| 端到端Agent链路 | 已实测 | Tool正确率 {_pct(execution.get('tool_correctness'))}，任务完成率 {_pct(execution.get('task_completion'))} | 真实聊天链路 |"
         if execution["status"] == "measured" else "| 端到端Agent链路 | 未运行 | 框架已实现，暂无真实落盘基线 | 不使用组件测试冒充端到端准确率 |"),
        (f"| RAGAS生成质量 | 已实测 | Faithfulness {_pct(generation['metrics'].get('faithfulness'))} | 真实回答＋检索上下文 |"
         if generation["status"] == "measured" else "| RAGAS生成质量 | 未运行 | 框架已实现，暂无真实落盘基线 | 不填造Faithfulness |"),
        "",
        "## 一、意图路由与异常降级",
        "",
        f"- 评测数据：{routing['dataset']['total']}条；校准{routing['dataset']['validation']}条，独立测试{routing['dataset']['test']}条。",
        f"- DeepSeek正常路由：Macro-F1 {_pct(routing['llm_macro_f1'])}，严格多标签命中率 {_pct(routing['llm_exact_match'])}。",
        f"- LLM完全不可用：本地BGE＋Pattern保持 {_pct(routing['full_outage_availability'])} 可用率、{_pct(routing['full_outage_exact_match'])} 正确路由率。",
    ]
    fusion_ab = routing.get("fusion_ab")
    if fusion_ab:
        weights = fusion_ab["selected_weights"]
        lines.extend([
            (
                f"- 同提示词公平A/B：DeepSeek-only Macro-F1 {_pct(fusion_ab['deepseek_only']['macro_f1'])}、"
                f"严格命中 {_pct(fusion_ab['deepseek_only']['exact_match'])}；验证集选择的三路融合 "
                f"{weights['llm']:.0%}/{weights['embedding']:.0%}/{weights['pattern']:.0%} 达到 "
                f"Macro-F1 {_pct(fusion_ab['selected_three_way']['macro_f1'])}、"
                f"严格命中 {_pct(fusion_ab['selected_three_way']['exact_match'])}。"
            ),
            (
                f"- 三路融合相对同轮LLM提升 Macro-F1 {_pct(fusion_ab['macro_f1_delta'])}、"
                f"严格命中 {_pct(fusion_ab['exact_match_delta'])}；修复{fusion_ab['rescued_case_count']}题，"
                f"同时回退{fusion_ab['regressed_case_count']}题。"
            ),
            (
                f"- 口径边界：该A/B使用新的逐领域置信度提示词，不能与前述{_pct(routing['llm_macro_f1'])}直接比较；"
                "100条测试集上的小幅提升仍需更大真实数据复验。"
            ),
        ])
    lines.extend([
        "",
        "## 二、RAG检索质量",
        "",
        f"- 数据：{retrieval['dataset_case_count']}题（可回答{retrieval['answerable_case_count']}，不可回答预留{retrieval['unanswerable_case_count']}），知识库{retrieval['corpus_document_count']}篇。",
        f"- 模型：`{retrieval['embedding_model']}`。",
        f"- Top-3：Hit@3 {_pct(top_3['hit_rate'])}，Recall@3 {_pct(top_3['recall_at_k'])}，MRR {top_3['mrr']:.4f}，NDCG@3 {top_3['ndcg_at_k']:.4f}。",
        f"- Chroma查询P95：{top_3['p95_ms']:.3f} ms。",
        "",
        "## 三、端到端执行质量",
        "",
    ])
    if tool_execution:
        metrics = tool_execution["metrics"]
        lines.extend([
            "### Tool组件实测",
            "",
            f"- 用例：{tool_execution['case_count']}条。",
            f"- 正常调用正确率：{_pct(metrics.get('tool_correctness'))}",
            f"- 可信上下文与身份隔离：{_pct(metrics.get('trusted_context_security'))}",
            f"- 幂等建单正确率：{_pct(metrics.get('idempotency_correctness'))}",
            f"- 安全Fallback正确率：{_pct(metrics.get('fallback_correctness'))}",
            f"- P95延迟：{tool_execution['latency_ms'].get('p95', 0.0):.3f} ms。",
            "- 边界：这是ToolManager＋SQLite组件基准，不等于完整Agent任务完成率。",
            "",
            "### 完整Agent端到端",
            "",
        ])
    diagnostic = retrieval.get("small_sample_ablation")
    if diagnostic:
        variants = diagnostic.get("variants", {})
        default = variants.get("default_chroma_vector", {})
        bge = variants.get("bge_zh_vector", {})
        if default and bge:
            lines.insert(lines.index("## 三、端到端执行质量") - 1, (
                f"- 12题诊断消融：默认Chroma Hit@3 {_pct(default.get('retrieval_hit_rate'))}，"
                f"BGE Hit@3 {_pct(bge.get('retrieval_hit_rate'))}；仅作小样本诊断。"
            ))
        if not diagnostic.get("rewrite_valid", False):
            lines.insert(lines.index("## 三、端到端执行质量") - 1, "- 查询改写实验无效：API未成功返回改写结果，不对外声称提升。")
    if execution["status"] == "measured":
        lines.extend([
            f"- Agent路由正确率：{_pct(execution.get('route_correctness'))}",
            f"- Tool调用正确率：{_pct(execution.get('tool_correctness'))}",
            f"- 工单状态正确率：{_pct(execution.get('ticket_correctness'))}",
            f"- 知识使用正确率：{_pct(execution.get('knowledge_correctness'))}",
            f"- 任务完成率：{_pct(execution.get('task_completion'))}",
        ])
    else:
        lines.append(f"- 未运行：{execution['reason']}")
    lines.extend(["", "## 四、RAGAS生成质量", ""])
    if generation["status"] == "measured":
        for name, value in generation["metrics"].items():
            lines.append(f"- {name}：{_pct(value)}")
    else:
        lines.append(f"- 未运行：{generation['reason']}")
    lines.extend([
        "",
        "## 口径说明",
        "",
        "- `已实测`表示存在可追溯的JSON运行结果。",
        "- `未运行`不等于指标为0，只表示当前没有真实基线。",
        "- 单元测试用于验证评测器逻辑，不作为模型准确率或任务成功率。",
        "",
    ])
    return "\n".join(lines)
