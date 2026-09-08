"""Run local intent ablations and deterministic LLM fault-injection experiments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.intent_axes import evaluate_routing_intents
from evaluation.intent_robustness import (
    FaultScenario,
    IntentPrototypeRouter,
    dataset_fingerprint,
    evaluate_fault_scenarios,
    expand_calibration_cases,
    tune_local_router,
)
from mcp.local_embeddings import DEFAULT_EMBEDDING_MODEL, FastEmbedTextModel


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _quality(cases, predictions):
    test_cases = [case for case in cases if case["split"] == "test"]
    return evaluate_routing_intents(test_cases, predictions)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# 意图路由分类、降级与可用性消融报告",
        "",
        "## 实验口径",
        "",
        f"- 扩充后数据：{report['dataset']['total']} 条（校准 {report['dataset']['validation']}，独立测试 {report['dataset']['test']}）",
        f"- 本地 Embedding：`{report['methodology']['embedding_model']}`",
        "- 权重和阈值只在校准集选择，100 条原测试集未改动。",
        "- 扩充数据由确定性话语改写生成，适合稳定性校准，不冒充真实用户标注数据。",
        "",
        "## 一、正常分类质量（独立测试集）",
        "",
        "| 方案 | Macro-F1 | 严格多标签命中率 |",
        "|---|---:|---:|",
    ]
    for name, row in report["classification_quality"].items():
        lines.append(f"| {name} | {row['macro_f1']:.2%} | {row['exact_match']:.2%} |")
    lines.extend([
        "",
        "## 二、异常降级与路由可用性",
        "",
        "| LLM 故障率 | 无降级可用率 | 无降级正确路由率 | 本地降级可用率 | 本地降级正确路由率 | 故障请求正确恢复率 |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["fault_ablation"].values():
        base = row["llm_without_fallback"]
        local = row["local_fallback"]
        lines.append(
            f"| {row['failure_rate']:.0%} | {base['response_availability']:.2%} | "
            f"{base['routing_exact_match_mean']:.2%} | {local['response_availability']:.2%} | "
            f"{local['routing_exact_match_mean']:.2%} | {local['failed_request_recovery_rate']:.2%} |"
        )
    config = report["selected_local_fallback"]
    lines.extend([
        "",
        "## 三、校准得到的本地降级参数",
        "",
        f"- BGE 权重：{config['embedding_weight']}",
        f"- Pattern 权重：{config['pattern_weight']}",
        f"- 各业务标签阈值：`{json.dumps(config['thresholds'], ensure_ascii=False)}`",
        f"- 本地批量推理耗时：{report['latency']['local_total_ms']:.2f} ms；平均 {report['latency']['local_ms_per_case']:.3f} ms/条（不含首次模型加载）。",
        "",
        "## 结论边界",
        "",
        "- LLM 是正常路径，本地 BGE＋Pattern 是独立降级路径，不把固定 70/20/10 当成未经验证的事实。",
        "- 故障恢复率表示本地分支在 LLM 请求失败时，既返回结果又命中正确专业 Agent 的比例。",
        "- 本实验只证明意图路由层；不等于下游 Agent、Tool 和最终回答也全部成功。",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    cases = expand_calibration_cases(
        base_payload["cases"],
        target_validation_count=args.validation_count,
    )
    expanded_payload = {
        **{key: value for key, value in base_payload.items() if key not in {"cases", "case_count"}},
        "description": "Expanded calibration split with an untouched held-out test split.",
        "case_count": len(cases),
        "cases": cases,
    }
    _write_json(args.expanded_dataset, expanded_payload)

    checkpoint = json.loads(args.predictions.read_text(encoding="utf-8"))["predictions"]
    test_ids = {case["case_id"] for case in cases if case["split"] == "test"}
    missing = test_ids - checkpoint.keys()
    if missing:
        raise ValueError(f"LLM checkpoint is missing {len(missing)} held-out predictions")

    embedder = FastEmbedTextModel(model_name=args.embedding_model, cache_dir=args.model_cache)
    router = IntentPrototypeRouter(embedder=embedder)
    start = time.perf_counter()
    pattern = tune_local_router(cases, router=router, candidate_embedding_weights=(0.0,))
    embedding = tune_local_router(cases, router=router, candidate_embedding_weights=(1.0,))
    hybrid = tune_local_router(cases, router=router)
    elapsed_ms = (time.perf_counter() - start) * 1000

    llm_test = {case_id: checkpoint[case_id] for case_id in test_ids}
    quality = {
        "pattern_only": _quality(cases, pattern.predictions),
        "bge_only": _quality(cases, embedding.predictions),
        "bge_plus_pattern": _quality(cases, hybrid.predictions),
        "deepseek_llm_only": _quality(cases, llm_test),
    }
    scenarios = [
        FaultScenario(failure_rate=rate, trials=args.trials, seed=args.seed)
        for rate in (0.0, 0.1, 0.3, 0.5, 1.0)
    ]
    fault_report = evaluate_fault_scenarios(
        cases,
        llm_test,
        hybrid.predictions,
        scenarios=scenarios,
    )
    report = {
        "schema_version": 1,
        "dataset": {
            "total": len(cases),
            "validation": sum(case["split"] == "validation" for case in cases),
            "test": sum(case["split"] == "test" for case in cases),
            "fingerprint": dataset_fingerprint(cases),
            "heldout_test_untouched": True,
        },
        "methodology": {
            "embedding_model": args.embedding_model,
            "selection_split": "expanded validation only",
            "final_split": "100-row untouched held-out test",
            "fault_trials_per_rate": args.trials,
            "fault_seed": args.seed,
        },
        "classification_quality": quality,
        "validation_selection": {
            "pattern_only_macro_f1": pattern.validation_macro_f1,
            "bge_only_macro_f1": embedding.validation_macro_f1,
            "bge_plus_pattern_macro_f1": hybrid.validation_macro_f1,
        },
        "selected_local_fallback": hybrid.config.to_dict(),
        "fault_ablation": fault_report,
        "latency": {
            "local_total_ms": round(elapsed_ms, 3),
            "local_ms_per_case": round(elapsed_ms / len(cases), 6),
            "note": "three local variants; excludes first model construction and download",
        },
    }
    _write_json(args.output, report)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/intent_axes_golden.json"))
    parser.add_argument("--expanded-dataset", type=Path, default=Path("data/eval/intent_robustness_golden.json"))
    parser.add_argument("--predictions", type=Path, default=Path("data/eval/results/intent_axes_predictions.json"))
    parser.add_argument("--output", type=Path, default=Path("data/eval/results/intent_robustness_latest.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/意图路由鲁棒性消融报告.md"))
    parser.add_argument("--validation-count", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--model-cache", default=None)
    return parser.parse_args()


def main() -> None:
    report = run(_parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
