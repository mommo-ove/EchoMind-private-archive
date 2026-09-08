"""Fair held-out comparison for multi-label intent fusion variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Mapping, Sequence

from evaluation.intent_axes import evaluate_routing_intents


ROUTING_LABELS = ("technical", "billing", "account")


@dataclass(frozen=True)
class FusionConfig:
    llm_weight: float
    embedding_weight: float
    pattern_weight: float
    thresholds: Mapping[str, float]

    def __post_init__(self) -> None:
        weights = (self.llm_weight, self.embedding_weight, self.pattern_weight)
        if any(weight < 0 for weight in weights):
            raise ValueError("fusion weights must be non-negative")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("fusion weights must sum to one")
        if set(self.thresholds) != set(ROUTING_LABELS):
            raise ValueError("thresholds must cover every routing label")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TuningResult:
    config: FusionConfig
    selection_case_ids: list[str]
    validation_metrics: Mapping[str, Any]
    candidate_count: int


def _prediction(domains: Sequence[str], scores: Mapping[str, float] | None = None) -> dict[str, Any]:
    selected = [label for label in ROUTING_LABELS if label in domains]
    result: dict[str, Any] = {
        "domains": selected or ["general"],
        "action": "other",
        "escalated": False,
    }
    if scores is not None:
        result["scores"] = {label: round(float(scores.get(label, 0.0)), 6) for label in ROUTING_LABELS}
    return result


def fused_scores(row: Mapping[str, Any], config: FusionConfig) -> dict[str, float]:
    weights = {
        "llm": config.llm_weight,
        "embedding": config.embedding_weight,
        "pattern": config.pattern_weight,
    }
    return {
        label: round(sum(
            weights[component] * float(row.get(component, {}).get(label, 0.0))
            for component in weights
        ), 6)
        for label in ROUTING_LABELS
    }


def fuse_predictions(
    score_rows: Mapping[str, Mapping[str, Any]],
    config: FusionConfig,
) -> dict[str, dict[str, Any]]:
    predictions = {}
    for case_id, row in score_rows.items():
        scores = fused_scores(row, config)
        selected = [
            label for label in ROUTING_LABELS
            if scores[label] >= float(config.thresholds[label])
        ]
        predictions[case_id] = _prediction(selected, scores)
    return predictions


def _binary_f1(expected: Sequence[bool], predicted: Sequence[bool]) -> float:
    tp = sum(e and p for e, p in zip(expected, predicted))
    fp = sum(not e and p for e, p in zip(expected, predicted))
    fn = sum(e and not p for e, p in zip(expected, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _weight_triples(values: Sequence[float], *, require_all_components: bool) -> list[tuple[float, float, float]]:
    triples = []
    for llm, embedding, pattern in product(values, repeat=3):
        if abs(llm + embedding + pattern - 1.0) > 1e-9:
            continue
        if require_all_components and min(llm, embedding, pattern) <= 0:
            continue
        triples.append((llm, embedding, pattern))
    if not triples:
        raise ValueError("weight grid contains no feasible triples")
    return triples


def tune_fusion(
    cases: Sequence[Mapping[str, Any]],
    score_rows: Mapping[str, Mapping[str, Any]],
    *,
    weight_values: Sequence[float] = tuple(value / 20 for value in range(21)),
    threshold_values: Sequence[float] = tuple(value / 20 for value in range(1, 20)),
    require_all_components: bool = False,
) -> TuningResult:
    """Choose weights and per-label thresholds using validation rows only."""
    validation = [case for case in cases if case["split"] == "validation"]
    if not validation:
        raise ValueError("validation cases are required")
    missing = {case["case_id"] for case in validation} - score_rows.keys()
    if missing:
        raise ValueError(f"score rows are missing {len(missing)} validation cases")

    best: tuple[tuple[float, ...], FusionConfig, Mapping[str, Any]] | None = None
    triples = _weight_triples(weight_values, require_all_components=require_all_components)
    for llm, embedding, pattern in triples:
        raw_config = FusionConfig(
            llm_weight=llm,
            embedding_weight=embedding,
            pattern_weight=pattern,
            thresholds={label: 0.5 for label in ROUTING_LABELS},
        )
        rows = {case["case_id"]: fused_scores(score_rows[case["case_id"]], raw_config) for case in validation}
        thresholds: dict[str, float] = {}
        for label in ROUTING_LABELS:
            expected = [label in case["expected"]["domains"] for case in validation]
            candidates = []
            for threshold in threshold_values:
                predicted = [rows[case["case_id"]][label] >= threshold for case in validation]
                candidates.append((_binary_f1(expected, predicted), float(threshold)))
            _, thresholds[label] = max(candidates, key=lambda item: (item[0], item[1]))
        config = FusionConfig(llm, embedding, pattern, thresholds)
        predictions = fuse_predictions(
            {case["case_id"]: score_rows[case["case_id"]] for case in validation},
            config,
        )
        metrics = evaluate_routing_intents(validation, predictions)
        # Prefer validation quality first. When tied, prefer more LLM evidence so an
        # apparently "best fusion" cannot be manufactured by arbitrary tie-breaking.
        rank = (
            float(metrics["macro_f1"]),
            float(metrics["exact_match"]),
            llm,
            min(llm, embedding, pattern),
        )
        if best is None or rank > best[0]:
            best = (rank, config, metrics)

    assert best is not None
    return TuningResult(
        config=best[1],
        selection_case_ids=[case["case_id"] for case in validation],
        validation_metrics=best[2],
        candidate_count=len(triples) * len(threshold_values) * len(ROUTING_LABELS),
    )


def _evaluate_config(
    test_cases: Sequence[Mapping[str, Any]],
    score_rows: Mapping[str, Mapping[str, Any]],
    config: FusionConfig,
) -> dict[str, Any]:
    subset = {case["case_id"]: score_rows[case["case_id"]] for case in test_cases}
    return evaluate_routing_intents(test_cases, fuse_predictions(subset, config))


def _predict_config(
    test_cases: Sequence[Mapping[str, Any]],
    score_rows: Mapping[str, Mapping[str, Any]],
    config: FusionConfig,
) -> dict[str, dict[str, Any]]:
    subset = {case["case_id"]: score_rows[case["case_id"]] for case in test_cases}
    return fuse_predictions(subset, config)


def _routing_set(prediction: Mapping[str, Any]) -> set[str]:
    return {label for label in prediction.get("domains", []) if label in ROUTING_LABELS}


def compare_fusion_variants(
    cases: Sequence[Mapping[str, Any]],
    score_rows: Mapping[str, Mapping[str, Any]],
    *,
    weight_values: Sequence[float] = tuple(value / 20 for value in range(21)),
    threshold_values: Sequence[float] = tuple(value / 20 for value in range(1, 20)),
) -> dict[str, Any]:
    test_cases = [case for case in cases if case["split"] == "test"]
    if not test_cases:
        raise ValueError("test cases are required")
    all_ids = {case["case_id"] for case in cases}
    missing = all_ids - score_rows.keys()
    if missing:
        raise ValueError(f"score rows are missing {len(missing)} cases")

    best = tune_fusion(
        cases,
        score_rows,
        weight_values=weight_values,
        threshold_values=threshold_values,
    )
    best_three_way = tune_fusion(
        cases,
        score_rows,
        weight_values=weight_values,
        threshold_values=threshold_values,
        require_all_components=True,
    )
    fixed_thresholds = {label: 0.6 for label in ROUTING_LABELS}
    runtime = FusionConfig(0.85, 0.0, 0.15, fixed_thresholds)
    fixed = FusionConfig(0.7, 0.2, 0.1, fixed_thresholds)
    without_pattern = FusionConfig(7 / 9, 2 / 9, 0.0, fixed_thresholds)
    without_embedding = FusionConfig(7 / 8, 0.0, 1 / 8, fixed_thresholds)
    without_llm = FusionConfig(0.0, 2 / 3, 1 / 3, fixed_thresholds)

    llm_predictions = {
        case["case_id"]: _prediction(score_rows[case["case_id"]].get("llm_domains", []))
        for case in test_cases
    }
    llm_metrics = evaluate_routing_intents(test_cases, llm_predictions)
    prediction_sets = {
        "deepseek_only": llm_predictions,
        "runtime_85_0_15": _predict_config(test_cases, score_rows, runtime),
        "fixed_70_20_10": _predict_config(test_cases, score_rows, fixed),
        "ablation_without_pattern": _predict_config(test_cases, score_rows, without_pattern),
        "ablation_without_embedding": _predict_config(test_cases, score_rows, without_embedding),
        "ablation_without_llm": _predict_config(test_cases, score_rows, without_llm),
        "grid_search_best": _predict_config(test_cases, score_rows, best.config),
        "grid_search_best_true_three_way": _predict_config(test_cases, score_rows, best_three_way.config),
    }
    variants: dict[str, dict[str, Any]] = {
        "deepseek_only": {
            "config": {"mode": "direct multi-label model output"},
            "metrics": llm_metrics,
        },
        "runtime_85_0_15": {
            "config": runtime.to_dict(),
            "metrics": _evaluate_config(test_cases, score_rows, runtime),
        },
        "fixed_70_20_10": {
            "config": fixed.to_dict(),
            "metrics": _evaluate_config(test_cases, score_rows, fixed),
        },
        "ablation_without_pattern": {
            "config": without_pattern.to_dict(),
            "metrics": _evaluate_config(test_cases, score_rows, without_pattern),
        },
        "ablation_without_embedding": {
            "config": without_embedding.to_dict(),
            "metrics": _evaluate_config(test_cases, score_rows, without_embedding),
        },
        "ablation_without_llm": {
            "config": without_llm.to_dict(),
            "metrics": _evaluate_config(test_cases, score_rows, without_llm),
        },
        "grid_search_best": {
            "config": best.config.to_dict(),
            "validation_metrics": dict(best.validation_metrics),
            "metrics": _evaluate_config(test_cases, score_rows, best.config),
        },
        "grid_search_best_true_three_way": {
            "config": best_three_way.config.to_dict(),
            "validation_metrics": dict(best_three_way.validation_metrics),
            "metrics": _evaluate_config(test_cases, score_rows, best_three_way.config),
        },
    }
    for row in variants.values():
        row["delta_vs_deepseek"] = {
            "macro_f1": round(row["metrics"]["macro_f1"] - llm_metrics["macro_f1"], 6),
            "exact_match": round(row["metrics"]["exact_match"] - llm_metrics["exact_match"], 6),
        }
    baseline_correct = {
        case["case_id"]: _routing_set(llm_predictions[case["case_id"]])
        == {label for label in case["expected"]["domains"] if label in ROUTING_LABELS}
        for case in test_cases
    }
    error_analysis = {}
    for name, predictions in prediction_sets.items():
        if name == "deepseek_only":
            continue
        variant_correct = {
            case["case_id"]: _routing_set(predictions[case["case_id"]])
            == {label for label in case["expected"]["domains"] if label in ROUTING_LABELS}
            for case in test_cases
        }
        error_analysis[name] = {
            "rescued_case_ids": [case_id for case_id in baseline_correct if not baseline_correct[case_id] and variant_correct[case_id]],
            "regressed_case_ids": [case_id for case_id in baseline_correct if baseline_correct[case_id] and not variant_correct[case_id]],
            "changed_case_ids": [
                case["case_id"] for case in test_cases
                if _routing_set(predictions[case["case_id"]]) != _routing_set(llm_predictions[case["case_id"]])
            ],
        }
    return {
        "selection_split": "validation",
        "final_split": "held-out test",
        "selection_case_ids": best.selection_case_ids,
        "test_case_ids": [case["case_id"] for case in test_cases],
        "grid_candidate_count": best.candidate_count,
        "variants": variants,
        "error_analysis": error_analysis,
    }
