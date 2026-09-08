"""Offline calibration of three-way intent fusion weights and thresholds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any, Dict, Mapping, Sequence


STRATEGIES = ("llm", "embedding", "pattern")


@dataclass(frozen=True)
class FusionConfig:
    llm_weight: float
    embedding_weight: float
    pattern_weight: float
    confidence_threshold: float = 0.5
    multi_label_threshold: float = 0.6

    @property
    def weights(self) -> Dict[str, float]:
        return {
            "llm": self.llm_weight,
            "embedding": self.embedding_weight,
            "pattern": self.pattern_weight,
        }


@dataclass(frozen=True)
class CalibrationConstraints:
    minimum_recall: Mapping[str, float] = field(default_factory=dict)
    minimum_precision: Mapping[str, float] = field(default_factory=dict)


def _fuse(case: Mapping[str, Any], config: FusionConfig) -> tuple[str, set[str]]:
    fused: Dict[str, float] = {}
    for strategy, weight in config.weights.items():
        for label, raw_score in case["scores"][strategy].items():
            fused[label] = fused.get(label, 0.0) + weight * float(raw_score)
    if not fused:
        return "other", set()
    primary = max(fused, key=fused.get)
    if fused[primary] < config.confidence_threshold:
        primary = "other"
    matched = {
        label for label, score in fused.items()
        if label != "other" and score >= config.multi_label_threshold
    }
    if not matched and primary != "other":
        matched = {primary}
    return primary, matched


def evaluate_config(
    cases: Sequence[Mapping[str, Any]],
    config: FusionConfig,
) -> Dict[str, Any]:
    labels = sorted({
        label
        for case in cases
        for label in case["expected_intents"]
    })
    expected_sets = [set(case["expected_intents"]) for case in cases]
    predicted_sets = []
    primary_correct = 0
    for case, expected in zip(cases, expected_sets):
        primary, predicted = _fuse(case, config)
        predicted_sets.append(predicted)
        expected_primary = case.get("expected_intent") or next(iter(expected), "other")
        primary_correct += int(primary == expected_primary)

    per_class: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = sum(label in p and label in e for p, e in zip(predicted_sets, expected_sets))
        fp = sum(label in p and label not in e for p, e in zip(predicted_sets, expected_sets))
        fn = sum(label not in p and label in e for p, e in zip(predicted_sets, expected_sets))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    macro_f1 = sum(row["f1"] for row in per_class.values()) / len(per_class) if per_class else 0.0
    subset_accuracy = (
        sum(predicted == expected for predicted, expected in zip(predicted_sets, expected_sets))
        / len(cases)
        if cases else 0.0
    )
    return {
        "total": len(cases),
        "accuracy": round(primary_correct / len(cases), 6) if cases else 0.0,
        "subset_accuracy": round(subset_accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "per_class": per_class,
    }


def _validate(cases: Sequence[Mapping[str, Any]]) -> None:
    if not cases:
        raise ValueError("at least one calibration case is required")
    for case in cases:
        for strategy in STRATEGIES:
            if strategy not in case.get("scores", {}):
                raise ValueError(f"case {case.get('case_id')} is missing {strategy} scores")


def _satisfies(metrics: Mapping[str, Any], constraints: CalibrationConstraints) -> bool:
    per_class = metrics["per_class"]
    recall_ok = all(
        per_class.get(label, {}).get("recall", 0.0) >= floor
        for label, floor in constraints.minimum_recall.items()
    )
    precision_ok = all(
        per_class.get(label, {}).get("precision", 0.0) >= floor
        for label, floor in constraints.minimum_precision.items()
    )
    return recall_ok and precision_ok


def audit_constraints(
    metrics: Mapping[str, Any],
    constraints: CalibrationConstraints,
) -> Dict[str, Any]:
    per_class = metrics["per_class"]
    violations = []
    for metric_name, floors in (
        ("recall", constraints.minimum_recall),
        ("precision", constraints.minimum_precision),
    ):
        for label, floor in floors.items():
            actual = per_class.get(label, {}).get(metric_name, 0.0)
            if actual < floor:
                violations.append({
                    "label": label,
                    "metric": metric_name,
                    "required": floor,
                    "actual": actual,
                })
    return {"passed": not violations, "violations": violations}


def grid_search(
    cases: Sequence[Mapping[str, Any]],
    *,
    weight_values: Sequence[float] = tuple(i / 10 for i in range(11)),
    confidence_thresholds: Sequence[float] = (0.4, 0.5, 0.6),
    multi_label_thresholds: Sequence[float] = (0.4, 0.5, 0.6, 0.7),
    constraints: CalibrationConstraints = CalibrationConstraints(),
    baseline: FusionConfig = FusionConfig(0.7, 0.2, 0.1),
) -> Dict[str, Any]:
    _validate(cases)
    candidates = []
    for llm, embedding, pattern in product(weight_values, repeat=3):
        if abs(llm + embedding + pattern - 1.0) > 1e-9:
            continue
        for confidence, multi_label in product(confidence_thresholds, multi_label_thresholds):
            config = FusionConfig(llm, embedding, pattern, confidence, multi_label)
            metrics = evaluate_config(cases, config)
            if _satisfies(metrics, constraints):
                candidates.append({"config": asdict(config), "metrics": metrics})
    if not candidates:
        raise ValueError("no candidate satisfies the business constraints")
    candidates.sort(
        key=lambda row: (
            row["metrics"]["macro_f1"],
            row["metrics"]["subset_accuracy"],
            row["metrics"]["accuracy"],
            -row["config"]["llm_weight"],
        ),
        reverse=True,
    )
    baseline_metrics = evaluate_config(cases, baseline)
    return {
        "candidate_count": sum(
            1 for weights in product(weight_values, repeat=3)
            if abs(sum(weights) - 1.0) <= 1e-9
        ) * len(confidence_thresholds) * len(multi_label_thresholds),
        "feasible_candidate_count": len(candidates),
        "constraints": {
            "minimum_recall": dict(constraints.minimum_recall),
            "minimum_precision": dict(constraints.minimum_precision),
        },
        "baseline": {"config": asdict(baseline), "metrics": baseline_metrics},
        "best": candidates[0],
        "delta": {
            "macro_f1": round(candidates[0]["metrics"]["macro_f1"] - baseline_metrics["macro_f1"], 6),
            "subset_accuracy": round(candidates[0]["metrics"]["subset_accuracy"] - baseline_metrics["subset_accuracy"], 6),
        },
    }


def calibrate_and_evaluate(
    *,
    validation_cases: Sequence[Mapping[str, Any]],
    test_cases: Sequence[Mapping[str, Any]],
    weight_values: Sequence[float] = tuple(i / 10 for i in range(11)),
    confidence_thresholds: Sequence[float] = (0.4, 0.5, 0.6),
    multi_label_thresholds: Sequence[float] = (0.4, 0.5, 0.6, 0.7),
    constraints: CalibrationConstraints = CalibrationConstraints(),
    baseline: FusionConfig = FusionConfig(0.7, 0.2, 0.1),
) -> Dict[str, Any]:
    validation = grid_search(
        validation_cases,
        weight_values=weight_values,
        confidence_thresholds=confidence_thresholds,
        multi_label_thresholds=multi_label_thresholds,
        constraints=constraints,
        baseline=baseline,
    )
    selected = FusionConfig(**validation["best"]["config"])
    test_metrics = evaluate_config(test_cases, selected)
    return {
        "selection_split": "validation",
        "final_evaluation_split": "test",
        "validation": validation,
        "test": {
            "config": asdict(selected),
            "metrics": test_metrics,
            "baseline_metrics": evaluate_config(test_cases, baseline),
            "constraint_audit": audit_constraints(test_metrics, constraints),
        },
    }
