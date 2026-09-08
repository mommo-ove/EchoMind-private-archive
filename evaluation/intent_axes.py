"""Evaluation primitives for independent domain, action and escalation axes."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


DOMAINS = frozenset({"general", "technical", "billing", "account"})
ACTIONS = frozenset({"greeting", "query", "request", "report", "complaint", "feedback", "other"})
ESCALATION_MARKERS = (
    "转人工", "接人工", "真人客服", "找真人", "人工客服", "找负责人",
    "联系负责人", "值班主管", "联系老师", "通知老师", "升级处理",
)


def _normalize_domains(values: Sequence[str]) -> list[str]:
    domains = list(dict.fromkeys(str(value) for value in values))
    if not domains or any(value not in DOMAINS for value in domains):
        raise ValueError("domains must contain known values")
    if "general" in domains and len(domains) > 1:
        raise ValueError("general cannot be combined with specialist domains")
    return domains


def validate_case(case: Mapping[str, Any]) -> dict[str, Any]:
    required = {"case_id", "split", "message", "expected"}
    if not required.issubset(case):
        raise ValueError("case is missing required fields")
    expected = case["expected"]
    if not isinstance(expected, Mapping):
        raise ValueError("expected must be an object")
    _normalize_domains(expected.get("domains", []))
    if expected.get("action") not in ACTIONS:
        raise ValueError("unknown action")
    if type(expected.get("escalated")) is not bool:
        raise ValueError("escalated must be boolean")
    return dict(case)


def normalize_prediction(prediction: Mapping[str, Any]) -> dict[str, Any]:
    raw_domains = prediction.get("domains", ["general"])
    try:
        domains = _normalize_domains(raw_domains if isinstance(raw_domains, list) else [raw_domains])
    except ValueError:
        domains = ["general"]
    action = prediction.get("action", "other")
    if action not in ACTIONS:
        action = "other"
    escalated = prediction.get("escalated", False)
    if type(escalated) is not bool:
        escalated = False
    return {"domains": domains, "action": action, "escalated": escalated}


def apply_safety_rules(message: str, prediction: Mapping[str, Any]) -> dict[str, Any]:
    enhanced = deepcopy(normalize_prediction(prediction))
    normalized = str(message).lower()
    if any(marker in normalized for marker in ESCALATION_MARKERS):
        enhanced["escalated"] = True
    return enhanced


def _binary_metrics(expected: Sequence[bool], predicted: Sequence[bool]) -> dict[str, float]:
    tp = sum(e and p for e, p in zip(expected, predicted))
    fp = sum(not e and p for e, p in zip(expected, predicted))
    fn = sum(e and not p for e, p in zip(expected, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}


def _multiclass_metrics(expected: Sequence[str], predicted: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    per_class = {}
    for label in labels:
        per_class[label] = _binary_metrics(
            [value == label for value in expected],
            [value == label for value in predicted],
        )
    macro = sum(row["f1"] for row in per_class.values()) / len(per_class) if per_class else 0.0
    accuracy = sum(e == p for e, p in zip(expected, predicted)) / len(expected) if expected else 0.0
    return {"accuracy": round(accuracy, 6), "macro_f1": round(macro, 6), "per_class": per_class}


def evaluate_predictions(cases: Sequence[Mapping[str, Any]], predictions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    expected_domains, predicted_domains = [], []
    expected_actions, predicted_actions = [], []
    expected_escalated, predicted_escalated = [], []
    exact = 0
    for case in cases:
        validate_case(case)
        expected = normalize_prediction(case["expected"])
        predicted = normalize_prediction(predictions.get(case["case_id"], {}))
        expected_set, predicted_set = set(expected["domains"]), set(predicted["domains"])
        expected_domains.append(expected_set)
        predicted_domains.append(predicted_set)
        expected_actions.append(expected["action"])
        predicted_actions.append(predicted["action"])
        expected_escalated.append(expected["escalated"])
        predicted_escalated.append(predicted["escalated"])
        exact += int(expected == predicted)

    domain_per_class = {}
    for label in sorted(DOMAINS):
        domain_per_class[label] = _binary_metrics(
            [label in values for values in expected_domains],
            [label in values for values in predicted_domains],
        )
    domain_macro = sum(row["f1"] for row in domain_per_class.values()) / len(domain_per_class)
    domain_subset = sum(e == p for e, p in zip(expected_domains, predicted_domains)) / len(cases) if cases else 0.0
    return {
        "total": len(cases),
        "domain": {
            "subset_accuracy": round(domain_subset, 6),
            "macro_f1": round(domain_macro, 6),
            "per_class": domain_per_class,
        },
        "action": _multiclass_metrics(expected_actions, predicted_actions, sorted(ACTIONS)),
        "escalation": _binary_metrics(expected_escalated, predicted_escalated),
        "exact_match": round(exact / len(cases), 6) if cases else 0.0,
    }


def evaluate_routing_intents(
    cases: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Score only specialist intents that map directly to runnable agents."""
    labels = ["technical", "billing", "account"]
    per_class = {}
    exact = 0
    for label in labels:
        expected = [label in case["expected"]["domains"] for case in cases]
        predicted = [
            label in normalize_prediction(predictions.get(case["case_id"], {}))["domains"]
            for case in cases
        ]
        metrics = _binary_metrics(expected, predicted)
        metrics["support"] = sum(expected)
        per_class[label] = metrics
    for case in cases:
        expected = {label for label in case["expected"]["domains"] if label in labels}
        predicted = {
            label
            for label in normalize_prediction(predictions.get(case["case_id"], {}))["domains"]
            if label in labels
        }
        exact += int(expected == predicted)
    return {
        "labels": labels,
        "total": len(cases),
        "exact_match": round(exact / len(cases), 6) if cases else 0.0,
        "macro_f1": round(sum(row["f1"] for row in per_class.values()) / len(labels), 6),
        "per_class": per_class,
    }


def audit_business_gates(metrics: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "domain.macro_f1": 0.85,
        "action.macro_f1": 0.80,
        "escalation.recall": 0.95,
        "escalation.precision": 0.85,
    }
    violations = []
    for name, required in gates.items():
        section, metric = name.split(".")
        actual = float(metrics.get(section, {}).get(metric, 0.0))
        if actual < required:
            violations.append({"metric": name, "required": required, "actual": actual})
    return {"passed": not violations, "gates": gates, "violations": violations}
