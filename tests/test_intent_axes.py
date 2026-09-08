import copy

import pytest

from evaluation.intent_axes import (
    apply_safety_rules,
    audit_business_gates,
    evaluate_predictions,
    evaluate_routing_intents,
    validate_case,
)


def _case(case_id, domains, action, escalated=False):
    return {
        "case_id": case_id,
        "split": "test",
        "message": "示例消息",
        "expected": {
            "domains": domains,
            "action": action,
            "escalated": escalated,
        },
    }


def test_case_schema_keeps_domain_action_and_escalation_independent():
    case = _case("axis-001", ["billing"], "complaint", True)

    assert validate_case(case) == case


@pytest.mark.parametrize(
    "expected",
    [
        {"domains": ["general", "billing"], "action": "query", "escalated": False},
        {"domains": ["unknown"], "action": "query", "escalated": False},
        {"domains": ["billing"], "action": "unknown", "escalated": False},
        {"domains": ["billing"], "action": "query", "escalated": "false"},
    ],
)
def test_case_schema_rejects_ambiguous_or_invalid_labels(expected):
    case = _case("axis-bad", ["billing"], "query")
    case["expected"] = expected

    with pytest.raises(ValueError):
        validate_case(case)


def test_safety_rules_only_force_explicit_handoff_without_mutating_input():
    prediction = {
        "domains": ["technical"],
        "action": "request",
        "escalated": False,
    }
    before = copy.deepcopy(prediction)

    enhanced = apply_safety_rules("宿舍断网了，马上给我转人工", prediction)

    assert enhanced == {
        "domains": ["technical"],
        "action": "request",
        "escalated": True,
    }
    assert prediction == before
    assert apply_safety_rules("宿舍断网了，有点着急", prediction)["escalated"] is False


def test_problem_report_is_a_first_class_action():
    case = _case("axis-report", ["technical"], "report", False)

    assert validate_case(case) == case


def test_axis_metrics_report_each_task_and_strict_exact_match():
    cases = [
        _case("one", ["billing"], "complaint", True),
        _case("two", ["technical"], "query", False),
    ]
    predictions = {
        "one": {"domains": ["billing"], "action": "complaint", "escalated": False},
        "two": {"domains": ["technical"], "action": "query", "escalated": False},
    }

    metrics = evaluate_predictions(cases, predictions)

    assert metrics["domain"]["subset_accuracy"] == 1.0
    assert metrics["action"]["accuracy"] == 1.0
    assert metrics["escalation"]["recall"] == 0.0
    assert metrics["exact_match"] == 0.5


def test_business_gate_audit_explains_failed_metric():
    metrics = {
        "domain": {"macro_f1": 0.90},
        "action": {"macro_f1": 0.82},
        "escalation": {"recall": 0.80, "precision": 0.90},
    }

    audit = audit_business_gates(metrics)

    assert audit["passed"] is False
    assert audit["violations"] == [
        {"metric": "escalation.recall", "required": 0.95, "actual": 0.8}
    ]


def test_routing_metrics_only_score_specialist_agent_intents():
    cases = [
        _case("one", ["technical", "billing"], "report", False),
        _case("two", ["general"], "greeting", False),
        _case("three", ["account"], "report", False),
    ]
    predictions = {
        "one": {"domains": ["technical", "billing"], "action": "report", "escalated": False},
        "two": {"domains": ["general"], "action": "greeting", "escalated": False},
        "three": {"domains": ["account"], "action": "report", "escalated": False},
    }

    metrics = evaluate_routing_intents(cases, predictions)

    assert metrics["labels"] == ["technical", "billing", "account"]
    assert metrics["exact_match"] == 1.0
    assert metrics["macro_f1"] == 1.0
