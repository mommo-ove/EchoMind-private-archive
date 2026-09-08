from evaluation.intent_calibration import (
    CalibrationConstraints,
    FusionConfig,
    evaluate_config,
    grid_search,
    calibrate_and_evaluate,
    audit_constraints,
)


CASES = [
    {
        "case_id": "tech-1",
        "expected_intents": ["technical"],
        "scores": {
            "llm": {"technical": 0.9},
            "embedding": {"technical": 0.7},
            "pattern": {"technical": 1.0},
        },
    },
    {
        "case_id": "billing-1",
        "expected_intents": ["billing"],
        "scores": {
            "llm": {"billing": 0.9},
            "embedding": {"billing": 0.7},
            "pattern": {"billing": 1.0},
        },
    },
    {
        "case_id": "escalation-1",
        "expected_intents": ["escalation"],
        "scores": {
            "llm": {"complaint": 0.8, "escalation": 0.2},
            "embedding": {"escalation": 0.9},
            "pattern": {"escalation": 1.0},
        },
    },
]


def test_evaluate_config_reports_multilabel_metrics_and_business_recall():
    report = evaluate_config(
        CASES,
        FusionConfig(0.4, 0.3, 0.3, confidence_threshold=0.5, multi_label_threshold=0.5),
    )

    assert report["total"] == 3
    assert report["per_class"]["technical"]["recall"] == 1.0
    assert report["per_class"]["billing"]["recall"] == 1.0
    assert report["per_class"]["escalation"]["recall"] == 1.0
    assert report["macro_f1"] == 1.0


def test_grid_search_rejects_candidate_that_breaks_escalation_recall_floor():
    report = grid_search(
        CASES,
        weight_values=[0.0, 0.5, 1.0],
        confidence_thresholds=[0.5],
        multi_label_thresholds=[0.5],
        constraints=CalibrationConstraints(
            minimum_recall={"escalation": 1.0},
            minimum_precision={"escalation": 1.0},
        ),
        baseline=FusionConfig(0.7, 0.2, 0.1, 0.5, 0.5),
    )

    assert report["feasible_candidate_count"] > 0
    assert report["best"]["metrics"]["per_class"]["escalation"]["recall"] == 1.0
    assert report["best"]["config"]["llm_weight"] < 1.0


def test_primary_accuracy_requires_expected_primary_not_any_secondary_label():
    case = {
        "case_id": "multi",
        "expected_intent": "technical",
        "expected_intents": ["technical", "billing"],
        "scores": {
            "llm": {"billing": 1.0, "technical": 0.8},
            "embedding": {},
            "pattern": {},
        },
    }

    report = evaluate_config([case], FusionConfig(1.0, 0.0, 0.0, 0.5, 0.5))

    assert report["accuracy"] == 0.0
    assert report["subset_accuracy"] == 1.0


def test_grid_search_requires_all_three_strategy_scores():
    incomplete = [{
        "case_id": "bad",
        "expected_intents": ["technical"],
        "scores": {"embedding": {"technical": 0.8}, "pattern": {}},
    }]

    try:
        grid_search(incomplete)
    except ValueError as error:
        assert "llm" in str(error)
    else:
        raise AssertionError("missing LLM scores must not produce a three-way report")


def test_calibration_selects_on_validation_then_scores_heldout_test():
    report = calibrate_and_evaluate(
        validation_cases=CASES,
        test_cases=CASES[:2],
        weight_values=[0.0, 0.5, 1.0],
        confidence_thresholds=[0.5],
        multi_label_thresholds=[0.5],
        constraints=CalibrationConstraints(minimum_recall={"escalation": 1.0}),
    )

    assert report["selection_split"] == "validation"
    assert report["final_evaluation_split"] == "test"
    assert report["validation"]["best"]["metrics"]["total"] == 3
    assert report["test"]["metrics"]["total"] == 2


def test_constraint_audit_lists_heldout_business_failures():
    constraints = CalibrationConstraints(
        minimum_recall={"escalation": 0.95, "technical": 0.8},
        minimum_precision={"escalation": 0.8},
    )
    metrics = {
        "per_class": {
            "escalation": {"recall": 0.6, "precision": 1.0},
            "technical": {"recall": 0.7, "precision": 1.0},
        }
    }

    audit = audit_constraints(metrics, constraints)

    assert audit["passed"] is False
    assert audit["violations"] == [
        {"label": "escalation", "metric": "recall", "required": 0.95, "actual": 0.6},
        {"label": "technical", "metric": "recall", "required": 0.8, "actual": 0.7},
    ]
