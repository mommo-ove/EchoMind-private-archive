import json

import pytest


def _build_cases():
    try:
        from tools.build_hallucination_golden import build_cases
    except ModuleNotFoundError as exc:
        pytest.fail(f"hallucination dataset builder is missing: {exc}")
    return build_cases()


def test_adversarial_dataset_is_large_stratified_and_deterministic():
    first = _build_cases()
    second = _build_cases()

    assert first == second
    assert len(first) == 200
    assert len({case["id"] for case in first}) == 200
    assert {case["split"] for case in first} == {"train", "validation", "test"}
    assert sum(case["split"] == "test" for case in first) == 40
    assert sum(case["split"] == "validation" for case in first) == 40
    assert sum(case["split"] == "train" for case in first) == 120
    assert {case["type"] for case in first} >= {
        "numeric_mismatch",
        "entity_fabrication",
        "status_mismatch",
        "missing_citation",
        "unsupported_causal_claim",
        "supported",
    }
    assert any(case["has_hallucination"] for case in first)
    assert any(not case["has_hallucination"] for case in first)


def test_checked_in_adversarial_dataset_matches_builder():
    path = "data/eval/hallucination_adversarial.json"
    try:
        payload = json.loads(open(path, encoding="utf-8").read())
    except FileNotFoundError:
        pytest.fail(f"checked-in dataset is missing: {path}")

    assert payload["metadata"]["synthetic"] is True
    assert payload["cases"] == _build_cases()
