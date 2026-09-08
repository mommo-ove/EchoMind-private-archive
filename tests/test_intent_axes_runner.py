import argparse
import asyncio
import json

import pytest

from tools.run_intent_axes_eval import AXIS_PROMPT_VERSION, SingleRunLock, collect_predictions, evaluate


class FakeClassifier:
    def __init__(self):
        self.calls = []
        self.attempts = 0

    async def classify_batch(self, cases):
        self.calls.append([case["case_id"] for case in cases])
        self.attempts += 1
        if self.attempts == 1:
            raise ValueError("temporary malformed JSON")
        return {
            case["case_id"]: {
                "domains": list(case["expected"]["domains"]),
                "action": case["expected"]["action"],
                "escalated": False,
            }
            for case in cases
        }


class SplitClassifier:
    def __init__(self):
        self.calls = []

    async def classify_batch(self, cases):
        self.calls.append([case["case_id"] for case in cases])
        if len(cases) > 1:
            raise ValueError("large batch returned no JSON")
        case = cases[0]
        return {
            case["case_id"]: {
                "domains": list(case["expected"]["domains"]),
                "action": case["expected"]["action"],
                "escalated": case["expected"]["escalated"],
            }
        }


class ConcurrencyClassifier:
    def __init__(self):
        self.active = 0
        self.maximum_active = 0

    async def classify_batch(self, cases):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1
        return {
            case["case_id"]: {
                "domains": list(case["expected"]["domains"]),
                "action": case["expected"]["action"],
                "escalated": case["expected"]["escalated"],
            }
            for case in cases
        }


def _dataset(path):
    path.write_text(json.dumps({
        "cases": [
            {
                "case_id": "one", "split": "test", "message": "请转人工",
                "expected": {"domains": ["general"], "action": "request", "escalated": True},
            },
            {
                "case_id": "two", "split": "test", "message": "查询校园网",
                "expected": {"domains": ["technical"], "action": "query", "escalated": False},
            },
        ]
    }, ensure_ascii=False), encoding="utf-8")


def test_collection_retries_batches_and_checkpoints(tmp_path):
    dataset = tmp_path / "dataset.json"
    scores = tmp_path / "predictions.json"
    _dataset(dataset)
    classifier = FakeClassifier()
    args = argparse.Namespace(dataset=dataset, predictions=scores, batch_size=2, max_attempts=2)

    result = asyncio.run(collect_predictions(args, classifier=classifier))

    assert result == scores
    assert classifier.calls == [["one", "two"], ["one", "two"]]
    assert len(json.loads(scores.read_text(encoding="utf-8"))["predictions"]) == 2
    assert json.loads(scores.read_text(encoding="utf-8"))["prompt_version"] == AXIS_PROMPT_VERSION


def test_collection_resumes_completed_cases(tmp_path):
    dataset = tmp_path / "dataset.json"
    scores = tmp_path / "predictions.json"
    _dataset(dataset)
    scores.write_text(json.dumps({"prompt_version": AXIS_PROMPT_VERSION, "predictions": {
        "one": {"domains": ["general"], "action": "request", "escalated": False}
    }}), encoding="utf-8")
    classifier = FakeClassifier()
    classifier.attempts = 1
    args = argparse.Namespace(dataset=dataset, predictions=scores, batch_size=2, max_attempts=2)

    asyncio.run(collect_predictions(args, classifier=classifier))

    assert classifier.calls == [["two"]]


def test_collection_discards_checkpoint_from_old_prompt(tmp_path):
    dataset = tmp_path / "dataset.json"
    scores = tmp_path / "predictions.json"
    _dataset(dataset)
    scores.write_text(json.dumps({
        "prompt_version": "old-prompt",
        "predictions": {"one": {"domains": ["general"], "action": "other", "escalated": False}},
    }), encoding="utf-8")
    classifier = ConcurrencyClassifier()
    args = argparse.Namespace(
        dataset=dataset, predictions=scores, batch_size=2, max_attempts=2,
        concurrency=1, collect_split="all",
    )

    asyncio.run(collect_predictions(args, classifier=classifier))

    assert classifier.maximum_active == 1
    assert set(json.loads(scores.read_text(encoding="utf-8"))["predictions"]) == {"one", "two"}


def test_collection_splits_batch_after_retries_are_exhausted(tmp_path):
    dataset = tmp_path / "dataset.json"
    scores = tmp_path / "predictions.json"
    _dataset(dataset)
    classifier = SplitClassifier()
    args = argparse.Namespace(dataset=dataset, predictions=scores, batch_size=2, max_attempts=2)

    asyncio.run(collect_predictions(args, classifier=classifier))

    assert classifier.calls == [["one", "two"], ["one", "two"], ["one"], ["two"]]
    assert set(json.loads(scores.read_text(encoding="utf-8"))["predictions"]) == {"one", "two"}


def test_collection_limits_concurrent_batches(tmp_path):
    dataset = tmp_path / "dataset.json"
    scores = tmp_path / "predictions.json"
    _dataset(dataset)
    classifier = ConcurrencyClassifier()
    args = argparse.Namespace(
        dataset=dataset, predictions=scores, batch_size=1, max_attempts=2, concurrency=2
    )

    asyncio.run(collect_predictions(args, classifier=classifier))

    assert classifier.maximum_active == 2


def test_collection_can_prioritize_heldout_split(tmp_path):
    dataset = tmp_path / "dataset.json"
    scores = tmp_path / "predictions.json"
    _dataset(dataset)
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    payload["cases"][0]["split"] = "validation"
    dataset.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    classifier = ConcurrencyClassifier()
    args = argparse.Namespace(
        dataset=dataset, predictions=scores, batch_size=1, max_attempts=2,
        concurrency=1, collect_split="test",
    )

    asyncio.run(collect_predictions(args, classifier=classifier))

    saved = json.loads(scores.read_text(encoding="utf-8"))["predictions"]
    assert set(saved) == {"two"}


def test_evaluate_compares_pure_llm_with_safety_rules(tmp_path):
    dataset = tmp_path / "dataset.json"
    scores = tmp_path / "predictions.json"
    report = tmp_path / "report.json"
    _dataset(dataset)
    scores.write_text(json.dumps({"predictions": {
        "one": {"domains": ["general"], "action": "request", "escalated": False},
        "two": {"domains": ["technical"], "action": "query", "escalated": False},
    }}, ensure_ascii=False), encoding="utf-8")
    args = argparse.Namespace(dataset=dataset, predictions=scores, output=report)

    evaluate(args)
    payload = json.loads(report.read_text(encoding="utf-8"))

    assert payload["variants"]["llm_only"]["escalation"]["recall"] == 0.0
    assert payload["variants"]["llm_plus_safety_rules"]["escalation"]["recall"] == 1.0
    rule_violations = payload["gate_audits"]["llm_plus_safety_rules"]["violations"]
    assert not any(row["metric"].startswith("escalation.") for row in rule_violations)


def test_evaluate_test_split_does_not_require_validation_predictions(tmp_path):
    dataset = tmp_path / "dataset.json"
    scores = tmp_path / "predictions.json"
    report = tmp_path / "report.json"
    _dataset(dataset)
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    payload["cases"][0]["split"] = "validation"
    dataset.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    scores.write_text(json.dumps({"predictions": {
        "two": {"domains": ["technical"], "action": "query", "escalated": False},
    }}), encoding="utf-8")
    args = argparse.Namespace(
        dataset=dataset, predictions=scores, output=report, evaluation_split="test"
    )

    evaluate(args)
    result = json.loads(report.read_text(encoding="utf-8"))

    assert set(result["splits"]) == {"test"}


def test_single_run_lock_rejects_second_live_instance(tmp_path):
    lock_path = tmp_path / "eval.lock"

    with SingleRunLock(lock_path):
        with pytest.raises(RuntimeError, match="already running"):
            with SingleRunLock(lock_path):
                pass

    assert not lock_path.exists()
