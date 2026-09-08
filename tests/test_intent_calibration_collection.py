import argparse
import asyncio
import json

from tools.run_intent_calibration import collect


class FakeRecognizer:
    def __init__(self):
        self.llm_calls = []

    async def _llm_recognize(self, message, _history):
        self.llm_calls.append(message)
        if message == "retry-me" and self.llm_calls.count(message) < 2:
            return {"failed": True}
        return {"scores": {"technical": 0.9}}

    async def _embedding_recognize(self, _message):
        return {"scores": {"technical": 0.8}}

    def _pattern_recognize(self, _message):
        return {"scores": {"technical": 1.0}}


def _args(tmp_path, cases):
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    return argparse.Namespace(
        golden=golden,
        scores=tmp_path / "scores.json",
        output=tmp_path / "report.json",
        max_attempts=3,
    )


def test_collect_retries_failed_format_and_persists_each_success(tmp_path):
    args = _args(tmp_path, [
        {"case_id": "1", "message": "retry-me", "split": "validation", "expected_intent": "technical", "expected_intents": ["technical"]},
        {"case_id": "2", "message": "normal", "split": "test", "expected_intent": "technical", "expected_intents": ["technical"]},
    ])
    recognizer = FakeRecognizer()

    asyncio.run(collect(args, recognizer=recognizer))

    saved = json.loads(args.scores.read_text(encoding="utf-8"))
    assert [case["case_id"] for case in saved["cases"]] == ["1", "2"]
    assert recognizer.llm_calls == ["retry-me", "retry-me", "normal"]


def test_collect_resumes_without_recalling_completed_cases(tmp_path):
    args = _args(tmp_path, [
        {"case_id": "1", "message": "completed", "split": "validation", "expected_intent": "technical", "expected_intents": ["technical"]},
        {"case_id": "2", "message": "pending", "split": "test", "expected_intent": "technical", "expected_intents": ["technical"]},
    ])
    args.scores.write_text(json.dumps({"schema_version": 1, "cases": [{
        "case_id": "1", "message": "completed", "split": "validation",
        "expected_intent": "technical", "expected_intents": ["technical"],
        "scores": {"llm": {"technical": 0.9}, "embedding": {"technical": 0.8}, "pattern": {"technical": 1.0}},
    }]}), encoding="utf-8")
    recognizer = FakeRecognizer()

    asyncio.run(collect(args, recognizer=recognizer))

    assert recognizer.llm_calls == ["pending"]
    assert len(json.loads(args.scores.read_text(encoding="utf-8"))["cases"]) == 2
