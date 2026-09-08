import asyncio
from types import SimpleNamespace

import pytest

from core.intent_recognizer import IntentCategory
from evaluation.evaluator import (
    EndToEndEvaluator,
    IntentEvaluator,
    IntentTestCase,
)


def test_intent_evaluator_scores_multi_label_predictions_per_class():
    predictions = {
        "network and billing": ("technical", ["technical", "billing"]),
        "network only": ("technical", ["technical", "billing"]),
        "billing missed": ("technical", ["technical"]),
    }

    class Recognizer:
        async def recognize(self, message):
            primary, matched = predictions[message]
            return SimpleNamespace(
                intent=IntentCategory(primary),
                matched_intents=[IntentCategory(label) for label in matched],
                confidence=0.9,
                reasoning="test",
            )

    cases = [
        IntentTestCase(
            "network and billing",
            "technical",
            expected_intents=["technical", "billing"],
        ),
        IntentTestCase(
            "network only",
            "technical",
            expected_intents=["technical"],
        ),
        IntentTestCase(
            "billing missed",
            "billing",
            expected_intents=["billing"],
        ),
    ]

    metrics = asyncio.run(IntentEvaluator(Recognizer()).evaluate(cases))

    assert metrics["accuracy"] == 0.6667
    assert metrics["subset_accuracy"] == 0.3333
    assert metrics["macro_f1"] == 0.65
    assert metrics["per_class"]["technical"] == {
        "precision": 2 / 3,
        "recall": 1.0,
        "f1": 0.8,
    }
    assert metrics["per_class"]["billing"] == {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert metrics["cases"][0]["expected_intents"] == [
        "technical",
        "billing",
    ]
    assert metrics["cases"][0]["predicted_intents"] == [
        "technical",
        "billing",
    ]


def test_eval_intent_input_accepts_multi_label_expectations():
    from api.main import EvalIntentInput

    payload = EvalIntentInput(
        message="校园网报错而且重复扣款",
        expected_intents=["technical", "billing"],
    ).model_dump(exclude_none=True)

    assert payload == {
        "message": "校园网报错而且重复扣款",
        "expected_intents": ["technical", "billing"],
    }


def test_eval_intent_input_requires_at_least_one_expected_label():
    from api.main import EvalIntentInput

    with pytest.raises(ValueError, match="expected_intent"):
        EvalIntentInput(message="没有标注")


def test_end_to_end_report_fails_when_secondary_intent_macro_f1_is_low():
    class Recognizer:
        async def recognize(self, _message):
            return SimpleNamespace(
                intent=IntentCategory.TECHNICAL,
                matched_intents=[IntentCategory.TECHNICAL],
                confidence=0.9,
                reasoning="test",
            )

    class ChatService:
        async def chat(self, _command):
            raise AssertionError("dialog evaluation is not part of this test")

    subject = EndToEndEvaluator(
        chat_service=ChatService(),
        recognizer=Recognizer(),
        judge=object(),
    )
    case = IntentTestCase(
        "network and billing",
        "technical",
        expected_intents=["technical", "billing"],
    )

    report = asyncio.run(subject.run(intent_cases=[case]))

    result = report.results[0]
    assert result.passed is False
    assert result.scores == {
        "accuracy": 1.0,
        "subset_accuracy": 0.0,
        "macro_f1": 0.5,
    }
    assert report.avg_scores["intent_macro_f1"] == 0.5
    assert report.avg_scores["intent_subset_accuracy"] == 0.0
