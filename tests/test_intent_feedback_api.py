import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api.main as main
from agents.agent_orchestrator import AgentOrchestrator


class FakeRecognizer:
    def __init__(self, *, changed=True, error=None):
        self.changed = changed
        self.error = error
        self.calls = []
        self.template_fingerprint = "fingerprint-v1"

    def learn(self, message, correct):
        self.calls.append((message, correct.value))
        if self.error:
            raise self.error
        return self.changed


def feedback_body(intent="technical"):
    return main.IntentFeedbackInput(
        message="网总是断",
        correct_intent=intent,
    )


def test_orchestrator_uses_injected_recognizer():
    recognizer = object()

    orchestrator = AgentOrchestrator(
        api_key="test-key",
        base_url="https://example.invalid",
        model="test-model",
        intent_recognizer=recognizer,
    )

    assert orchestrator.intent_recognizer is recognizer


def test_feedback_endpoint_is_disabled_without_admin_token(monkeypatch):
    monkeypatch.delenv("INTENT_FEEDBACK_ADMIN_TOKEN", raising=False)
    monkeypatch.setattr(main, "_intent_recognizer", FakeRecognizer())

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            main.submit_intent_feedback(
                feedback_body(),
                x_intent_admin_token="anything",
            )
        )

    assert error.value.status_code == 503


def test_feedback_endpoint_rejects_wrong_admin_token(monkeypatch):
    monkeypatch.setenv("INTENT_FEEDBACK_ADMIN_TOKEN", "correct-token")
    monkeypatch.setattr(main, "_intent_recognizer", FakeRecognizer())

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            main.submit_intent_feedback(
                feedback_body(),
                x_intent_admin_token="wrong-token",
            )
        )

    assert error.value.status_code == 403


@pytest.mark.parametrize("token", [None, "wrong-token"])
def test_feedback_admin_dependency_rejects_before_body_validation(
    token,
    monkeypatch,
):
    monkeypatch.setenv("INTENT_FEEDBACK_ADMIN_TOKEN", "correct-token")
    headers = {} if token is None else {"X-Intent-Admin-Token": token}

    client = TestClient(main.app)
    response = client.post(
        "/intent/feedback",
        json={"private": "raw-input"},
        headers=headers,
    )
    client.close()

    assert response.status_code == 403
    assert "raw-input" not in response.text


def test_feedback_valid_admin_token_allows_body_validation(monkeypatch):
    monkeypatch.setenv("INTENT_FEEDBACK_ADMIN_TOKEN", "correct-token")

    client = TestClient(main.app)
    response = client.post(
        "/intent/feedback",
        json={"private": "raw-input"},
        headers={"X-Intent-Admin-Token": "correct-token"},
    )
    client.close()

    assert response.status_code == 422


def test_feedback_endpoint_rejects_unknown_intent(monkeypatch):
    monkeypatch.setenv("INTENT_FEEDBACK_ADMIN_TOKEN", "correct-token")
    monkeypatch.setattr(main, "_intent_recognizer", FakeRecognizer())

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            main.submit_intent_feedback(
                feedback_body("not-a-real-intent"),
                x_intent_admin_token="correct-token",
            )
        )

    assert error.value.status_code == 400


def test_feedback_endpoint_rejects_blank_message(monkeypatch):
    monkeypatch.setenv("INTENT_FEEDBACK_ADMIN_TOKEN", "correct-token")
    monkeypatch.setattr(main, "_intent_recognizer", FakeRecognizer())
    body = main.IntentFeedbackInput(message="   ", correct_intent="technical")

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            main.submit_intent_feedback(
                body,
                x_intent_admin_token="correct-token",
            )
        )

    assert error.value.status_code == 400


def test_feedback_endpoint_applies_reviewed_sample(monkeypatch):
    recognizer = FakeRecognizer()
    monkeypatch.setenv("INTENT_FEEDBACK_ADMIN_TOKEN", "correct-token")
    monkeypatch.setattr(main, "_intent_recognizer", recognizer)

    result = asyncio.run(
        main.submit_intent_feedback(
            feedback_body(),
            x_intent_admin_token="correct-token",
        )
    )

    assert recognizer.calls == [("网总是断", "technical")]
    assert result == {
        "message": "网总是断",
        "correct_intent": "technical",
        "changed": True,
        "template_fingerprint": "fingerprint-v1",
    }


def test_feedback_endpoint_reports_persistence_failure(monkeypatch):
    monkeypatch.setenv("INTENT_FEEDBACK_ADMIN_TOKEN", "correct-token")
    monkeypatch.setattr(
        main,
        "_intent_recognizer",
        FakeRecognizer(error=RuntimeError("redis unavailable")),
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            main.submit_intent_feedback(
                feedback_body(),
                x_intent_admin_token="correct-token",
            )
        )

    assert error.value.status_code == 503
