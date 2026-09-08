import asyncio
from types import SimpleNamespace

from fastapi import Request

import api.main as main


def base_response(**updates):
    values = {
        "conv_id": "conv-1",
        "response": "ok",
        "intent": "technical",
        "agent_type": "technical",
        "escalated": False,
        "latency_ms": 12.3,
        "knowledge_used": True,
        "intent_scores": {"technical": 0.91, "billing": 0.86},
        "matched_intents": ["technical", "billing"],
        "agent_types": ["technical", "billing"],
    }
    values.update(updates)
    return values


def make_request():
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/chat",
            "query_string": b"",
            "headers": [],
        }
    )


def dump(model):
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def test_chat_response_exposes_multi_label_routing_details():
    payload = dump(main.ChatResponse(**base_response()))

    assert payload["intent_scores"] == {
        "technical": 0.91,
        "billing": 0.86,
    }
    assert payload["matched_intents"] == ["technical", "billing"]
    assert payload["agent_types"] == ["technical", "billing"]


def test_chat_response_legacy_construction_has_backward_compatible_defaults():
    first = main.ChatResponse(**base_response())
    second = main.ChatResponse(**base_response())

    assert first.trace_id is None
    assert first.tool_calls == []
    assert first.ticket_ids == []
    assert first.citations == []
    assert first.evidence_verification == {}

    first.tool_calls.append(main.ToolTraceOutput(name="lookup", success=True))
    first.ticket_ids.append("ticket-1")
    first.citations.append({"id": "citation-1"})

    assert second.tool_calls == []
    assert second.ticket_ids == []
    assert second.citations == []


def test_chat_maps_new_fields_to_typed_bounded_safe_output(monkeypatch):
    class FakeService:
        async def chat(self, _command):
            return SimpleNamespace(
                **base_response(),
                trace_id="a" * 32,
                tool_calls=[
                    {
                        "name": "get_ticket",
                        "success": True,
                        "latency_ms": 3.5,
                        "cached": False,
                        "error_type": None,
                        "params": {"description": "private tool input"},
                        "token": "private-token",
                    },
                    {"name": object(), "success": "yes"},
                ],
                ticket_ids=["ticket-1", "ticket-1", object()],
                citations=[
                    {
                        "id": "cite-1",
                        "title": "Network guide",
                        "content": "approved citation content",
                        "score": 0.9,
                        "token": "private-token",
                    },
                    {"id": object(), "content": "invalid"},
                ],
                evidence_verification={
                    "checked": True,
                    "passed": True,
                    "checked_claims": 2,
                    "issue_codes": [],
                    "abstained": False,
                    "reflection_count": 1,
                    "corrected": True,
                    "safe_fallback_used": False,
                },
            )

    monkeypatch.setattr(main, "_chat_service", FakeService())

    response = asyncio.run(
        main.chat(
            main.ChatRequest(message="Help", user_id="display"),
            make_request(),
        )
    )
    payload = dump(response)

    assert payload["trace_id"] == "a" * 32
    assert payload["tool_calls"] == [
        {
            "name": "get_ticket",
            "success": True,
            "latency_ms": 3.5,
            "cached": False,
            "error_type": None,
        }
    ]
    assert payload["ticket_ids"] == ["ticket-1"]
    assert payload["citations"] == [
        {
            "id": "cite-1",
            "title": "Network guide",
            "content": "approved citation content",
            "score": 0.9,
        }
    ]
    assert payload["evidence_verification"] == {
        "checked": True,
        "passed": True,
        "checked_claims": 2,
        "issue_codes": [],
        "abstained": False,
        "reflection_count": 1,
        "corrected": True,
        "safe_fallback_used": False,
    }
    assert "private" not in repr(payload["tool_calls"])
    assert "private-token" not in repr(payload)


def test_chat_metadata_sanitizer_preserves_layout_and_removes_unsafe_unicode(
    monkeypatch,
):
    class FakeService:
        async def chat(self, _command):
            return SimpleNamespace(
                **base_response(),
                trace_id="b" * 32,
                tool_calls=[
                    {
                        "name": "get\u200b_ticket",
                        "success": True,
                        "error_type": "tool\u2060_error",
                    }
                ],
                ticket_ids=["ticket_\u200b1"],
                citations=[
                    {
                        "id": "cite\u200b-1",
                        "title": "Network\nGuide\tA\u200bB",
                        "content": "Line 1\n\tLine 2\x00\ud800\ufe0f end",
                        "score": 0.8,
                    }
                ],
            )

    monkeypatch.setattr(main, "_chat_service", FakeService())

    payload = dump(asyncio.run(
        main.chat(
            main.ChatRequest(message="Help", user_id="display"),
            make_request(),
        )
    ))

    assert payload["tool_calls"][0]["name"] == "get_ticket"
    assert payload["tool_calls"][0]["error_type"] == "tool_error"
    assert payload["ticket_ids"] == ["ticket_1"]
    assert payload["citations"] == [
        {
            "id": "cite-1",
            "title": "Network\nGuide\tAB",
            "content": "Line 1\n\tLine 2 end",
            "score": 0.8,
        }
    ]
