import asyncio
import json
import re
import time
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from anthropic.types import TextBlock, ToolUseBlock

from agents.agent_runtime import (
    MAX_STEPS_RESPONSE,
    AgentRuntime,
    RuntimeResult,
    sanitize_text_content,
)
from core.llm_utils import content_blocks


@dataclass
class SDKBlock:
    payload: dict

    def model_dump(self):
        return dict(self.payload)


class FakeMessages:
    def __init__(self, responses, *, synchronous=False):
        self._responses = list(responses)
        self.calls = []
        self._synchronous = synchronous

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        response = self._responses.pop(0)
        if self._synchronous:
            return response

        async def deliver():
            if callable(response):
                return await response()
            return response

        return deliver()


class FakeClient:
    def __init__(self, responses, *, synchronous=False):
        self.messages = FakeMessages(responses, synchronous=synchronous)


class FakeManager:
    SCHEMA_PROPERTIES = {
        "site": {"type": "string"},
        "days": {"type": "integer"},
        "title": {"type": "string"},
        "nested": {"type": "object"},
        "items": {"type": "array"},
        "value": {},
        "ticket_id": {"type": "string"},
    }

    def __init__(self, results=None, *, schema_error=None):
        self.calls = []
        self.schema_requests = []
        self._results = list(results or [])
        self._schema_error = schema_error

    def schemas(self, allowed):
        if self._schema_error:
            raise self._schema_error
        self.schema_requests.append(list(allowed))
        return [
            {
                "name": name,
                "description": f"Call {name}",
                "input_schema": {
                    "type": "object",
                    "properties": self.SCHEMA_PROPERTIES,
                },
            }
            for name in allowed
        ]

    async def call(self, name, params, context):
        self.calls.append((name, deepcopy(params), deepcopy(context)))
        if self._results:
            return self._results.pop(0)
        return SimpleNamespace(
            success=True,
            data={"tool": name, "params": params},
            error=None,
            rejected=False,
            cached=False,
            latency_ms=1.5,
        )


def response(*blocks):
    return SimpleNamespace(content=list(blocks))


def tool_use(tool_id, name, params):
    return {"type": "tool_use", "id": tool_id, "name": name, "input": params}


def cyclic_tool_input():
    value = {}
    value["self"] = value
    return {"value": value}


def deeply_nested_tool_input():
    value = "leaf"
    for _ in range(10):
        value = {"nested": value}
    return {"value": value}


def text(value):
    return {"type": "text", "text": value}


def run(coro):
    return asyncio.run(coro)


def test_runtime_reflects_once_and_returns_grounded_tool_answer():
    try:
        from core.evidence_verifier import EvidenceVerifier
    except ModuleNotFoundError as exc:
        pytest.fail(f"EvidenceVerifier is not implemented: {exc}")

    client = FakeClient(
        [
            response(tool_use("tool-1", "create_ticket", {"title": "网络报修"})),
            response(text("已创建工单 ticket_wrong。")),
            response(text("已创建工单 ticket_cafebabe。")),
        ]
    )
    manager = FakeManager(results=[SimpleNamespace(
        success=True,
        data={"id": "ticket_cafebabe", "status": "OPEN"},
        error=None,
        rejected=False,
        cached=False,
        fallback_used=False,
        latency_ms=1.0,
    )])
    runtime = AgentRuntime(
        client,
        "model",
        manager,
        max_steps=4,
        verifier=EvidenceVerifier(),
        max_reflections=1,
    )

    result = run(runtime.run(
        system_prompt="只依据工具结果回答。",
        messages=[{"role": "user", "content": "请创建网络报修工单"}],
        allowed_tools=["create_ticket"],
        context={"user_id": "student_001", "idempotency_scope": "scope-1"},
        question="请创建网络报修工单",
    ))

    assert result.text == "已创建工单 ticket_cafebabe。"
    assert result.verification is not None
    assert result.verification.passed is True
    assert result.reflection_count == 1
    assert result.corrected is True


def test_runtime_uses_safe_fallback_when_reflection_is_still_unsupported():
    try:
        from core.evidence_verifier import EvidenceVerifier
    except ModuleNotFoundError as exc:
        pytest.fail(f"EvidenceVerifier is not implemented: {exc}")

    client = FakeClient(
        [
            response(tool_use("tool-1", "create_ticket", {"title": "网络报修"})),
            response(text("已创建工单 ticket_wrong。")),
            response(text("已创建工单 ticket_still_wrong。")),
        ]
    )
    manager = FakeManager(results=[SimpleNamespace(
        success=True,
        data={"id": "ticket_cafebabe", "status": "OPEN"},
        error=None,
        rejected=False,
        cached=False,
        fallback_used=False,
        latency_ms=1.0,
    )])
    runtime = AgentRuntime(
        client,
        "model",
        manager,
        max_steps=4,
        verifier=EvidenceVerifier(),
        max_reflections=1,
    )

    result = run(runtime.run(
        system_prompt="只依据工具结果回答。",
        messages=[{"role": "user", "content": "请创建网络报修工单"}],
        allowed_tools=["create_ticket"],
        context={"user_id": "student_001", "idempotency_scope": "scope-1"},
        question="请创建网络报修工单",
    ))

    assert "无法根据当前证据可靠确认" in result.text
    assert result.verification is not None
    assert result.verification.passed is False
    assert result.reflection_count == 1
    assert result.safe_fallback_used is True


def test_sanitizer_removes_explicit_default_ignorables_and_preserves_text():
    sanitized = sanitize_text_content(
        "Cafe\u0301 中文 😀 A\u034fB\ufe0fC\ufff0"
    )

    assert sanitized == "Cafe\u0301 中文 😀 ABC"
    assert "\u0301" in sanitized
    json.dumps({"text": sanitized}, ensure_ascii=False).encode("utf-8")


def test_content_blocks_normalizes_sdk_objects_and_dicts():
    original = {"type": "text", "text": "dictionary"}

    assert content_blocks(
        [
            SDKBlock({"type": "text", "text": "sdk"}),
            original,
            object(),
        ]
    ) == [
        {"type": "text", "text": "sdk"},
        original,
    ]


def test_content_blocks_deep_copies_nested_model_data():
    original = tool_use(
        "tool-1",
        "query_network_status",
        {"site": "campus", "nested": {"value": "original"}},
    )

    normalized = content_blocks([original])
    normalized[0]["input"]["site"] = "mutated"
    normalized[0]["input"]["nested"]["value"] = "mutated"

    assert original["input"] == {
        "site": "campus",
        "nested": {"value": "original"},
    }


def test_runtime_executes_tool_and_returns_final_text_without_mutating_input():
    client = FakeClient(
        [
            response(tool_use("tool-1", "query_network_status", {"site": "campus"})),
            response(text("Campus network authentication is healthy.")),
        ]
    )
    manager = FakeManager()
    runtime = AgentRuntime(client, "model", manager, max_steps=4)
    messages = [{"role": "user", "content": "I cannot sign in."}]
    original_messages = deepcopy(messages)

    result = run(
        runtime.run(
            system_prompt="You are technical support.",
            messages=messages,
            allowed_tools=["query_network_status"],
            context={"user_id": "demo_user", "idempotency_scope": "scope-1"},
        )
    )

    assert result.text == "Campus network authentication is healthy."
    assert result.stop_reason == "end_turn"
    assert result.tool_calls[0].name == "query_network_status"
    assert result.tool_calls[0].success is True
    assert manager.schema_requests == [
        ["query_network_status"],
        ["query_network_status"],
    ]
    assert [schema["name"] for schema in client.messages.calls[0]["tools"]] == [
        "query_network_status"
    ]
    assert messages == original_messages


def test_forbidden_tool_is_not_executed_and_matching_error_is_returned_to_model():
    client = FakeClient(
        [
            response(tool_use("forbidden-1", "delete_ticket", {"ticket_id": "T1"})),
            response(text("I cannot delete that ticket.")),
        ]
    )
    manager = FakeManager()
    runtime = AgentRuntime(client, "model", manager)

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Delete my ticket."}],
            allowed_tools=["get_ticket"],
        )
    )

    assert manager.calls == []
    tool_results = client.messages.calls[1]["messages"][-1]["content"]
    assert tool_results == [
        {
            "type": "tool_result",
            "tool_use_id": "forbidden-1",
            "content": "Requested tool is not allowed.",
            "is_error": True,
        }
    ]
    assert result.tool_calls[0].name == "forbidden_tool"
    assert result.tool_calls[0].success is False


def test_tool_parameters_are_forwarded_and_context_gets_opaque_operation_key():
    client = FakeClient(
        [
            response(tool_use("tool-1", "query_network_status", {"site": "campus"})),
            response(text("Done.")),
        ]
    )
    manager = FakeManager()
    runtime = AgentRuntime(client, "model", manager)
    context = {"user_id": "u-1", "idempotency_scope": "scope-1"}

    run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Check the network."}],
            allowed_tools=["query_network_status"],
            context=context,
        )
    )

    assert manager.calls[0][:2] == (
        "query_network_status",
        {"site": "campus"},
    )
    forwarded_context = manager.calls[0][2]
    assert forwarded_context["user_id"] == "u-1"
    assert "idempotency_scope" not in forwarded_context
    assert re.fullmatch(r"op_[0-9a-f]{64}", forwarded_context["idempotency_key"])
    assert context == {"user_id": "u-1", "idempotency_scope": "scope-1"}


def test_operation_idempotency_is_canonical_stable_distinct_and_opaque():
    client = FakeClient(
        [
            response(
                tool_use(
                    "tool-1",
                    "create_ticket",
                    {"title": "A", "nested": {"b": 2, "a": 1}},
                ),
                tool_use(
                    "tool-2",
                    "create_ticket",
                    {"nested": {"a": 1, "b": 2}, "title": "A"},
                ),
                tool_use(
                    "tool-3",
                    "create_ticket",
                    {"title": "B", "nested": {"a": 1, "b": 2}},
                ),
            ),
            response(text("Done.")),
        ]
    )
    manager = FakeManager()
    runtime = AgentRuntime(client, "model", manager)
    raw_scope = "conversation/request/technical"
    principal = "authenticated-user"

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Create tickets."}],
            allowed_tools=["create_ticket"],
            context={
                "user_id": principal,
                "idempotency_scope": raw_scope,
            },
        )
    )

    keys = [call[2]["idempotency_key"] for call in manager.calls]
    assert keys[0] == keys[1]
    assert keys[2] != keys[0]
    assert all(re.fullmatch(r"op_[0-9a-f]{64}", key) for key in keys)
    assert raw_scope not in repr(keys)
    assert principal not in repr(keys)
    assert "title" not in repr(keys)
    assert raw_scope not in repr(result.tool_calls)
    assert principal not in repr(result.tool_calls)


def test_missing_idempotency_scope_does_not_invent_operation_key():
    client = FakeClient(
        [
            response(tool_use("tool-1", "create_ticket", {"title": "A"})),
            response(text("Done.")),
        ]
    )
    manager = FakeManager()

    run(
        AgentRuntime(client, "model", manager).run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Create it."}],
            allowed_tools=["create_ticket"],
            context={"user_id": "principal"},
        )
    )

    assert manager.calls[0][2] == {"user_id": "principal"}


def test_multiple_tool_uses_receive_ordered_matching_results():
    client = FakeClient(
        [
            response(
                text("Checking both systems."),
                tool_use("card-1", "query_campus_card", {"days": 2}),
                tool_use("network-1", "query_network_status", {"site": "campus"}),
            ),
            response(text("Both checks are complete.")),
        ]
    )
    manager = FakeManager()
    runtime = AgentRuntime(client, "model", manager)

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Check both."}],
            allowed_tools=["query_campus_card", "query_network_status"],
        )
    )

    tool_results = client.messages.calls[1]["messages"][-1]["content"]
    assert [block["tool_use_id"] for block in tool_results] == [
        "card-1",
        "network-1",
    ]
    assert all(block["type"] == "tool_result" for block in tool_results)
    assert [call.name for call in result.tool_calls] == [
        "query_campus_card",
        "query_network_status",
    ]


def test_maximum_step_limit_returns_controlled_result():
    client = FakeClient(
        [
            response(tool_use("tool-1", "lookup", {})),
            response(tool_use("tool-2", "lookup", {})),
        ]
    )
    manager = FakeManager()
    runtime = AgentRuntime(client, "model", manager, max_steps=2)

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Keep checking."}],
            allowed_tools=["lookup"],
        )
    )

    assert result.stop_reason == "max_steps"
    assert "step limit" in result.text.lower()
    assert len(client.messages.calls) == 2
    assert len(manager.calls) == 2


def test_runtime_result_ticket_ids_have_independent_safe_defaults():
    first = RuntimeResult(text="one")
    second = RuntimeResult(text="two")

    first.ticket_ids.append("ticket_1")

    assert second.ticket_ids == []


def test_successful_create_ticket_collects_structured_id():
    manager = FakeManager(
        [
            SimpleNamespace(
                success=True,
                data={"id": "ticket_123", "status": "OPEN"},
                error=None,
                rejected=False,
                cached=False,
                latency_ms=1,
                fallback_used=False,
            )
        ]
    )
    client = FakeClient(
        [
            response(
                tool_use(
                    "tool-1",
                    "create_ticket",
                    {
                        "title": "Issue",
                        "category": "technical",
                        "description": "Details",
                    },
                )
            ),
            response(text("Created.")),
        ]
    )

    result = run(
        AgentRuntime(client, "model", manager).run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Create it."}],
            allowed_tools=["create_ticket"],
            context={
                "user_id": "principal",
                "idempotency_scope": "scope-1",
            },
        )
    )

    assert result.ticket_ids == ["ticket_123"]


def test_successful_tool_collects_standard_ticket_id_field():
    manager = FakeManager(
        [
            SimpleNamespace(
                success=True,
                data={"ticket_id": "ticket_456", "status": "OPEN"},
                error=None,
                rejected=False,
                cached=False,
                latency_ms=1,
                fallback_used=False,
            )
        ]
    )
    client = FakeClient(
        [
            response(tool_use("tool-1", "get_ticket", {"ticket_id": "old"})),
            response(text("Found.")),
        ]
    )

    result = run(
        AgentRuntime(client, "model", manager).run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Find it."}],
            allowed_tools=["get_ticket"],
        )
    )

    assert result.ticket_ids == ["ticket_456"]


def test_ticket_id_is_not_collected_from_unsafe_tool_payload():
    manager = FakeManager(
        [
            SimpleNamespace(
                success=True,
                data={"ticket_id": "ticket_unsafe", "value": object()},
                error=None,
                rejected=False,
                cached=False,
                latency_ms=1,
                fallback_used=False,
            )
        ]
    )
    client = FakeClient(
        [
            response(tool_use("tool-1", "get_ticket", {"ticket_id": "old"})),
            response(text("Handled.")),
        ]
    )

    result = run(
        AgentRuntime(client, "model", manager).run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Find it."}],
            allowed_tools=["get_ticket"],
        )
    )

    assert result.tool_calls[0].success is False
    assert result.ticket_ids == []


def test_ticket_id_survives_controlled_max_steps_completion():
    manager = FakeManager(
        [
            SimpleNamespace(
                success=True,
                data={"id": "ticket_123"},
                error=None,
                rejected=False,
                cached=False,
                latency_ms=1,
                fallback_used=False,
            )
        ]
    )
    client = FakeClient(
        [
            response(
                tool_use(
                    "tool-1",
                    "create_ticket",
                    {
                        "title": "Issue",
                        "category": "technical",
                        "description": "Details",
                    },
                )
            ),
        ]
    )

    result = run(
        AgentRuntime(client, "model", manager, max_steps=1).run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Create it."}],
            allowed_tools=["create_ticket"],
        )
    )

    assert result.stop_reason == "max_steps"
    assert result.ticket_ids == ["ticket_123"]


def test_create_ticket_fallback_unknown_does_not_infer_ticket_id():
    manager = FakeManager(
        [
            SimpleNamespace(
                success=True,
                data={"id": "must-not-be-collected"},
                error="raw failure",
                rejected=False,
                cached=False,
                latency_ms=1,
                fallback_used=True,
            )
        ]
    )
    client = FakeClient(
        [
            response(tool_use("tool-1", "create_ticket", {})),
            response(text("Handled.")),
        ]
    )

    result = run(
        AgentRuntime(client, "model", manager).run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Create it."}],
            allowed_tools=["create_ticket"],
        )
    )

    assert result.ticket_ids == []


def test_ticket_ids_are_bounded_and_stably_deduplicated():
    manager = FakeManager(
        [
            SimpleNamespace(
                success=True,
                data={"id": "ticket_first"},
                rejected=False,
                cached=False,
                latency_ms=1,
                fallback_used=False,
            ),
            SimpleNamespace(
                success=True,
                data={"id": "ticket_first"},
                rejected=False,
                cached=False,
                latency_ms=1,
                fallback_used=False,
            ),
            SimpleNamespace(
                success=True,
                data={"id": "x" * 129},
                rejected=False,
                cached=False,
                latency_ms=1,
                fallback_used=False,
            ),
            SimpleNamespace(
                success=True,
                data={"id": "ticket_second"},
                rejected=False,
                cached=False,
                latency_ms=1,
                fallback_used=False,
            ),
        ]
    )
    client = FakeClient(
        [
            response(
                *[
                    tool_use(f"tool-{index}", "create_ticket", {})
                    for index in range(4)
                ]
            ),
            response(text("Handled.")),
        ]
    )

    result = run(
        AgentRuntime(client, "model", manager).run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Create tickets."}],
            allowed_tools=["create_ticket"],
        )
    )

    assert result.ticket_ids == ["ticket_first", "ticket_second"]


def test_runtime_ticket_id_collection_has_explicit_global_cap():
    tool_steps = 9
    calls_per_step = 16
    manager = FakeManager(
        [
            SimpleNamespace(
                success=True,
                data={"id": f"ticket_{index:03d}"},
                rejected=False,
                cached=False,
                latency_ms=1,
                fallback_used=False,
            )
            for index in range(tool_steps * calls_per_step)
        ]
    )
    responses = []
    for step in range(tool_steps):
        responses.append(
            response(
                *[
                    tool_use(
                        f"tool-{step}-{offset}",
                        "create_ticket",
                        {"title": f"Ticket {step}-{offset}"},
                    )
                    for offset in range(calls_per_step)
                ]
            )
        )
    result = run(
        AgentRuntime(
            FakeClient(responses),
            "model",
            manager,
            max_steps=tool_steps,
        ).run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Create tickets."}],
            allowed_tools=["create_ticket"],
        )
    )

    assert len(result.ticket_ids) == 128
    assert result.ticket_ids[0] == "ticket_000"
    assert result.ticket_ids[-1] == "ticket_127"
    assert result.stop_reason == "max_steps"
    assert result.text == MAX_STEPS_RESPONSE


def test_runtime_output_is_unicode_safe_and_json_utf8_encodable():
    unsafe = "正常😀\nnext\ud800\x00\x85\u202e\u2066"
    client = FakeClient([response(text(unsafe))])

    result = run(
        AgentRuntime(client, "model", FakeManager()).run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Hello."}],
            allowed_tools=[],
        )
    )

    assert result.text.startswith("正常😀\nnext")
    assert "\n" in result.text
    assert all(
        character == "\n"
        or (
            unicodedata.category(character) not in {"Cc", "Cf", "Cs"}
        )
        for character in result.text
    )
    json.dumps(
        {"text": result.text},
        ensure_ascii=False,
    ).encode("utf-8")


def test_total_timeout_includes_model_calls_and_returns_controlled_result():
    async def slow_response():
        await asyncio.sleep(0.05)
        return response(text("Too late."))

    client = FakeClient([slow_response])
    runtime = AgentRuntime(
        client,
        "model",
        FakeManager(),
        total_timeout_s=0.005,
    )

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Hello."}],
            allowed_tools=[],
        )
    )

    assert result.stop_reason == "timeout"
    assert "deadline" in result.text.lower()


def test_total_timeout_includes_tool_calls():
    class SlowManager(FakeManager):
        async def call(self, name, params, context):
            await asyncio.sleep(0.05)
            return await super().call(name, params, context)

    client = FakeClient([response(tool_use("tool-1", "lookup", {}))])
    manager = SlowManager()
    runtime = AgentRuntime(
        client,
        "model",
        manager,
        total_timeout_s=0.005,
    )

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Look it up."}],
            allowed_tools=["lookup"],
        )
    )

    assert result.stop_reason == "timeout"
    assert "deadline" in result.text.lower()


def test_failed_tool_result_is_visible_to_the_model_with_is_error():
    manager = FakeManager(
        [
            SimpleNamespace(
                success=False,
                data=None,
                error="network backend unavailable",
                rejected=False,
                cached=False,
                latency_ms=2.0,
            )
        ]
    )
    client = FakeClient(
        [
            response(tool_use("tool-1", "query_network_status", {"site": "campus"})),
            response(text("The status service is unavailable.")),
        ]
    )
    runtime = AgentRuntime(client, "model", manager)

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Check the network."}],
            allowed_tools=["query_network_status"],
        )
    )

    tool_result = client.messages.calls[1]["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "tool-1"
    assert tool_result["is_error"] is True
    assert tool_result["content"] == "Tool execution failed."
    assert "network backend unavailable" not in repr(result)
    assert "network backend unavailable" not in repr(client.messages.calls[1])
    assert result.tool_calls[0].success is False
    assert result.tool_calls[0].error_type == "tool_failure"


def test_trace_parameters_are_redacted_and_context_is_never_recorded():
    secret = "do-not-leak"
    idempotency_scope = "conv-1:billing"
    client = FakeClient(
        [
            response(
                tool_use(
                    "tool-1",
                    "create_ticket",
                    {
                        "title": "Duplicate charge",
                        "nested": {"password": secret},
                        "items": [secret],
                    },
                )
            ),
            response(text("Ticket created.")),
        ]
    )
    manager = FakeManager()
    runtime = AgentRuntime(client, "model", manager)

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Create a ticket."}],
            allowed_tools=["create_ticket"],
            context={
                "user_id": "demo-user",
                "idempotency_scope": idempotency_scope,
                "secret": secret,
            },
        )
    )

    trace_repr = repr(result.tool_calls)
    assert secret not in trace_repr
    assert idempotency_scope not in trace_repr
    assert "demo-user" not in trace_repr
    assert result.tool_calls[0].params == {
        "title": "[REDACTED]",
        "nested": "[REDACTED]",
        "items": "[REDACTED]",
    }
    assert manager.calls[0][1]["nested"]["password"] == secret


def test_tool_call_fingerprint_is_stable_private_and_call_specific(monkeypatch):
    monkeypatch.setattr("secrets.token_bytes", lambda count: b"x" * count)
    tool_id = "provider-call-secret"
    secret = "sensitive-ticket-value"

    def fingerprint(call_id, params):
        client = FakeClient(
            [
                response(tool_use(call_id, "lookup", params)),
                response(text("Handled.")),
            ]
        )
        result = run(
            AgentRuntime(client, "model", FakeManager()).run(
                system_prompt="Support.",
                messages=[{"role": "user", "content": "Look it up."}],
                allowed_tools=["lookup"],
            )
        )
        return result.tool_calls[0].fingerprint

    first = fingerprint(
        tool_id,
        {"value": secret, "nested": {"site": "north"}},
    )
    reordered = fingerprint(
        tool_id,
        {"nested": {"site": "north"}, "value": secret},
    )
    different_id = fingerprint(
        "provider-call-other",
        {"value": secret, "nested": {"site": "north"}},
    )
    different_params = fingerprint(
        tool_id,
        {"value": "different", "nested": {"site": "north"}},
    )

    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert first == reordered
    assert len({first, different_id, different_params}) == 3
    assert tool_id not in first
    assert secret not in first


def test_identical_tool_calls_in_separate_runs_have_distinct_fingerprints():
    client = FakeClient(
        [
            response(tool_use("tool-1", "lookup", {"value": "same"})),
            response(text("First handled.")),
            response(tool_use("tool-1", "lookup", {"value": "same"})),
            response(text("Second handled.")),
        ]
    )
    runtime = AgentRuntime(client, "model", FakeManager())

    def invoke():
        return run(
            runtime.run(
                system_prompt="Support.",
                messages=[{"role": "user", "content": "Look it up."}],
                allowed_tools=["lookup"],
                context={
                    "idempotency_scope": "private-request-scope",
                    "secret": "private-context-value",
                },
            )
        )

    first = invoke().tool_calls[0].fingerprint
    second = invoke().tool_calls[0].fingerprint

    assert first != second
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert re.fullmatch(r"[0-9a-f]{64}", second)
    assert "private-request-scope" not in first + second
    assert "private-context-value" not in first + second


def test_tool_call_fingerprint_is_present_on_every_execution_outcome():
    unsafe_data = {}
    unsafe_data["self"] = unsafe_data
    manager = FakeManager(
        [
            SimpleNamespace(
                success=False,
                rejected=True,
                cached=False,
                latency_ms=1,
            ),
            SimpleNamespace(
                success=False,
                rejected=False,
                cached=False,
                latency_ms=1,
            ),
            SimpleNamespace(
                success=True,
                data=unsafe_data,
                rejected=False,
                cached=False,
                latency_ms=1,
            ),
            SimpleNamespace(
                success=True,
                data={"status": "available"},
                rejected=False,
                cached=False,
                latency_ms=1,
            ),
        ]
    )
    client = FakeClient(
        [
            response(
                tool_use("forbidden-1", "forbidden", {"value": "secret"}),
                tool_use("rejected-1", "lookup", {"value": "secret"}),
                tool_use("failed-1", "lookup", {"value": "secret"}),
                tool_use("unsafe-1", "lookup", {"value": "secret"}),
                tool_use("success-1", "lookup", {"value": "secret"}),
            ),
            response(text("Handled.")),
        ]
    )

    result = run(
        AgentRuntime(client, "model", manager).run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Run all."}],
            allowed_tools=["lookup"],
        )
    )

    assert [call.error_type for call in result.tool_calls] == [
        "forbidden_tool",
        "tool_rejected",
        "tool_failure",
        "unsafe_tool_result",
        None,
    ]
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", call.fingerprint)
        for call in result.tool_calls
    )
    assert len({call.fingerprint for call in result.tool_calls}) == 5
    assert "secret" not in repr(result.tool_calls)


def test_duplicate_tool_use_id_stops_before_sending_invalid_protocol():
    client = FakeClient(
        [
            response(
                tool_use("duplicate", "lookup", {"value": 1}),
                tool_use("duplicate", "lookup", {"value": 2}),
            ),
        ]
    )
    manager = FakeManager()
    runtime = AgentRuntime(client, "model", manager)

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Look it up."}],
            allowed_tools=["lookup"],
        )
    )

    assert manager.calls == []
    assert len(client.messages.calls) == 1
    assert result.stop_reason == "protocol_error"
    assert "protocol" in result.text.lower()
    assert result.tool_calls[-1].name == "invalid_tool_use"
    assert result.tool_calls[-1].params == {}


def test_duplicate_tool_use_id_across_steps_is_not_executed_twice():
    client = FakeClient(
        [
            response(tool_use("duplicate", "lookup", {"value": 1})),
            response(tool_use("duplicate", "lookup", {"value": 2})),
        ]
    )
    manager = FakeManager()
    runtime = AgentRuntime(client, "model", manager)

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Look it up twice."}],
            allowed_tools=["lookup"],
        )
    )

    assert len(manager.calls) == 1
    assert len(client.messages.calls) == 2
    assert result.stop_reason == "protocol_error"
    assert [call.success for call in result.tool_calls] == [True, False]
    assert result.tool_calls[-1].name == "invalid_tool_use"


def test_non_object_tool_input_stops_before_invalid_history_is_sent():
    client = FakeClient(
        [
            response(tool_use("bad-1", "lookup", ["not", "an", "object"])),
        ]
    )
    manager = FakeManager()
    runtime = AgentRuntime(client, "model", manager)

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Look it up."}],
            allowed_tools=["lookup"],
        )
    )

    assert manager.calls == []
    assert len(client.messages.calls) == 1
    assert result.stop_reason == "protocol_error"
    assert result.tool_calls[-1].name == "invalid_tool_use"


def test_missing_tool_use_id_stops_without_sending_synthetic_result():
    malformed = {
        "type": "tool_use",
        "name": "lookup",
        "input": {"value": 1},
    }
    client = FakeClient([response(malformed)])
    manager = FakeManager()
    runtime = AgentRuntime(client, "model", manager)

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Look it up."}],
            allowed_tools=["lookup"],
        )
    )

    assert result.stop_reason == "protocol_error"
    assert manager.calls == []
    assert len(client.messages.calls) == 1
    assert result.tool_calls[-1].name == "invalid_tool_use"


def test_oversized_tool_use_id_stops_without_next_provider_call():
    client = FakeClient(
        [response(tool_use("x" * 1000, "lookup", {"value": 1}))]
    )
    runtime = AgentRuntime(client, "model", FakeManager())

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Look it up."}],
            allowed_tools=["lookup"],
        )
    )

    assert result.stop_reason == "protocol_error"
    assert len(client.messages.calls) == 1
    assert "x" * 100 not in repr(result)


@pytest.mark.parametrize(
    "malformed_block",
    [
        {
            "type": "tool_use",
            "id": "tool-1",
            "input": {},
        },
        {
            "type": "tool_use",
            "id": "tool-1",
            "name": 123,
            "input": {},
        },
        {
            "type": "tool_use",
            "id": "tool-1",
            "name": "x" * 200,
            "input": {},
        },
        {
            "type": "tool_use",
            "id": "tool-1",
            "name": "bad name!",
            "input": {},
        },
        {
            "type": "tool_use",
            "id": "tool-1",
            "name": "lookup",
            "input": {},
            "unexpected": "must-not-be-forwarded",
        },
    ],
    ids=[
        "missing-name",
        "non-string-name",
        "oversized-name",
        "invalid-name-characters",
        "extra-field",
    ],
)
def test_malformed_tool_block_stops_before_provider_reuse(malformed_block):
    client = FakeClient([response(malformed_block)])
    manager = FakeManager()
    runtime = AgentRuntime(client, "model", manager)

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Look it up."}],
            allowed_tools=["lookup"],
        )
    )

    assert result.stop_reason == "protocol_error"
    assert len(client.messages.calls) == 1
    assert manager.calls == []
    assert "must-not-be-forwarded" not in repr(result)


def test_valid_tool_block_is_reconstructed_with_exact_safe_fields():
    original = tool_use("tool-1", "lookup", {"value": {"nested": 1}})
    client = FakeClient(
        [
            response(original),
            response(text("Done.")),
        ]
    )
    runtime = AgentRuntime(client, "model", FakeManager())

    run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Look it up."}],
            allowed_tools=["lookup"],
        )
    )

    assistant_block = client.messages.calls[1]["messages"][-2]["content"][0]
    assert set(assistant_block) == {"type", "id", "name", "input"}
    assert assistant_block == original
    assert assistant_block is not original
    assert assistant_block["input"] is not original["input"]


@pytest.mark.parametrize(
    "params_factory",
    [
        lambda: {"value": object()},
        cyclic_tool_input,
        lambda: {"value": float("nan")},
        lambda: {"value": {1: "non-string key"}},
        deeply_nested_tool_input,
        lambda: {"value": list(range(51))},
        lambda: {
            "value": [
                {"items": [1, 2, 3, 4]}
                for _ in range(50)
            ]
        },
        lambda: {"value": "x" * 2049},
        lambda: {
            "value": {
                f"part_{index}": "x" * 1000
                for index in range(20)
            }
        },
    ],
    ids=[
        "object",
        "cycle",
        "nan",
        "non-string-key",
        "depth",
        "container-length",
        "node-count",
        "string-length",
        "serialized-bytes",
    ],
)
def test_invalid_tool_input_stops_before_execution_or_next_provider_call(
    params_factory,
):
    client = FakeClient(
        [response(tool_use("tool-1", "lookup", params_factory()))]
    )
    manager = FakeManager()
    runtime = AgentRuntime(client, "model", manager)

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Look it up."}],
            allowed_tools=["lookup"],
        )
    )

    assert result.stop_reason == "protocol_error"
    assert manager.calls == []
    assert len(client.messages.calls) == 1


def test_valid_nested_tool_input_is_reconstructed_without_secret_redaction():
    params = {
        "value": {
            "nothing": None,
            "enabled": True,
            "attempts": 2,
            "ratio": 1.5,
            "label": "valid",
            "items": [False, 3, {"apiKeys": ["input-secret"]}],
        }
    }
    client = FakeClient(
        [
            response(tool_use("tool-1", "lookup", params)),
            response(text("Done.")),
        ]
    )
    manager = FakeManager()
    runtime = AgentRuntime(client, "model", manager)

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Look it up."}],
            allowed_tools=["lookup"],
        )
    )

    assert result.text == "Done."
    assert manager.calls[0][1] == params
    assistant_input = client.messages.calls[1]["messages"][-2]["content"][0][
        "input"
    ]
    assert assistant_input == params
    assert assistant_input is not params
    assert assistant_input["value"] is not params["value"]


def test_anthropic_040_text_and_tool_blocks_are_protocol_safe():
    sdk_text = TextBlock(type="text", text="Checking.")
    sdk_tool = ToolUseBlock(
        type="tool_use",
        id="tool-1",
        name="lookup",
        input={"value": 1},
    )
    client = FakeClient(
        [
            response(sdk_text, sdk_tool),
            response(TextBlock(type="text", text="Done.")),
        ]
    )
    manager = FakeManager()
    runtime = AgentRuntime(client, "model", manager)

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Look it up."}],
            allowed_tools=["lookup"],
        )
    )

    assert result.text == "Done."
    assistant_blocks = client.messages.calls[1]["messages"][-2]["content"]
    assert assistant_blocks == [
        {"type": "text", "text": "Checking."},
        {
            "type": "tool_use",
            "id": "tool-1",
            "name": "lookup",
            "input": {"value": 1},
        },
    ]


@pytest.mark.parametrize(
    "malformed_companion",
    [
        {
            "type": "text",
            "text": "Checking.",
            "unexpected": "must-not-be-forwarded",
        },
        {"type": "text", "text": 123},
        {"type": "text", "text": "x" * 20000},
        "raw string block",
    ],
    ids=[
        "text-extra-field",
        "non-string-text",
        "oversized-text",
        "raw-string-block",
    ],
)
def test_tool_turn_rejects_malformed_companion_blocks(malformed_companion):
    client = FakeClient(
        [
            response(
                malformed_companion,
                tool_use("tool-1", "lookup", {}),
            )
        ]
    )
    manager = FakeManager()
    runtime = AgentRuntime(client, "model", manager)

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Look it up."}],
            allowed_tools=["lookup"],
        )
    )

    assert result.stop_reason == "protocol_error"
    assert manager.calls == []
    assert len(client.messages.calls) == 1
    assert "must-not-be-forwarded" not in repr(result)


def test_deepseek_thinking_block_is_dropped_before_returning_final_text():
    client = FakeClient(
        [
            response(
                {
                    "type": "thinking",
                    "text": None,
                    "thinking": "private reasoning must not be forwarded",
                    "signature": "deepseek-signature",
                },
                {"type": "text", "text": "你好，请问需要什么帮助？"},
            )
        ]
    )
    runtime = AgentRuntime(client, "model", FakeManager())

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "你好"}],
            allowed_tools=["lookup"],
        )
    )

    assert result.text == "你好，请问需要什么帮助？"
    assert result.stop_reason == "end_turn"
    assert result.tool_calls == []


def test_deepseek_thinking_block_is_not_forwarded_with_a_tool_call():
    client = FakeClient(
        [
            response(
                {
                    "type": "thinking",
                    "text": None,
                    "thinking": "private reasoning must not be forwarded",
                    "signature": "deepseek-signature",
                },
                tool_use("tool-1", "lookup", {}),
            ),
            response({"type": "text", "text": "Done."}),
        ]
    )
    manager = FakeManager()
    runtime = AgentRuntime(client, "model", manager)

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Look it up."}],
            allowed_tools=["lookup"],
        )
    )

    assert result.text == "Done."
    assert client.messages.calls[1]["messages"][-2]["content"] == [
        tool_use("tool-1", "lookup", {})
    ]


def test_sync_anthropic_like_client_response_is_supported():
    client = FakeClient(
        [{"content": [SDKBlock({"type": "text", "text": "Synchronous."})]}],
        synchronous=True,
    )
    runtime = AgentRuntime(client, "model", FakeManager())

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Hello."}],
            allowed_tools=[],
        )
    )

    assert result.text == "Synchronous."


def test_sync_model_creation_time_counts_toward_total_deadline():
    class BlockingMessages:
        def create(self, **_kwargs):
            time.sleep(0.02)
            return response(text("Too late."))

    client = SimpleNamespace(messages=BlockingMessages())
    runtime = AgentRuntime(
        client,
        "model",
        FakeManager(),
        total_timeout_s=0.005,
    )

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Hello."}],
            allowed_tools=[],
        )
    )

    assert result.stop_reason == "timeout"


def test_sync_model_create_does_not_block_event_loop():
    class BlockingMessages:
        def create(self, **_kwargs):
            time.sleep(0.08)
            return response(text("Too late."))

    async def scenario():
        loop = asyncio.get_running_loop()
        ticked_at = None

        async def heartbeat():
            nonlocal ticked_at
            await asyncio.sleep(0.005)
            ticked_at = loop.time()

        runtime = AgentRuntime(
            SimpleNamespace(messages=BlockingMessages()),
            "model",
            FakeManager(),
            total_timeout_s=0.02,
        )
        started = loop.time()
        heartbeat_task = asyncio.create_task(heartbeat())
        result = await runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Hello."}],
            allowed_tools=[],
        )
        elapsed = loop.time() - started
        await heartbeat_task
        return result, elapsed, ticked_at - started

    result, elapsed, heartbeat_elapsed = run(scenario())

    assert result.stop_reason == "timeout"
    assert elapsed < 0.05
    assert heartbeat_elapsed < 0.04


def test_sync_manager_call_is_rejected_without_invocation():
    class SideEffectManager(FakeManager):
        side_effects = 0

        def call(self, name, params, context):
            self.side_effects += 1
            return SimpleNamespace(
                success=True,
                data={"ok": True},
                error=None,
                rejected=False,
                cached=False,
                latency_ms=1,
            )

    manager = SideEffectManager()
    client = FakeClient(
        [
            response(tool_use("tool-1", "lookup", {})),
            response(text("Tool execution was unavailable.")),
        ]
    )
    runtime = AgentRuntime(client, "model", manager)

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Look it up."}],
            allowed_tools=["lookup"],
        )
    )

    assert manager.side_effects == 0
    assert result.stop_reason == "end_turn"
    tool_result = client.messages.calls[1]["messages"][-1]["content"][0]
    assert tool_result["content"] == "Tool execution failed."
    assert tool_result["is_error"] is True


def test_sync_callable_returning_awaitable_is_awaited():
    class AwaitableMessages:
        def create(self, **_kwargs):
            async def deliver():
                await asyncio.sleep(0)
                return response(text("Awaited."))

            return deliver()

    runtime = AgentRuntime(
        SimpleNamespace(messages=AwaitableMessages()),
        "model",
        FakeManager(),
    )

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Hello."}],
            allowed_tools=[],
        )
    )

    assert result.text == "Awaited."


def test_mutating_manager_cannot_change_response_history_or_later_inputs():
    first_response = response(
        tool_use("tool-1", "lookup", {"value": "first"}),
        tool_use("tool-2", "lookup", {"value": "second"}),
    )

    class MutatingManager(FakeManager):
        async def call(self, name, params, context):
            self.calls.append((name, deepcopy(params), deepcopy(context)))
            params["value"] = "mutated"
            context["user_id"] = "mutated"
            return SimpleNamespace(
                success=True,
                data={"ok": True},
                error=None,
                rejected=False,
                cached=False,
                latency_ms=1,
            )

    context = {"user_id": "original", "idempotency_scope": "original:key"}
    client = FakeClient([first_response, response(text("Done."))])
    manager = MutatingManager()
    runtime = AgentRuntime(client, "model", manager)

    run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Look both up."}],
            allowed_tools=["lookup"],
            context=context,
        )
    )

    assert [call[1]["value"] for call in manager.calls] == ["first", "second"]
    assert [call[2]["user_id"] for call in manager.calls] == [
        "original",
        "original",
    ]
    assert context["user_id"] == "original"
    assert first_response.content[0]["input"]["value"] == "first"
    assistant_blocks = client.messages.calls[1]["messages"][-2]["content"]
    assert [block["input"]["value"] for block in assistant_blocks] == [
        "first",
        "second",
    ]


def test_successful_tool_data_redacts_credential_fields():
    secret = "credential-must-not-leak"
    manager = FakeManager(
        [
            SimpleNamespace(
                success=True,
                data={
                    "status": "ok",
                    "password": secret,
                    "userId": secret,
                    "nested": {"api_key": secret, "value": 3},
                },
                error="backend warning with secret " + secret,
                rejected=False,
                cached=False,
                latency_ms=float("nan"),
            )
        ]
    )
    client = FakeClient(
        [
            response(tool_use("tool-1", "lookup", {})),
            response(text("Done.")),
        ]
    )
    runtime = AgentRuntime(client, "model", manager)

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Look it up."}],
            allowed_tools=["lookup"],
        )
    )

    model_call = repr(client.messages.calls[1])
    assert secret not in model_call
    assert secret not in repr(result)
    assert '"status":"ok"' in model_call
    assert "[REDACTED]" in model_call
    assert result.tool_calls[0].latency_ms == 0.0


def test_result_sanitizer_uses_exact_normalized_secret_key_taxonomy():
    secret = "must-not-leak"
    sensitive = {
        "password": secret,
        "PASSWD": secret,
        "pwd": secret,
        "token": secret,
        "access-token": secret,
        "refresh.token": secret,
        "idToken": secret,
        "IDToken": secret,
        "apiKey": secret,
        "APIKey": secret,
        "ACCESS KEY": secret,
        "secret_key": secret,
        "privateKey": secret,
        "client-secret": secret,
        "sessionCookie": secret,
        "cookie": secret,
        "set-cookie": secret,
        "Authorization": secret,
        "authHeader": secret,
        "credential": secret,
        "credentials": secret,
        "idempotencyKey": secret,
        "userId": secret,
        "rawError": secret,
        "errorMessage": secret,
        "backendError": secret,
        "trustedContext": secret,
        "openaiApiKey": secret,
        "awsSecretAccessKey": secret,
        "x-api-key": secret,
        "apiToken": secret,
        "bearerToken": secret,
        "passwordHash": secret,
        "databaseUrl": secret,
        "redisDsn": secret,
        "postgresConnectionUrl": secret,
    }
    sensitive_variants = {
        "apiKeys": [secret],
        "refreshTokens": [secret],
        "passwords": [secret],
        "cookies": [secret],
        "clientSecrets": [secret],
        "credentialValues": {"primary": secret},
        "connectionString": secret,
        "connectionUrl": secret,
        "DSN": secret,
        "connectionStrings": [secret],
        "connectionUrls": [secret],
        "connectionUris": [secret],
        "DSNs": [secret],
        "databaseUrls": [secret],
    }
    legitimate = {
        "error_code": "TIMEOUT",
        "backend_status": "degraded",
        "token_count": 12,
        "authorization_status": "approved",
        "retry_with_same_idempotency_key": True,
    }
    legitimate_variants = {
        "credential_status": "missing",
        "password_required": True,
        "access_key_status": "active",
        "password_status": "invalid",
        "apiKeysConfigured": False,
        "refreshTokensPresent": True,
        "cookiesEnabled": False,
        "secretsCount": 0,
        "passwordsCount": 2,
        "credentialsCount": 1.5,
        "apiKeysCount": 10**309,
    }
    metadata_bypasses = {
        "apiKeysConfigured": [secret],
        "refreshTokensPresent": [secret],
        "cookiesEnabled": secret,
        "secretRequired": {"value": secret},
        "credentialsCount": -1,
        "passwordsCount": float("nan"),
        "cookiesCount": float("inf"),
        "secretsCount": True,
        "apiKeysCount": 10**5000,
        "password_status": "p@ssword",
        "retry_with_same_idempotency_key": secret,
        "token_count": [secret],
        "authorization_status": secret,
        "backend_status": {"detail": secret},
        "error_code": "p@ssword",
    }
    known_error_codes = [
        "TIMEOUT",
        "SERVICE_UNAVAILABLE",
        "UNAUTHORIZED",
        "FORBIDDEN",
        "NOT_FOUND",
        "VALIDATION_ERROR",
        "CONFLICT",
        "RATE_LIMITED",
        "INTERNAL_ERROR",
        "CIRCUIT_OPEN",
        "TOOL_REJECTED",
        "TOOL_FAILED",
        "PROTOCOL_ERROR",
    ]
    unknown_error_codes = [
        "AKIAIOSFODNN7EXAMPLE",
        "SUPERSECRETTOKEN123",
    ]
    manager = FakeManager(
        [
            SimpleNamespace(
                success=True,
                data={
                    "nested": {
                        **sensitive,
                        **legitimate,
                    },
                    "variants": {
                        **sensitive_variants,
                        **legitimate_variants,
                    },
                    "metadata_bypasses": metadata_bypasses,
                    "known_error_codes": [
                        {"error_code": code}
                        for code in known_error_codes
                    ],
                    "unknown_error_codes": [
                        {"error_code": code}
                        for code in unknown_error_codes
                    ],
                },
                error=None,
                rejected=False,
                cached=False,
                latency_ms=1,
            )
        ]
    )
    client = FakeClient(
        [
            response(tool_use("tool-1", "lookup", {})),
            response(text("Handled.")),
        ]
    )
    runtime = AgentRuntime(client, "model", manager)

    run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Look it up."}],
            allowed_tools=["lookup"],
        )
    )

    tool_result = client.messages.calls[1]["messages"][-1]["content"][0]
    payload = json.loads(tool_result["content"])
    nested = payload["data"]["nested"]
    variants = payload["data"]["variants"]
    bypasses = payload["data"]["metadata_bypasses"]
    preserved_codes = payload["data"]["known_error_codes"]
    redacted_codes = payload["data"]["unknown_error_codes"]
    assert secret not in tool_result["content"]
    assert "p@ssword" not in tool_result["content"]
    assert all(nested[key] == "[REDACTED]" for key in sensitive)
    assert {key: nested[key] for key in legitimate} == legitimate
    assert all(
        variants[key] == "[REDACTED]"
        for key in sensitive_variants
    )
    assert {
        key: variants[key]
        for key in legitimate_variants
    } == legitimate_variants
    assert all(
        bypasses[key] == "[REDACTED]"
        for key in metadata_bypasses
    )
    assert [
        item["error_code"]
        for item in preserved_codes
    ] == known_error_codes
    assert all(
        item["error_code"] == "[REDACTED]"
        for item in redacted_codes
    )


def test_successful_ticket_fallback_retry_control_remains_visible():
    manager = FakeManager(
        [
            SimpleNamespace(
                success=True,
                data={
                    "created": None,
                    "status": "unknown",
                    "retry_with_same_idempotency_key": True,
                    "message": "Retry with the same idempotency key.",
                },
                error="raw backend detail",
                rejected=False,
                cached=False,
                latency_ms=1,
                fallback_used=True,
            )
        ]
    )
    client = FakeClient(
        [
            response(tool_use("tool-1", "lookup", {})),
            response(text("Handled.")),
        ]
    )
    runtime = AgentRuntime(client, "model", manager)

    run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Create it."}],
            allowed_tools=["lookup"],
        )
    )

    tool_result = client.messages.calls[1]["messages"][-1]["content"][0]
    payload = json.loads(tool_result["content"])
    assert payload["data"]["retry_with_same_idempotency_key"] is True
    assert "raw backend detail" not in tool_result["content"]


def _deep_result(depth):
    value = "leaf"
    for _ in range(depth):
        value = {"next": value}
    return value


@pytest.mark.parametrize(
    "unsafe_data",
    [
        _deep_result(20),
        "x" * 20000,
        object(),
    ],
    ids=["too-deep", "oversized-string", "unserializable-object"],
)
def test_unsafe_tool_data_becomes_controlled_error(unsafe_data):
    manager = FakeManager(
        [
            SimpleNamespace(
                success=True,
                data=unsafe_data,
                error=None,
                rejected=False,
                cached=False,
                latency_ms=1,
            )
        ]
    )
    client = FakeClient(
        [
            response(tool_use("tool-1", "lookup", {})),
            response(text("Handled.")),
        ]
    )
    runtime = AgentRuntime(client, "model", manager)

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Look it up."}],
            allowed_tools=["lookup"],
        )
    )

    tool_result = client.messages.calls[1]["messages"][-1]["content"][0]
    assert tool_result == {
        "type": "tool_result",
        "tool_use_id": "tool-1",
        "content": "Tool result could not be safely processed.",
        "is_error": True,
    }
    assert result.tool_calls[0].success is False
    assert result.tool_calls[0].error_type == "unsafe_tool_result"
    assert len(repr(client.messages.calls[1])) < 10000


@pytest.mark.parametrize(
    ("failure_source", "manager"),
    [
        ("schema", FakeManager(schema_error=RuntimeError("schema-secret"))),
        ("provider", FakeManager()),
    ],
)
def test_provider_and_schema_exceptions_return_controlled_result(
    failure_source,
    manager,
):
    class RaisingMessages:
        def create(self, **_kwargs):
            raise RuntimeError("provider-secret")

    client = (
        SimpleNamespace(messages=RaisingMessages())
        if failure_source == "provider"
        else FakeClient([])
    )
    runtime = AgentRuntime(client, "model", manager)

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Hello."}],
            allowed_tools=["lookup"],
        )
    )

    assert result.stop_reason == "error"
    assert "temporarily unavailable" in result.text.lower()
    assert "secret" not in repr(result)


def test_manager_exception_returns_matching_generic_error_to_model():
    secret = "manager-backend-secret"

    class RaisingManager(FakeManager):
        async def call(self, name, params, context):
            raise RuntimeError(secret)

    client = FakeClient(
        [
            response(tool_use("tool-1", "lookup", {})),
            response(text("Handled.")),
        ]
    )
    runtime = AgentRuntime(client, "model", RaisingManager())

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Look it up."}],
            allowed_tools=["lookup"],
        )
    )

    tool_result = client.messages.calls[1]["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "tool-1"
    assert tool_result["content"] == "Tool execution failed."
    assert tool_result["is_error"] is True
    assert secret not in repr(client.messages.calls[1])
    assert secret not in repr(result)


def test_runtime_does_not_swallow_caller_cancellation():
    started = asyncio.Event()

    async def never_returns():
        started.set()
        await asyncio.Event().wait()

    async def scenario():
        runtime = AgentRuntime(
            FakeClient([never_returns]),
            "model",
            FakeManager(),
            total_timeout_s=30,
        )
        task = asyncio.create_task(
            runtime.run(
                system_prompt="Support.",
                messages=[{"role": "user", "content": "Hello."}],
                allowed_tools=[],
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(scenario())


def test_forbidden_or_unknown_param_names_are_not_recorded_in_trace():
    attacker_name = "steal_" + "x" * 50
    attacker_key = "credential_" + "y" * 50
    client = FakeClient(
        [
            response(
                tool_use(
                    "tool-1",
                    attacker_name,
                    {attacker_key: "secret"},
                )
            ),
            response(text("Handled.")),
        ]
    )
    runtime = AgentRuntime(client, "model", FakeManager())

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Do it."}],
            allowed_tools=["lookup"],
        )
    )

    trace_repr = repr(result.tool_calls)
    assert attacker_name not in trace_repr
    assert attacker_key not in trace_repr
    assert result.tool_calls[0].name == "forbidden_tool"
    assert result.tool_calls[0].params == {}


def test_allowed_trace_only_contains_bounded_schema_property_names():
    unknown_key = "unknown_" + "z" * 100
    client = FakeClient(
        [
            response(
                tool_use(
                    "tool-1",
                    "lookup",
                    {"value": 1, unknown_key: "secret"},
                )
            ),
            response(text("Handled.")),
        ]
    )
    runtime = AgentRuntime(client, "model", FakeManager())

    result = run(
        runtime.run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Look it up."}],
            allowed_tools=["lookup"],
        )
    )

    assert result.tool_calls[0].params == {"value": "[REDACTED]"}
    assert unknown_key not in repr(result.tool_calls)
