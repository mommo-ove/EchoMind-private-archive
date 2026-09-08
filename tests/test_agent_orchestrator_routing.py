import asyncio
import json
import re
import unicodedata
from copy import deepcopy
from types import SimpleNamespace

import pytest

from agents.agent_runtime import (
    AgentRuntime,
    RUNTIME_ERROR_RESPONSE,
    RuntimeResult,
)
from agents.agent_orchestrator import (
    AccountAgent,
    AgentOrchestrator,
    AgentResponse,
    AgentType,
    GeneralAgent,
    Request,
    TechnicalAgent,
)
from campus.store import CampusStore
from campus.tools import CampusToolset, register_campus_tools
from core.intent_recognizer import IntentCategory, UrgencyLevel
from mcp.tool_manager import AGENT_TOOL_ALLOWLIST, MCPToolManager


def make_orchestrator_with_recorded_execution():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [object()],
        AgentType.TECHNICAL: [object()],
        AgentType.BILLING: [object()],
        AgentType.ACCOUNT: [object()],
    }
    calls = []

    async def execute(_req, agent_type):
        calls.append(agent_type)
        return AgentResponse(
            agent_type=agent_type,
            content=agent_type.value,
            success=True,
        )

    orchestrator._execute = execute
    return orchestrator, calls


def test_account_intent_routes_to_dedicated_account_agent():
    orchestrator, calls = make_orchestrator_with_recorded_execution()
    request = Request(
        message="账号被锁定了",
        user_id="student-1",
        conv_id="conv-account",
        intent=IntentCategory.ACCOUNT,
        intent_scores={IntentCategory.ACCOUNT: 0.94},
        matched_intents=[IntentCategory.ACCOUNT],
        urgency=UrgencyLevel.LOW,
    )

    result = asyncio.run(orchestrator.run(request))

    assert calls == [AgentType.ACCOUNT]
    assert result.agent_type == AgentType.ACCOUNT


def test_three_business_intents_can_route_to_three_agents():
    orchestrator, calls = make_orchestrator_with_recorded_execution()
    request = Request(
        message="网络、扣款和账号都有问题",
        user_id="student-1",
        conv_id="conv-three",
        intent=IntentCategory.TECHNICAL,
        matched_intents=[
            IntentCategory.TECHNICAL,
            IntentCategory.BILLING,
            IntentCategory.ACCOUNT,
        ],
        urgency=UrgencyLevel.LOW,
    )

    result = asyncio.run(orchestrator.run(request))

    assert calls == [AgentType.TECHNICAL, AgentType.BILLING, AgentType.ACCOUNT]
    assert result.agent_types == [AgentType.TECHNICAL, AgentType.BILLING, AgentType.ACCOUNT]


def test_single_matched_technical_intent_routes_to_technical_agent():
    orchestrator, calls = make_orchestrator_with_recorded_execution()
    request = Request(
        message="请帮我处理一下 500 错误",
        user_id="student-1",
        conv_id="conv-1",
        intent=IntentCategory.REQUEST,
        intent_scores={IntentCategory.TECHNICAL: 0.82},
        matched_intents=[IntentCategory.TECHNICAL],
        urgency=UrgencyLevel.LOW,
    )

    result = asyncio.run(orchestrator.run(request))

    assert calls == [AgentType.TECHNICAL]
    assert result.agent_type == AgentType.TECHNICAL


def test_critical_urgency_still_overrides_single_matched_intent():
    orchestrator, calls = make_orchestrator_with_recorded_execution()
    request = Request(
        message="系统 500，已经造成严重事故",
        user_id="student-1",
        conv_id="conv-2",
        intent=IntentCategory.REQUEST,
        intent_scores={IntentCategory.TECHNICAL: 0.82},
        matched_intents=[IntentCategory.TECHNICAL],
        urgency=UrgencyLevel.CRITICAL,
    )

    result = asyncio.run(orchestrator.run(request))

    assert calls == [AgentType.ESCALATION]
    assert result.escalated is True


def test_multi_label_scores_route_to_two_agents_without_domain_keywords():
    orchestrator, calls = make_orchestrator_with_recorded_execution()
    request = Request(
        message="这个情况两边都有问题",
        user_id="student-1",
        conv_id="conv-3",
        intent=IntentCategory.TECHNICAL,
        intent_scores={
            IntentCategory.TECHNICAL: 0.91,
            IntentCategory.BILLING: 0.86,
        },
        matched_intents=[
            IntentCategory.TECHNICAL,
            IntentCategory.BILLING,
        ],
        urgency=UrgencyLevel.LOW,
    )

    result = asyncio.run(orchestrator.run(request))

    assert calls == [AgentType.TECHNICAL, AgentType.BILLING]
    assert result.agent_type == AgentType.TECHNICAL
    assert result.intent_scores == {
        IntentCategory.TECHNICAL: 0.91,
        IntentCategory.BILLING: 0.86,
    }
    assert result.matched_intents == [
        IntentCategory.TECHNICAL,
        IntentCategory.BILLING,
    ]
    assert result.agent_types == [AgentType.TECHNICAL, AgentType.BILLING]


def test_intent_below_multi_label_threshold_does_not_add_second_agent():
    orchestrator, calls = make_orchestrator_with_recorded_execution()
    request = Request(
        message="这个情况两边都有问题",
        user_id="student-1",
        conv_id="conv-4",
        intent=IntentCategory.TECHNICAL,
        intent_scores={
            IntentCategory.TECHNICAL: 0.91,
            IntentCategory.BILLING: 0.52,
        },
        matched_intents=[IntentCategory.TECHNICAL],
        urgency=UrgencyLevel.LOW,
    )

    asyncio.run(orchestrator.run(request))

    assert calls == [AgentType.TECHNICAL]


def test_escalation_overrides_multiple_matched_specialist_intents():
    orchestrator, calls = make_orchestrator_with_recorded_execution()
    request = Request(
        message="马上处理这个复合问题",
        user_id="student-1",
        conv_id="conv-5",
        intent=IntentCategory.ESCALATION,
        intent_scores={
            IntentCategory.ESCALATION: 0.92,
            IntentCategory.TECHNICAL: 0.88,
            IntentCategory.BILLING: 0.84,
        },
        matched_intents=[
            IntentCategory.ESCALATION,
            IntentCategory.TECHNICAL,
            IntentCategory.BILLING,
        ],
        urgency=UrgencyLevel.CRITICAL,
    )

    result = asyncio.run(orchestrator.run(request))

    assert calls == [AgentType.ESCALATION]
    assert result.escalated is True


class RecordingRuntime:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if self.results:
            return self.results.pop(0)
        return RuntimeResult(
            text="runtime response",
            stop_reason="end_turn",
            tool_calls=[],
            ticket_ids=[],
        )


class FakeMessages:
    def __init__(self, text="legacy response"):
        self.text = text
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return SimpleNamespace(
            content=[{"type": "text", "text": self.text}],
        )


def test_orchestrator_injects_exact_per_agent_tool_allowlists():
    runtime = RecordingRuntime()
    orchestrator = AgentOrchestrator(
        api_key="test-key",
        runtime=runtime,
        intent_recognizer=object(),
    )

    assert orchestrator._pool[AgentType.GENERAL][0]._allowed_tools == tuple(
        AGENT_TOOL_ALLOWLIST["general"]
    )
    assert orchestrator._pool[AgentType.TECHNICAL][0]._allowed_tools == tuple(
        AGENT_TOOL_ALLOWLIST["technical"]
    )
    assert orchestrator._pool[AgentType.BILLING][0]._allowed_tools == tuple(
        AGENT_TOOL_ALLOWLIST["billing"]
    )
    assert isinstance(orchestrator._pool[AgentType.ACCOUNT][0], AccountAgent)
    assert orchestrator._pool[AgentType.ACCOUNT][0]._allowed_tools == tuple(
        AGENT_TOOL_ALLOWLIST["account"]
    )


def test_runtime_receives_trusted_request_context_and_agent_allowlist():
    runtime = RecordingRuntime()
    agent = TechnicalAgent(
        SimpleNamespace(messages=FakeMessages()),
        "model",
        runtime=runtime,
        allowed_tools=AGENT_TOOL_ALLOWLIST["technical"],
    )
    request = Request(
        message="Check campus networking",
        user_id="student-1",
        conv_id="conv-1",
        principal_id="authenticated-student",
        request_id="request-1",
        intent=IntentCategory.TECHNICAL,
        urgency=UrgencyLevel.LOW,
        citations=[{
            "id": "kb-network-401",
            "title": "Campus network",
            "content": "Clear stale authentication state.",
        }],
    )

    response = asyncio.run(agent.handle(request))

    assert response.success is True
    assert runtime.calls[0]["allowed_tools"] == AGENT_TOOL_ALLOWLIST["technical"]
    assert runtime.calls[0]["question"] == "Check campus networking"
    assert runtime.calls[0]["citations"] == request.citations
    trusted_context = runtime.calls[0]["context"]
    assert trusted_context["user_id"] == "authenticated-student"
    assert re.fullmatch(
        r"scope_[0-9a-f]{64}",
        trusted_context["idempotency_scope"],
    )
    model_payload = json.dumps(
        {
            "system_prompt": runtime.calls[0]["system_prompt"],
            "messages": runtime.calls[0]["messages"],
        }
    )
    assert "student-1" not in model_payload
    assert "authenticated-student" not in model_payload
    assert "conv-1" not in repr(trusted_context.values())
    assert "request-1" not in repr(trusted_context.values())
    assert "user_id" not in model_payload
    assert "idempotency_scope" not in model_payload


def test_runtime_verification_metadata_is_propagated_by_agent():
    from core.evidence_verifier import VerificationReport

    runtime = RecordingRuntime(results=[RuntimeResult(
        text="grounded",
        stop_reason="end_turn",
        verification=VerificationReport(
            passed=True,
            checked_claims=2,
        ),
        reflection_count=1,
        corrected=True,
    )])
    agent = TechnicalAgent(
        SimpleNamespace(messages=FakeMessages()),
        "model",
        runtime=runtime,
        allowed_tools=AGENT_TOOL_ALLOWLIST["technical"],
    )

    response = asyncio.run(agent.handle(Request(
        message="Check the ticket",
        user_id="student-1",
        conv_id="conv-1",
        intent=IntentCategory.TECHNICAL,
        urgency=UrgencyLevel.LOW,
    )))

    assert response.evidence_verification == {
        "checked": True,
        "passed": True,
        "checked_claims": 2,
        "issue_codes": [],
        "abstained": False,
        "reflection_count": 1,
        "corrected": True,
        "safe_fallback_used": False,
    }


def test_no_runtime_uses_exact_legacy_llm_call_shape():
    messages = FakeMessages()
    agent = TechnicalAgent(
        SimpleNamespace(messages=messages),
        "legacy-model",
    )
    request = Request(
        message="Legacy request",
        user_id="student-1",
        conv_id="conv-legacy",
        context="Trusted background",
    )

    response = asyncio.run(agent.handle(request))

    assert response.content == "legacy response"
    assert messages.calls == [
        {
            "model": "legacy-model",
            "max_tokens": 1024,
            "system": agent.system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": "[背景信息]\nTrusted background",
                },
                {
                    "role": "assistant",
                    "content": "好的，我已了解背景信息。",
                },
                {"role": "user", "content": "Legacy request"},
            ],
        }
    ]


def test_knowledge_context_adds_strict_grounding_rules_to_agent_prompt():
    agent = TechnicalAgent(
        SimpleNamespace(messages=FakeMessages()),
        "model",
    )
    request = Request(
        message="校园网401怎么处理",
        user_id="student-1",
        conv_id="conv-grounded",
        context=(
            "[Knowledge base references]\n"
            "[Citation doc-401]\n"
            "Content: 清除旧认证状态后重新连接。"
        ),
    )

    prompt = agent._build_system_prompt(request)

    assert "只能依据知识库引用和成功的工具结果陈述事实" in prompt
    assert "无法从证据确认时，明确说明无法确认" in prompt
    assert "不得补充证据中没有的时间、概率、政策或根因" in prompt
    assert "引用对应的 Citation ID" in prompt


def test_legacy_llm_output_is_unicode_safe():
    messages = FakeMessages("正常😀\nnext\ud800\x00\x85\u202e\u2066")
    agent = TechnicalAgent(
        SimpleNamespace(messages=messages),
        "legacy-model",
    )

    response = asyncio.run(
        agent.handle(Request("Help", "display", "conv"))
    )

    assert response.content.startswith("正常😀\nnext")
    assert all(
        character == "\n"
        or unicodedata.category(character) not in {"Cc", "Cf", "Cs"}
        for character in response.content
    )
    json.dumps(
        {"content": response.content},
        ensure_ascii=False,
    ).encode("utf-8")


def test_runtime_failure_is_unsuccessful_and_does_not_expose_raw_error():
    runtime = RecordingRuntime(
        [
            SimpleNamespace(
                text="backend-secret",
                stop_reason="error",
                tool_calls=[
                    SimpleNamespace(
                        name="query_network_status",
                        success=False,
                        params={},
                        latency_ms=1.0,
                        cached=False,
                        error_type="tool_failure",
                    )
                ],
                ticket_ids=[],
                raw_error="backend-secret",
            )
        ]
    )
    agent = TechnicalAgent(
        SimpleNamespace(messages=FakeMessages()),
        "model",
        runtime=runtime,
        allowed_tools=AGENT_TOOL_ALLOWLIST["technical"],
    )

    response = asyncio.run(
        agent.handle(
            Request(
                message="Check network",
                user_id="student-1",
                conv_id="conv-1",
            )
        )
    )

    assert response.success is False
    assert response.content == "The assistant service is temporarily unavailable."
    assert "backend-secret" not in repr(response)
    assert response.tool_calls == [
        {
            "name": "query_network_status",
            "success": False,
            "params": {},
            "latency_ms": 1.0,
            "cached": False,
            "error_type": "tool_failure",
        }
    ]


def test_parallel_result_preserves_all_traces_and_unique_ticket_ids_in_order():
    orchestrator, _calls = make_orchestrator_with_recorded_execution()

    async def execute(_req, agent_type):
        if agent_type == AgentType.TECHNICAL:
            return AgentResponse(
                agent_type=agent_type,
                content="Network checked. Save a screenshot.",
                tool_calls=[{"name": "query_network_status", "success": True}],
                ticket_ids=["T-technical", "T-shared"],
            )
        return AgentResponse(
            agent_type=agent_type,
            content="Charge checked. Save a screenshot.",
            tool_calls=[{"name": "query_campus_card", "success": True}],
            ticket_ids=["T-shared", "T-billing"],
        )

    orchestrator._execute = execute
    request = Request(
        message="Check both",
        user_id="student-1",
        conv_id="conv-1",
        intent=IntentCategory.TECHNICAL,
        urgency=UrgencyLevel.LOW,
    )

    result = asyncio.run(
        orchestrator.run_parallel(
            request,
            [AgentType.TECHNICAL, AgentType.BILLING],
        )
    )

    assert result.tool_calls == [
        {"name": "query_network_status", "success": True},
        {"name": "query_campus_card", "success": True},
    ]
    assert result.ticket_ids == ["T-technical", "T-shared", "T-billing"]
    assert result.response.count("Save a screenshot.") == 1
    assert result.response.index("[technical]") < result.response.index("[billing]")


def test_failed_specialist_falls_back_to_general_and_keeps_both_traces():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)

    class StubAgent:
        def __init__(self, response):
            self.response = response
            self.calls = 0

        async def handle(self, _req):
            self.calls += 1
            return self.response

    specialist = StubAgent(
        AgentResponse(
            agent_type=AgentType.TECHNICAL,
            content="controlled specialist failure",
            success=False,
            tool_calls=[{"name": "query_network_status", "success": False}],
            ticket_ids=["T-partial"],
        )
    )
    general = StubAgent(
        AgentResponse(
            agent_type=AgentType.GENERAL,
            content="General fallback",
            tool_calls=[{"name": "get_ticket", "success": True}],
            ticket_ids=["T-partial", "T-general"],
        )
    )
    orchestrator._best_agent = lambda agent_type: {
        AgentType.TECHNICAL: specialist,
        AgentType.GENERAL: general,
    }.get(agent_type)

    response = asyncio.run(
        orchestrator._execute(
            Request("Help", "student-1", "conv-1"),
            AgentType.TECHNICAL,
        )
    )

    assert specialist.calls == 1
    assert general.calls == 1
    assert response.agent_type == AgentType.GENERAL
    assert response.tool_calls == [
        {"name": "query_network_status", "success": False},
        {"name": "get_ticket", "success": True},
    ]
    assert response.ticket_ids == ["T-partial", "T-general"]


def test_single_result_preserves_tool_traces_and_unique_ticket_ids():
    orchestrator, _calls = make_orchestrator_with_recorded_execution()

    async def execute(_req, agent_type):
        return AgentResponse(
            agent_type=agent_type,
            content="Network checked.",
            tool_calls=[{"name": "query_network_status", "success": True}],
            ticket_ids=["T-network", "T-network"],
        )

    orchestrator._execute = execute
    request = Request(
        message="Check network",
        user_id="student-1",
        conv_id="conv-1",
        intent=IntentCategory.TECHNICAL,
        urgency=UrgencyLevel.LOW,
    )

    result = asyncio.run(orchestrator.run(request))

    assert result.tool_calls == [
        {"name": "query_network_status", "success": True}
    ]
    assert result.ticket_ids == ["T-network"]


def test_escalation_bypasses_runtime_tool_use():
    runtime = RecordingRuntime()
    messages = FakeMessages("Please contact human support.")
    agent = GeneralAgent(
        SimpleNamespace(messages=messages),
        "model",
        runtime=runtime,
        allowed_tools=AGENT_TOOL_ALLOWLIST["general"],
    )

    response = asyncio.run(
        agent.handle(
            Request(
                message="Escalate now",
                user_id="student-1",
                conv_id="conv-1",
                intent=IntentCategory.ESCALATION,
                urgency=UrgencyLevel.CRITICAL,
                matched_intents=[IntentCategory.ESCALATION],
            )
        )
    )

    assert response.success is True
    assert runtime.calls == []
    assert len(messages.calls) == 1


def test_display_user_id_alone_is_not_forwarded_as_tool_principal():
    runtime = RecordingRuntime()
    agent = TechnicalAgent(
        SimpleNamespace(messages=FakeMessages()),
        "model",
        runtime=runtime,
        allowed_tools=AGENT_TOOL_ALLOWLIST["technical"],
    )

    asyncio.run(
        agent.handle(
            Request(
                message="Create a ticket",
                user_id="attacker-selected-user",
                conv_id="conv-1",
                request_id="request-1",
            )
        )
    )

    trusted_context = runtime.calls[0]["context"]
    assert set(trusted_context) == {"idempotency_scope"}
    assert re.fullmatch(
        r"scope_[0-9a-f]{64}",
        trusted_context["idempotency_scope"],
    )
    assert "attacker-selected-user" not in repr(runtime.calls[0])


@pytest.mark.parametrize(
    ("text_value", "stop_reason"),
    [
        ("", "end_turn"),
        ("   \n\t", "end_turn"),
        (None, "end_turn"),
        ({"text": "forged"}, "end_turn"),
        ("Looks valid", "unknown_success"),
    ],
)
def test_runtime_requires_recognized_stop_and_nonempty_string_text(
    text_value,
    stop_reason,
):
    runtime = RecordingRuntime(
        [
            SimpleNamespace(
                text=text_value,
                stop_reason=stop_reason,
                tool_calls=[],
                ticket_ids=[],
            )
        ]
    )
    agent = TechnicalAgent(
        SimpleNamespace(messages=FakeMessages()),
        "model",
        runtime=runtime,
        allowed_tools=[],
    )

    response = asyncio.run(
        agent.handle(Request("Help", "display", "conv"))
    )

    assert response.success is False
    assert response.content == RUNTIME_ERROR_RESPONSE
    assert agent.stats.success == 0


def test_empty_runtime_end_turn_enables_general_fallback():
    technical = TechnicalAgent(
        SimpleNamespace(messages=FakeMessages()),
        "model",
        runtime=RecordingRuntime(
            [
                SimpleNamespace(
                    text=" ",
                    stop_reason="end_turn",
                    tool_calls=[],
                    ticket_ids=[],
                )
            ]
        ),
        allowed_tools=[],
    )
    general = GeneralAgent(
        SimpleNamespace(messages=FakeMessages("general fallback")),
        "model",
    )
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [general],
        AgentType.TECHNICAL: [technical],
        AgentType.BILLING: [],
    }

    result = asyncio.run(
        orchestrator.run(
            Request(
                "Help",
                "display",
                "conv",
                intent=IntentCategory.TECHNICAL,
                urgency=UrgencyLevel.LOW,
            )
        )
    )

    assert result.response == "general fallback"
    assert result.agent_type == AgentType.GENERAL


def test_idempotency_scope_is_stable_per_request_and_unique_across_requests():
    runtime = RecordingRuntime()
    agent = TechnicalAgent(
        SimpleNamespace(messages=FakeMessages()),
        "model",
        runtime=runtime,
        allowed_tools=AGENT_TOOL_ALLOWLIST["technical"],
    )
    same_request = Request(
        "Help",
        "display-user",
        "conv",
        principal_id="principal",
        request_id="request-a",
    )
    other_request = Request(
        "Help",
        "display-user",
        "conv",
        principal_id="principal",
        request_id="request-b",
    )

    async def exercise_concurrently():
        await asyncio.gather(
            agent.handle(same_request),
            agent.handle(same_request),
            agent.handle(other_request),
        )

    asyncio.run(exercise_concurrently())

    scopes = [call["context"]["idempotency_scope"] for call in runtime.calls]
    assert scopes[0] == scopes[1]
    assert scopes[2] != scopes[0]
    assert all(re.fullmatch(r"scope_[0-9a-f]{64}", scope) for scope in scopes)
    assert "conv" not in repr(scopes)
    assert "request-a" not in repr(scopes)
    assert "request-b" not in repr(scopes)


def test_idempotency_scope_hashes_delimiters_without_ambiguity():
    runtime = RecordingRuntime()
    agent = TechnicalAgent(
        SimpleNamespace(messages=FakeMessages()),
        "model",
        runtime=runtime,
        allowed_tools=AGENT_TOOL_ALLOWLIST["technical"],
    )

    asyncio.run(
        agent.handle(
            Request(
                "Help",
                "display",
                "a:b",
                request_id="c",
            )
        )
    )
    asyncio.run(
        agent.handle(
            Request(
                "Help",
                "display",
                "a",
                request_id="b:c",
            )
        )
    )

    scopes = [call["context"]["idempotency_scope"] for call in runtime.calls]
    assert scopes[0] != scopes[1]
    assert all(re.fullmatch(r"scope_[0-9a-f]{64}", scope) for scope in scopes)
    assert "a:b" not in repr(scopes)


def test_matched_escalation_marks_result_escalated_when_primary_is_not():
    orchestrator, calls = make_orchestrator_with_recorded_execution()
    request = Request(
        message="Please escalate this technical issue",
        user_id="student-1",
        conv_id="conv-1",
        intent=IntentCategory.TECHNICAL,
        urgency=UrgencyLevel.LOW,
        matched_intents=[
            IntentCategory.TECHNICAL,
            IntentCategory.ESCALATION,
        ],
    )

    result = asyncio.run(orchestrator.run(request))

    assert calls == [AgentType.ESCALATION]
    assert result.escalated is True


@pytest.mark.parametrize(
    "malformed",
    [
        "ticket_123",
        b"ticket_123",
        {"ticket_123": True},
    ],
)
def test_malformed_ticket_id_collection_is_rejected(malformed):
    runtime = RecordingRuntime(
        [
            SimpleNamespace(
                text="Done.",
                stop_reason="end_turn",
                tool_calls=[],
                ticket_ids=malformed,
            )
        ]
    )
    agent = TechnicalAgent(
        SimpleNamespace(messages=FakeMessages()),
        "model",
        runtime=runtime,
        allowed_tools=[],
    )

    response = asyncio.run(
        agent.handle(Request("Help", "display", "conv"))
    )

    assert response.ticket_ids == []


def test_parallel_preserves_identical_and_distinct_tool_invocations_in_order():
    orchestrator, _calls = make_orchestrator_with_recorded_execution()
    first_fingerprint = "a" * 64
    second_fingerprint = "b" * 64

    async def execute(_req, agent_type):
        first_call = {
            "name": "get_ticket",
            "success": True,
            "params": {"ticket_id": "ticket-1"},
            "latency_ms": 0,
            "cached": False,
            "error_type": None,
            "fingerprint": first_fingerprint,
        }
        if agent_type == AgentType.TECHNICAL:
            return AgentResponse(
                agent_type=agent_type,
                content="Technical.",
                tool_calls=[first_call],
            )
        return AgentResponse(
            agent_type=agent_type,
            content="Billing.",
            tool_calls=[
                {
                    **first_call,
                    "latency_ms": 25,
                    "cached": True,
                },
                {
                    **first_call,
                    "params": {"ticket_id": "ticket-2"},
                    "fingerprint": second_fingerprint,
                },
            ],
        )

    orchestrator._execute = execute
    result = asyncio.run(
        orchestrator.run_parallel(
            Request("Both", "display", "conv"),
            [AgentType.TECHNICAL, AgentType.BILLING],
        )
    )

    assert result.tool_calls == [
        {
            "name": "get_ticket",
            "success": True,
            "params": {"ticket_id": "[REDACTED]"},
            "latency_ms": 0.0,
            "cached": False,
            "error_type": None,
            "fingerprint": first_fingerprint,
        },
        {
            "name": "get_ticket",
            "success": True,
            "params": {"ticket_id": "[REDACTED]"},
            "latency_ms": 25.0,
            "cached": True,
            "error_type": None,
            "fingerprint": first_fingerprint,
        },
        {
            "name": "get_ticket",
            "success": True,
            "params": {"ticket_id": "[REDACTED]"},
            "latency_ms": 0.0,
            "cached": False,
            "error_type": None,
            "fingerprint": second_fingerprint,
        },
    ]
    assert "ticket-1" not in repr(result.tool_calls)
    assert "ticket-2" not in repr(result.tool_calls)


def test_fallback_preserves_identical_legacy_tool_invocations_in_order():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)

    class StubAgent:
        def __init__(self, response):
            self.response = response

        async def handle(self, _req):
            return self.response

    specialist_trace = {
        "name": "get_ticket",
        "success": True,
        "params": {"ticket_id": "raw-ticket-secret"},
    }
    general_trace = {
        "name": "get_ticket",
        "success": True,
        "params": {"ticket_id": "raw-ticket-secret"},
    }
    specialist = StubAgent(
        AgentResponse(
            agent_type=AgentType.TECHNICAL,
            content="failed",
            success=False,
            tool_calls=[specialist_trace],
        )
    )
    general = StubAgent(
        AgentResponse(
            agent_type=AgentType.GENERAL,
            content="fallback",
            tool_calls=[general_trace],
        )
    )
    orchestrator._best_agent = lambda agent_type: {
        AgentType.TECHNICAL: specialist,
        AgentType.GENERAL: general,
    }.get(agent_type)

    response = asyncio.run(
        orchestrator._execute(
            Request("Help", "display", "conv"),
            AgentType.TECHNICAL,
        )
    )

    assert response.tool_calls == [
        {
            "name": "get_ticket",
            "success": True,
            "params": {"ticket_id": "[REDACTED]"},
        },
        {
            "name": "get_ticket",
            "success": True,
            "params": {"ticket_id": "[REDACTED]"},
        },
    ]
    assert "raw-ticket-secret" not in repr(response.tool_calls)


def test_parallel_tool_invocation_audit_trail_has_global_bound():
    orchestrator, _calls = make_orchestrator_with_recorded_execution()

    async def execute(_req, agent_type):
        return AgentResponse(
            agent_type=agent_type,
            content=f"{agent_type.value}.",
            tool_calls=[
                {"name": "get_ticket", "success": True}
                for _ in range(80)
            ],
        )

    orchestrator._execute = execute
    result = asyncio.run(
        orchestrator.run_parallel(
            Request("Both", "display", "conv"),
            [AgentType.TECHNICAL, AgentType.BILLING],
        )
    )

    assert len(result.tool_calls) == 128
    assert result.tool_calls == [
        {"name": "get_ticket", "success": True}
        for _ in range(128)
    ]


@pytest.mark.parametrize(
    "fingerprint",
    [
        "a" * 63,
        "A" * 64,
        "g" * 64,
        "a" * 64 + "suffix",
        123,
    ],
)
def test_runtime_trace_rejects_non_strict_fingerprint_values(fingerprint):
    runtime_result = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                name="lookup",
                success=True,
                params={},
                latency_ms=1,
                cached=False,
                error_type=None,
                fingerprint=fingerprint,
            )
        ]
    )

    traces = TechnicalAgent._runtime_tool_calls(runtime_result)

    assert traces == [
        {
            "name": "lookup",
            "success": True,
            "params": {},
            "latency_ms": 1.0,
            "cached": False,
            "error_type": None,
        }
    ]


def test_real_create_ticket_id_flows_through_runtime_and_orchestrator(tmp_path):
    store = CampusStore(tmp_path / "campus.db")
    manager = MCPToolManager(api_key="test-key")
    register_campus_tools(manager, CampusToolset(store))

    class ScriptedMessages:
        def __init__(self):
            self.responses = [
                SimpleNamespace(
                    content=[
                        {
                            "type": "tool_use",
                            "id": "create-1",
                            "name": "create_ticket",
                            "input": {
                                "category": "technical",
                                "title": "Cannot connect",
                                "description": "Campus network login fails",
                            },
                        }
                    ]
                ),
                SimpleNamespace(
                    content=[
                        {
                            "type": "text",
                            "text": "Ticket created.",
                        }
                    ]
                ),
            ]

        async def create(self, **_kwargs):
            return self.responses.pop(0)

    runtime = AgentRuntime(
        SimpleNamespace(messages=ScriptedMessages()),
        "model",
        manager,
    )
    technical = TechnicalAgent(
        SimpleNamespace(messages=FakeMessages()),
        "model",
        runtime=runtime,
        allowed_tools=AGENT_TOOL_ALLOWLIST["technical"],
    )
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [],
        AgentType.TECHNICAL: [technical],
        AgentType.BILLING: [],
    }
    request = Request(
        "Create a ticket",
        "display-user",
        "conv-1",
        principal_id="authenticated-user",
        request_id="request-1",
        intent=IntentCategory.TECHNICAL,
        urgency=UrgencyLevel.LOW,
    )

    result = asyncio.run(orchestrator.run(request))

    assert len(result.ticket_ids) == 1
    assert store.get_ticket(
        result.ticket_ids[0],
        user_id="authenticated-user",
    )["id"] == result.ticket_ids[0]


def test_real_get_ticket_id_flows_through_manager_and_runtime(tmp_path):
    store = CampusStore(tmp_path / "campus.db")
    ticket = store.create_ticket(
        idempotency_key="seed-ticket",
        user_id="authenticated-user",
        category="technical",
        title="Cannot connect",
        description="Campus network login fails",
    )
    manager = MCPToolManager(api_key="test-key")
    register_campus_tools(manager, CampusToolset(store))

    class ScriptedMessages:
        def __init__(self):
            self.responses = [
                SimpleNamespace(
                    content=[
                        {
                            "type": "tool_use",
                            "id": "get-1",
                            "name": "get_ticket",
                            "input": {"ticket_id": ticket["id"]},
                        }
                    ]
                ),
                SimpleNamespace(
                    content=[{"type": "text", "text": "Ticket found."}]
                ),
            ]

        async def create(self, **_kwargs):
            return self.responses.pop(0)

    result = asyncio.run(
        AgentRuntime(
            SimpleNamespace(messages=ScriptedMessages()),
            "model",
            manager,
        ).run(
            system_prompt="Support.",
            messages=[{"role": "user", "content": "Find my ticket."}],
            allowed_tools=["get_ticket"],
            context={
                "user_id": "authenticated-user",
                "idempotency_scope": "scope-for-get",
            },
        )
    )

    assert result.ticket_ids == [ticket["id"]]
    assert ticket["id"] not in repr(result.tool_calls)


def test_untrusted_display_user_is_rejected_for_user_tool_but_public_tool_works(
    tmp_path,
):
    store = CampusStore(tmp_path / "campus.db")
    manager = MCPToolManager(api_key="test-key")
    register_campus_tools(manager, CampusToolset(store))

    class ScriptedMessages:
        def __init__(self, name, params):
            self.responses = [
                SimpleNamespace(
                    content=[
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": name,
                            "input": params,
                        }
                    ]
                ),
                SimpleNamespace(
                    content=[{"type": "text", "text": "Handled."}]
                ),
            ]

        async def create(self, **_kwargs):
            return self.responses.pop(0)

    async def invoke(name, params):
        runtime = AgentRuntime(
            SimpleNamespace(messages=ScriptedMessages(name, params)),
            "model",
            manager,
        )
        return await TechnicalAgent(
            SimpleNamespace(messages=FakeMessages()),
            "model",
            runtime=runtime,
            allowed_tools=AGENT_TOOL_ALLOWLIST["technical"],
        ).handle(
            Request(
                "Run tool",
                "attacker-selected-user",
                "conv",
                request_id="request",
            )
        )

    rejected = asyncio.run(
        invoke(
            "create_ticket",
            {
                "category": "technical",
                "title": "Unauthorized",
                "description": "Must not be created",
            },
        )
    )
    public = asyncio.run(
        invoke("query_network_status", {"site": "campus"})
    )

    assert rejected.tool_calls[0]["success"] is False
    assert rejected.tool_calls[0]["error_type"] == "tool_rejected"
    assert rejected.ticket_ids == []
    assert public.tool_calls[0]["success"] is True


def test_run_parallel_rejects_empty_agent_list():
    orchestrator, _calls = make_orchestrator_with_recorded_execution()

    with pytest.raises(ValueError, match="agent_types"):
        asyncio.run(
            orchestrator.run_parallel(
                Request("Help", "display", "conv"),
                [],
            )
        )


def test_runtime_text_sanitization_preserves_chinese_emoji_and_newline():
    runtime = RecordingRuntime(
        [
            SimpleNamespace(
                text=(
                    "正常😀\n"
                    "A\u034fB\ufe0fC\ud800\x00\x85\u202e\u2066"
                ),
                stop_reason="end_turn",
                tool_calls=[],
                ticket_ids=[],
            )
        ]
    )
    agent = TechnicalAgent(
        SimpleNamespace(messages=FakeMessages()),
        "model",
        runtime=runtime,
        allowed_tools=[],
    )

    response = asyncio.run(
        agent.handle(Request("Help", "display", "conv"))
    )

    assert response.success is True
    assert response.content.startswith("正常😀\nABC")
    assert "\u034f" not in response.content
    assert "\ufe0f" not in response.content
    assert all(
        character == "\n"
        or unicodedata.category(character) not in {"Cc", "Cf", "Cs"}
        for character in response.content
    )
    json.dumps(
        {"content": response.content},
        ensure_ascii=False,
    ).encode("utf-8")


def test_display_user_cannot_authorize_real_create_ticket(tmp_path):
    manager = MCPToolManager(api_key="test-key")
    register_campus_tools(
        manager,
        CampusToolset(CampusStore(tmp_path / "campus.db")),
    )

    class ScriptedMessages:
        def __init__(self):
            self.responses = [
                SimpleNamespace(
                    content=[
                        {
                            "type": "tool_use",
                            "id": "create-1",
                            "name": "create_ticket",
                            "input": {
                                "category": "technical",
                                "title": "Cannot connect",
                                "description": "Campus network login fails",
                            },
                        }
                    ]
                ),
                SimpleNamespace(
                    content=[
                        {"type": "text", "text": "Authorization required."}
                    ]
                ),
            ]

        async def create(self, **_kwargs):
            return self.responses.pop(0)

    agent = TechnicalAgent(
        SimpleNamespace(messages=FakeMessages()),
        "model",
        runtime=AgentRuntime(
            SimpleNamespace(messages=ScriptedMessages()),
            "model",
            manager,
        ),
        allowed_tools=AGENT_TOOL_ALLOWLIST["technical"],
    )

    response = asyncio.run(
        agent.handle(
            Request(
                "Create a ticket",
                "attacker-selected-user",
                "conv-1",
                request_id="request-1",
            )
        )
    )

    assert response.ticket_ids == []
    assert response.tool_calls[0]["success"] is False
    assert response.tool_calls[0]["error_type"] == "tool_rejected"
