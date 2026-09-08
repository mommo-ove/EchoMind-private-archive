import asyncio
import dataclasses
import json
import re
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request

from agents.agent_orchestrator import AgentType, OrchestratorResult
from core.intent_recognizer import (
    IntentCategory,
    IntentResult,
    UrgencyLevel,
)
from core.retrieval_policy import RetrievalDecision
from memory.conversation_memory import MemoryContext, Message, MsgRole
from memory.conversation_memory import OperationClaim, OperationStatus
from services.chat_service import (
    ChatCommand,
    ChatIdempotencyConflict,
    ChatOperationInProgress,
    ChatPipelineError,
    ChatResult,
    ChatService,
    _normalize_knowledge_item,
)


def test_knowledge_citation_preserves_stable_source_id_for_retrieval_evaluation():
    citation = _normalize_knowledge_item({
        "id": "chroma-chunk-123",
        "title": "Campus network policy",
        "content": "Clear stale authentication state.",
        "chunk": 0,
        "score": 0.9,
    })

    assert citation["id"] == "chroma-chunk-123"


class FakeMemory:
    def __init__(
        self,
        events,
        *,
        profile_failure=False,
        operation_store=None,
    ):
        self.events = events
        self.context_calls = []
        self.messages = []
        self.exchange_calls = []
        self._exchange_keys = set()
        self.profile_calls = []
        self.profile_failure = profile_failure
        self.operation_store = (
            operation_store if operation_store is not None else {}
        )
        self.operation_calls = []

    async def get_context(self, user_id, conv_id, query=""):
        self.events.append("memory.read")
        self.context_calls.append((user_id, conv_id, query))
        return MemoryContext([], [], {}, "")

    async def add_exchange(
        self,
        user_id,
        conv_id,
        user_content,
        assistant_content,
        *,
        exchange_key,
        user_metadata=None,
        assistant_metadata=None,
    ):
        self.exchange_calls.append((
            user_id,
            conv_id,
            user_content,
            assistant_content,
            exchange_key,
        ))
        scoped_key = (user_id, conv_id, exchange_key)
        if scoped_key in self._exchange_keys:
            return False
        self._exchange_keys.add(scoped_key)
        self.events.extend([
            "memory.write.user",
            "memory.write.assistant",
        ])
        self.messages.extend([
            (user_id, conv_id, MsgRole.USER, user_content),
            (user_id, conv_id, MsgRole.ASSISTANT, assistant_content),
        ])
        return True
    async def add_message(self, *_args, **_kwargs):
        raise AssertionError("ChatService must persist exchanges atomically")

    async def update_profile(self, user_id, conv_id):
        self.profile_calls.append((user_id, conv_id))
        if self.profile_failure:
            raise RuntimeError("private profile backend detail")

    async def claim_operation(
        self,
        user_id,
        conv_id,
        operation_id,
        payload_hash,
        owner_id,
    ):
        self.operation_calls.append(
            ("claim", user_id, conv_id, operation_id, payload_hash, owner_id)
        )
        key = (user_id, conv_id, operation_id)
        record = self.operation_store.get(key)
        if record is None:
            self.operation_store[key] = {
                "payload_hash": payload_hash,
                "owner_id": owner_id,
                "payload": None,
            }
            return OperationClaim(OperationStatus.CLAIMED)
        if record["payload_hash"] != payload_hash:
            return OperationClaim(OperationStatus.CONFLICT)
        if record["payload"] is not None:
            return OperationClaim(
                OperationStatus.COMPLETED,
                record["payload"],
            )
        return OperationClaim(OperationStatus.IN_PROGRESS)

    async def commit_operation(
        self,
        user_id,
        conv_id,
        operation_id,
        payload_hash,
        owner_id,
        user_content,
        assistant_content,
        payload,
    ):
        self.operation_calls.append(
            ("commit", user_id, conv_id, operation_id, payload_hash, owner_id)
        )
        key = (user_id, conv_id, operation_id)
        record = self.operation_store.get(key)
        if (
            record is None
            or record["payload_hash"] != payload_hash
            or record["owner_id"] != owner_id
            or record["payload"] is not None
        ):
            return False
        self.exchange_calls.append((
            user_id,
            conv_id,
            user_content,
            assistant_content,
            operation_id,
        ))
        self.events.extend([
            "memory.write.user",
            "memory.write.assistant",
        ])
        self.messages.extend([
            (user_id, conv_id, MsgRole.USER, user_content),
            (user_id, conv_id, MsgRole.ASSISTANT, assistant_content),
        ])
        record["payload"] = payload
        record["owner_id"] = ""
        return True

    async def release_operation(
        self,
        user_id,
        conv_id,
        operation_id,
        payload_hash,
        owner_id,
    ):
        self.operation_calls.append(
            ("release", user_id, conv_id, operation_id, payload_hash, owner_id)
        )
        key = (user_id, conv_id, operation_id)
        record = self.operation_store.get(key)
        if (
            record is None
            or record["payload_hash"] != payload_hash
            or record["owner_id"] != owner_id
            or record["payload"] is not None
        ):
            return False
        del self.operation_store[key]
        return True


class FakeRecognizer:
    def __init__(self, events, intent=IntentCategory.TECHNICAL):
        self.events = events
        self.calls = []
        self.intent = intent

    async def recognize(self, message, history=None):
        self.events.append("intent.recognize")
        self.calls.append((message, deepcopy(history)))
        return IntentResult(
            intent=self.intent,
            confidence=0.93,
            urgency=UrgencyLevel.LOW,
            entities={},
            reasoning="test",
            latency_ms=1.0,
            intent_scores={
                self.intent: 0.93,
                IntentCategory.BILLING: 0.72,
            },
            matched_intents=[self.intent, IntentCategory.BILLING],
        )


class FakePolicy:
    def __init__(self, events, use_knowledge=True):
        self.events = events
        self.use_knowledge = use_knowledge
        self.calls = []

    def decide(self, intent, message):
        self.events.append("retrieval.decide")
        self.calls.append((intent, message))
        return RetrievalDecision(
            self.use_knowledge,
            f"test_policy:{getattr(intent, 'value', 'unknown')}",
        )


class FakeKnowledge:
    def __init__(
        self,
        events,
        *,
        data=None,
        success=True,
        fallback_used=False,
        failure=None,
    ):
        self.events = events
        self.calls = []
        self.data = data if data is not None else [
            {
                "title": "Campus Wi-Fi",
                "score": 0.91,
                "content": "Reconnect after clearing the saved network.",
                "chunk": 2,
            }
        ]
        self.success = success
        self.fallback_used = fallback_used
        self.failure = failure

    async def search_with_rewrite(self, tool_name, query, top_k=3, context=None):
        self.events.append("knowledge.search")
        self.calls.append((tool_name, query, top_k, deepcopy(context)))
        if self.failure:
            raise self.failure
        return SimpleNamespace(
            success=self.success,
            data=self.data,
            fallback_used=self.fallback_used,
            error="private retrieval backend detail",
        )


class FakeOrchestrator:
    def __init__(self, events, *, failure=None):
        self.events = events
        self.calls = []
        self.failure = failure

    async def run(self, request):
        self.events.append("orchestrator.run")
        self.calls.append(request)
        if self.failure:
            raise self.failure
        return OrchestratorResult(
            request_id=request.request_id,
            response="Reset the saved network and sign in again.",
            agent_type=AgentType.TECHNICAL,
            intent=request.intent,
            escalated=False,
            latency_ms=12.34,
            intent_scores=dict(request.intent_scores),
            matched_intents=list(request.matched_intents),
            agent_types=[AgentType.TECHNICAL, AgentType.BILLING],
            tool_calls=[
                {
                    "name": "query_network_status",
                    "success": True,
                    "params": {"site": "[REDACTED]"},
                    "fingerprint": "a" * 64,
                }
            ],
            ticket_ids=["T-123"],
        )


def build_service(
    *,
    intent=IntentCategory.TECHNICAL,
    use_knowledge=True,
    knowledge=None,
    profile_failure=False,
    orchestrator_failure=None,
    operation_store=None,
):
    events = []
    memory = FakeMemory(
        events,
        profile_failure=profile_failure,
        operation_store=operation_store,
    )
    recognizer = FakeRecognizer(events, intent=intent)
    policy = FakePolicy(events, use_knowledge=use_knowledge)
    knowledge = knowledge or FakeKnowledge(events)
    orchestrator = FakeOrchestrator(events, failure=orchestrator_failure)
    service = ChatService(
        memory=memory,
        intent_recognizer=recognizer,
        retrieval_policy=policy,
        orchestrator=orchestrator,
        knowledge_search=knowledge,
    )
    return service, events, memory, recognizer, policy, knowledge, orchestrator


def test_chat_service_runs_exact_pipeline_and_recognizes_once():
    service, events, memory, recognizer, _policy, _knowledge, orchestrator = (
        build_service()
    )

    result = asyncio.run(
        service.chat(
            message="Campus Wi-Fi returns 401.",
            user_id="display-user",
            conv_id="conv-1",
            principal_id="authenticated-user",
        )
    )

    assert events == [
        "memory.read",
        "intent.recognize",
        "retrieval.decide",
        "knowledge.search",
        "orchestrator.run",
        "memory.write.user",
        "memory.write.assistant",
    ]
    assert len(recognizer.calls) == 1
    request = orchestrator.calls[0]
    assert request.intent == IntentCategory.TECHNICAL
    assert request.urgency == UrgencyLevel.LOW
    assert request.intent_scores == {
        IntentCategory.TECHNICAL: 0.93,
        IntentCategory.BILLING: 0.72,
    }
    assert request.matched_intents == [
        IntentCategory.TECHNICAL,
        IntentCategory.BILLING,
    ]
    assert request.principal_id == "authenticated-user"
    assert request.citations == result.citations
    assert memory.messages == [
        (
            "authenticated-user",
            "conv-1",
            MsgRole.USER,
            "Campus Wi-Fi returns 401.",
        ),
        (
            "authenticated-user",
            "conv-1",
            MsgRole.ASSISTANT,
            "Reset the saved network and sign in again.",
        ),
    ]
    assert isinstance(result, ChatResult)
    assert result.intent == "technical"
    assert result.agent_type == "technical"
    assert result.matched_intents == ["technical", "billing"]
    assert result.agent_types == ["technical", "billing"]
    assert result.knowledge_used is True
    assert result.tool_calls[0]["fingerprint"] == "a" * 64


def test_evidence_verification_is_exposed_in_result_and_trace():
    service, _events, _memory, _recognizer, _policy, _knowledge, orchestrator = (
        build_service()
    )

    async def verified_run(request):
        orchestrator.calls.append(request)
        return OrchestratorResult(
            request_id=request.request_id,
            response="Grounded response",
            agent_type=AgentType.TECHNICAL,
            intent=request.intent,
            evidence_verification={
                "checked": True,
                "passed": True,
                "checked_claims": 1,
                "issue_codes": [],
                "abstained": False,
                "reflection_count": 1,
                "corrected": True,
                "safe_fallback_used": False,
            },
        )

    orchestrator.run = verified_run
    result = asyncio.run(service.chat(
        message="Campus Wi-Fi returns 401.",
        user_id="display",
        conv_id="conv",
        principal_id="principal",
    ))

    assert result.evidence_verification["passed"] is True
    assert result.evidence_verification["corrected"] is True
    assert result.trace["evidence_verification"] == result.evidence_verification


def test_authenticated_turns_are_serialized_through_atomic_write():
    events = []

    class StatefulMemory(FakeMemory):
        async def get_context(self, user_id, conv_id, query=""):
            self.events.append(f"memory.read.{query}")
            self.context_calls.append((user_id, conv_id, query))
            recent = [
                Message(role=role, content=content, timestamp=datetime.now())
                for _uid, _cid, role, content in self.messages
            ]
            return MemoryContext(recent, [], {}, "")

    class GatedOrchestrator(FakeOrchestrator):
        def __init__(self, events):
            super().__init__(events)
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        async def run(self, request):
            if request.message == "first":
                self.first_started.set()
                await self.release_first.wait()
            return await super().run(request)

    memory = StatefulMemory(events)
    recognizer = FakeRecognizer(events)
    orchestrator = GatedOrchestrator(events)
    service = ChatService(
        memory=memory,
        intent_recognizer=recognizer,
        retrieval_policy=FakePolicy(events, use_knowledge=False),
        orchestrator=orchestrator,
    )

    async def exercise():
        first = asyncio.create_task(service.chat(
            message="first",
            user_id="display",
            principal_id="principal",
            conv_id="conv",
        ))
        await orchestrator.first_started.wait()
        second = asyncio.create_task(service.chat(
            message="second",
            user_id="display",
            principal_id="principal",
            conv_id="conv",
        ))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert [call[2] for call in memory.context_calls] == ["first"]
        orchestrator.release_first.set()
        await asyncio.gather(first, second)

    asyncio.run(exercise())

    assert recognizer.calls[1][1] == [
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": "Reset the saved network and sign in again.",
        },
    ]
    assert service.active_conversation_locks == 0


def test_unrelated_authenticated_conversations_are_not_serialized():
    service, _events, _memory, _recognizer, _policy, _knowledge, orchestrator = (
        build_service(use_knowledge=False)
    )

    async def exercise():
        both_entered = asyncio.Event()
        release = asyncio.Event()
        entered = 0

        async def gated_run(request):
            nonlocal entered
            orchestrator.calls.append(request)
            entered += 1
            if entered == 2:
                both_entered.set()
            await release.wait()
            return OrchestratorResult(
                request_id=request.request_id,
                response="ok",
                agent_type=AgentType.TECHNICAL,
                intent=request.intent,
            )

        orchestrator.run = gated_run
        calls = [
            asyncio.create_task(service.chat(
                message="one",
                user_id="display",
                principal_id="principal",
                conv_id="conv-1",
            )),
            asyncio.create_task(service.chat(
                message="two",
                user_id="display",
                principal_id="principal",
                conv_id="conv-2",
            )),
        ]
        await asyncio.wait_for(both_entered.wait(), timeout=1)
        release.set()
        await asyncio.gather(*calls)

    asyncio.run(exercise())


def test_cancelling_a_serialized_turn_releases_its_conversation_lock():
    service, _events, _memory, _recognizer, _policy, _knowledge, orchestrator = (
        build_service(use_knowledge=False)
    )

    async def exercise():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def cancellable_run(request):
            orchestrator.calls.append(request)
            if request.message == "cancel me":
                entered.set()
                await release.wait()
            return OrchestratorResult(
                request_id=request.request_id,
                response="ok",
                agent_type=AgentType.TECHNICAL,
                intent=request.intent,
            )

        orchestrator.run = cancellable_run
        first = asyncio.create_task(service.chat(
            message="cancel me",
            user_id="display",
            principal_id="principal",
            conv_id="conv",
        ))
        await entered.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        result = await asyncio.wait_for(service.chat(
            message="next",
            user_id="display",
            principal_id="principal",
            conv_id="conv",
        ), timeout=1)
        assert result.response == "ok"

    asyncio.run(exercise())
    assert service.active_conversation_locks == 0


def test_greeting_policy_skips_knowledge_search():
    service, events, *_ = build_service(
        intent=IntentCategory.GREETING,
        use_knowledge=False,
    )

    result = asyncio.run(service.chat(
        message="Hello",
        user_id="visitor",
        conv_id="conv",
        principal_id="principal",
    ))

    assert "knowledge.search" not in events
    assert events[:3] == [
        "memory.read",
        "intent.recognize",
        "retrieval.decide",
    ]
    assert result.knowledge_used is False
    assert result.citations == []


def test_profile_failure_is_observed_without_failing_response(caplog):
    service, _events, memory, *_ = build_service(profile_failure=True)

    async def run():
        result = await service.chat(
            ChatCommand(
                message="Campus Wi-Fi returns 401.",
                user_id="display",
                conv_id="conv",
                principal_id="principal",
            )
        )
        await asyncio.sleep(0)
        return result

    result = asyncio.run(run())

    assert result.response.startswith("Reset")
    assert memory.profile_calls == [("principal", "conv")]
    assert "profile update failed" in caplog.text.lower()
    assert "private profile backend detail" not in caplog.text


def test_result_is_frozen_and_trace_is_bounded_and_safe():
    oversized = "x" * 20_000
    secret = "raw-error-secret"
    events = []
    knowledge = FakeKnowledge(
        events,
        data=[
            {
                "title": oversized,
                "content": oversized,
                "score": float("inf"),
                "error": secret,
                "api_key": secret,
            }
        ],
    )
    service, _events, *_ = build_service(knowledge=knowledge)

    result = asyncio.run(
        service.chat(
            message="Campus Wi-Fi returns 401.",
            user_id="display",
            principal_id="principal",
        )
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.response = "changed"
    assert set(result.trace) == {
        "policy",
        "route",
        "knowledge",
        "memory",
        "tools",
        "duration_ms",
        "evidence_verification",
    }
    assert len(json.dumps(result.trace, ensure_ascii=False)) < 12_000
    assert secret not in repr(result)
    assert len(result.citations[0]["title"]) <= 160
    assert len(result.citations[0]["content"]) <= 600
    assert result.citations[0]["score"] == 0.0
    assert result.citations[0]["id"] in orchestrator_context(service)


def test_bounded_context_keeps_knowledge_citations_with_large_memory():
    service, _events, memory, *_ = build_service()

    async def large_context(user_id, conv_id, query=""):
        return SimpleNamespace(
            recent_messages=[],
            to_prompt_text=lambda: "memory-" * 4_000,
        )

    memory.get_context = large_context
    result = asyncio.run(
        service.chat(
            message="Campus Wi-Fi fails",
            user_id="display",
            conv_id="conv",
            principal_id="principal",
        )
    )
    context = orchestrator_context(service)

    assert len(context) <= 16_000
    assert result.citations[0]["id"] in context
    assert result.citations[0]["content"] in context


def test_every_reported_citation_is_wholly_present_with_large_memory():
    items = [
        {
            "title": f"Reference {index}",
            "content": chr(ord("a") + index) * 600,
            "score": 0.9 - index / 10,
            "chunk": index,
        }
        for index in range(3)
    ]
    events = []
    service, _events, memory, *_ = build_service(
        knowledge=FakeKnowledge(events, data=items)
    )

    async def large_context(user_id, conv_id, query=""):
        return SimpleNamespace(
            recent_messages=[],
            to_prompt_text=lambda: "memory-" * 10_000,
        )

    memory.get_context = large_context
    result = asyncio.run(service.chat(
        message="Compare all references",
        user_id="display",
        conv_id="conv",
        principal_id="principal",
    ))
    context = orchestrator_context(service)

    assert len(result.citations) == 3
    assert len(context) <= 16_000
    for citation in result.citations:
        complete_block = (
            f"[Citation {citation['id']}]\n"
            f"Title: {citation['title']}\n"
            f"Score: {citation['score']:.4f}\n"
            f"Content: {citation['content']}"
        )
        assert complete_block in context


def orchestrator_context(service):
    return service._orchestrator.calls[0].context


def test_knowledge_fallback_or_failure_is_not_used():
    async def exercise(knowledge):
        service, _events, *_ = build_service(knowledge=knowledge)
        return await service.chat(
            message="Campus Wi-Fi returns 401.",
            user_id="display",
        )

    fallback_events = []
    fallback = FakeKnowledge(
        fallback_events,
        data=[{
            "title": "Fallback",
            "content": "backend unavailable: private detail",
            "fallback": True,
        }],
        fallback_used=True,
    )
    failed_events = []
    failed = FakeKnowledge(
        failed_events,
        failure=RuntimeError("private retrieval failure"),
    )

    fallback_result = asyncio.run(exercise(fallback))
    failed_result = asyncio.run(exercise(failed))

    assert fallback_result.knowledge_used is False
    assert fallback_result.citations == []
    assert failed_result.knowledge_used is False
    assert failed_result.citations == []
    assert "private" not in repr(failed_result.trace)


def test_malformed_knowledge_result_degrades_without_leaking():
    class BrokenResult:
        @property
        def success(self):
            raise RuntimeError("private malformed result detail")

    class BrokenKnowledge:
        async def search_with_rewrite(self, *_args, **_kwargs):
            return BrokenResult()

    service, *_ = build_service(knowledge=BrokenKnowledge())

    result = asyncio.run(
        service.chat(message="Campus Wi-Fi fails", user_id="display")
    )

    assert result.knowledge_used is False
    assert result.citations == []
    assert "private" not in repr(result.trace)


def test_non_boolean_knowledge_fallback_flag_degrades_safely():
    events = []
    knowledge = FakeKnowledge(
        events,
        fallback_used="false",
        data=[{
            "title": "Private fallback",
            "content": "private fallback content",
            "score": 1.0,
        }],
    )
    service, *_ = build_service(knowledge=knowledge)

    result = asyncio.run(
        service.chat(message="Campus Wi-Fi fails", user_id="display")
    )

    assert result.knowledge_used is False
    assert result.citations == []
    assert "private fallback content" not in orchestrator_context(service)


def test_knowledge_iteration_failure_degrades_without_raw_leak():
    class BrokenList(list):
        def __iter__(self):
            raise RuntimeError("private iteration detail")

    events = []
    knowledge = FakeKnowledge(events, data=BrokenList([{
        "title": "Private",
        "content": "private content",
        "score": 1.0,
    }]))
    service, *_ = build_service(knowledge=knowledge)

    result = asyncio.run(
        service.chat(message="Campus Wi-Fi fails", user_id="display")
    )

    assert result.knowledge_used is False
    assert result.citations == []
    assert "private" not in repr(result.trace)


def test_non_boolean_item_fallback_flag_is_never_used():
    events = []
    knowledge = FakeKnowledge(events, data=[{
        "title": "Private fallback",
        "content": "private fallback content",
        "score": 1.0,
        "fallback": "false",
    }])
    service, *_ = build_service(knowledge=knowledge)

    result = asyncio.run(
        service.chat(message="Campus Wi-Fi fails", user_id="display")
    )

    assert result.knowledge_used is False
    assert result.citations == []
    assert "private fallback content" not in orchestrator_context(service)


def test_orchestrator_failure_is_controlled_and_memory_is_not_written():
    from services.chat_service import ChatPipelineError

    service, _events, memory, *_ = build_service(
        orchestrator_failure=RuntimeError("private orchestrator detail")
    )

    with pytest.raises(ChatPipelineError, match="orchestrator failed") as error:
        asyncio.run(
            service.chat(message="Need help", user_id="display")
        )

    assert "private orchestrator detail" not in str(error.value)
    assert memory.messages == []
    assert memory.profile_calls == []


@pytest.mark.parametrize("malformed_escalated", ["false", 0, None])
def test_malformed_escalated_is_rejected_before_memory_write(
    malformed_escalated,
):
    from services.chat_service import ChatPipelineError

    (
        service,
        _events,
        memory,
        _recognizer,
        _policy,
        _knowledge,
        orchestrator,
    ) = build_service()

    async def malformed_run(request):
        return OrchestratorResult(
            request_id=request.request_id,
            response="unsafe response",
            agent_type=AgentType.TECHNICAL,
            intent=request.intent,
            escalated=malformed_escalated,
            latency_ms=1.0,
            intent_scores=dict(request.intent_scores),
            matched_intents=list(request.matched_intents),
            agent_types=[AgentType.TECHNICAL],
        )

    orchestrator.run = malformed_run

    with pytest.raises(ChatPipelineError, match="orchestrator failed"):
        asyncio.run(service.chat(
            message="Need help",
            user_id="display",
            conv_id="conv",
            principal_id="principal",
        ))

    assert memory.messages == []


def test_raising_orchestrator_property_is_safe_before_memory_write():
    from services.chat_service import ChatPipelineError

    (
        service,
        _events,
        memory,
        _recognizer,
        _policy,
        _knowledge,
        orchestrator,
    ) = build_service()

    class RaisingResult:
        request_id = "request"
        response = "unsafe response"
        agent_type = AgentType.TECHNICAL
        intent = IntentCategory.TECHNICAL
        latency_ms = 1.0
        intent_scores = {IntentCategory.TECHNICAL: 1.0}
        matched_intents = [IntentCategory.TECHNICAL]
        agent_types = [AgentType.TECHNICAL]
        tool_calls = []
        ticket_ids = []

        @property
        def escalated(self):
            raise RuntimeError("private escalation property detail")

    async def malformed_run(_request):
        return RaisingResult()

    orchestrator.run = malformed_run

    with pytest.raises(ChatPipelineError, match="orchestrator failed") as error:
        asyncio.run(service.chat(message="Need help", user_id="display"))

    assert "private escalation property detail" not in str(error.value)
    assert memory.messages == []


def test_memory_read_failure_is_controlled_and_short_circuits_pipeline():
    from services.chat_service import ChatPipelineError

    service, events, memory, *_ = build_service()

    async def fail_read(*_args, **_kwargs):
        events.append("memory.read")
        raise RuntimeError("private memory detail")

    memory.get_context = fail_read

    with pytest.raises(ChatPipelineError, match="memory read failed") as error:
        asyncio.run(service.chat(
            message="Need help",
            user_id="display",
            conv_id="conv",
            principal_id="principal",
        ))

    assert "private memory detail" not in str(error.value)
    assert events == ["memory.read"]
    assert memory.messages == []


def test_cancellation_propagates_without_becoming_a_pipeline_failure():
    service, _events, _memory, recognizer, *_ = build_service()

    async def cancel(_message, history=None):
        raise asyncio.CancelledError()

    recognizer.recognize = cancel

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(service.chat(message="Need help", user_id="display"))


def test_multiline_message_reaches_pipeline_and_atomic_memory_unchanged():
    service, _events, memory, recognizer, _policy, _knowledge, orchestrator = (
        build_service(use_knowledge=False)
    )
    message = "first line\n\tindented second line"

    asyncio.run(service.chat(
        message=message,
        user_id="display",
        principal_id="principal",
        conv_id="conv",
    ))

    assert recognizer.calls[0][0] == message
    assert orchestrator.calls[0].message == message
    assert memory.messages[0] == (
        "principal",
        "conv",
        MsgRole.USER,
        message,
    )


def test_generated_conversation_and_trace_identifiers_are_stable_and_valid():
    service, _events, memory, *_ = build_service()

    result = asyncio.run(
        service.chat(
            message="Need help",
            user_id="display",
            principal_id="principal",
        )
    )

    assert str(__import__("uuid").UUID(result.conv_id)) == result.conv_id
    assert len(result.trace_id) == 32
    assert int(result.trace_id, 16) >= 0
    exchange_key = memory.exchange_calls[0][-1]
    assert re.fullmatch(r"[0-9a-f]{64}", exchange_key)
    assert exchange_key != result.trace_id


def test_retry_with_same_request_id_persists_one_complete_turn_without_leak():
    service, events, memory, _recognizer, _policy, _knowledge, orchestrator = (
        build_service()
    )

    async def exercise():
        first = await service.chat(
            message="Need help",
            user_id="display",
            conv_id="conv",
            principal_id="principal",
            request_id="private-operation-1",
        )
        second = await service.chat(
            message="Need help",
            user_id="display",
            conv_id="conv",
            principal_id="principal",
            request_id="private-operation-1",
        )
        return first, second

    first, second = asyncio.run(exercise())

    assert first == second
    assert len(memory.exchange_calls) == 1
    assert len(memory.messages) == 2
    assert events.count("memory.read") == 1
    assert events.count("intent.recognize") == 1
    assert events.count("orchestrator.run") == 1
    stable_id = memory.exchange_calls[0][-1]
    assert re.fullmatch(r"[0-9a-f]{64}", stable_id)
    assert orchestrator.calls[0].request_id == stable_id
    assert "private-operation-1" not in repr(memory.exchange_calls)
    assert "private-operation-1" not in repr(memory.operation_calls)
    assert "private-operation-1" not in repr(first)
    assert "private-operation-1" not in repr(second)


def test_same_key_with_different_payload_is_a_conflict_before_pipeline():
    service, events, memory, *_ = build_service()

    asyncio.run(service.chat(
        message="First",
        user_id="display",
        conv_id="conv",
        principal_id="principal",
        request_id="operation",
    ))

    with pytest.raises(ChatIdempotencyConflict):
        asyncio.run(service.chat(
            message="Changed",
            user_id="display",
            conv_id="conv",
            principal_id="principal",
            request_id="operation",
        ))

    assert events.count("memory.read") == 1
    assert len(memory.exchange_calls) == 1


def test_completed_operation_replays_across_service_instances_deeply_isolated():
    operation_store = {}
    first_service, _events, _memory, *_ = build_service(
        operation_store=operation_store
    )
    second_service, second_events, second_memory, *_ = build_service(
        operation_store=operation_store
    )

    first = asyncio.run(first_service.chat(
        message="Need help",
        user_id="display",
        conv_id="conv",
        principal_id="principal",
        request_id="operation",
    ))
    replay = asyncio.run(second_service.chat(
        message="Need help",
        user_id="display",
        conv_id="conv",
        principal_id="principal",
        request_id="operation",
    ))

    assert replay == first
    assert second_events == []
    assert second_memory.exchange_calls == []
    replay.trace["route"]["intent"] = "mutated"
    replay.tool_calls[0]["name"] = "mutated"
    replay_again = asyncio.run(second_service.chat(
        message="Need help",
        user_id="display",
        conv_id="conv",
        principal_id="principal",
        request_id="operation",
    ))
    assert replay_again.trace["route"]["intent"] == first.intent
    assert replay_again.tool_calls[0]["name"] != "mutated"


def test_concurrent_duplicate_across_services_executes_pipeline_once():
    operation_store = {}
    first_service, first_events, first_memory, *_first, first_orchestrator = (
        build_service(operation_store=operation_store)
    )
    second_service, second_events, second_memory, *_ = build_service(
        operation_store=operation_store
    )

    async def exercise():
        started = asyncio.Event()
        release = asyncio.Event()
        original_run = first_orchestrator.run

        async def blocking_run(request):
            started.set()
            await release.wait()
            return await original_run(request)

        first_orchestrator.run = blocking_run
        first_task = asyncio.create_task(first_service.chat(
            message="Need help",
            user_id="display",
            conv_id="conv",
            principal_id="principal",
            request_id="operation",
        ))
        await started.wait()
        second_task = asyncio.create_task(second_service.chat(
            message="Need help",
            user_id="display",
            conv_id="conv",
            principal_id="principal",
            request_id="operation",
        ))
        await asyncio.sleep(0.02)
        release.set()
        return await asyncio.gather(first_task, second_task)

    first, second = asyncio.run(exercise())

    assert first == second
    assert first_events.count("orchestrator.run") == 1
    assert second_events == []
    assert len(first_memory.exchange_calls) == 1
    assert second_memory.exchange_calls == []


def test_pending_operation_times_out_without_running_pipeline(monkeypatch):
    service, events, memory, *_ = build_service()

    async def always_pending(*_args):
        return OperationClaim(OperationStatus.IN_PROGRESS)

    memory.claim_operation = always_pending
    monkeypatch.setattr(service, "_IDEMPOTENCY_POLL_ATTEMPTS", 2)
    monkeypatch.setattr(service, "_IDEMPOTENCY_POLL_SECONDS", 0)

    with pytest.raises(ChatOperationInProgress):
        asyncio.run(service.chat(
            message="Need help",
            user_id="display",
            conv_id="conv",
            principal_id="principal",
            request_id="operation",
        ))

    assert events == []


def test_cancelled_new_operation_releases_claim():
    service, _events, memory, *_rest, orchestrator = build_service()

    async def exercise():
        started = asyncio.Event()

        async def blocking_run(_request):
            started.set()
            await asyncio.Event().wait()

        orchestrator.run = blocking_run
        task = asyncio.create_task(service.chat(
            message="Need help",
            user_id="display",
            conv_id="conv",
            principal_id="principal",
            request_id="operation",
        ))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    assert memory.operation_store == {}
    assert any(call[0] == "release" for call in memory.operation_calls)


def test_failed_new_operation_is_released_and_retry_can_run():
    service, events, memory, *_rest, orchestrator = build_service(
        orchestrator_failure=RuntimeError("temporary")
    )

    with pytest.raises(ChatPipelineError):
        asyncio.run(service.chat(
            message="Need help",
            user_id="display",
            conv_id="conv",
            principal_id="principal",
            request_id="operation",
        ))

    assert any(call[0] == "release" for call in memory.operation_calls)
    orchestrator.failure = None
    result = asyncio.run(service.chat(
        message="Need help",
        user_id="display",
        conv_id="conv",
        principal_id="principal",
        request_id="operation",
    ))
    assert result.response.startswith("Reset")
    assert events.count("orchestrator.run") == 2


@pytest.mark.parametrize(
    "payload",
    ["not-json", "x" * 70_000],
    ids=["malformed", "oversized"],
)
def test_malformed_or_oversized_replay_is_rejected_safely(payload):
    operation_store = {}
    service, events, memory, *_ = build_service(
        operation_store=operation_store
    )

    original_commit = memory.commit_operation

    async def corrupt_commit(*args):
        committed = await original_commit(*args)
        if committed:
            next(iter(operation_store.values()))["payload"] = payload
        return committed

    memory.commit_operation = corrupt_commit
    asyncio.run(service.chat(
        message="Need help",
        user_id="display",
        conv_id="conv",
        principal_id="principal",
        request_id="operation",
    ))

    with pytest.raises(ChatPipelineError, match="replay"):
        asyncio.run(service.chat(
            message="Need help",
            user_id="display",
            conv_id="conv",
            principal_id="principal",
            request_id="operation",
        ))

    assert events.count("orchestrator.run") == 1


def test_different_request_ids_persist_distinct_complete_exchanges():
    service, _events, memory, *_ = build_service()

    async def exercise():
        await service.chat(
            message="First",
            user_id="display",
            conv_id="conv",
            principal_id="principal",
            request_id="operation-1",
        )
        await service.chat(
            message="Second",
            user_id="display",
            conv_id="conv",
            principal_id="principal",
            request_id="operation-2",
        )

    asyncio.run(exercise())

    assert len(memory.messages) == 4
    assert memory.exchange_calls[0][-1] != memory.exchange_calls[1][-1]


@pytest.mark.parametrize(
    "request_id",
    ["", " ", "has whitespace", "x" * 129, "bad\nkey"],
)
def test_invalid_request_ids_are_rejected_before_dependencies(
    request_id,
):
    service, events, *_ = build_service()

    with pytest.raises((TypeError, ValueError)):
        asyncio.run(
            service.chat(
                message="Need help",
                user_id="display",
                request_id=request_id,
            )
        )

    assert events == []


def test_service_does_not_mutate_recognizer_or_orchestrator_results():
    service, _events, _memory, recognizer, *_rest = build_service()
    original = asyncio.run(recognizer.recognize("seed"))
    recognizer.calls.clear()

    async def recognize(_message, history=None):
        recognizer.calls.append((_message, deepcopy(history)))
        return original

    recognizer.recognize = recognize
    before_scores = deepcopy(original.intent_scores)
    before_matched = list(original.matched_intents)

    asyncio.run(service.chat(message="Need help", user_id="display"))

    assert original.intent_scores == before_scores
    assert original.matched_intents == before_matched


def test_tool_fingerprints_are_strict_bounded_and_repeated_events_survive():
    service, _events, _memory, _recognizer, _policy, _knowledge, orchestrator = (
        build_service(use_knowledge=False)
    )
    valid = "0123456789abcdef" * 4
    calls = [
        {
            "name": "lookup",
            "success": True,
            "params": {"secret_id": "raw-secret"},
            "id": "raw-call-id",
            "fingerprint": valid,
        },
        {
            "name": "lookup",
            "success": True,
            "params": {"secret_id": "other-secret"},
            "fingerprint": valid,
        },
        {
            "name": "invalid",
            "fingerprint": "A" * 64,
        },
    ] + [
        {"name": f"extra-{index}", "fingerprint": f"{index:064x}"}
        for index in range(40)
    ]

    async def run(request):
        result = await FakeOrchestrator([]).run(request)
        result.tool_calls = calls
        return result

    orchestrator.run = run
    result = asyncio.run(service.chat(
        message="Need help",
        user_id="display",
    ))

    assert len(result.tool_calls) == 32
    assert result.tool_calls[:2] == [
        {
            "name": "lookup",
            "success": True,
            "params": {"secret_id": "[REDACTED]"},
            "fingerprint": valid,
        },
        {
            "name": "lookup",
            "success": True,
            "params": {"secret_id": "[REDACTED]"},
            "fingerprint": valid,
        },
    ]
    assert "fingerprint" not in result.tool_calls[2]
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", item["fingerprint"])
        for item in result.tool_calls
        if "fingerprint" in item
    )
    assert "raw-call-id" not in repr(result.tool_calls)
    assert "raw-secret" not in repr(result.tool_calls)
    assert "other-secret" not in repr(result.trace)


def test_result_trace_monitor_and_orchestrator_metadata_are_deeply_isolated():
    events = []

    class RetainingMonitor:
        def __init__(self):
            self.trace = None

        def record_chat_trace(self, trace):
            self.trace = trace
            trace["tools"][0]["params"]["site"] = "recorder mutation"

    monitor = RetainingMonitor()
    memory = FakeMemory(events)
    orchestrator = FakeOrchestrator(events)
    service = ChatService(
        memory=memory,
        intent_recognizer=FakeRecognizer(events),
        retrieval_policy=FakePolicy(events),
        orchestrator=orchestrator,
        knowledge_search=FakeKnowledge(events),
        monitor=monitor,
    )

    result = asyncio.run(service.chat(
        message="Need help",
        user_id="display",
    ))
    source_params = orchestrator.calls

    assert result.tool_calls[0]["params"]["site"] == "[REDACTED]"
    assert result.trace["tools"][0]["params"]["site"] == "[REDACTED]"
    assert monitor.trace["tools"][0]["params"]["site"] == "recorder mutation"

    result.tool_calls[0]["params"]["site"] = "caller tool mutation"
    assert result.trace["tools"][0]["params"]["site"] == "[REDACTED]"
    assert monitor.trace["tools"][0]["params"]["site"] == "recorder mutation"

    result.trace["tools"][0]["params"]["site"] = "caller trace mutation"
    assert result.tool_calls[0]["params"]["site"] == "caller tool mutation"
    assert monitor.trace["tools"][0]["params"]["site"] == "recorder mutation"

    result.citations[0]["title"] = "caller citation mutation"
    assert "caller citation mutation" not in orchestrator_context(service)
    assert source_params[0].context == orchestrator_context(service)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"message": "", "user_id": "display"},
        {"message": "   ", "user_id": "display"},
        {"message": 123, "user_id": "display"},
        {"message": "hello", "user_id": ""},
        {"message": "hello", "user_id": object()},
        {"message": "hello", "user_id": "display", "principal_id": ""},
        {"message": "hello", "user_id": "display", "conv_id": 123},
        {"message": "bad\rreturn", "user_id": "display"},
        {"message": "bad\u0085control", "user_id": "display"},
        {"message": "bad\u200bformat", "user_id": "display"},
        {"message": "bad\ud800surrogate", "user_id": "display"},
    ],
)
def test_invalid_or_blank_command_fields_fail_controlled(kwargs):
    service, *_ = build_service()

    with pytest.raises((TypeError, ValueError)):
        asyncio.run(service.chat(**kwargs))


def test_unauthenticated_display_user_uses_no_persistent_memory_identity():
    service, _events, memory, _recognizer, _policy, _knowledge, orchestrator = (
        build_service()
    )

    result = asyncio.run(
        service.chat(
            message="Create a ticket",
            user_id="attacker-selected-user",
            conv_id="shared-conversation",
        )
    )

    assert memory.context_calls == []
    assert memory.messages == []
    assert memory.profile_calls == []
    assert result.trace["memory"]["mode"] == "stateless_anonymous"
    assert orchestrator.calls[0].principal_id is None
    assert orchestrator.calls[0].user_id == "attacker-selected-user"


def test_each_unauthenticated_call_remains_stateless():
    service, events, memory, *_ = build_service()

    async def run():
        await service.chat(
            message="first",
            user_id="victim",
            conv_id="conv",
        )
        await service.chat(
            message="second",
            user_id="victim",
            conv_id="conv",
        )

    asyncio.run(run())

    assert memory.context_calls == []
    assert memory.messages == []
    assert memory.profile_calls == []
    assert events.count("intent.recognize") == 2


def test_unauthenticated_chat_is_stateless_even_with_idempotency_key():
    service, events, memory, *_ = build_service()

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("anonymous chat must not use persistent memory")

    memory.get_context = forbidden
    memory.add_exchange = forbidden
    memory.claim_operation = forbidden
    memory.update_profile = forbidden

    first = asyncio.run(service.chat(
        message="Need public help",
        user_id="display",
        conv_id="public",
        request_id="ignored-for-anonymous",
    ))
    second = asyncio.run(service.chat(
        message="Need public help",
        user_id="display",
        conv_id="public",
        request_id="ignored-for-anonymous",
    ))

    assert first.trace["memory"]["mode"] == "stateless_anonymous"
    assert second.trace["memory"]["mode"] == "stateless_anonymous"
    assert events.count("intent.recognize") == 2


def test_anthropic_config_strips_key_and_rejects_whitespace(monkeypatch):
    from api import main

    monkeypatch.setenv("ANTHROPIC_API_KEY", "  secret-key  ")
    assert main._anthropic_cfg()["api_key"] == "secret-key"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "  \t ")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        main._anthropic_cfg()


def test_valid_principal_is_the_only_persistent_memory_identity():
    service, _events, memory, *_ = build_service()

    asyncio.run(
        service.chat(
            message="Remember my preference",
            user_id="attacker-controlled-display",
            conv_id="conv",
            principal_id="server-authenticated-principal",
        )
    )

    assert memory.context_calls[0][0] == "server-authenticated-principal"
    assert all(
        call[0] == "server-authenticated-principal"
        for call in memory.messages
    )


def make_request(
    *,
    principal_id=None,
    state_user_id=None,
    headers=(),
):
    request = Request({
        "type": "http",
        "method": "POST",
        "path": "/chat",
        "headers": list(headers),
    })
    if principal_id is not None:
        request.state.principal_id = principal_id
    if state_user_id is not None:
        request.state.user_id = state_user_id
    return request


def test_api_principal_resolver_only_accepts_bounded_server_request_state():
    from api.main import resolve_chat_principal

    assert resolve_chat_principal(
        make_request(principal_id="principal-1")
    ) == "principal-1"
    assert resolve_chat_principal(
        make_request(state_user_id="principal-2")
    ) == "principal-2"
    assert resolve_chat_principal(
        make_request(principal_id=" ")
    ) is None
    assert resolve_chat_principal(
        make_request(principal_id="x" * 129)
    ) is None
    assert resolve_chat_principal(
        make_request(
            headers=[(b"x-echomind-user-token", b"attacker-token")]
        )
    ) is None
    assert resolve_chat_principal(
        make_request(
            headers=[(b"authorization", b"Bearer attacker-token")]
        )
    ) is None


def test_api_chat_passes_only_resolved_principal_and_preserves_legacy_fields(
    monkeypatch,
):
    from api import main

    class FakeService:
        def __init__(self):
            self.commands = []

        async def chat(self, command):
            self.commands.append(command)
            return ChatResult(
                conv_id="conv",
                response="ok",
                intent="technical",
                agent_type="technical",
                escalated=False,
                latency_ms=1.2,
                knowledge_used=True,
                intent_scores={"technical": 0.9},
                matched_intents=["technical"],
                agent_types=["technical"],
                citations=[],
                tool_calls=[],
                ticket_ids=[],
                trace_id="trace",
                trace={},
            )

    fake = FakeService()
    monkeypatch.setattr(main, "_chat_service", fake)

    response = asyncio.run(
        main.chat(
            main.ChatRequest(
                message="Help",
                user_id="attacker-display",
                conv_id="conv",
            ),
            make_request(principal_id="authenticated-principal"),
            idempotency_key="private-operation-1",
        )
    )
    asyncio.run(
        main.chat(
            main.ChatRequest(
                message="Public help",
                user_id="attacker-display",
                conv_id="public-conv",
            ),
            make_request(
                headers=[(b"x-echomind-user-token", b"attacker-token")]
            ),
        )
    )

    assert fake.commands[0].user_id == "attacker-display"
    assert fake.commands[0].principal_id == "authenticated-principal"
    assert fake.commands[0].request_id == "private-operation-1"
    assert "private-operation-1" not in repr(response)
    assert fake.commands[1].principal_id is None
    assert response.model_dump() == {
        "conv_id": "conv",
        "response": "ok",
        "intent": "technical",
        "agent_type": "technical",
        "escalated": False,
        "latency_ms": 1.2,
        "knowledge_used": True,
        "intent_scores": {"technical": 0.9},
        "matched_intents": ["technical"],
        "agent_types": ["technical"],
    }


def test_api_chat_maps_pipeline_failures_to_a_safe_503(monkeypatch):
    from api import main
    from services.chat_service import ChatPipelineError

    class FailingService:
        async def chat(self, _command):
            raise ChatPipelineError("memory read failed")

    monkeypatch.setattr(main, "_chat_service", FailingService())

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            main.chat(
                main.ChatRequest(message="Help", user_id="display"),
                make_request(),
            )
        )

    assert error.value.status_code == 503
    assert error.value.detail == "聊天服务暂时不可用"


def test_api_chat_maps_idempotency_conflict_to_safe_409(monkeypatch):
    from api import main

    class FailingService:
        async def chat(self, _command):
            raise ChatIdempotencyConflict("private payload hash")

    monkeypatch.setattr(main, "_chat_service", FailingService())

    with pytest.raises(HTTPException) as error:
        asyncio.run(main.chat(
            main.ChatRequest(message="Help", user_id="display"),
            make_request(principal_id="principal"),
            idempotency_key="operation",
        ))

    assert error.value.status_code == 409
    assert "private" not in error.value.detail


def test_api_chat_maps_pending_operation_to_safe_409(monkeypatch):
    from api import main

    class PendingService:
        async def chat(self, _command):
            raise ChatOperationInProgress("private owner")

    monkeypatch.setattr(main, "_chat_service", PendingService())

    with pytest.raises(HTTPException) as error:
        asyncio.run(main.chat(
            main.ChatRequest(message="Help", user_id="display"),
            make_request(principal_id="principal"),
            idempotency_key="operation",
        ))

    assert error.value.status_code == 409
    assert "private" not in error.value.detail


def test_service_close_cancels_and_observes_background_profile_tasks():
    service, _events, memory, *_ = build_service()

    async def exercise():
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def blocking_update(_user_id, _conv_id):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        memory.update_profile = blocking_update
        await service.chat(
            message="Need help",
            user_id="display",
            conv_id="conv",
            principal_id="principal",
        )
        await started.wait()
        await service.aclose()
        assert cancelled.is_set()
        assert not service.pending_background_tasks

    asyncio.run(exercise())


def test_service_rejects_chat_after_close_and_close_is_idempotent():
    from services.chat_service import ChatPipelineError

    service, events, *_ = build_service()

    async def exercise():
        await service.aclose()
        await service.aclose()
        with pytest.raises(ChatPipelineError, match="closed"):
            await service.chat(message="Need help", user_id="display")

    asyncio.run(exercise())

    assert events == []


def test_in_flight_chat_finishing_after_close_registers_no_background_task():
    (
        service,
        _events,
        memory,
        _recognizer,
        _policy,
        _knowledge,
        orchestrator,
    ) = build_service()

    async def exercise():
        started = asyncio.Event()
        release = asyncio.Event()
        original_run = orchestrator.run

        async def blocking_run(request):
            started.set()
            await release.wait()
            return await original_run(request)

        orchestrator.run = blocking_run
        chat_task = asyncio.create_task(service.chat(
            message="Need help",
            user_id="display",
            conv_id="conv",
            principal_id="principal",
        ))
        await started.wait()
        await service.aclose()
        release.set()
        result = await chat_task
        await asyncio.sleep(0)
        return result

    result = asyncio.run(exercise())

    assert result.response.startswith("Reset")
    assert memory.profile_calls == []
    assert service.pending_background_tasks == 0


def test_close_caller_cancellation_propagates_and_retry_drains_tasks():
    service, _events, memory, *_ = build_service()

    async def exercise():
        profile_started = asyncio.Event()
        release_profile = asyncio.Event()

        async def resistant_update(_user_id, _conv_id):
            profile_started.set()
            while not release_profile.is_set():
                try:
                    await release_profile.wait()
                except asyncio.CancelledError:
                    continue

        memory.update_profile = resistant_update
        await service.chat(
            message="Need help",
            user_id="display",
            conv_id="conv",
            principal_id="principal",
        )
        await profile_started.wait()

        close_task = asyncio.create_task(service.aclose())
        await asyncio.sleep(0)
        close_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_task

        release_profile.set()
        await service.aclose()
        assert service.pending_background_tasks == 0

    asyncio.run(exercise())


@pytest.mark.parametrize("awaitable_kind", ["future", "task", "custom"])
def test_monitor_tracks_all_awaitable_results_until_close(awaitable_kind):
    service, _events, _memory, *_ = build_service()

    async def exercise():
        loop = asyncio.get_running_loop()
        pending = loop.create_future()

        class CustomAwaitable:
            def __await__(self):
                return pending.__await__()

        async def wait_forever():
            await pending

        class Monitor:
            def record_chat_trace(self, _trace):
                if awaitable_kind == "future":
                    return pending
                if awaitable_kind == "task":
                    return asyncio.create_task(wait_forever())
                return CustomAwaitable()

        service._monitor = Monitor()
        await service.chat(
            message="Need help",
            user_id="display",
            conv_id="conv",
            principal_id="principal",
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert service.pending_background_tasks == 1
        await service.aclose()
        assert service.pending_background_tasks == 0
        assert pending.cancelled()

    asyncio.run(exercise())
