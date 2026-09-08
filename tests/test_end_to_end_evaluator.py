import asyncio
from collections import deque

from evaluation.evaluator import (
    EndToEndEvaluator,
    QualityScores,
    RagasScores,
    retrieval_ranking_metrics,
)
from services.chat_service import ChatCommand, ChatResult
import api.main as main


class RecordingChatService:
    def __init__(self, results):
        self._results = deque(results)
        self.calls = []

    async def chat(self, command):
        assert isinstance(command, ChatCommand)
        self.calls.append(command)
        result = self._results.popleft()
        return ChatResult(
            conv_id=command.conv_id,
            response=result.response,
            intent=result.intent,
            agent_type=result.agent_type,
            escalated=result.escalated,
            latency_ms=result.latency_ms,
            knowledge_used=result.knowledge_used,
            intent_scores=dict(result.intent_scores),
            matched_intents=list(result.matched_intents),
            agent_types=list(result.agent_types),
            citations=list(result.citations),
            tool_calls=list(result.tool_calls),
            ticket_ids=list(result.ticket_ids),
            trace_id=result.trace_id,
            trace=dict(result.trace),
            evidence_verification=dict(result.evidence_verification),
        )


class PassingJudge:
    async def judge(self, question, response, context=None):
        assert question
        assert response
        return QualityScores(0.9, 0.9, 0.9, 0.9)


def chat_result(
    *,
    response="ok",
    intent="technical",
    agent_type="technical",
    latency_ms=12.0,
    knowledge_used=False,
    citations=None,
    tool_calls=None,
    ticket_ids=None,
    agent_types=None,
    matched_intents=None,
    model_call_count=None,
    evidence_verification=None,
):
    trace = {}
    if model_call_count is not None:
        trace["model_call_count"] = model_call_count
    return ChatResult(
        conv_id="placeholder",
        response=response,
        intent=intent,
        agent_type=agent_type,
        escalated=False,
        latency_ms=latency_ms,
        knowledge_used=knowledge_used,
        citations=list(citations or []),
        tool_calls=list(tool_calls or []),
        ticket_ids=list(ticket_ids or []),
        agent_types=list(agent_types or [agent_type]),
        matched_intents=list(matched_intents or []),
        trace=trace,
        evidence_verification=dict(evidence_verification or {}),
    )


def test_dialog_evaluation_records_grounding_and_reflection_outcome():
    service = RecordingChatService([chat_result(
        response="已创建工单 ticket_1",
        tool_calls=[{"name": "create_ticket", "success": True}],
        ticket_ids=["ticket_1"],
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
    )])

    report = asyncio.run(evaluator(service).run(dialog_cases=[{
        "question": "创建工单",
        "expected_tools": ["create_ticket"],
        "expect_ticket": True,
    }]))

    result = report.results[0]
    assert result.scores["evidence_grounding"] == 1.0
    assert result.scores["reflection_correction"] == 1.0
    assert result.metadata["evidence_verification"]["checked_claims"] == 2
    assert result.metadata["evidence_passed"] is True


def test_failed_evidence_verification_fails_task_even_when_judge_passes():
    service = RecordingChatService([chat_result(
        response="我无法根据当前证据可靠确认该信息。",
        evidence_verification={
            "checked": True,
            "passed": False,
            "checked_claims": 1,
            "issue_codes": ["conflicting_status"],
            "abstained": False,
            "reflection_count": 1,
            "corrected": False,
            "safe_fallback_used": True,
        },
    )])

    report = asyncio.run(evaluator(service).run(dialog_cases=[{
        "question": "工单处理完了吗？",
    }]))

    result = report.results[0]
    assert result.metadata["quality_passed"] is True
    assert result.metadata["evidence_passed"] is False
    assert result.metadata["task_completed"] is False


def evaluator(service, *, baseline_path=None):
    return EndToEndEvaluator(
        chat_service=service,
        recognizer=object(),
        judge=PassingJudge(),
        baseline_path=baseline_path,
    )


def test_dialog_evaluation_uses_chat_service_and_one_trusted_conversation():
    service = RecordingChatService(
        [chat_result(response="hello"), chat_result(response="check network")]
    )

    report = asyncio.run(
        evaluator(service).run(
            dialog_cases=[{"turns": ["你好", "校园网报401"]}]
        )
    )

    assert len(service.calls) == 2
    first, second = service.calls
    assert first.message == "你好"
    assert second.message == "校园网报401"
    assert first.user_id == second.user_id
    assert first.principal_id == second.principal_id == first.user_id
    assert first.conv_id == second.conv_id
    assert first.request_id != second.request_id
    assert [result.metadata["response"] for result in report.results] == [
        "hello",
        "check network",
    ]


def test_dialog_route_accepts_expected_domain_in_multi_label_result():
    service = RecordingChatService([
        chat_result(
            intent="query",
            agent_type="technical",
            matched_intents=["query", "technical"],
        )
    ])

    report = asyncio.run(
        evaluator(service).run(dialog_cases=[{
            "question": "校园网401应该怎么排查",
            "expected_intents": ["technical"],
            "expected_agents": ["technical"],
            "expect_knowledge": False,
        }])
    )

    result = report.results[0]
    assert result.metadata["actual_intents"] == ["query", "technical"]
    assert result.metadata["route_correct"] is True
    assert result.passed is True


def test_api_evaluator_builder_injects_shared_service_and_embedding_cache(monkeypatch):
    service = RecordingChatService([])
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", "C:/models/bge")

    subject = main._build_end_to_end_evaluator(
        chat_service=service,
        recognizer=object(),
        cfg={
            "api_key": "test-key",
            "model": "test-model",
            "base_url": "https://api.deepseek.com/anthropic",
        },
        baseline_path=None,
    )

    assert subject._chat_service is service
    assert subject._ragas_evaluator._embedding_cache_dir == "C:/models/bge"
    assert subject._ragas_evaluator._llm_options == {
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    assert subject._ragas_evaluator._judge_max_tokens == 8192


def test_eval_dialog_input_accepts_optional_pipeline_expectations():
    payload = main.EvalDialogInput(
        question="校园卡重复扣费",
        expected_intents=["billing"],
        expected_agents=["billing"],
        expected_tools=["query_campus_card", "create_ticket"],
        expect_ticket=True,
        expect_knowledge=True,
    ).model_dump(exclude_none=True)

    assert payload["expected_intents"] == ["billing"]
    assert payload["expected_agents"] == ["billing"]
    assert payload["expected_tools"] == [
        "query_campus_card",
        "create_ticket",
    ]
    assert payload["expect_ticket"] is True
    assert payload["expect_knowledge"] is True


def test_report_covers_routes_tools_tickets_knowledge_completion_and_counts():
    service = RecordingChatService(
        [
            chat_result(
                intent="billing",
                agent_type="billing",
                latency_ms=25.5,
                knowledge_used=True,
                citations=[{"id": "cite-1", "title": "校园卡规则"}],
                tool_calls=[
                    {"name": "query_campus_card", "success": True},
                    {"name": "create_ticket", "success": True},
                ],
                ticket_ids=["ticket-1"],
                model_call_count=3,
            )
        ]
    )

    report = asyncio.run(
        evaluator(service).run(
            dialog_cases=[
                {
                    "question": "校园卡重复扣费",
                    "expected_intents": ["billing"],
                    "expected_agents": ["billing"],
                    "expected_tools": ["query_campus_card", "create_ticket"],
                    "expect_ticket": True,
                    "expect_knowledge": True,
                }
            ]
        )
    )

    result = report.results[0]
    assert result.passed is True
    assert result.metadata == {
        **result.metadata,
        "expected_intent": "billing",
        "actual_intent": "billing",
        "expected_agent": "billing",
        "actual_agents": ["billing"],
        "route_correct": True,
        "expected_tools": ["query_campus_card", "create_ticket"],
        "tool_names": ["query_campus_card", "create_ticket"],
        "tools_correct": True,
        "expect_ticket": True,
        "ticket_ids": ["ticket-1"],
        "ticket_correct": True,
        "expect_knowledge": True,
        "knowledge_used": True,
        "citation_count": 1,
        "knowledge_correct": True,
        "task_completed": True,
        "latency_ms": 25.5,
        "model_call_count": 3,
        "tool_call_count": 2,
    }
    assert result.scores["route_correctness"] == 1.0
    assert result.scores["task_completion"] == 1.0
    assert report.metrics == {
        "route_correctness": 1.0,
        "tool_correctness": 1.0,
        "ticket_correctness": 1.0,
        "knowledge_correctness": 1.0,
        "task_completion": 1.0,
        "avg_latency_ms": 25.5,
        "model_call_count": 3,
        "model_call_count_observed_turns": 1,
        "tool_call_count": 2,
        "ragas_case_count": 0,
        "ragas_pass_rate": None,
    }


def test_unmet_optional_expectation_fails_task_and_is_regression():
    service = RecordingChatService(
        [
            chat_result(
                intent="billing",
                agent_type="billing",
                tool_calls=[{"name": "create_ticket", "success": True}],
                ticket_ids=["ticket-1"],
            ),
            chat_result(intent="technical", agent_type="technical"),
        ]
    )
    subject = evaluator(service)

    first = asyncio.run(
        subject.run(
            dialog_cases=[
                {
                    "question": "扣费异常",
                    "expected_intents": ["billing"],
                    "expected_agents": ["billing"],
                    "expected_tools": ["create_ticket"],
                    "expect_ticket": True,
                }
            ]
        )
    )
    second = asyncio.run(
        subject.run(
            dialog_cases=[
                {
                    "question": "扣费异常",
                    "expected_intents": ["billing"],
                    "expected_agents": ["billing"],
                    "expected_tools": ["create_ticket"],
                    "expect_ticket": True,
                }
            ]
        )
    )

    assert first.results[0].passed is True
    assert second.results[0].passed is False
    assert second.results[0].metadata["task_completed"] is False
    assert second.metrics["route_correctness"] == 0.0
    assert second.metrics["tool_correctness"] == 0.0
    assert second.metrics["ticket_correctness"] == 0.0
    assert any("route_correctness" in item for item in second.regressions)
    assert any("task_completion" in item for item in second.regressions)


def test_legacy_question_case_needs_no_new_expectation_fields():
    service = RecordingChatService([chat_result(response="legacy works")])

    report = asyncio.run(
        evaluator(service).run(dialog_cases=[{"question": "旧用例"}])
    )

    assert report.total == 1
    assert report.passed == 1
    assert report.results[0].metadata["task_completed"] is True
    assert report.metrics["route_correctness"] is None
    assert report.metrics["tool_correctness"] is None
    assert report.metrics["ticket_correctness"] is None
    assert report.metrics["knowledge_correctness"] is None
    assert report.results[0].metadata["model_call_count"] is None
    assert report.results[0].metadata["model_call_count_available"] is False
    assert report.metrics["model_call_count"] == 0
    assert report.metrics["model_call_count_observed_turns"] == 0


def test_case_level_outcomes_are_scored_only_after_the_final_turn():
    service = RecordingChatService(
        [
            chat_result(response="need more details"),
            chat_result(
                response="ticket created",
                tool_calls=[{"name": "create_ticket", "success": True}],
                ticket_ids=["ticket-2"],
            ),
        ]
    )

    report = asyncio.run(
        evaluator(service).run(
            dialog_cases=[
                {
                    "turns": ["网络坏了", "宿舍楼A无法联网"],
                    "expected_tools": ["create_ticket"],
                    "expect_ticket": True,
                }
            ]
        )
    )

    first, final = report.results
    assert first.metadata["tools_correct"] is None
    assert first.metadata["ticket_correct"] is None
    assert first.metadata["task_completed"] is True
    assert final.metadata["tools_correct"] is True
    assert final.metadata["ticket_correct"] is True
    assert final.metadata["task_completed"] is True
    assert report.metrics["tool_correctness"] == 1.0
    assert report.metrics["ticket_correctness"] == 1.0


def test_expect_no_knowledge_rejects_either_knowledge_or_citation_use():
    service = RecordingChatService(
        [chat_result(knowledge_used=True, citations=[])]
    )

    report = asyncio.run(
        evaluator(service).run(
            dialog_cases=[
                {"question": "不要检索", "expect_knowledge": False}
            ]
        )
    )

    assert report.results[0].metadata["knowledge_correct"] is False
    assert report.results[0].metadata["task_completed"] is False


class RecordingRagasEvaluator:
    def __init__(self, scores=None):
        self.calls = []
        self._scores = scores or RagasScores(
            faithfulness=0.9,
            answer_relevancy=0.85,
            context_precision=0.8,
            context_recall=0.75,
            answer_correctness=0.88,
        )

    async def evaluate(self, **sample):
        self.calls.append(sample)
        return self._scores


def test_ragas_receives_the_actual_retrieved_contexts_and_reference_answer():
    ragas = RecordingRagasEvaluator()
    service = RecordingChatService([
        chat_result(
            response="clear stale auth and reconnect",
            knowledge_used=True,
            citations=[
                {
                    "id": "network-auth",
                    "title": "Network authentication",
                    "content": "Clear stale authentication state after a 401 error.",
                    "score": 0.91,
                }
            ],
        )
    ])
    subject = EndToEndEvaluator(
        chat_service=service,
        recognizer=object(),
        judge=PassingJudge(),
        ragas_evaluator=ragas,
    )

    report = asyncio.run(subject.run(dialog_cases=[{
        "question": "how to handle network 401",
        "reference_answer": "clear stale auth and reconnect",
        "reference_context_ids": ["network-auth"],
        "expect_knowledge": True,
    }]))

    assert ragas.calls == [{
        "user_input": "how to handle network 401",
        "response": "clear stale auth and reconnect",
        "retrieved_contexts": ["Clear stale authentication state after a 401 error."],
        "reference": "clear stale auth and reconnect",
    }]
    result = report.results[0]
    assert result.scores["faithfulness"] == 0.9
    assert result.scores["answer_relevancy"] == 0.85
    assert result.scores["context_precision"] == 0.8
    assert result.scores["context_recall"] == 0.75
    assert result.scores["answer_correctness"] == 0.88
    assert result.scores["retrieval_hit_rate"] == 1.0
    assert result.scores["retrieval_recall_at_k"] == 1.0
    assert result.scores["retrieval_mrr"] == 1.0
    assert result.scores["retrieval_ndcg_at_k"] == 1.0
    assert result.metadata["retrieved_context_ids"] == ["network-auth"]
    assert report.metrics["ragas_case_count"] == 1
    assert report.metrics["ragas_pass_rate"] == 1.0


def test_faithfulness_failure_keeps_quality_scores_but_fails_rag_case():
    ragas = RecordingRagasEvaluator(RagasScores(
        faithfulness=0.4,
        answer_relevancy=0.9,
        context_precision=0.9,
        context_recall=0.9,
        answer_correctness=0.9,
    ))
    service = RecordingChatService([chat_result(
        response="unsupported claim",
        knowledge_used=True,
        citations=[{"id": "doc-1", "content": "evidence", "title": "evidence"}],
    )])
    subject = EndToEndEvaluator(
        chat_service=service, recognizer=object(), judge=PassingJudge(),
        ragas_evaluator=ragas,
    )

    report = asyncio.run(subject.run(dialog_cases=[{
        "question": "what is the rule",
        "reference_answer": "evidence",
        "reference_context_ids": ["doc-1"],
        "expect_knowledge": True,
    }]))

    result = report.results[0]
    assert result.scores["overall"] == 0.9
    assert result.scores["faithfulness"] == 0.4
    assert result.metadata["quality_passed"] is True
    assert result.metadata["ragas_passed"] is False
    assert result.metadata["task_completed"] is False


def test_retrieval_ranking_metrics_respect_rank_and_graded_relevance():
    metrics = retrieval_ranking_metrics(
        ["noise", "partial", "gold"],
        {"gold": 3.0, "partial": 1.0},
    )

    assert metrics["retrieval_hit_rate"] == 1.0
    assert metrics["retrieval_recall_at_k"] == 1.0
    assert metrics["retrieval_mrr"] == 0.5
    assert 0.0 < metrics["retrieval_ndcg_at_k"] < 1.0
    assert retrieval_ranking_metrics([], {"gold": 3.0}) == {
        "retrieval_hit_rate": 0.0,
        "retrieval_recall_at_k": 0.0,
        "retrieval_mrr": 0.0,
        "retrieval_ndcg_at_k": 0.0,
    }


def test_low_quality_score_is_retained_when_the_case_fails():
    class LowQualityJudge:
        async def judge(self, question, response, context=None):
            return QualityScores(0.6, 0.6, 0.6, 0.6)

    service = RecordingChatService([chat_result(
        response="ticket created but answer quality is weak",
        tool_calls=[{"name": "create_ticket", "success": True}],
        ticket_ids=["ticket-1"],
    )])
    subject = EndToEndEvaluator(
        chat_service=service,
        recognizer=object(),
        judge=LowQualityJudge(),
    )

    report = asyncio.run(subject.run(dialog_cases=[{
        "question": "create a ticket",
        "expected_tools": ["create_ticket"],
        "expect_ticket": True,
    }]))

    result = report.results[0]
    assert result.scores["overall"] == 0.6
    assert result.metadata["tools_correct"] is True
    assert result.metadata["ticket_correct"] is True
    assert result.metadata["quality_passed"] is False
    assert result.metadata["task_completed"] is False


def test_eval_dialog_input_accepts_ragas_ground_truth_fields():
    payload = main.EvalDialogInput(
        question="how to handle network 401",
        reference_answer="clear stale auth and login again",
        reference_context_ids=["network-auth"],
        reference_context_relevance={"network-auth": 3.0},
    ).model_dump(exclude_none=True)

    assert payload["reference_answer"].startswith("clear stale auth")
    assert payload["reference_context_ids"] == ["network-auth"]
    assert payload["reference_context_relevance"] == {"network-auth": 3.0}
