"""
亮点：端到端 Agent 评测框架

核心问题：如何评测端到端 Agent？

评测维度：
  1. 意图识别准确率 —— 预测意图 vs 标注意图，计算 Accuracy / F1
  2. 响应质量评分 —— 用 LLM 作为评判者（LLM-as-Judge），
     从相关性、准确性、完整性、有用性四个维度打分
  3. 端到端对话评测 —— 模拟完整多轮对话，评估整体体验
  4. 回归测试 —— 与历史基线对比，防止性能退化

LLM-as-Judge 是评测 Agent 质量的关键技术：
  人工标注成本高、主观性强；用 LLM 评判可以规模化、可重复。
"""
import asyncio
import json
import logging
import math
import pathlib
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content

from core.intent_recognizer import IntentCategory, IntentRecognizer

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class IntentTestCase:
    message:          str
    expected_intent:  str
    context:          Optional[Dict[str, Any]] = None
    expected_intents: Optional[List[str]] = None


@dataclass
class QualityScores:
    """LLM-as-Judge 评分结果。"""
    relevance:    float   # 相关性：回答是否针对问题
    accuracy:     float   # 准确性：信息是否正确
    completeness: float   # 完整性：是否完整解决问题
    helpfulness:  float   # 有用性：用户是否能据此行动
    judge_failed: bool = False
    error: Optional[str] = None

    @property
    def overall(self) -> float:
        return statistics.mean([self.relevance, self.accuracy, self.completeness, self.helpfulness])


@dataclass
class RagasScores:
    """RAGAS retrieval/generation scores for one answered turn."""

    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_correctness: Optional[float] = None
    judge_failed: bool = False
    error: Optional[str] = None

    def available_scores(self) -> Dict[str, float]:
        return {
            name: value
            for name, value in (
                ("faithfulness", self.faithfulness),
                ("answer_relevancy", self.answer_relevancy),
                ("context_precision", self.context_precision),
                ("context_recall", self.context_recall),
                ("answer_correctness", self.answer_correctness),
            )
            if value is not None
        }


def retrieval_ranking_metrics(
    retrieved_context_ids: List[str],
    reference_relevance: Dict[str, float],
) -> Dict[str, float]:
    """Compute deterministic retrieval metrics from stable context IDs.

    ``reference_relevance`` supports binary labels (0/1) and graded relevance
    (for example 1=partly useful, 3=direct answer evidence).  NDCG rewards both
    finding relevant chunks and placing the most relevant chunks near the top.
    """

    relevant = {
        str(context_id): max(0.0, float(relevance))
        for context_id, relevance in reference_relevance.items()
        if isinstance(relevance, (int, float))
        and not isinstance(relevance, bool)
        and math.isfinite(float(relevance))
        and float(relevance) > 0.0
    }
    if not relevant:
        return {}

    retrieved = [str(context_id) for context_id in retrieved_context_ids]
    hits = [context_id for context_id in retrieved if context_id in relevant]
    hit_rate = float(bool(hits))
    recall = len(set(hits)) / len(relevant)

    first_rank = next(
        (index for index, context_id in enumerate(retrieved, start=1) if context_id in relevant),
        None,
    )
    mrr = 1.0 / first_rank if first_rank is not None else 0.0

    def dcg(values: List[float]) -> float:
        return sum(
            (2.0 ** relevance - 1.0) / math.log2(rank + 1.0)
            for rank, relevance in enumerate(values, start=1)
        )

    gains = [relevant.get(context_id, 0.0) for context_id in retrieved]
    ideal = sorted(relevant.values(), reverse=True)[:len(retrieved)]
    ideal_dcg = dcg(ideal)
    ndcg = dcg(gains) / ideal_dcg if ideal_dcg > 0.0 else 0.0
    return {
        "retrieval_hit_rate": round(hit_rate, 6),
        "retrieval_recall_at_k": round(recall, 6),
        "retrieval_mrr": round(mrr, 6),
        "retrieval_ndcg_at_k": round(ndcg, 6),
    }


@dataclass
class EvalResult:
    test_id:    str
    passed:     bool
    scores:     Dict[str, float]
    detail:     str = ""
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """评测报告。"""
    timestamp:        str
    total:            int
    passed:           int
    pass_rate:        float
    avg_scores:       Dict[str, float]
    regressions:      List[str]          # 相比基线退化的指标
    recommendations:  List[str]
    results:          List[EvalResult]
    metrics:          Dict[str, Any] = field(default_factory=dict)


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────

class LLMJudge:
    """
    用 LLM 评判 Agent 响应质量。

    为什么用 LLM 而不是人工？
    - 可规模化：数千条测试用例自动评测
    - 可重复：相同输入得到稳定评分
    - 多维度：同时评估相关性、准确性等多个维度

    注意：LLM Judge 本身也有偏差，建议定期用人工标注校准。
    """

    JUDGE_PROMPT = """你是一个客服质量评估专家。请对以下客服响应进行评分。

用户问题: {question}
Agent 响应: {response}
{context_section}

请从以下四个维度评分（0.0-1.0），返回 JSON：
- relevance: 响应是否直接针对用户问题（0=完全无关，1=完全相关）
- accuracy: 信息是否准确无误（0=明显错误，1=完全正确）
- completeness: 是否完整解决了用户需求（0=完全没解决，1=完全解决）
- helpfulness: 用户能否据此采取行动（0=毫无帮助，1=非常有帮助）

只返回 JSON，例如: {{"relevance": 0.9, "accuracy": 0.8, "completeness": 0.7, "helpfulness": 0.85}}"""

    def __init__(self, client: AsyncAnthropic, model: str):
        self._client = client
        self._model  = model

    async def judge(
        self,
        question: str,
        response: str,
        context: Optional[str] = None,
    ) -> QualityScores:
        ctx_section = f"背景信息: {context}" if context else ""
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            response=response,
            context_section=ctx_section,
        )
        prompt = self._clean_text(prompt)
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=256, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            return QualityScores(
                relevance=float(data.get("relevance", 0.5)),
                accuracy=float(data.get("accuracy", 0.5)),
                completeness=float(data.get("completeness", 0.5)),
                helpfulness=float(data.get("helpfulness", 0.5)),
            )
        except Exception as ex:
            logger.warning(f"LLM Judge 失败: {ex}")
            return QualityScores(
                0.5, 0.5, 0.5, 0.5,
                judge_failed=True,
                error=str(ex),
            )

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 LLM 请求编码失败。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")


# ── 意图识别评测 ──────────────────────────────────────────────────────────────

class IntentEvaluator:
    """评测意图识别的准确率和 F1。"""

    def __init__(self, recognizer: IntentRecognizer):
        self._recognizer = recognizer

    async def evaluate(self, cases: List[IntentTestCase]) -> Dict[str, Any]:
        predictions, ground_truth = [], []
        prediction_sets: List[set[str]] = []
        ground_truth_sets: List[set[str]] = []
        case_details: List[Dict[str, Any]] = []

        for case in cases:
            result = await self._recognizer.recognize(case.message)
            predicted = result.intent.value
            expected_labels = list(dict.fromkeys(
                case.expected_intents or [case.expected_intent]
            ))
            predicted_labels = list(dict.fromkeys(
                item.value if isinstance(item, IntentCategory) else str(item)
                for item in getattr(result, "matched_intents", [])
            ))
            if not predicted_labels:
                predicted_labels = [predicted]
            predictions.append(predicted)
            ground_truth.append(case.expected_intent)
            prediction_sets.append(set(predicted_labels))
            ground_truth_sets.append(set(expected_labels))
            case_details.append({
                "message": case.message,
                "expected": case.expected_intent,
                "predicted": predicted,
                "expected_intents": expected_labels,
                "predicted_intents": predicted_labels,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
            })

        # 纯 Python 计算指标
        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        accuracy = correct / len(predictions) if predictions else 0.0

        # 多标签逐类 F1；单标签用例会自然退化为原有计算方式。
        labels = sorted(set().union(*ground_truth_sets, *prediction_sets))
        per_class: Dict[str, Dict[str, float]] = {}
        for label in labels:
            tp = sum(
                label in predicted and label in expected
                for predicted, expected in zip(prediction_sets, ground_truth_sets)
            )
            fp = sum(
                label in predicted and label not in expected
                for predicted, expected in zip(prediction_sets, ground_truth_sets)
            )
            fn = sum(
                label not in predicted and label in expected
                for predicted, expected in zip(prediction_sets, ground_truth_sets)
            )
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_class[label] = {"precision": prec, "recall": rec, "f1": f1}

        macro_f1 = statistics.mean(v["f1"] for v in per_class.values()) if per_class else 0.0
        subset_correct = sum(
            predicted == expected
            for predicted, expected in zip(prediction_sets, ground_truth_sets)
        )
        subset_accuracy = subset_correct / len(cases) if cases else 0.0

        return {
            "accuracy":   round(accuracy, 4),
            "subset_accuracy": round(subset_accuracy, 4),
            "macro_f1":   round(macro_f1, 4),
            "per_class":  per_class,
            "total":      len(cases),
            "correct":    correct,
            "subset_correct": subset_correct,
            "cases":      case_details,
        }


# ── 端到端评测器 ──────────────────────────────────────────────────────────────

class EndToEndEvaluator:
    """
    端到端 Agent 评测。

    评测流程：
      1. 运行意图识别评测（准确率/F1）
      2. 运行对话质量评测（LLM-as-Judge）
      3. 与历史基线对比（回归检测）
      4. 生成可操作的优化建议
    """

    # 质量及格线
    PASS_THRESHOLD = 0.75

    def __init__(
        self,
        chat_service,
        recognizer: IntentRecognizer,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        baseline_path: Optional[str] = None,
        judge: Optional[Any] = None,
        ragas_evaluator: Optional[Any] = None,
    ):
        if chat_service is None or not callable(
            getattr(chat_service, "chat", None)
        ):
            raise TypeError("chat_service must expose an async chat method")
        if judge is None:
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError("api_key is required when judge is not provided")
            kwargs: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            client = AsyncAnthropic(**kwargs)
            judge = LLMJudge(client, model)

        self._chat_service     = chat_service
        self._judge            = judge
        self._ragas_evaluator  = ragas_evaluator
        self._intent_evaluator = IntentEvaluator(recognizer)
        self._history:         List[EvalReport] = []
        self._baseline_path = pathlib.Path(baseline_path) if baseline_path else None
        self._baseline: Optional[EvalReport] = self._load_baseline()

    async def run(
        self,
        intent_cases:    Optional[List[IntentTestCase]] = None,
        dialog_cases:    Optional[List[Dict[str, Any]]] = None,
    ) -> EvalReport:
        """
        运行完整评测。

        intent_cases: 意图识别测试用例
        dialog_cases:
          - 单轮: [{"question": "..."}]
          - 多轮: [{"turns": ["第一轮", "第二轮", ...]}]
        """
        results: List[EvalResult] = []
        all_scores: Dict[str, List[float]] = {
            "relevance": [], "accuracy": [], "completeness": [], "helpfulness": [],
            "faithfulness": [], "answer_relevancy": [],
            "context_precision": [], "context_recall": [],
            "answer_correctness": [], "retrieval_hit_rate": [],
            "retrieval_recall_at_k": [], "retrieval_mrr": [],
            "retrieval_ndcg_at_k": [],
            "evidence_grounding": [], "reflection_correction": [],
        }

        # 1. 意图识别评测
        intent_metrics: Dict[str, Any] = {}
        if intent_cases:
            intent_metrics = await self._intent_evaluator.evaluate(intent_cases)
            passed = (
                intent_metrics["accuracy"] >= self.PASS_THRESHOLD
                and intent_metrics["macro_f1"] >= self.PASS_THRESHOLD
            )
            results.append(EvalResult(
                test_id="intent_recognition",
                passed=passed,
                scores={
                    "accuracy": intent_metrics["accuracy"],
                    "subset_accuracy": intent_metrics["subset_accuracy"],
                    "macro_f1": intent_metrics["macro_f1"],
                },
                detail=(
                    f"主意图准确率 {intent_metrics['accuracy']:.1%}，"
                    f"子集准确率 {intent_metrics['subset_accuracy']:.1%}，"
                    f"多标签 Macro-F1 {intent_metrics['macro_f1']:.3f}"
                ),
                metadata={
                    "total": intent_metrics.get("total", 0),
                    "correct": intent_metrics.get("correct", 0),
                    "cases": intent_metrics.get("cases", []),
                },
            ))

        # 2. 对话质量评测（调用 orchestrator 产出回复，再用 LLM Judge 评分）
        if dialog_cases:
            for i, case in enumerate(dialog_cases):
                case_results = await self._evaluate_dialog_case(case, i)
                results.extend(case_results)
                for r in case_results:
                    for k in all_scores:
                        if k in r.scores:
                            all_scores[k].append(r.scores[k])

        # 3. 汇总
        dialog_results = [
            result
            for result in results
            if "model_call_count" in result.metadata
        ]
        metrics = self._dialog_metrics(dialog_results)
        avg_scores = {
            k: round(statistics.mean(v), 4) for k, v in all_scores.items() if v
        }
        if intent_metrics:
            avg_scores["intent_accuracy"] = intent_metrics["accuracy"]
            avg_scores["intent_subset_accuracy"] = intent_metrics[
                "subset_accuracy"
            ]
            avg_scores["intent_macro_f1"] = intent_metrics["macro_f1"]
        for name in (
            "route_correctness",
            "tool_correctness",
            "ticket_correctness",
            "knowledge_correctness",
            "task_completion",
        ):
            value = metrics.get(name)
            if value is not None:
                avg_scores[name] = value

        passed_count = sum(1 for r in results if r.passed)
        pass_rate    = passed_count / len(results) if results else 0.0

        # 4. 回归检测
        regressions = self._detect_regressions(avg_scores)

        # 5. 优化建议
        recommendations = self._recommendations(avg_scores, intent_metrics)

        report = EvalReport(
            timestamp=datetime.now().isoformat(),
            total=len(results),
            passed=passed_count,
            pass_rate=round(pass_rate, 4),
            avg_scores=avg_scores,
            regressions=regressions,
            recommendations=recommendations,
            results=results,
            metrics=metrics,
        )
        self._history.append(report)
        self._save_baseline(report)
        return report

    @staticmethod
    def _dialog_turns(case: Dict[str, Any]) -> List[str]:
        turns = case.get("turns")
        if isinstance(turns, list):
            return [str(t) for t in turns if str(t).strip()]
        question = case.get("question")
        return [str(question)] if question else []

    @staticmethod
    def _history_context(history: List[Dict[str, str]]) -> str:
        if not history:
            return ""
        lines = [f"{m['role']}: {m['content']}" for m in history[-8:]]
        return "[评测多轮历史]\n" + "\n".join(lines)

    async def _evaluate_dialog_case(self, case: Dict[str, Any], case_idx: int) -> List[EvalResult]:
        """Evaluate every turn through the same ChatService used by the API."""
        from services.chat_service import ChatCommand

        questions = self._dialog_turns(case)
        if not questions:
            return []
        namespace = uuid.uuid4().hex
        conv_id = str(case.get("conv_id") or f"eval-conv-{namespace}")
        user_id = str(case.get("user_id") or f"eval-user-{namespace}")
        history: List[Dict[str, str]] = []
        observations: List[Dict[str, Any]] = []

        for turn_idx, question in enumerate(questions):
            context = self._history_context(history)
            chat_result = await self._chat_service.chat(ChatCommand(
                message=question,
                user_id=user_id,
                conv_id=conv_id,
                principal_id=user_id,
                request_id=f"eval-{case_idx}-{turn_idx}-{namespace}",
            ))
            answer = str(chat_result.response)
            scores = await self._judge.judge(
                question,
                answer,
                context=context or None,
            )
            history.extend((
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ))
            agents = self._result_agents(chat_result)
            tool_calls = self._result_tool_calls(chat_result)
            citations = self._result_citations(chat_result)
            evidence_verification = getattr(
                chat_result,
                "evidence_verification",
                {},
            )
            if not isinstance(evidence_verification, dict):
                evidence_verification = {}
            retrieved_contexts = [
                citation["content"]
                for citation in citations
                if citation["content"]
            ]
            reference = self._reference_at(case, "reference_answers", turn_idx)
            if reference is None:
                value = case.get("reference_answer")
                reference = value.strip() if isinstance(value, str) and value.strip() else None
            ragas_scores = RagasScores()
            if self._ragas_evaluator is not None and retrieved_contexts:
                ragas_scores = await self._ragas_evaluator.evaluate(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=retrieved_contexts,
                    reference=reference,
                )
            actual_intent = str(getattr(chat_result, "intent", ""))
            actual_intents = self._string_list(
                getattr(chat_result, "matched_intents", [])
            )
            if not actual_intents and actual_intent:
                actual_intents = [actual_intent]
            observations.append({
                "turn": turn_idx,
                "question": question,
                "response": answer,
                "scores": scores,
                "ragas_scores": ragas_scores,
                "reference_answer": reference,
                "intent": actual_intent,
                "actual_intents": actual_intents,
                "agents": agents,
                "tool_calls": tool_calls,
                "ticket_ids": self._string_list(
                    getattr(chat_result, "ticket_ids", [])
                ),
                "knowledge_used": getattr(
                    chat_result,
                    "knowledge_used",
                    False,
                ) is True,
                "citations": citations,
                "citation_count": len(citations),
                "latency_ms": self._nonnegative_number(
                    getattr(chat_result, "latency_ms", 0.0)
                ),
                "model_call_count": self._model_call_count(chat_result),
                "trace_id": str(getattr(chat_result, "trace_id", "")),
                "evidence_verification": dict(evidence_verification),
            })

        expected_intents = self._string_list(case.get("expected_intents"))
        expected_agents = self._string_list(case.get("expected_agents"))
        expected_tools = self._string_list(case.get("expected_tools"))
        expect_ticket = self._optional_bool(case.get("expect_ticket"))
        expect_knowledge = self._optional_bool(case.get("expect_knowledge"))
        reference_context_ids = self._string_list(case.get("reference_context_ids"))
        reference_relevance = self._reference_relevance(
            case.get("reference_context_relevance"),
            reference_context_ids,
        )
        all_tool_calls = [
            call for item in observations for call in item["tool_calls"]
        ]
        tool_names = [call["name"] for call in all_tool_calls]
        successful_tools = {
            call["name"] for call in all_tool_calls if call["success"]
        }
        ticket_ids = list(dict.fromkeys(
            ticket_id
            for item in observations
            for ticket_id in item["ticket_ids"]
        ))
        knowledge_used = any(item["knowledge_used"] for item in observations)
        citation_count = sum(item["citation_count"] for item in observations)
        tools_correct = (
            all(name in successful_tools for name in expected_tools)
            if expected_tools else None
        )
        ticket_correct = (
            bool(ticket_ids) is expect_ticket
            if expect_ticket is not None else None
        )
        knowledge_correct = (
            (
                knowledge_used and citation_count > 0
                if expect_knowledge
                else not knowledge_used and citation_count == 0
            )
            if expect_knowledge is not None else None
        )

        results: List[EvalResult] = []
        for item in observations:
            turn = item["turn"]
            is_final_turn = turn == len(observations) - 1
            scores = item["scores"]
            ragas_scores = item["ragas_scores"]
            expected_intent = self._expected_at(expected_intents, turn)
            expected_agent = self._expected_at(expected_agents, turn)
            route_checks = []
            if expected_intent is not None:
                route_checks.append(expected_intent in item["actual_intents"])
            if expected_agent is not None:
                route_checks.append(expected_agent in item["agents"])
            route_correct = all(route_checks) if route_checks else None
            turn_tools_correct = tools_correct if is_final_turn else None
            turn_ticket_correct = ticket_correct if is_final_turn else None
            turn_knowledge_correct = knowledge_correct if is_final_turn else None
            checks = [
                check for check in (
                    route_correct,
                    turn_tools_correct,
                    turn_ticket_correct,
                    turn_knowledge_correct,
                )
                if check is not None
            ]
            retrieved_context_ids = [
                citation["id"] for citation in item["citations"]
            ]
            ranking_scores = retrieval_ranking_metrics(
                retrieved_context_ids,
                reference_relevance,
            )
            ragas_values = ragas_scores.available_scores()
            ragas_required = (
                bool(item["citations"])
                and (
                    item["reference_answer"] is not None
                    or bool(reference_relevance)
                )
            )
            ragas_passed = (
                all(value >= self.PASS_THRESHOLD for value in ragas_values.values())
                and all(value >= self.PASS_THRESHOLD for value in ranking_scores.values())
                and not ragas_scores.judge_failed
            ) if ragas_required else None
            quality_passed = scores.overall >= self.PASS_THRESHOLD
            verification = item["evidence_verification"]
            evidence_checked = verification.get("checked") is True
            evidence_passed = (
                verification.get("passed") is True
                if evidence_checked else None
            )
            reflection_attempted = (
                evidence_checked
                and isinstance(verification.get("reflection_count"), int)
                and verification.get("reflection_count", 0) > 0
            )
            evidence_scores = {}
            if evidence_checked:
                evidence_scores["evidence_grounding"] = float(
                    evidence_passed is True
                )
            if reflection_attempted:
                evidence_scores["reflection_correction"] = float(
                    verification.get("corrected") is True
                )
            task_completed = (
                quality_passed
                and all(checks)
                and ragas_passed is not False
                and evidence_passed is not False
            )
            test_id = (
                f"dialog_{case_idx}"
                if len(questions) == 1
                else f"dialog_{case_idx}_turn_{turn}"
            )
            results.append(EvalResult(
                test_id=test_id,
                passed=task_completed,
                scores={
                    "relevance": scores.relevance,
                    "accuracy": scores.accuracy,
                    "completeness": scores.completeness,
                    "helpfulness": scores.helpfulness,
                    "overall": scores.overall,
                    "route_correctness": (
                        float(route_correct)
                        if route_correct is not None else None
                    ),
                    "task_completion": float(task_completed),
                    **ragas_values,
                    **ranking_scores,
                    **evidence_scores,
                },
                detail=(
                    f"Q: {item['question'][:30]}... -> "
                    f"overall {scores.overall:.3f}"
                ),
                metadata={
                    "question": item["question"],
                    "response": item["response"],
                    "agent_type": item["agents"][0] if item["agents"] else "",
                    "intent": item["intent"],
                    "turn": turn,
                    "conv_id": conv_id,
                    "judge_failed": scores.judge_failed,
                    "judge_error": scores.error,
                    "ragas_judge_failed": ragas_scores.judge_failed,
                    "ragas_judge_error": ragas_scores.error,
                    "quality_passed": quality_passed,
                    "ragas_passed": ragas_passed,
                    "reference_answer": item["reference_answer"],
                    "reference_context_ids": list(reference_relevance),
                    "retrieved_context_ids": retrieved_context_ids,
                    "expected_intent": expected_intent,
                    "actual_intent": item["intent"],
                    "actual_intents": item["actual_intents"],
                    "expected_agent": expected_agent,
                    "actual_agents": item["agents"],
                    "route_correct": route_correct,
                    "expected_tools": expected_tools,
                    "tool_names": tool_names,
                    "tools_correct": turn_tools_correct,
                    "expect_ticket": expect_ticket,
                    "ticket_ids": ticket_ids,
                    "ticket_correct": turn_ticket_correct,
                    "expect_knowledge": expect_knowledge,
                    "knowledge_used": knowledge_used,
                    "citation_count": citation_count,
                    "knowledge_correct": turn_knowledge_correct,
                    "task_completed": task_completed,
                    "latency_ms": item["latency_ms"],
                    "model_call_count": item["model_call_count"],
                    "model_call_count_available": (
                        item["model_call_count"] is not None
                    ),
                    "tool_call_count": len(item["tool_calls"]),
                    "trace_id": item["trace_id"],
                    "evidence_verification": dict(verification),
                    "evidence_passed": evidence_passed,
                    "reflection_attempted": reflection_attempted,
                },
            ))
        return results

    @staticmethod
    def _string_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        ]

    @staticmethod
    def _optional_bool(value: Any) -> Optional[bool]:
        return value if isinstance(value, bool) else None

    @staticmethod
    def _expected_at(values: List[str], index: int) -> Optional[str]:
        return values[index] if index < len(values) else None

    @staticmethod
    def _reference_at(case: Dict[str, Any], field: str, index: int) -> Optional[str]:
        values = case.get(field)
        if not isinstance(values, list) or index >= len(values):
            return None
        value = values[index]
        return value.strip() if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _reference_relevance(
        value: Any,
        fallback_ids: List[str],
    ) -> Dict[str, float]:
        if isinstance(value, dict):
            normalized: Dict[str, float] = {}
            for context_id, relevance in value.items():
                if (
                    isinstance(context_id, str)
                    and context_id.strip()
                    and isinstance(relevance, (int, float))
                    and not isinstance(relevance, bool)
                    and math.isfinite(float(relevance))
                    and float(relevance) > 0
                ):
                    normalized[context_id.strip()] = float(relevance)
            if normalized:
                return normalized
        return {context_id: 1.0 for context_id in fallback_ids}

    @staticmethod
    def _result_citations(result: Any) -> List[Dict[str, Any]]:
        raw = getattr(result, "citations", [])
        if not isinstance(raw, list):
            return []
        citations = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            context_id = item.get("id")
            if not isinstance(context_id, str) or not context_id.strip():
                continue
            citations.append({
                "id": context_id.strip(),
                "title": str(item.get("title", "")),
                "content": str(item.get("content", "")),
                "score": item.get("score"),
            })
        return citations

    @staticmethod
    def _nonnegative_number(value: Any) -> float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, float(value))
        return 0.0

    @classmethod
    def _result_agents(cls, result: Any) -> List[str]:
        agents = cls._string_list(getattr(result, "agent_types", []))
        primary = getattr(result, "agent_type", None)
        if not agents and isinstance(primary, str) and primary.strip():
            agents = [primary.strip()]
        return list(dict.fromkeys(agents))

    @staticmethod
    def _result_tool_calls(result: Any) -> List[Dict[str, Any]]:
        raw_calls = getattr(result, "tool_calls", [])
        if not isinstance(raw_calls, list):
            return []
        calls = []
        for raw in raw_calls:
            if not isinstance(raw, dict):
                continue
            name = raw.get("name")
            if isinstance(name, str) and name.strip():
                calls.append({
                    "name": name.strip(),
                    "success": raw.get("success") is True,
                })
        return calls

    @classmethod
    def _model_call_count(cls, result: Any) -> Optional[int]:
        trace = getattr(result, "trace", {})
        observed = trace.get("model_call_count") if isinstance(trace, dict) else None
        if (
            isinstance(observed, int)
            and not isinstance(observed, bool)
            and observed >= 0
        ):
            return observed
        return None

    @staticmethod
    def _dialog_metrics(results: List[EvalResult]) -> Dict[str, Any]:
        def optional_average(field: str) -> Optional[float]:
            values = [
                result.metadata[field]
                for result in results
                if result.metadata.get(field) is not None
            ]
            if not values:
                return None
            return round(statistics.mean(float(value) for value in values), 4)

        latencies = [
            float(result.metadata.get("latency_ms", 0.0))
            for result in results
        ]
        return {
            "route_correctness": optional_average("route_correct"),
            "tool_correctness": optional_average("tools_correct"),
            "ticket_correctness": optional_average("ticket_correct"),
            "knowledge_correctness": optional_average("knowledge_correct"),
            "task_completion": optional_average("task_completed"),
            "avg_latency_ms": (
                round(statistics.mean(latencies), 4) if latencies else 0.0
            ),
            "model_call_count": sum(
                int(result.metadata["model_call_count"])
                for result in results
                if result.metadata.get("model_call_count") is not None
            ),
            "model_call_count_observed_turns": sum(
                1
                for result in results
                if result.metadata.get("model_call_count") is not None
            ),
            "tool_call_count": sum(
                int(result.metadata.get("tool_call_count", 0))
                for result in results
            ),
            "ragas_case_count": sum(
                result.metadata.get("ragas_passed") is not None
                for result in results
            ),
            "ragas_pass_rate": optional_average("ragas_passed"),
        }

    def _detect_regressions(self, current: Dict[str, float]) -> List[str]:
        """与上一次评测对比，找出退化超过 5% 的指标。"""
        prev_report = self._history[-1] if self._history else self._baseline
        if prev_report is None:
            return []
        prev = prev_report.avg_scores
        regressions = []
        for metric, value in current.items():
            if metric in prev and prev[metric] > 0:
                delta = (value - prev[metric]) / prev[metric]
                if delta < -0.05:
                    regressions.append(
                        f"{metric}: {prev[metric]:.3f} → {value:.3f} (退化 {abs(delta):.1%})"
                    )
        return regressions

    def _recommendations(
        self,
        scores: Dict[str, float],
        intent_metrics: Dict[str, Any],
    ) -> List[str]:
        recs = []
        if scores.get("intent_accuracy", 1.0) < 0.90:
            recs.append("意图识别准确率 < 90%：增加 Few-shot 示例，或对低 F1 的意图类别补充训练数据")
        if scores.get("relevance", 1.0) < 0.75:
            recs.append("相关性偏低：检查 Agent system_prompt，确保 Agent 聚焦于用户问题")
        if scores.get("completeness", 1.0) < 0.75:
            recs.append("完整性偏低：Agent 可能过早结束回答，考虑在 prompt 中要求提供完整解决方案")
        if scores.get("helpfulness", 1.0) < 0.75:
            recs.append("有用性偏低：回答可能过于抽象，考虑要求 Agent 提供具体操作步骤")
        if scores.get("faithfulness", 1.0) < 0.75:
            recs.append("Faithfulness 偏低：回答包含检索证据无法支持的事实，收紧引用约束并检查检索上下文")
        if scores.get("context_precision", 1.0) < 0.75:
            recs.append("Context Precision 偏低：Top-K 中噪声片段偏多，检查混合检索权重或重排模型")
        if scores.get("context_recall", 1.0) < 0.75:
            recs.append("Context Recall 偏低：关键证据未召回，优化切块、查询改写或扩大候选召回数")
        if scores.get("retrieval_ndcg_at_k", 1.0) < 0.75:
            recs.append("NDCG@K 偏低：相关证据排序靠后，检查 Cross-Encoder 重排和相关性标注")
        if not recs:
            recs.append("所有指标均达标，继续保持")
        return recs

    @property
    def history(self) -> List[EvalReport]:
        return self._history

    def _load_baseline(self) -> Optional[EvalReport]:
        if not self._baseline_path or not self._baseline_path.exists():
            return None
        try:
            data = json.loads(self._baseline_path.read_text(encoding="utf-8"))
            return self._report_from_dict(data)
        except Exception as ex:
            logger.warning(f"读取评测基线失败: {ex}")
            return None

    def _save_baseline(self, report: EvalReport) -> None:
        if not self._baseline_path:
            return
        try:
            self._baseline_path.parent.mkdir(parents=True, exist_ok=True)
            self._baseline_path.write_text(
                json.dumps(asdict(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._baseline = report
        except Exception as ex:
            logger.warning(f"保存评测基线失败: {ex}")

    @staticmethod
    def _report_from_dict(data: Dict[str, Any]) -> EvalReport:
        return EvalReport(
            timestamp=data.get("timestamp", ""),
            total=int(data.get("total", 0)),
            passed=int(data.get("passed", 0)),
            pass_rate=float(data.get("pass_rate", 0.0)),
            avg_scores=dict(data.get("avg_scores", {})),
            regressions=list(data.get("regressions", [])),
            recommendations=list(data.get("recommendations", [])),
            results=[
                EvalResult(
                    test_id=r.get("test_id", ""),
                    passed=bool(r.get("passed", False)),
                    scores=dict(r.get("scores", {})),
                    detail=r.get("detail", ""),
                    metadata=dict(r.get("metadata", {})),
                )
                for r in data.get("results", [])
            ],
            metrics=dict(data.get("metrics", {})),
        )


# ── 内置测试用例（开箱即用）──────────────────────────────────────────────────

DEFAULT_INTENT_CASES: List[IntentTestCase] = [
    IntentTestCase("我的订单什么时候到？",       "query"),
    IntentTestCase("帮我取消订单",               "request"),
    IntentTestCase("你们服务太差了！",            "complaint"),
    IntentTestCase("应用一直报500错误",           "technical"),
    IntentTestCase("为什么扣了两次款？",          "billing"),
    IntentTestCase("我要投诉，转人工！",          "escalation"),
    IntentTestCase("你好",                        "greeting"),
    IntentTestCase("修改我的邮箱地址",            "account"),
]

DEFAULT_DIALOG_CASES: List[Dict[str, Any]] = [
    {"question": "我的订单 #12345 还没到，已经超时了"},
    {"question": "应用登录一直报错 401"},
    {"question": "为什么这个月多扣了 50 块钱？"},
    {"question": "帮我把收货地址改成北京市朝阳区"},
    {"turns": ["你好，我想退款", "订单号是 #12345", "退款多久能到账？"]},
]
