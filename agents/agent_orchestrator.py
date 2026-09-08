"""
亮点：多 Agent 路由与编排

核心问题：多 Agent 情况下如何做 Routing？

路由策略（三层决策）：
  1. 意图路由 —— 根据 IntentCategory 直接映射到专属 Agent
  2. 性能路由 —— 同类 Agent 有多个时，选成功率最高、延迟最低的
  3. 降级路由 —— 专属 Agent 不可用时，自动降级到 GeneralAgent

并行协作：
  - 复杂问题（如"技术问题 + 账单问题"）可同时派发给多个 Agent
  - 结果由 Orchestrator 合并后返回

升级机制：
  - Agent 置信度低于阈值 → 自动升级到更高级 Agent 或转人工
"""
import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from itertools import islice
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from agents.agent_runtime import (
    MAX_STEPS_RESPONSE,
    PROTOCOL_ERROR_RESPONSE,
    RUNTIME_ERROR_RESPONSE,
    TIMEOUT_RESPONSE,
    sanitize_text_content,
)
from agents.result_composer import AgentPart, ResultComposer
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from core.llm_utils import extract_text_content
from mcp.tool_manager import AGENT_TOOL_ALLOWLIST

_TOOL_CALL_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")

logger = logging.getLogger(__name__)

_RUNTIME_FAILURE_RESPONSES = {
    "max_steps": MAX_STEPS_RESPONSE,
    "protocol_error": PROTOCOL_ERROR_RESPONSE,
    "timeout": TIMEOUT_RESPONSE,
    "error": RUNTIME_ERROR_RESPONSE,
}


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class AgentType(Enum):
    GENERAL   = "general"    # 通用客服
    TECHNICAL = "technical"  # 技术支持
    BILLING   = "billing"    # 账单/退款
    ACCOUNT   = "account"
    ESCALATION = "escalation" # 人工升级（占位）


@dataclass
class AgentStats:
    """Agent 运行时统计，供 Monitor 和路由决策使用。"""
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
        """路由评分：成功率高、延迟低的 Agent 得分高。"""
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)


@dataclass
class AgentResponse:
    agent_type:  AgentType
    content:     str
    success:     bool = True
    confidence:  float = 1.0
    latency_ms:  float = 0.0
    escalate:    bool  = False   # 是否需要升级
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    ticket_ids: List[str] = field(default_factory=list)
    evidence_verification: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # 来自 MemoryManager 的格式化上下文
    history:     Optional[List[Dict[str, str]]] = None  # 对话历史，传给意图识别
    intent:      Optional[IntentCategory] = None
    urgency:     Optional[UrgencyLevel]   = None
    intent_scores: Dict[IntentCategory, float] = field(default_factory=dict)
    matched_intents: List[IntentCategory] = field(default_factory=list)
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    principal_id: Optional[str] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    escalated:   bool  = False
    latency_ms:  float = 0.0
    intent_scores: Dict[IntentCategory, float] = field(default_factory=dict)
    matched_intents: List[IntentCategory] = field(default_factory=list)
    agent_types: List[AgentType] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    ticket_ids: List[str] = field(default_factory=list)
    evidence_verification: Dict[str, Any] = field(default_factory=dict)


# ── 基础 Agent ────────────────────────────────────────────────────────────────

class BaseAgent:
    """所有 Agent 的基类，封装 LLM 调用和统计。"""

    agent_type: AgentType
    system_prompt: str

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        skill_manager: Optional[Any] = None,
        runtime: Optional[Any] = None,
        allowed_tools: Optional[List[str]] = None,
    ):
        self._client = client
        self._model  = model
        self._skill_manager = skill_manager
        self._runtime = runtime
        self._allowed_tools = tuple(allowed_tools or ())
        self.stats   = AgentStats()

    async def handle(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        try:
            tool_calls: List[Dict[str, Any]] = []
            ticket_ids: List[str] = []
            evidence_verification: Dict[str, Any] = {}
            success = True
            if self._runtime is not None and not self._is_escalation(req):
                runtime_result = await self._call_runtime(req)
                content = self._safe_runtime_text(runtime_result)
                tool_calls = self._runtime_tool_calls(runtime_result)
                ticket_ids = _unique_ticket_ids(
                    getattr(runtime_result, "ticket_ids", [])
                )
                evidence_verification = self._runtime_verification(
                    runtime_result
                )
                success = (
                    getattr(runtime_result, "stop_reason", None) == "end_turn"
                    and isinstance(
                        getattr(runtime_result, "text", None),
                        str,
                    )
                    and bool(
                        sanitize_text_content(
                            getattr(runtime_result, "text", ""),
                            16384,
                        ).strip()
                    )
                )
            else:
                content = sanitize_text_content(
                    await self._call_llm(req),
                    16384,
                )
            ms = (time.monotonic() - t0) * 1000
            if success:
                self.stats.success += 1
            self.stats.total_ms += ms
            escalate = self._needs_escalation(content)
            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=success,
                latency_ms=ms,
                escalate=escalate,
                tool_calls=tool_calls,
                ticket_ids=ticket_ids,
                evidence_verification=evidence_verification,
            )
        except Exception:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error("%s Agent processing failed", self.agent_type.value)
            return AgentResponse(
                agent_type=self.agent_type,
                content="抱歉，处理您的请求时出现问题，请稍后重试。",
                success=False,
                latency_ms=ms,
            )

    async def _call_llm(self, req: Request) -> str:
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        messages = []
        if req.context:
            messages.append({"role": "user", "content": f"[背景信息]\n{_clean(req.context)}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})
        messages.append({"role": "user", "content": _clean(req.message)})

        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=self._build_system_prompt(req),
            messages=messages,
        )
        return extract_text_content(resp.content)

    async def _call_runtime(self, req: Request) -> Any:
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        messages = []
        if req.context:
            messages.append(
                {
                    "role": "user",
                    "content": f"[背景信息]\n{_clean(req.context)}",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": "好的，我已了解背景信息。",
                }
            )
        messages.append({"role": "user", "content": _clean(req.message)})

        return await self._runtime.run(
            system_prompt=self._build_system_prompt(req),
            messages=messages,
            allowed_tools=list(self._allowed_tools),
            context=_runtime_context(req, self.agent_type),
            question=req.message,
            citations=deepcopy(req.citations),
        )

    @staticmethod
    def _safe_runtime_text(runtime_result: Any) -> str:
        stop_reason = getattr(runtime_result, "stop_reason", None)
        if stop_reason != "end_turn":
            return _RUNTIME_FAILURE_RESPONSES.get(
                stop_reason,
                RUNTIME_ERROR_RESPONSE,
            )
        text = getattr(runtime_result, "text", "")
        if not isinstance(text, str):
            return RUNTIME_ERROR_RESPONSE
        sanitized = sanitize_text_content(text, 16384)
        return sanitized if sanitized.strip() else RUNTIME_ERROR_RESPONSE

    @staticmethod
    def _runtime_tool_calls(runtime_result: Any) -> List[Dict[str, Any]]:
        traces = getattr(runtime_result, "tool_calls", [])
        if not isinstance(traces, (list, tuple)):
            return []

        safe_traces = []
        for trace in traces[:64]:
            def field_value(name: str, default: Any) -> Any:
                if isinstance(trace, dict):
                    return trace.get(name, default)
                return getattr(trace, name, default)

            name = field_value("name", "")
            if not isinstance(name, str) or not name:
                continue
            params = field_value("params", {})
            safe_params = {}
            if isinstance(params, dict):
                for key, value in islice(params.items(), 32):
                    if not isinstance(key, str) or not key:
                        continue
                    safe_params[key[:64]] = (
                        value
                        if value == "[REDACTED]"
                        else "[REDACTED]"
                    )
            error_type = field_value("error_type", None)
            safe_trace = {
                "name": name[:64],
                "success": bool(field_value("success", False)),
                "params": safe_params,
                "latency_ms": _safe_latency(
                    field_value("latency_ms", 0.0)
                ),
                "cached": bool(field_value("cached", False)),
                "error_type": (
                    error_type[:64]
                    if isinstance(error_type, str)
                    else None
                ),
            }
            fingerprint = field_value("fingerprint", "")
            if (
                isinstance(fingerprint, str)
                and _TOOL_CALL_FINGERPRINT_PATTERN.fullmatch(fingerprint)
            ):
                safe_trace["fingerprint"] = fingerprint
            safe_traces.append(safe_trace)
        return safe_traces

    @staticmethod
    def _runtime_verification(runtime_result: Any) -> Dict[str, Any]:
        report = getattr(runtime_result, "verification", None)
        if report is None:
            return {}
        issues = getattr(report, "issues", ())
        issue_codes = []
        if isinstance(issues, (list, tuple)):
            for issue in issues[:16]:
                code = getattr(issue, "code", None)
                if isinstance(code, str) and code and code not in issue_codes:
                    issue_codes.append(code[:64])
        checked_claims = getattr(report, "checked_claims", 0)
        if isinstance(checked_claims, bool) or not isinstance(checked_claims, int):
            checked_claims = 0
        return {
            "checked": True,
            "passed": getattr(report, "passed", False) is True,
            "checked_claims": max(0, min(checked_claims, 10_000)),
            "issue_codes": issue_codes,
            "abstained": getattr(report, "abstained", False) is True,
            "reflection_count": max(0, min(
                int(getattr(runtime_result, "reflection_count", 0) or 0),
                2,
            )),
            "corrected": getattr(runtime_result, "corrected", False) is True,
            "safe_fallback_used": (
                getattr(runtime_result, "safe_fallback_used", False) is True
            ),
        }

    @staticmethod
    def _is_escalation(req: Request) -> bool:
        return (
            req.urgency == UrgencyLevel.CRITICAL
            or req.intent == IntentCategory.ESCALATION
            or IntentCategory.ESCALATION in req.matched_intents
        )

    def _build_system_prompt(self, req: Request) -> str:
        """把动态加载的 Skills 拼入 system prompt，让业务规则随请求生效。"""
        prompt = self.system_prompt
        if "[Knowledge base references]" in req.context:
            prompt += (
                "\n\n[知识库证据约束]\n"
                "只能依据知识库引用和成功的工具结果陈述事实。"
                "无法从证据确认时，明确说明无法确认。"
                "不得补充证据中没有的时间、概率、政策或根因。"
                "使用知识事实时引用对应的 Citation ID。"
                "回答保持简洁，优先给出证据直接支持的步骤。"
            )
        if self._skill_manager is None:
            return prompt
        skill_prompt = self._skill_manager.prompt_for(req.message, self.agent_type.value)
        if not skill_prompt:
            return prompt
        return f"{prompt}\n\n[动态 Skills]\n{skill_prompt}"

    def _needs_escalation(self, content: str) -> bool:
        """检测 Agent 是否建议升级（简单关键词检测）。"""
        keywords = ["转人工", "人工客服", "escalate", "specialist", "无法处理"]
        return any(kw in content for kw in keywords)


class GeneralAgent(BaseAgent):
    agent_type    = AgentType.GENERAL
    system_prompt = (
        "你是 EchoMind 智能客服。友好、简洁地回答用户问题。"
        "如果问题超出你的能力范围，明确说明并建议转接专业客服。"
    )


class TechnicalAgent(BaseAgent):
    agent_type    = AgentType.TECHNICAL
    system_prompt = (
        "你是技术支持专家。专注于：故障排查、错误诊断、系统配置。"
        "提供清晰的步骤化解决方案。遇到需要后台操作的问题，说明需要升级处理。"
    )


class BillingAgent(BaseAgent):
    agent_type    = AgentType.BILLING
    system_prompt = (
        "你是账单服务专家。专注于：账单查询、退款申请、发票问题、订阅管理。"
        "对财务问题保持准确和专业。涉及实际退款操作时，说明需要人工审核。"
    )


# ── 编排器 ────────────────────────────────────────────────────────────────────

class AccountAgent(BaseAgent):
    agent_type = AgentType.ACCOUNT
    system_prompt = (
        "你是校园账号服务专家。专注于账号锁定、密码重置指引、设备数量超限、"
        "统一身份认证和个人资料问题。不得声称已经修改密码、解锁账号或变更身份信息；"
        "需要后台权限时，应查询知识库、创建工单或建议人工核验。"
    )


class AgentOrchestrator:
    """
    多 Agent 编排器。

    路由逻辑（三层）：
      1. 意图 → Agent 类型映射
      2. 同类多实例时按 routing_score() 选最优
      3. 专属 Agent 失败时降级到 GeneralAgent
    """

    # 意图 → Agent 类型的静态映射（路由表）
    _INTENT_ROUTING: Dict[IntentCategory, AgentType] = {
        IntentCategory.TECHNICAL:  AgentType.TECHNICAL,
        IntentCategory.BILLING:    AgentType.BILLING,
        IntentCategory.ACCOUNT:    AgentType.ACCOUNT,
        IntentCategory.ESCALATION: AgentType.ESCALATION,
        # 其余意图 → GENERAL（默认）
    }

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        skill_manager: Optional[Any] = None,
        intent_recognizer: Optional[IntentRecognizer] = None,
        runtime: Optional[Any] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncAnthropic(**kwargs)

        self._intent_recognizer = intent_recognizer or IntentRecognizer(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        self._skill_manager = skill_manager
        self._result_composer = ResultComposer()

        # Agent 池：每种类型可有多个实例（水平扩展）
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.GENERAL: [
                GeneralAgent(
                    client,
                    model,
                    skill_manager,
                    runtime,
                    AGENT_TOOL_ALLOWLIST["general"],
                )
            ],
            AgentType.TECHNICAL: [
                TechnicalAgent(
                    client,
                    model,
                    skill_manager,
                    runtime,
                    AGENT_TOOL_ALLOWLIST["technical"],
                )
            ],
            AgentType.BILLING: [
                BillingAgent(
                    client,
                    model,
                    skill_manager,
                    runtime,
                    AGENT_TOOL_ALLOWLIST["billing"],
                )
            ],
            AgentType.ACCOUNT: [
                AccountAgent(
                    client,
                    model,
                    skill_manager,
                    runtime,
                    AGENT_TOOL_ALLOWLIST["account"],
                )
            ],
        }

    @property
    def intent_recognizer(self) -> IntentRecognizer:
        """Recognizer shared with evaluators and admin feedback endpoints."""
        return self._intent_recognizer

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        """更新 SkillManager 引用，供运行时重载或测试替换使用。"""
        self._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def run(self, req: Request) -> OrchestratorResult:
        """
        处理一次请求的完整流程：
          意图识别 → 路由选 Agent → 执行 → 检查升级 → 返回结果
        """
        t0 = time.monotonic()

        # 1. 意图识别（如果调用方已识别则跳过）
        if req.intent is None:
            intent_result = await self._intent_recognizer.recognize(req.message, history=req.history)
            req.intent  = intent_result.intent
            req.urgency = intent_result.urgency
            req.intent_scores = intent_result.intent_scores
            req.matched_intents = intent_result.matched_intents

        # 复杂问题自动并行协作，例如同一句同时涉及登录故障和扣款/退款。
        collaboration = self._collaboration_targets(req)
        must_escalate = (
            req.urgency == UrgencyLevel.CRITICAL
            or req.intent == IntentCategory.ESCALATION
            or IntentCategory.ESCALATION in req.matched_intents
        )
        if not must_escalate and len(collaboration) > 1:
            return await self.run_parallel(req, collaboration)

        # 2. 路由：一个专业目标直接路由，多个目标并行；没有目标时按主意图映射。
        if must_escalate:
            agent_type = AgentType.ESCALATION
        elif collaboration:
            agent_type = collaboration[0]
        else:
            agent_type = self._route(req.intent, req.urgency)

        # 3. 执行（含降级）
        response = await self._execute(req, agent_type)

        # 4. 升级检查
        escalated = False
        if (
            response.escalate
            or req.urgency == UrgencyLevel.CRITICAL
            or req.intent == IntentCategory.ESCALATION
            or IntentCategory.ESCALATION in req.matched_intents
        ):
            escalated = True
            logger.warning(f"请求 {req.request_id} 触发升级: urgency={req.urgency}")
            # 生产环境：此处创建工单、通知人工客服

        return OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            intent_scores=req.intent_scores,
            matched_intents=req.matched_intents,
            agent_types=[response.agent_type],
            tool_calls=_copy_tool_calls(response.tool_calls),
            ticket_ids=_unique_ticket_ids(response.ticket_ids),
            evidence_verification=dict(response.evidence_verification),
        )

    async def run_parallel(self, req: Request, agent_types: List[AgentType]) -> OrchestratorResult:
        """
        并行派发给多个 Agent，合并结果。
        适用于复杂问题（如同时涉及技术和账单）。
        """
        if not agent_types:
            raise ValueError("agent_types must not be empty")
        t0 = time.monotonic()
        tasks = [self._execute(req, at) for at in agent_types]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并：拼接所有成功响应
        parts = [
            AgentPart(r.agent_type.value, r.content, r.success)
            for r in responses
            if isinstance(r, AgentResponse)
        ]

        composer = getattr(self, "_result_composer", None) or ResultComposer()
        combined = composer.compose(parts)
        escalated = any(isinstance(r, AgentResponse) and r.escalate for r in responses)
        successful_types = list(dict.fromkeys(
            r.agent_type
            for r in responses
            if isinstance(r, AgentResponse) and r.success
        ))

        return OrchestratorResult(
            request_id=req.request_id,
            response=combined,
            agent_type=successful_types[0] if successful_types else agent_types[0],
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            intent_scores=req.intent_scores,
            matched_intents=req.matched_intents,
            agent_types=successful_types or agent_types,
            tool_calls=_copy_tool_calls(
                call
                for response in responses
                if isinstance(response, AgentResponse)
                for call in response.tool_calls
            ),
            ticket_ids=_unique_ticket_ids(
                ticket_id
                for response in responses
                if isinstance(response, AgentResponse)
                for ticket_id in response.ticket_ids
            ),
            evidence_verification=_merge_evidence_verification(
                response.evidence_verification
                for response in responses
                if isinstance(response, AgentResponse)
            ),
        )

    # ── 路由逻辑 ──────────────────────────────────────────────────────────────

    def _route(self, intent: Optional[IntentCategory], urgency: Optional[UrgencyLevel]) -> AgentType:
        """
        三层路由决策：
          1. 意图映射
          2. 紧急度覆盖（CRITICAL 直接升级）
          3. 默认 GENERAL
        """
        if urgency == UrgencyLevel.CRITICAL:
            return AgentType.ESCALATION

        if intent and intent in self._INTENT_ROUTING:
            target = self._INTENT_ROUTING[intent]
            # 如果目标类型有可用实例则使用，否则降级
            if target in self._pool and self._pool[target]:
                return target

        return AgentType.GENERAL

    def _collaboration_targets(self, req: Request) -> List[AgentType]:
        """
        把多标签意图结果映射为一个或两个专业 Agent。

        matched_intents 来自 LLM、Embedding、Pattern 的逐标签加权融合，
        不再在编排层额外扫描领域关键词。
        """
        targets: List[AgentType] = []

        intents = list(req.matched_intents)
        if not intents and req.intent is not None:
            intents.append(req.intent)

        for intent in intents:
            if intent == IntentCategory.TECHNICAL:
                targets.append(AgentType.TECHNICAL)
            elif intent == IntentCategory.BILLING:
                targets.append(AgentType.BILLING)
            elif intent == IntentCategory.ACCOUNT:
                targets.append(AgentType.ACCOUNT)

        # 保持顺序去重，并只返回当前有实例的 Agent 类型。
        deduped = list(dict.fromkeys(targets))
        return [
            agent_type
            for agent_type in deduped[:3]
            if self._pool.get(agent_type)
        ]

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """
        性能路由：从同类 Agent 中选 routing_score() 最高的。
        这是"基于在线表现动态调整路由"的核心。
        """
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None
        return max(agents, key=lambda a: a.stats.routing_score())

    async def _execute(self, req: Request, agent_type: AgentType) -> AgentResponse:
        """执行 Agent，失败时降级到 GeneralAgent。"""
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.GENERAL)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.GENERAL,
                content="服务暂时不可用，请稍后重试。",
                success=False,
            )

        response = await agent.handle(req)

        # 专属 Agent 失败时降级到 GeneralAgent
        if not response.success and agent_type != AgentType.GENERAL:
            logger.warning(f"{agent_type.value} 失败，降级到 GeneralAgent")
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback:
                fallback_response = await fallback.handle(req)
                response = AgentResponse(
                    agent_type=fallback_response.agent_type,
                    content=fallback_response.content,
                    success=fallback_response.success,
                    confidence=fallback_response.confidence,
                    latency_ms=fallback_response.latency_ms,
                    escalate=(
                        response.escalate or fallback_response.escalate
                    ),
                    tool_calls=_copy_tool_calls(
                        [*response.tool_calls, *fallback_response.tool_calls]
                    ),
                    ticket_ids=_unique_ticket_ids(
                        [*response.ticket_ids, *fallback_response.ticket_ids]
                    ),
                    evidence_verification=_merge_evidence_verification([
                        response.evidence_verification,
                        fallback_response.evidence_verification,
                    ]),
                )

        return response

    # ── 统计（供 Monitor 读取）────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                result[key] = {
                    "total":        agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms":       round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        """
        接收 Monitor 的在线表现反馈，动态调整路由惩罚项。

        penalties 的 key 使用 get_stats() 中的 agent key，例如 technical_0。
        """
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, 0.0)
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)


def _merge_evidence_verification(
    reports: Any,
) -> Dict[str, Any]:
    checked_reports = [
        report
        for report in reports
        if isinstance(report, dict) and report.get("checked") is True
    ]
    if not checked_reports:
        return {}
    issue_codes = list(dict.fromkeys(
        code
        for report in checked_reports
        for code in report.get("issue_codes", [])
        if isinstance(code, str) and code
    ))[:16]
    return {
        "checked": True,
        "passed": all(report.get("passed") is True for report in checked_reports),
        "checked_claims": min(10_000, sum(
            value
            for report in checked_reports
            for value in [report.get("checked_claims", 0)]
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        )),
        "issue_codes": issue_codes,
        "abstained": all(
            report.get("abstained") is True for report in checked_reports
        ),
        "reflection_count": min(6, sum(
            value
            for report in checked_reports
            for value in [report.get("reflection_count", 0)]
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        )),
        "corrected": any(
            report.get("corrected") is True for report in checked_reports
        ),
        "safe_fallback_used": any(
            report.get("safe_fallback_used") is True
            for report in checked_reports
        ),
    }


def _copy_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    """Copy bounded normalized tool traces without mutating source responses."""
    try:
        items = islice(iter(tool_calls), 128)
    except (TypeError, ValueError):
        return []

    copied = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        normalized: Dict[str, Any] = {"name": name[:64]}
        if "success" in item:
            normalized["success"] = bool(item["success"])
        if "params" in item:
            params = item["params"]
            normalized["params"] = {
                key[:64]: (
                    value
                    if value == "[REDACTED]"
                    else "[REDACTED]"
                )
                for key, value in islice(
                    params.items() if isinstance(params, dict) else (),
                    32,
                )
                if isinstance(key, str) and key
            }
        if "latency_ms" in item:
            normalized["latency_ms"] = _safe_latency(item["latency_ms"])
        if "cached" in item:
            normalized["cached"] = bool(item["cached"])
        if "error_type" in item:
            error_type = item["error_type"]
            normalized["error_type"] = (
                error_type[:64] if isinstance(error_type, str) else None
            )
        fingerprint = item.get("fingerprint")
        if (
            isinstance(fingerprint, str)
            and _TOOL_CALL_FINGERPRINT_PATTERN.fullmatch(fingerprint)
        ):
            normalized["fingerprint"] = fingerprint
        copied.append(normalized)
    return copied


def _unique_ticket_ids(ticket_ids: Any) -> List[str]:
    """Return bounded ticket IDs in first-seen order."""
    if isinstance(ticket_ids, (str, bytes, bytearray, Mapping)):
        return []
    if not isinstance(ticket_ids, (list, tuple, Iterator)):
        return []
    try:
        items = islice(iter(ticket_ids), 128)
    except (TypeError, ValueError):
        return []

    unique = []
    seen = set()
    for ticket_id in items:
        if (
            not isinstance(ticket_id, str)
            or not ticket_id
            or len(ticket_id) > 128
            or not ticket_id.isprintable()
            or any(character.isspace() for character in ticket_id)
            or ticket_id in seen
        ):
            continue
        seen.add(ticket_id)
        unique.append(ticket_id)
    return unique


def _runtime_context(
    req: Request,
    agent_type: AgentType,
) -> Dict[str, str]:
    conv_id = _trusted_context_component(req.conv_id, "conv_id")
    request_id = _trusted_context_component(
        req.request_id,
        "request_id",
    )
    context = {
        "idempotency_scope": (
            "scope_"
            + hashlib.sha256(
                json.dumps(
                    [conv_id, request_id, agent_type.value],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
    }
    if req.principal_id is not None:
        context["user_id"] = _trusted_context_component(
            req.principal_id,
            "principal_id",
        )
    return context


def _trusted_context_component(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not value.isprintable()
    ):
        raise ValueError(f"invalid trusted {field_name}")
    return value


def _safe_latency(value: Any) -> float:
    try:
        latency = float(value)
    except (TypeError, ValueError):
        return 0.0
    if latency < 0 or latency != latency or latency == float("inf"):
        return 0.0
    return min(latency, 86_400_000.0)
