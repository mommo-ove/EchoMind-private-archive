"""
亮点：MCP 工具调用框架

核心问题：工具调用出错（检索不全、召回不好）怎么优化？

本模块的答案：
  1. 查询改写（Query Rewriting）—— 用 LLM 把用户原始问题扩写成多个角度的子查询，
     再合并去重，解决"召回不全"问题。
  2. 结果重排（Reranking）—— 用专用 Cross-Encoder 对召回结果打分并重新排序，
     解决"召回不好/排序差"问题。
  3. 熔断器（Circuit Breaker）—— 连续失败超阈值时自动断开，防止雪崩。
  4. 结果缓存（TTL Cache）—— 相同参数直接返回缓存，减少重复调用。
  5. 降级策略（Fallback）—— 工具不可用时返回有意义的降级结果。
"""
import asyncio
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)

AGENT_TOOL_ALLOWLIST = {
    "general": ["get_ticket"],
    "technical": [
        "knowledge_search",
        "query_network_status",
        "create_ticket",
        "get_ticket",
    ],
    "billing": [
        "knowledge_search",
        "query_campus_card",
        "create_ticket",
        "get_ticket",
    ],
    "account": [
        "knowledge_search",
        "create_ticket",
        "get_ticket",
    ],
}


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED    = "closed"     # 正常
    OPEN      = "open"       # 熔断，拒绝请求
    HALF_OPEN = "half_open"  # 探测恢复


@dataclass(frozen=True)
class CircuitPermit:
    generation: int
    probe: bool = False


@dataclass
class ToolResult:
    success:        bool
    data:           Any
    tool_name:      str
    error:          Optional[str] = None
    cached:         bool = False
    latency_ms:     float = 0.0
    reranked:       bool = False   # 是否经过重排
    fallback_used:  bool = False
    rejected:       bool = False


@dataclass
class ToolStats:
    """
    工具运行时统计，供 Monitor 读取。

    legacy success/failed 表示后端执行结果；result_* 表示调用方最终看到的结果。
    """
    total:              int = 0
    success:            int = 0
    failed:             int = 0
    total_latency_ms:   float = 0.0
    consecutive_fails:  int = 0
    executed:           int = 0
    cache_hits:         int = 0
    rejected:           int = 0
    result_success:     int = 0
    result_failed:      int = 0
    fallback_success:   int = 0
    fallback_failed:    int = 0

    @property
    def success_rate(self) -> float:
        return self.success / self.executed if self.executed else 1.0

    @property
    def result_success_rate(self) -> float:
        return self.result_success / self.total if self.total else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.executed if self.executed else 0.0


# ── 熔断器 ────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    三态熔断器：CLOSED → OPEN → HALF_OPEN → CLOSED

    连续失败 failure_threshold 次后打开；
    打开 recovery_s 秒后进入 HALF_OPEN 探测；
    探测成功则关闭，失败则重新打开。
    """

    def __init__(self, failure_threshold: int = 5, recovery_s: float = 60.0):
        self.threshold   = failure_threshold
        self.recovery_s  = recovery_s
        self.state       = CircuitState.CLOSED
        self.fail_count  = 0
        self.opened_at:  Optional[float] = None
        self._generation = 0
        self._lock = Lock()
        self._legacy_permits: List[CircuitPermit] = []

    def allow(self) -> bool:
        """兼容旧调用方：原子保留许可，并由无 token 的 record_*() 结算。"""
        with self._lock:
            permit = self._acquire_locked()
            if permit is None:
                return False
            self._legacy_permits.append(permit)
            return True

    def can_acquire(self) -> bool:
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                return (
                    self.opened_at is not None
                    and time.monotonic() - self.opened_at >= self.recovery_s
                )
            return False

    def current_state(self) -> CircuitState:
        with self._lock:
            return self.state

    def acquire(self) -> Optional[CircuitPermit]:
        """原子获取执行许可；冷却完成后只保留一个 HALF_OPEN 探针。"""
        with self._lock:
            return self._acquire_locked()

    def _acquire_locked(self) -> Optional[CircuitPermit]:
        if self.state == CircuitState.CLOSED:
            return CircuitPermit(self._generation)
        if (
            self.state == CircuitState.OPEN
            and self.opened_at is not None
            and time.monotonic() - self.opened_at >= self.recovery_s
        ):
            self.state = CircuitState.HALF_OPEN
            return CircuitPermit(self._generation, probe=True)
        return None

    def record_success(self, permit: Optional[CircuitPermit] = None) -> bool:
        with self._lock:
            if permit is None:
                if not self._legacy_permits:
                    return False
                permit = self._legacy_permits.pop()
            return self._record_success_locked(permit)

    def _record_success_locked(self, permit: CircuitPermit) -> bool:
        if permit.generation != self._generation:
            return False
        if permit.probe:
            if self.state != CircuitState.HALF_OPEN:
                return False
            self.fail_count = 0
            self.state = CircuitState.CLOSED
            self.opened_at = None
            self._generation += 1
            return True
        if self.state != CircuitState.CLOSED:
            return False
        self.fail_count = 0
        return True

    def record_failure(self, permit: Optional[CircuitPermit] = None) -> bool:
        with self._lock:
            if permit is None:
                if not self._legacy_permits:
                    return False
                permit = self._legacy_permits.pop()
            return self._record_failure_locked(permit)

    def _record_failure_locked(self, permit: CircuitPermit) -> bool:
        if permit.generation != self._generation:
            return False
        is_probe = permit.probe
        expected_state = CircuitState.HALF_OPEN if is_probe else CircuitState.CLOSED
        if self.state != expected_state:
            return False

        self.fail_count += 1
        if is_probe or self.fail_count >= self.threshold:
            self.state = CircuitState.OPEN
            self.opened_at = time.monotonic()
            self._generation += 1
            logger.warning(f"熔断器打开（连续失败 {self.fail_count} 次）")
        return True


# ── 工具定义 ──────────────────────────────────────────────────────────────────

@dataclass
class Tool:
    name:        str
    description: str
    handler:     Callable                    # async (params, context) -> Any
    schema:      Dict[str, Any]              # JSON Schema
    cache_ttl:   float = 0.0                 # 0 = 不缓存
    timeout_s:   float = 30.0
    supports_rerank: bool = False            # 是否支持结果重排
    fallback:    Optional[Callable] = None    # sync/async (params, context, error) -> Any
    context_validator: Optional[Callable] = None  # sync/async (params, context) -> None

    # 运行时状态（不参与构造）
    stats:   ToolStats    = field(default_factory=ToolStats, init=False)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker, init=False)


# ── MCP 工具管理器 ────────────────────────────────────────────────────────────

class MCPToolManager:
    """
    MCP 工具调用框架。

    核心优化链路（针对检索类工具）：
      用户查询 → 查询改写（多角度子查询）→ 并行召回 → 结果重排 → 返回 Top-K
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        reranker: Optional[Any] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model  = model
        self._reranker = reranker
        self._structured_llm_options: Dict[str, Any] = {}
        if base_url and "api.deepseek.com" in base_url.lower():
            # DeepSeek thinking output is useful for agents, but structured
            # JSON helpers need a plain response that can be parsed reliably.
            self._structured_llm_options = {
                "extra_body": {"thinking": {"type": "disabled"}}
            }
        self._tools: Dict[str, Tool] = {}
        self._cache: Dict[str, tuple] = {}   # key → (result, expire_at, reranked)

    # ── 注册 / 注销 ───────────────────────────────────────────────────────────

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def schemas(self, allowed: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        names = allowed if allowed is not None else list(self._tools)
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.schema,
            }
            for name in names
            if (tool := self._tools.get(name)) is not None
        ]

    def availability(self, name: str) -> tuple[bool, Optional[str]]:
        tool = self._tools.get(name)
        if tool is None:
            return False, f"工具不存在: {name}"
        if not tool.breaker.can_acquire():
            return False, f"工具熔断中: {name}，请稍后重试"
        return True, None

    # ── 核心调用 ──────────────────────────────────────────────────────────────

    async def call(
        self,
        name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
        rerank_top_k: int = 0,
    ) -> ToolResult:
        """Run one tool call and account for caller cancellation exactly once."""
        tool = self._tools.get(name)
        try:
            return await self._call_impl(
                tool,
                name,
                params,
                context,
                use_cache=use_cache,
                rerank_top_k=rerank_top_k,
            )
        except asyncio.CancelledError:
            if tool is not None:
                tool.stats.result_failed += 1
            raise

    async def _call_impl(
        self,
        tool: Optional[Tool],
        name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
        rerank_top_k: int = 0,          # >0 时对结果重排，取 Top-K
    ) -> ToolResult:
        """
        调用工具，完整执行链：
          参数校验 → 熔断许可 → CLOSED 状态缓存检查 → 执行（含超时）→ 可选重排

        缓存不会绕过已打开或 HALF_OPEN 的熔断器；冷却后的首个许可必须真实探测后端。
        """
        if not tool:
            return ToolResult(
                success=False,
                data=None,
                tool_name=name,
                error=f"工具不存在: {name}",
                rejected=True,
            )

        tool.stats.total += 1
        cache_rerank_top_k = rerank_top_k if rerank_top_k > 0 and tool.supports_rerank else 0

        try:
            self._validate_params(tool, params)
        except ValueError as ex:
            tool.stats.rejected += 1
            logger.warning(f"工具参数被拒绝: {name} — {ex}")
            return await self._fallback_result(tool, params, context, str(ex), rejected=True)

        if tool.context_validator is not None:
            try:
                validation = tool.context_validator(params, context)
                if asyncio.iscoroutine(validation):
                    await validation
            except Exception:
                tool.stats.rejected += 1
                error = f"工具 {name} 缺少或包含无效的可信上下文"
                logger.warning(error)
                return self._finish_result(
                    tool,
                    ToolResult(
                        success=False,
                        data=None,
                        tool_name=name,
                        error=error,
                        rejected=True,
                    ),
                )

        permit = tool.breaker.acquire()
        if permit is None:
            tool.stats.rejected += 1
            error = f"工具熔断中: {name}，请稍后重试"
            return await self._fallback_result(tool, params, context, error, rejected=True)

        # 缓存只在 CLOSED 常规许可下使用；HALF_OPEN 探针必须执行真实后端。
        if not permit.probe and use_cache and tool.cache_ttl > 0:
            cached = self._get_cache(name, params, cache_rerank_top_k)
            if cached is not None:
                cached_data, cached_reranked = cached
                tool.stats.cache_hits += 1
                return self._finish_result(
                    tool,
                    ToolResult(
                        success=True,
                        data=cached_data,
                        tool_name=name,
                        cached=True,
                        reranked=cached_reranked,
                    ),
                )

        t0 = time.monotonic()
        tool.stats.executed += 1
        try:
            data = await asyncio.wait_for(tool.handler(params, context), timeout=tool.timeout_s)
        except asyncio.CancelledError:
            latency = (time.monotonic() - t0) * 1000
            tool.stats.failed += 1
            tool.stats.total_latency_ms += latency
            if tool.breaker.record_failure(permit):
                tool.stats.consecutive_fails = tool.breaker.fail_count
            logger.info(f"宸ュ叿璋冪敤琚彇娑? {name}")
            raise
        except asyncio.TimeoutError:
            latency = (time.monotonic() - t0) * 1000
            tool.stats.failed += 1
            tool.stats.total_latency_ms += latency
            if tool.breaker.record_failure(permit):
                tool.stats.consecutive_fails = tool.breaker.fail_count
            logger.error(f"工具超时: {name} ({tool.timeout_s}s)")
            return await self._fallback_result(
                tool,
                params,
                context,
                "执行超时",
                latency_ms=latency,
            )

        except Exception as ex:
            latency = (time.monotonic() - t0) * 1000
            tool.stats.failed += 1
            tool.stats.total_latency_ms += latency
            if tool.breaker.record_failure(permit):
                tool.stats.consecutive_fails = tool.breaker.fail_count
            logger.error(f"工具异常: {name} — {ex}")
            return await self._fallback_result(
                tool,
                params,
                context,
                str(ex),
                latency_ms=latency,
            )

        latency = (time.monotonic() - t0) * 1000
        tool.stats.success += 1
        tool.stats.total_latency_ms += latency
        if tool.breaker.record_success(permit):
            tool.stats.consecutive_fails = 0

        # 重排（针对返回列表的检索工具）
        reranked = False
        if rerank_top_k > 0 and tool.supports_rerank and isinstance(data, list):
            query = params.get("query", "")
            data, reranked = await self._rerank(query, data, rerank_top_k), True

        # 写缓存：缓存最终返回结果，避免下次命中未重排的原始结果。
        if tool.cache_ttl > 0:
            self._set_cache(name, params, data, tool.cache_ttl, cache_rerank_top_k, reranked)

        return self._finish_result(
            tool,
            ToolResult(
                success=True,
                data=data,
                tool_name=name,
                latency_ms=latency,
                reranked=reranked,
            ),
        )

    async def _fallback_result(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        error: str,
        *,
        rejected: bool = False,
        latency_ms: float = 0.0,
    ) -> ToolResult:
        """工具不可用时返回降级结果，而不是把空错误直接暴露给调用方。"""
        if tool.fallback is None:
            return self._finish_result(
                tool,
                ToolResult(
                    success=False,
                    data=None,
                    tool_name=tool.name,
                    error=error,
                    latency_ms=latency_ms,
                    rejected=rejected,
                ),
            )
        try:
            data = tool.fallback(params, context, error)
            if asyncio.iscoroutine(data):
                data = await data
            tool.stats.fallback_success += 1
            return self._finish_result(
                tool,
                ToolResult(
                    success=True,
                    data=data,
                    tool_name=tool.name,
                    error=error,
                    latency_ms=latency_ms,
                    fallback_used=True,
                    rejected=rejected,
                ),
            )
        except Exception as ex:
            tool.stats.fallback_failed += 1
            logger.error(f"工具降级失败: {tool.name} — {ex}")
            return self._finish_result(
                tool,
                ToolResult(
                    success=False,
                    data=None,
                    tool_name=tool.name,
                    error=f"{error}; fallback失败: {ex}",
                    latency_ms=latency_ms,
                    fallback_used=True,
                    rejected=rejected,
                ),
            )

    @staticmethod
    def _finish_result(tool: Tool, result: ToolResult) -> ToolResult:
        if result.success:
            tool.stats.result_success += 1
        else:
            tool.stats.result_failed += 1
        return result

    # ── 查询改写（解决召回不全）────────────────────────────────────────────────

    async def rewrite_query(self, query: str, n: int = 3) -> List[str]:
        """
        用 LLM 将原始查询改写为 n 个不同角度的子查询。

        目的：单一查询往往只能召回某一角度的文档，
        多角度子查询并行检索后合并，显著提升召回率。

        示例：
          原始: "退款流程"
          改写: ["如何申请退款", "退款需要多少天", "退款政策是什么"]
        """
        prompt = f"""将以下用户查询改写为 {n} 个不同角度的搜索子查询，用于检索知识库。
要求：每个子查询角度不同，覆盖原始问题的不同方面。
原始查询: "{query}"
返回 JSON 数组，例如: ["子查询1", "子查询2", "子查询3"]"""
        prompt = self._clean_text(prompt)
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=256, temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
                **self._structured_llm_options,
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("["), raw.rfind("]") + 1
            queries = json.loads(raw[s:e])
            # 原始查询也保留，去重
            return list(dict.fromkeys([query] + queries))
        except Exception as ex:
            logger.warning(f"查询改写失败，使用原始查询: {ex}")
            return [query]

    async def search_with_rewrite(
        self,
        tool_name: str,
        query: str,
        top_k: int = 5,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        完整的检索优化链路：查询改写 → 并行召回 → 去重 → 重排 → Top-K

        这是解决"检索不全、召回不好"的完整方案。
        非 CLOSED 状态只执行一次原始查询调用：冷却前快速降级，冷却后作为单探针。
        """
        tool = self._tools.get(tool_name)
        if tool is None or tool.breaker.current_state() != CircuitState.CLOSED:
            return await self.call(
                tool_name,
                {"query": query, "top_k": top_k},
                context,
                use_cache=False,
            )

        # 1. 查询改写：生成多角度子查询
        sub_queries = await self.rewrite_query(query, n=3)
        logger.info(f"查询改写: {query!r} → {sub_queries}")

        # 2. 并行召回：所有子查询同时检索
        recall_k = max(top_k, 5)
        tasks = [
            self.call(tool_name, {"query": q, "top_k": recall_k}, context, use_cache=True)
            for q in sub_queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. 合并去重（按内容哈希去重）
        seen, merged = set(), []
        for r in results:
            if isinstance(r, ToolResult) and r.success and isinstance(r.data, list):
                for item in r.data:
                    if isinstance(item, dict) and item.get("id") not in (None, ""):
                        # Scores vary by rewritten query; the stable document
                        # id identifies the evidence chunk across all recalls.
                        key = f"id:{item['id']}"
                    else:
                        identity = (
                            {k: v for k, v in item.items() if k != "score"}
                            if isinstance(item, dict)
                            else item
                        )
                        key = hashlib.md5(
                            json.dumps(
                                identity,
                                sort_keys=True,
                                ensure_ascii=False,
                                default=str,
                            ).encode("utf-8")
                        ).hexdigest()
                    if key not in seen:
                        seen.add(key)
                        merged.append(item)

        if not merged:
            return ToolResult(success=False, data=[], tool_name=tool_name, error="所有子查询均无结果")

        # 4. 重排：用专用 Cross-Encoder 对合并结果按相关性打分，取 Top-K
        reranked = await self._rerank(query, merged, top_k)
        return ToolResult(success=True, data=reranked, tool_name=tool_name, reranked=True)

    # ── 结果重排（解决召回不好）──────────────────────────────────────────────

    async def _rerank(self, query: str, items: List[Any], top_k: int) -> List[Any]:
        """Use a dedicated cross-encoder to rerank recalled candidates."""
        if len(items) <= top_k:
            return items
        if self._reranker is None:
            return items[:top_k]

        try:
            documents = [self._rerank_text(item) for item in items]
            scores = list(await self._reranker.score(query, documents))
            if len(scores) != len(items):
                raise ValueError("reranker score count does not match candidate count")
            numeric_scores = [float(score) for score in scores]
            if not all(math.isfinite(score) for score in numeric_scores):
                raise ValueError("reranker returned a non-finite score")
            order = sorted(
                range(len(items)),
                key=lambda index: numeric_scores[index],
                reverse=True,
            )
            reranked = [items[index] for index in order]
            return reranked[:top_k]
        except Exception as ex:
            logger.warning("专用重排器失败，返回原始召回顺序: %s", ex)
            return items[:top_k]

    @staticmethod
    def _rerank_text(item: Any) -> str:
        if isinstance(item, dict):
            parts = [
                value.strip()
                for key in ("title", "content", "document", "text")
                if isinstance((value := item.get(key)), str) and value.strip()
            ]
            if parts:
                return "\n".join(parts)
        return str(item)

    # ── 缓存 ──────────────────────────────────────────────────────────────────

    def _cache_key(self, name: str, params: Dict, rerank_top_k: int = 0) -> str:
        payload = {"params": params, "rerank_top_k": rerank_top_k}
        return f"{name}:{hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()}"

    def _get_cache(self, name: str, params: Dict, rerank_top_k: int = 0) -> Optional[Tuple[Any, bool]]:
        key = self._cache_key(name, params, rerank_top_k)
        if key in self._cache:
            data, expire_at, reranked = self._cache[key]
            if time.monotonic() < expire_at:
                return data, reranked
            del self._cache[key]
        return None

    def _set_cache(
        self,
        name: str,
        params: Dict,
        data: Any,
        ttl: float,
        rerank_top_k: int = 0,
        reranked: bool = False,
    ) -> None:
        if len(self._cache) >= 5000:
            # 清掉最旧的 1/4
            for k in list(self._cache)[:1250]:
                del self._cache[k]
        self._cache[self._cache_key(name, params, rerank_top_k)] = (data, time.monotonic() + ttl, reranked)

    # ── 参数校验 ──────────────────────────────────────────────────────────────

    _TYPE_MAP = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}
    _SENSITIVE_FIELD_MARKERS = (
        "apikey",
        "api_key",
        "authorization",
        "credential",
        "password",
        "passwd",
        "secret",
        "token",
    )

    @classmethod
    def _is_sensitive_field_name(cls, key: Any) -> bool:
        normalized = str(key).lower().replace("-", "_")
        compact = normalized.replace("_", "")
        return any(
            marker in normalized or marker.replace("_", "") in compact
            for marker in cls._SENSITIVE_FIELD_MARKERS
        )

    def _validate_params(self, tool: Tool, params: Dict[str, Any]) -> None:
        """根据工具的 JSON Schema 校验参数，不合法时抛出 ValueError。"""
        schema = tool.schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        if not isinstance(params, dict):
            raise ValueError(f"工具 {tool.name} 参数必须是对象")

        for field in required:
            if field not in params:
                raise ValueError(f"工具 {tool.name} 缺少必需参数: {field}")

        if schema.get("additionalProperties") is False:
            unexpected = [key for key in params if key not in properties]
            if unexpected:
                if any(self._is_sensitive_field_name(key) for key in unexpected):
                    raise ValueError(
                        f"工具 {tool.name} 包含禁止的凭据字段"
                    )
                raise ValueError(f"工具 {tool.name} 包含未声明参数: {unexpected[0]}")

        for key, value in params.items():
            if key in properties:
                field_schema = properties[key]
                expected_type = field_schema.get("type")
                if expected_type and expected_type in self._TYPE_MAP:
                    matches_type = isinstance(value, self._TYPE_MAP[expected_type])
                    if expected_type in {"integer", "number"} and isinstance(value, bool):
                        matches_type = False
                    if not matches_type:
                        raise ValueError(
                            f"工具 {tool.name} 参数 {key} 类型错误: 期望 {expected_type}，实际 {type(value).__name__}"
                        )
                if "enum" in field_schema and value not in field_schema["enum"]:
                    raise ValueError(f"工具 {tool.name} 参数 {key} 不在允许值中")
                if (
                    "minimum" in field_schema
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and value < field_schema["minimum"]
                ):
                    raise ValueError(f"工具 {tool.name} 参数 {key} 小于最小值")
                if (
                    "maximum" in field_schema
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and value > field_schema["maximum"]
                ):
                    raise ValueError(f"工具 {tool.name} 参数 {key} 大于最大值")
                if (
                    "minLength" in field_schema
                    and isinstance(value, str)
                    and len(value) < field_schema["minLength"]
                ):
                    raise ValueError(f"工具 {tool.name} 参数 {key} 短于最小长度")
                if (
                    "maxLength" in field_schema
                    and isinstance(value, str)
                    and len(value) > field_schema["maxLength"]
                ):
                    raise ValueError(f"工具 {tool.name} 参数 {key} 超过最大长度")

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 LLM 请求编码失败。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    # ── 统计 ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """legacy success_rate/avg_latency_ms 均表示真实后端执行，不含缓存、拒绝和 fallback。"""
        return {
            name: {
                "total": t.stats.total,
                "executed": t.stats.executed,
                "cache_hits": t.stats.cache_hits,
                "rejected": t.stats.rejected,
                "success": t.stats.success,
                "failed": t.stats.failed,
                "execution_success": t.stats.success,
                "execution_failed": t.stats.failed,
                "result_success": t.stats.result_success,
                "result_failed": t.stats.result_failed,
                "fallback_success": t.stats.fallback_success,
                "fallback_failed": t.stats.fallback_failed,
                "success_rate": round(t.stats.success_rate, 3),
                "result_success_rate": round(t.stats.result_success_rate, 3),
                "avg_latency_ms": round(t.stats.avg_latency_ms, 1),
                "execution_avg_latency_ms": round(t.stats.avg_latency_ms, 1),
                "consecutive_fails": t.stats.consecutive_fails,
                "circuit_state": t.breaker.current_state().value,
            }
            for name, t in self._tools.items()
        }
