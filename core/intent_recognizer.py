"""
亮点：端到端意图识别

三路融合策略：
  1. LLM 语义理解（权重 70%）—— 主力，理解复杂语义和上下文
  2. Embedding 向量相似度（权重 20%）—— 快速匹配常见表达
  3. 关键词模式匹配（权重 10%）—— 零延迟兜底

三路结果通过加权投票合并，置信度低于阈值时降级为 OTHER。
LLM 和 Embedding 并行调用，不串行等待。
"""
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    QUERY      = "query"       # 查询信息
    COMPLAINT  = "complaint"   # 投诉不满
    REQUEST    = "request"     # 请求操作
    GREETING   = "greeting"    # 问候
    ESCALATION = "escalation"  # 要求升级/转人工
    TECHNICAL  = "technical"   # 技术问题
    BILLING    = "billing"     # 账单/退款
    ACCOUNT    = "account"     # 账户管理
    FEEDBACK   = "feedback"    # 正面反馈
    OTHER      = "other"


class UrgencyLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    urgency:    UrgencyLevel
    entities:   Dict[str, List[str]]   # 从消息中提取的实体
    reasoning:  str
    latency_ms: float
    intent_scores: Dict[IntentCategory, float] = field(default_factory=dict)
    matched_intents: List[IntentCategory] = field(default_factory=list)


# ── Few-shot 模板（同时用于 LLM 示例和 Embedding 匹配）────────────────────────
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.QUERY:      ["我的订单状态是什么？", "如何重置密码？", "快递什么时候到？"],
    IntentCategory.COMPLAINT:  ["等了好几个小时！", "服务太差了！", "一直没人处理！"],
    IntentCategory.REQUEST:    ["帮我取消订单", "我需要修改地址", "请协助退款"],
    IntentCategory.GREETING:   ["你好", "嗨，有人吗", "早上好"],
    IntentCategory.ESCALATION: ["我要投诉！", "转人工客服", "找你们经理"],
    IntentCategory.TECHNICAL:  ["应用一直崩溃", "无法登录", "出现500错误"],
    IntentCategory.BILLING:    ["为什么扣了两次款？", "申请退款", "发票问题"],
    IntentCategory.ACCOUNT:    ["修改邮箱", "注销账户", "更新个人信息"],
    IntentCategory.FEEDBACK:   ["服务很棒！", "非常满意", "给个好评"],
}

# 紧急关键词
_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["紧急", "emergency", "urgent", "asap", "立刻"],
    UrgencyLevel.HIGH:     ["今天", "马上", "尽快", "hurry", "now"],
    UrgencyLevel.MEDIUM:   ["这周", "soon", "快点"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """
    端到端意图识别器。

    初始化时不加载任何本地模型，所有 AI 能力通过 Anthropic API 调用。
    模板 Embedding 在首次请求时懒加载并缓存，后续复用。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: float = 0.5,
        multi_label_threshold: float = 0.6,
        feedback_store: Optional[Any] = None,
        embedding_enabled: Optional[bool] = None,
        strategy_weights: Optional[Dict[str, float]] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client    = AsyncAnthropic(**kwargs)
        self.model     = model
        self._structured_llm_options: Dict[str, Any] = {}
        if base_url and "api.deepseek.com" in base_url.lower():
            self._structured_llm_options = {
                "extra_body": {"thinking": {"type": "disabled"}}
            }
        self.threshold = confidence_threshold
        self.multi_label_threshold = multi_label_threshold
        # 第三方兼容 API（如 DeepSeek）通常不支持 Embedding，禁用该策略。
        # 官方 Anthropic SDK 当前没有 embeddings 资源，因此下面会使用稳定的
        # 本地字符 n-gram 向量作为轻量兜底，保证三路融合链路真实可跑。
        self._embedding_enabled = (
            not bool(base_url)
            if embedding_enabled is None
            else bool(embedding_enabled)
        )
        defaults = (
            {"llm": 0.7, "embedding": 0.2, "pattern": 0.1}
            if self._embedding_enabled
            else {"llm": 0.85, "embedding": 0.0, "pattern": 0.15}
        )
        self._strategy_weights = dict(strategy_weights or defaults)
        if set(self._strategy_weights) != {"llm", "embedding", "pattern"}:
            raise ValueError("strategy_weights must contain llm, embedding, and pattern")
        if (
            any(weight < 0 for weight in self._strategy_weights.values())
            or abs(sum(self._strategy_weights.values()) - 1.0) > 1e-9
        ):
            raise ValueError("strategy_weights must be non-negative and sum to 1")

        self._templates: Dict[IntentCategory, List[str]] = {
            category: list(templates)
            for category, templates in _TEMPLATES.items()
        }
        self._feedback_store = feedback_store
        self._tpl_embeddings: Dict[IntentCategory, List[List[float]]] = {}
        self._cache: Dict[str, IntentResult] = {}
        self.cache_hits   = 0
        self.cache_misses = 0
        self._load_persisted_feedback()

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """
        识别用户意图。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        """
        key = self._cache_key(message, history)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        # LLM 和 Embedding 并行（Embedding 不可用时跳过）
        llm_task = asyncio.create_task(self._llm_recognize(message, history))
        emb_task = asyncio.create_task(self._embedding_recognize(message)) if self._embedding_enabled else None
        pat      = self._pattern_recognize(message)

        if emb_task:
            llm, emb = await asyncio.gather(llm_task, emb_task)
        else:
            llm = await llm_task
            emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}

        intent, fused_confidence, intent_scores, matched_intents = self._vote(llm, emb, pat)
        entities = await self._extract_entities(message)
        urgency  = self._urgency(message, intent)

        result = IntentResult(
            intent=intent,
            confidence=fused_confidence,
            urgency=urgency,
            entities=entities,
            reasoning=llm.get("reasoning", ""),
            latency_ms=(time.monotonic() - t0) * 1000,
            intent_scores=intent_scores,
            matched_intents=matched_intents,
        )

        # LRU 缓存
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
        self._cache[key] = result
        return result

    def learn(
        self,
        message: str,
        correct: IntentCategory,
        *,
        persist: bool = True,
    ) -> bool:
        """应用一条人工纠正样本；返回模板是否发生变化。"""
        message = self._clean_text(message).strip()
        if not message:
            return False

        correct_templates = self._templates.setdefault(correct, [])
        conflicting_categories = {
            category
            for category, templates in self._templates.items()
            if category != correct and message in templates
        }
        needs_add = message not in correct_templates
        if not conflicting_categories and not needs_add:
            return False

        # 先写持久层，避免 Redis 失败时内存模板已经被修改，形成“半成功”状态。
        if persist and self._feedback_store is not None:
            self._feedback_store.upsert(message, correct.value)

        for category in conflicting_categories:
            templates = self._templates[category]
            self._templates[category] = [
                template for template in templates if template != message
            ]
        if needs_add:
            correct_templates.append(message)

        for category in conflicting_categories | {correct}:
            self._tpl_embeddings.pop(category, None)
        self._cache.clear()

        logger.info("应用人工意图反馈 → %s: %s", correct.value, message[:40])
        return True

    def _load_persisted_feedback(self) -> None:
        """启动时加载已审核样本；Redis异常不阻塞主服务。"""
        if self._feedback_store is None:
            return
        try:
            samples = self._feedback_store.load_all()
        except Exception as ex:
            logger.warning("加载持久化意图反馈失败: %s", ex)
            return

        loaded = 0
        for sample in samples:
            try:
                correct = IntentCategory(sample.intent)
            except (AttributeError, ValueError):
                logger.warning("跳过未知意图反馈标签: %r", getattr(sample, "intent", None))
                continue
            if self.learn(sample.message, correct, persist=False):
                loaded += 1
        if loaded:
            logger.info("已加载 %d 条持久化意图反馈", loaded)

    # ── 三路识别策略 ──────────────────────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """策略 1：LLM 语义理解（Few-shot + 上下文）。"""
        message = self._clean_text(message)
        # 每类保留 1 条稳定内置示例，并补充最近 2 条人工审核样本。
        # 这样反馈在第三方 API 禁用 Embedding 时仍能影响 LLM，同时限制 Prompt 长度。
        examples = "\n".join(
            f'  消息: "{t}" → 意图: {cat.value}'
            for cat in self._templates
            for t in self._few_shot_templates(cat)
        )
        # 最近 3 轮对话上下文
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""你是客服意图分析专家。根据示例判断用户可能同时包含的一个或多个意图，返回 JSON。

示例:
{examples}

{ctx}
用户消息: "{message}"

返回格式（仅 JSON，不要其他文字）:
{{"intents": [{{"intent": "<意图值>", "confidence": <0-1>}}], "reasoning": "<一句话说明>"}}

重要分类规则：
1. technical、billing、account 是业务领域意图；query、request、complaint、feedback 是动作意图。
2. 领域意图和动作意图不是互斥类别，必须分别判断；命中专业领域时，即使用户使用“怎么、为什么、查询”等问法，也要同时输出领域意图和动作意图。
3. 边界示例：“校园网401应该怎么排查”同时输出 technical 和 query；“校园卡为什么重复扣费”同时输出 billing 和 query；“帮我重置登录密码”同时输出 account 和 request。
4. 校园网、WiFi、认证错误码和网络故障属于 technical；校园卡消费、充值、余额、扣款、退款属于 billing；统一身份认证、密码、绑定邮箱、账号锁定属于 account。
5. 用户明确要求转人工、找人工客服或负责人时，必须输出 escalation；如果同时描述业务问题，也保留对应领域意图。

要求：
1. 每个确实存在的意图单独给出 confidence。
2. 不要为了凑数量输出无关意图。
3. intents 按 confidence 从高到低排列。

可选意图: {", ".join(c.value for c in IntentCategory)}"""
        prompt = self._clean_text(prompt)

        try:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=384,
                # Routing must be reproducible: small sampling changes can
                # swap a close action/domain pair (for example query vs
                # technical) and send the same request to a different Agent.
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                **self._structured_llm_options,
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            scores: Dict[IntentCategory, float] = {}
            for item in data.get("intents", []):
                try:
                    category = IntentCategory(item["intent"])
                    confidence = min(max(float(item["confidence"]), 0.0), 1.0)
                except (KeyError, TypeError, ValueError):
                    continue
                scores[category] = max(scores.get(category, 0.0), confidence)

            # 兼容仍按旧格式返回单标签 JSON 的模型。
            if not scores and "intent" in data:
                try:
                    category = IntentCategory(data["intent"])
                    confidence = min(max(float(data.get("confidence", 0.0)), 0.0), 1.0)
                except (TypeError, ValueError):
                    category, confidence = IntentCategory.OTHER, 0.0
                if confidence > 0:
                    scores[category] = confidence

            if scores:
                primary = max(scores, key=scores.get)
                primary_confidence = scores[primary]
            else:
                primary, primary_confidence = IntentCategory.OTHER, 0.0
            return {
                "intent": primary,
                "confidence": primary_confidence,
                "scores": scores,
                "reasoning": data.get("reasoning", ""),
            }
        except Exception as ex:
            logger.warning(f"LLM 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "LLM 失败", "failed": True}

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配。"""
        try:
            await self._load_template_embeddings()
            msg_vec = await self._embed_text(message)

            scores: Dict[IntentCategory, float] = {}
            for cat, vecs in self._tpl_embeddings.items():
                scores[cat] = min(
                    max(max(_cosine(msg_vec, v) for v in vecs), 0.0),
                    1.0,
                )

            best_cat = max(scores, key=scores.get) if scores else IntentCategory.OTHER
            best_score = scores.get(best_cat, 0.0)
            return {
                "intent": best_cat,
                "confidence": best_score,
                "scores": scores,
            }
        except Exception as ex:
            logger.warning(f"Embedding 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """策略 3：关键词模式匹配（同步，零延迟兜底）。"""
        msg = message.lower()
        patterns = {
            IntentCategory.ESCALATION: ["投诉", "经理", "转人工", "supervisor"],
            IntentCategory.COMPLAINT:  ["太差", "糟糕", "horrible", "等了很久"],
            IntentCategory.QUERY:      ["?", "？", "怎么", "什么", "status"],
            IntentCategory.REQUEST:    ["帮我", "需要", "please", "help"],
            IntentCategory.GREETING:   ["你好", "嗨", "hello", "hi"],
            IntentCategory.BILLING:    [
                "退款", "扣款", "扣费", "发票", "消费", "充值", "余额", "refund",
            ],
            IntentCategory.TECHNICAL:  ["崩溃", "报错", "error", "crash"],
            IntentCategory.ACCOUNT:    ["密码", "邮箱", "账户", "password"],
        }
        scores: Dict[IntentCategory, float] = {}
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw in msg)
            if hits:
                # Keep runtime semantics aligned with the calibration code:
                # one exact business keyword is strong binary evidence.  The
                # global fusion weight still limits Pattern's final influence.
                scores[cat] = 1.0
        best_cat = max(scores, key=scores.get) if scores else IntentCategory.OTHER
        best_score = scores.get(best_cat, 0.0)
        return {
            "intent": best_cat,
            "confidence": best_score,
            "scores": scores,
        }

    # ── 投票合并 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _strategy_scores(result: Dict[str, Any]) -> Dict[IntentCategory, float]:
        """把新旧策略输出统一为每个意图一个分数。"""
        normalized: Dict[IntentCategory, float] = {}
        for raw_category, raw_score in (result.get("scores") or {}).items():
            try:
                category = (
                    raw_category
                    if isinstance(raw_category, IntentCategory)
                    else IntentCategory(str(raw_category))
                )
                score = min(max(float(raw_score), 0.0), 1.0)
            except (TypeError, ValueError):
                continue
            if score > 0:
                normalized[category] = max(normalized.get(category, 0.0), score)

        if normalized:
            return normalized

        raw_category = result.get("intent", IntentCategory.OTHER)
        try:
            category = (
                raw_category
                if isinstance(raw_category, IntentCategory)
                else IntentCategory(str(raw_category))
            )
            score = min(max(float(result.get("confidence", 0.0)), 0.0), 1.0)
        except (TypeError, ValueError):
            return {}
        return {category: score} if score > 0 else {}

    def _vote(
        self,
        llm: Dict,
        emb: Dict,
        pat: Dict,
    ) -> tuple[
        IntentCategory,
        float,
        Dict[IntentCategory, float],
        List[IntentCategory],
    ]:
        """逐意图加权融合，并返回主意图、分数表和阈值内的多标签。"""
        strategies = [
            (llm, self._strategy_weights["llm"]),
            (emb, self._strategy_weights["embedding"]),
            (pat, self._strategy_weights["pattern"]),
        ]

        active = []
        for result, weight in strategies:
            if result.get("failed"):
                continue
            strategy_scores = self._strategy_scores(result)
            if strategy_scores:
                active.append((strategy_scores, weight))
        total_weight = sum(weight for _, weight in active)
        if total_weight <= 0:
            return IntentCategory.OTHER, 0.0, {}, []

        scores: Dict[IntentCategory, float] = {}
        for strategy_scores, weight in active:
            normalized_weight = weight / total_weight
            for category, score in strategy_scores.items():
                scores[category] = scores.get(category, 0.0) + normalized_weight * score

        best = max(scores, key=scores.get)
        best_score = float(scores[best])
        intent = best if best_score >= self.threshold else IntentCategory.OTHER
        matched = sorted(
            (
                category
                for category, score in scores.items()
                if category != IntentCategory.OTHER
                and score >= self.multi_label_threshold
            ),
            key=lambda category: scores[category],
            reverse=True,
        )
        return intent, best_score, scores, matched

    # ── 实体提取 ──────────────────────────────────────────────────────────────

    async def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """用 LLM 从消息中提取结构化实体。"""
        message = self._clean_text(message)
        prompt = f"""从客服消息中提取实体，返回 JSON（字段值为列表，没有则为空列表）:
消息: "{message}"
格式: {{"order_id":[],"product":[],"date":[],"amount":[],"error_code":[]}}"""
        prompt = self._clean_text(prompt)
        try:
            resp = await self.client.messages.create(
                model=self.model, max_tokens=256, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                **self._structured_llm_options,
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            return json.loads(raw[s:e])
        except Exception:
            return {"order_id": [], "product": [], "date": [], "amount": [], "error_code": []}

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    async def _load_template_embeddings(self) -> None:
        """懒加载所有模板的 Embedding（只在首次调用时执行）。"""
        missing = [cat for cat in self._templates if cat not in self._tpl_embeddings]
        if not missing:
            return

        all_texts = [t for cat in missing for t in self._templates[cat]]
        vecs = [await self._embed_text(text) for text in all_texts]
        idx = 0
        for cat in missing:
            n = len(self._templates[cat])
            self._tpl_embeddings[cat] = vecs[idx: idx + n]
            idx += n

    async def _embed_text(self, text: str) -> List[float]:
        """
        生成文本向量。

        如果未来接入的官方/兼容客户端提供 embeddings.create，会优先使用远端向量；
        当前 Anthropic SDK 没有该资源时，退化为字符 n-gram 哈希向量。这样不会因为
        Embedding 服务缺失导致三路融合中断。
        """
        embeddings = getattr(self.client, "embeddings", None)
        if embeddings is not None:
            try:
                resp = await embeddings.create(model="voyage-3-lite", input=[text])
                return list(resp.data[0].embedding)
            except Exception as ex:
                logger.warning(f"远端 Embedding 失败，使用本地向量兜底: {ex}")

        return self._local_embedding(text)

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        """稳定的字符 n-gram 哈希向量，用于无远端 Embedding 时的语义近似匹配。"""
        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(normalized[i:i + n] for i in range(len(normalized) - n + 1))
        if not tokens:
            tokens.add(normalized)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        for level, kws in _URGENCY_KEYWORDS.items():
            if any(kw in msg for kw in kws):
                return level
        if intent == IntentCategory.ESCALATION:
            return UrgencyLevel.HIGH
        if intent == IntentCategory.COMPLAINT:
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _template_fingerprint(self) -> str:
        payload = {
            category.value: self._templates.get(category, [])
            for category in sorted(self._templates, key=lambda item: item.value)
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _few_shot_templates(self, category: IntentCategory) -> List[str]:
        """Return one stable example plus at most two reviewed examples."""
        templates = self._templates.get(category, [])
        if not templates:
            return []

        built_in = set(_TEMPLATES.get(category, []))
        reviewed = [template for template in templates if template not in built_in]
        return [templates[0], *reviewed[-2:]]

    @property
    def template_fingerprint(self) -> str:
        """Public fingerprint for diagnostics and cache-version visibility."""
        return self._template_fingerprint()

    def _cache_key(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        normalized_history = [
            {
                "role": self._clean_text(item.get("role", "user")),
                "content": self._clean_text(item.get("content", "")),
            }
            for item in (history or [])[-3:]
        ]
        payload = {
            "message": self._clean_text(message)[:500],
            "history": normalized_history,
            "model": self.model,
            "templates": self._template_fingerprint(),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 HTTP 客户端编码 prompt 时崩溃。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
