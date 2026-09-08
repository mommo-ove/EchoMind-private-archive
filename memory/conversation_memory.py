"""
亮点：多轮对话记忆管理

三级记忆架构，模拟人类记忆机制：
  1. 工作记忆（Redis）—— 当前会话的最近 N 条消息，毫秒级读写
  2. 情景记忆（ChromaDB）—— 跨会话的历史对话，按语义相似度检索
  3. 用户画像（ChromaDB）—— 从对话中提炼的长期偏好和实体

关键设计：
  - 上下文构建时三级记忆融合，按重要性 + 时效性排序
  - 工作记忆超过阈值时自动压缩（LLM 摘要），防止 context 爆炸
  - 所有 Embedding 通过 Anthropic API 生成，无本地模型
"""
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import chromadb
import redis
from anthropic import AsyncAnthropic
from redis.exceptions import WatchError

from core.llm_utils import extract_text_content
from mcp.local_embeddings import FastEmbedTextModel

logger = logging.getLogger(__name__)


class MsgRole(Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


class OperationStatus(Enum):
    CLAIMED = "claimed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class OperationClaim:
    status: OperationStatus
    payload: Optional[str] = None


@dataclass
class Message:
    role:       MsgRole
    content:    str
    timestamp:  datetime = field(default_factory=datetime.now)
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContext:
    """传给 Agent 的完整上下文。"""
    recent_messages:  List[Message]   # 工作记忆：最近对话
    relevant_history: List[str]       # 情景记忆：语义相关的历史片段
    user_profile:     Dict[str, Any]  # 用户画像：偏好、常用实体
    summary:          str             # 当前会话摘要（压缩后）

    @staticmethod
    def _clean(text: str) -> str:
        """移除 Unicode 代理字符，防止编码错误。"""
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    def to_prompt_text(self) -> str:
        """将记忆上下文格式化为 LLM 可用的文本。"""
        parts = []
        if self.summary:
            parts.append(f"[会话摘要]\n{self._clean(self.summary)}")
        if self.relevant_history:
            parts.append("[相关历史]\n" + "\n".join(f"- {self._clean(h)}" for h in self.relevant_history[:3]))
        if self.user_profile:
            parts.append(f"[用户画像]\n{json.dumps(self.user_profile, ensure_ascii=True)}")
        if self.recent_messages:
            parts.append("[最近对话]")
            for m in self.recent_messages:
                parts.append(f"{m.role.value}: {self._clean(m.content)}")
        return "\n\n".join(parts)


class MemoryManager:
    """
    三级记忆管理器。

    工作记忆存 Redis（TTL 24h），情景记忆和用户画像存 ChromaDB（持久化）。
    """

    WORKING_MAX   = 20    # 工作记忆最大条数，超过则触发压缩
    COMPRESS_AT   = 15    # 达到此条数时压缩，保留摘要 + 最近 5 条
    HISTORY_TOP_K = 5     # 情景记忆检索返回条数

    OPERATION_PENDING_TTL_SECONDS = 60
    OPERATION_COMPLETED_TTL_SECONDS = 86400
    OPERATION_PAYLOAD_MAX_BYTES = 65536
    COMPRESSION_LEASE_SECONDS = 30

    def __init__(
        self,
        redis_url:    str = "redis://localhost:6379/0",
        chroma_host:  str = "localhost",
        chroma_port:  int = 8000,
        chroma_path:  str = "./data/chroma",
        api_key:      str = "",
        base_url:     Optional[str] = None,
        model:        str = "claude-3-5-sonnet-20241022",
        embedder:      Optional[FastEmbedTextModel] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model  = model
        self._embedder = embedder or FastEmbedTextModel()

        self._redis = redis.from_url(redis_url, decode_responses=True)

        # ChromaDB：优先连接独立服务（docker compose 模式），连不上则降级为本地嵌入式
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            chroma = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            chroma.heartbeat()  # 测试连接
            logger.info(f"ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"ChromaDB 服务不可用，使用本地嵌入式模式: {chroma_path}")
            chroma = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 情景记忆：存储历史对话片段
        self._episodic = chroma.get_or_create_collection("episodic_bge_v1")
        # 用户画像：存储提炼出的偏好和实体
        self._profile  = chroma.get_or_create_collection("user_profile_bge_v1")

    # ── 写入 ──────────────────────────────────────────────────────────────────

    async def add_message(
        self,
        user_id: str,
        conv_id: str,
        role:    MsgRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """将一条消息写入工作记忆，超阈值时自动压缩。"""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        msg = Message(
            role=role,
            content=self._safe_text(content),
            metadata=self._clean_metadata(metadata),
        )
        key = self._wm_key(user_id, conv_id)
        revision_key = self._revision_key(user_id, conv_id)
        pipeline = self._redis.pipeline(transaction=True)
        pipeline.lpush(key, self._encode_message(msg))
        pipeline.expire(key, 86400)
        pipeline.incr(revision_key)
        pipeline.expire(revision_key, 86400)
        pipeline.llen(key)
        results = pipeline.execute()

        if results[-1] >= self.COMPRESS_AT:
            await self._compress(user_id, conv_id)

    async def add_exchange(
        self,
        user_id: str,
        conv_id: str,
        user_content: str,
        assistant_content: str,
        *,
        exchange_key: str,
        user_metadata: Optional[Dict[str, Any]] = None,
        assistant_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Atomically append one idempotent user/assistant exchange."""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        self._validate_exchange_key(exchange_key)
        user_message = self._encode_message(Message(
            role=MsgRole.USER,
            content=self._safe_text(user_content),
            metadata=self._clean_metadata(user_metadata),
        ))
        assistant_message = self._encode_message(Message(
            role=MsgRole.ASSISTANT,
            content=self._safe_text(assistant_content),
            metadata=self._clean_metadata(assistant_metadata),
        ))
        marker_key = self._exchange_marker_key(
            user_id,
            conv_id,
            exchange_key,
        )
        script = """
local memory_type = redis.call('TYPE', KEYS[1]).ok
local marker_type = redis.call('TYPE', KEYS[2]).ok
local revision_type = redis.call('TYPE', KEYS[3]).ok
if memory_type ~= 'none' and memory_type ~= 'list' then
  return redis.error_reply('working memory key has invalid type')
end
if marker_type ~= 'none' and marker_type ~= 'string' then
  return redis.error_reply('exchange marker has invalid type')
end
if revision_type ~= 'none' and revision_type ~= 'string' then
  return redis.error_reply('revision key has invalid type')
end
if redis.call('EXISTS', KEYS[2]) == 1 then
  return -1
end
redis.call('LPUSH', KEYS[1], ARGV[1], ARGV[2])
redis.call('EXPIRE', KEYS[1], ARGV[3])
redis.call('SET', KEYS[2], '1', 'EX', ARGV[3])
redis.call('INCR', KEYS[3])
redis.call('EXPIRE', KEYS[3], ARGV[3])
return redis.call('LLEN', KEYS[1])
"""
        length = self._redis.eval(
            script,
            3,
            self._wm_key(user_id, conv_id),
            marker_key,
            self._revision_key(user_id, conv_id),
            user_message,
            assistant_message,
            86400,
        )
        if length == -1:
            return False
        if length >= self.COMPRESS_AT:
            await self._compress(user_id, conv_id)
        return True

    async def claim_operation(
        self,
        user_id: str,
        conv_id: str,
        operation_id: str,
        payload_hash: str,
        owner_id: str,
    ) -> OperationClaim:
        """Atomically claim or replay one scoped idempotent operation."""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        self._validate_digest(operation_id, "operation id")
        self._validate_digest(payload_hash, "payload hash")
        self._validate_owner_id(owner_id)
        script = """-- CLAIM_OPERATION
local key_type = redis.call('TYPE', KEYS[1]).ok
if key_type ~= 'none' and key_type ~= 'hash' then
  return redis.error_reply('operation key has invalid type')
end
if key_type == 'none' then
  redis.call('HSET', KEYS[1], 'state', 'pending')
  redis.call('HSET', KEYS[1], 'payload_hash', ARGV[1])
  redis.call('HSET', KEYS[1], 'owner', ARGV[2])
  redis.call('HSET', KEYS[1], 'payload', '')
  redis.call('EXPIRE', KEYS[1], ARGV[3])
  return {'claimed', ''}
end
local stored_hash = redis.call('HGET', KEYS[1], 'payload_hash')
if stored_hash ~= ARGV[1] then
  return {'conflict', ''}
end
local state = redis.call('HGET', KEYS[1], 'state')
if state == 'completed' then
  return {'completed', redis.call('HGET', KEYS[1], 'payload') or ''}
end
if state == 'pending' then
  return {'in_progress', ''}
end
return redis.error_reply('operation record has invalid state')
"""
        raw = self._redis.eval(
            script,
            1,
            self._operation_key(user_id, conv_id, operation_id),
            payload_hash,
            owner_id,
            self.OPERATION_PENDING_TTL_SECONDS,
        )
        if (
            not isinstance(raw, (list, tuple))
            or len(raw) != 2
            or not isinstance(raw[0], str)
            or not isinstance(raw[1], str)
        ):
            raise RuntimeError("invalid operation claim response")
        try:
            status = OperationStatus(raw[0])
        except ValueError as exc:
            raise RuntimeError("invalid operation claim status") from exc
        payload = raw[1] if status is OperationStatus.COMPLETED else None
        if (
            payload is not None
            and len(payload.encode("utf-8")) > self.OPERATION_PAYLOAD_MAX_BYTES
        ):
            raise RuntimeError("operation replay payload is too large")
        return OperationClaim(status=status, payload=payload)

    async def commit_operation(
        self,
        user_id: str,
        conv_id: str,
        operation_id: str,
        payload_hash: str,
        owner_id: str,
        user_content: str,
        assistant_content: str,
        payload: str,
    ) -> bool:
        """Atomically append the exchange and publish its replay result."""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        self._validate_digest(operation_id, "operation id")
        self._validate_digest(payload_hash, "payload hash")
        self._validate_owner_id(owner_id)
        if not isinstance(payload, str):
            raise TypeError("operation payload must be a string")
        if len(payload.encode("utf-8")) > self.OPERATION_PAYLOAD_MAX_BYTES:
            raise ValueError("operation payload is too large")
        user_message = self._encode_message(Message(
            role=MsgRole.USER,
            content=self._safe_text(user_content),
        ))
        assistant_message = self._encode_message(Message(
            role=MsgRole.ASSISTANT,
            content=self._safe_text(assistant_content),
        ))
        script = """-- COMMIT_OPERATION
if redis.call('TYPE', KEYS[1]).ok ~= 'hash' then
  return 0
end
local memory_type = redis.call('TYPE', KEYS[2]).ok
local revision_type = redis.call('TYPE', KEYS[3]).ok
if memory_type ~= 'none' and memory_type ~= 'list' then
  return redis.error_reply('working memory key has invalid type')
end
if revision_type ~= 'none' and revision_type ~= 'string' then
  return redis.error_reply('revision key has invalid type')
end
if redis.call('HGET', KEYS[1], 'state') ~= 'pending'
   or redis.call('HGET', KEYS[1], 'payload_hash') ~= ARGV[1]
   or redis.call('HGET', KEYS[1], 'owner') ~= ARGV[2] then
  return 0
end
redis.call('LPUSH', KEYS[2], ARGV[3], ARGV[4])
redis.call('EXPIRE', KEYS[2], ARGV[6])
redis.call('INCR', KEYS[3])
redis.call('EXPIRE', KEYS[3], ARGV[6])
redis.call('HSET', KEYS[1], 'state', 'completed')
redis.call('HSET', KEYS[1], 'owner', '')
redis.call('HSET', KEYS[1], 'payload', ARGV[5])
redis.call('EXPIRE', KEYS[1], ARGV[6])
return redis.call('LLEN', KEYS[2])
"""
        length = self._redis.eval(
            script,
            3,
            self._operation_key(user_id, conv_id, operation_id),
            self._wm_key(user_id, conv_id),
            self._revision_key(user_id, conv_id),
            payload_hash,
            owner_id,
            user_message,
            assistant_message,
            payload,
            self.OPERATION_COMPLETED_TTL_SECONDS,
        )
        if not isinstance(length, int) or isinstance(length, bool):
            raise RuntimeError("invalid operation commit response")
        if length <= 0:
            return False
        if length >= self.COMPRESS_AT:
            await self._compress(user_id, conv_id)
        return True

    async def release_operation(
        self,
        user_id: str,
        conv_id: str,
        operation_id: str,
        payload_hash: str,
        owner_id: str,
    ) -> bool:
        """Release only the caller-owned pending operation."""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        self._validate_digest(operation_id, "operation id")
        self._validate_digest(payload_hash, "payload hash")
        self._validate_owner_id(owner_id)
        script = """-- RELEASE_OPERATION
if redis.call('TYPE', KEYS[1]).ok ~= 'hash' then
  return 0
end
if redis.call('HGET', KEYS[1], 'state') ~= 'pending'
   or redis.call('HGET', KEYS[1], 'payload_hash') ~= ARGV[1]
   or redis.call('HGET', KEYS[1], 'owner') ~= ARGV[2] then
  return 0
end
return redis.call('DEL', KEYS[1])
"""
        return bool(self._redis.eval(
            script,
            1,
            self._operation_key(user_id, conv_id, operation_id),
            payload_hash,
            owner_id,
        ))

    async def update_profile(self, user_id: str, conv_id: str) -> None:
        """
        从当前工作记忆中提炼用户偏好，更新用户画像。
        用 LLM 提炼偏好，然后存入 ChromaDB（ChromaDB 内置 embedding，不依赖外部 API）。
        """
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        messages = await self._get_working_memory(user_id, conv_id)
        if not messages:
            return

        text = self._safe_text("\n".join(f"{m.role.value}: {m.content}" for m in messages[-10:]))
        prompt = f"""从以下对话中提炼用户偏好和关键实体，返回 JSON。
对话:
{text}

返回格式: {{"preferences": ["..."], "entities": {{"产品": [], "问题类型": []}}}}"""
        prompt = self._safe_text(prompt)

        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=512, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            profile_data = json.loads(raw[s:e])

            doc_id = self._profile_doc_id(user_id, conv_id)
            doc_text = self._safe_text(json.dumps(profile_data, ensure_ascii=False))

            try:
                self._profile.delete(ids=[doc_id])
            except Exception:
                pass

            # 直接传 documents，让 ChromaDB 内置模型生成 embedding（不依赖 Voyage API）
            self._profile.add(
                ids=[doc_id],
                documents=[doc_text],
                embeddings=self._embedder.embed([doc_text]),
                metadatas=[{"user_id": user_id, "conv_id": conv_id,
                            "ts": datetime.now().isoformat()}],
            )
            logger.info(f"用户画像已更新: {user_id}")
        except Exception as ex:
            logger.warning(f"更新用户画像失败: {ex}")

    # ── 读取 ──────────────────────────────────────────────────────────────────

    async def get_context(self, user_id: str, conv_id: str, query: str = "") -> MemoryContext:
        """
        构建完整的记忆上下文。

        query 用于从情景记忆中检索语义相关的历史片段。
        """
        # 1. 工作记忆（当前会话最近消息）
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        query = self._safe_text(query)

        recent = await self._get_working_memory(user_id, conv_id)

        # 2. 情景记忆（跨会话语义检索）
        history = await self._search_episodic(user_id, query or (recent[-1].content if recent else ""))

        # 3. 用户画像
        profile = await self._get_profile(user_id)

        # 4. 会话摘要（如果已压缩过）
        summary = self._redis.get(self._summary_key(user_id, conv_id)) or ""

        return MemoryContext(
            recent_messages=recent,
            relevant_history=history,
            user_profile=profile,
            summary=summary,
        )

    # ── 压缩（防止 context 爆炸）─────────────────────────────────────────────

    async def _compress(self, user_id: str, conv_id: str) -> None:
        lease_key = self._compression_lease_key(user_id, conv_id)
        owner_id = uuid.uuid4().hex
        try:
            acquired = self._redis.set(
                lease_key,
                owner_id,
                nx=True,
                ex=self.COMPRESSION_LEASE_SECONDS,
            )
        except Exception as exc:
            logger.warning(
                "Working memory compression lease failed (%s)",
                type(exc).__name__,
            )
            return
        if not acquired:
            return
        try:
            await self._compress_owned(user_id, conv_id)
        finally:
            script = """-- RELEASE_COMPRESSION_LEASE
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
  return 0
end
return redis.call('DEL', KEYS[1])
"""
            try:
                self._redis.eval(script, 1, lease_key, owner_id)
            except Exception as exc:
                logger.warning(
                    "Working memory compression lease release failed (%s)",
                    type(exc).__name__,
                )

    async def _compress_owned(self, user_id: str, conv_id: str) -> None:
        """
        工作记忆压缩：
          1. 用 LLM 对旧消息生成摘要
          2. 摘要存 Redis（覆盖旧摘要）
          3. 旧消息存入情景记忆（ChromaDB）供跨会话检索
          4. 工作记忆只保留最近 5 条
        """
        key = self._wm_key(user_id, conv_id)
        revision_key = self._revision_key(user_id, conv_id)
        summary_key = self._summary_key(user_id, conv_id)
        try:
            snapshot = self._redis.pipeline(transaction=True)
            snapshot.get(revision_key)
            snapshot.lrange(key, 0, self.WORKING_MAX - 1)
            snapshot.get(summary_key)
            revision, raws, old_summary = snapshot.execute()
            revision_number = int(revision or 0)
            if revision_number < 0:
                return
            messages = self._messages_from_raw(raws)
        except Exception as exc:
            logger.warning(
                "Working memory compression snapshot failed (%s)",
                type(exc).__name__,
            )
            return
        if len(messages) < self.COMPRESS_AT:
            return

        to_compress = messages[:-5]   # 保留最近 5 条
        keep        = messages[-5:]

        # LLM 摘要
        text = self._safe_text("\n".join(f"{m.role.value}: {m.content}" for m in to_compress))
        prompt = self._safe_text(f"用 2-3 句话总结以下对话的关键信息：\n{text}")
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=256, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = self._safe_text(extract_text_content(resp.content)).strip()
        except Exception:
            summary = f"对话包含 {len(to_compress)} 条消息（摘要生成失败）"

        new_summary = self._safe_text(
            f"{old_summary or ''}\n{summary}"
        ).strip()
        summary_message = Message(
            role=MsgRole.SYSTEM,
            content=new_summary,
            metadata={"compressed": True},
        )
        retained = [
            self._encode_message(message)
            for message in [summary_message, *keep]
        ]
        expected_revision = revision_number
        try:
            with self._redis.pipeline(transaction=True) as pipeline:
                pipeline.watch(revision_key)
                current_revision = int(
                    pipeline.get(revision_key) or 0
                )
                if current_revision != expected_revision:
                    pipeline.unwatch()
                    return
                pipeline.multi()
                pipeline.setex(summary_key, 86400, new_summary)
                pipeline.delete(key)
                pipeline.lpush(key, *retained)
                pipeline.expire(key, 86400)
                pipeline.incr(revision_key)
                pipeline.expire(revision_key, 86400)
                pipeline.execute()
        except WatchError:
            return
        except Exception as exc:
            logger.warning(
                "Working memory compression commit failed (%s)",
                type(exc).__name__,
            )
            return

        await self._store_episodic(user_id, conv_id, text, summary)
        logger.info(f"工作记忆压缩完成: {user_id}/{conv_id}，摘要 {len(summary)} 字")

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    async def _get_working_memory(self, user_id: str, conv_id: str) -> List[Message]:
        key  = self._wm_key(user_id, conv_id)
        raws = self._redis.lrange(key, 0, self.WORKING_MAX - 1)
        return self._messages_from_raw(raws)

    @staticmethod
    def _messages_from_raw(raws: List[str]) -> List[Message]:
        msgs = []
        for raw in reversed(raws):  # Redis lpush 最新在前，reversed 还原时序
            d = json.loads(raw)
            msgs.append(Message(
                role=MsgRole(d["role"]),
                content=d["content"],
                timestamp=datetime.fromisoformat(d["ts"]),
                metadata=d.get("metadata", {}),
            ))
        return msgs

    async def _search_episodic(self, user_id: str, query: str) -> List[str]:
        """语义检索情景记忆。ChromaDB 内置 embedding，不依赖外部 API。"""
        query_text = self._safe_text(query).strip()
        if not query_text:
            return []
        try:
            # 直接传 query_texts，ChromaDB 内置模型自动生成向量做匹配
            results = self._episodic.query(
                query_embeddings=[self._embedder.embed_query(query_text)],
                n_results=self.HISTORY_TOP_K,
                where={"user_id": self._safe_text(user_id)},
            )
            docs = results["documents"][0] if results["documents"] else []
            return [self._safe_text(doc) for doc in docs if isinstance(doc, str) and doc.strip()]
        except Exception as ex:
            logger.warning(f"情景记忆检索失败: {ex}")
            return []

    async def _store_episodic(self, user_id: str, conv_id: str, text: str, summary: str) -> None:
        """将压缩后的对话片段存入情景记忆。ChromaDB 内置 embedding，不依赖外部 API。"""
        try:
            user_id = self._safe_text(user_id)
            conv_id = self._safe_text(conv_id)
            text = self._safe_text(text)
            summary = self._safe_text(summary)
            doc_id = self._episodic_doc_id(user_id, conv_id)
            # 直接传 documents，ChromaDB 内置模型自动生成 embedding
            self._episodic.add(
                ids=[doc_id],
                documents=[summary],
                embeddings=self._embedder.embed([summary]),
                metadatas=[{"user_id": user_id, "conv_id": conv_id,
                            "ts": datetime.now().isoformat(), "full_text": self._safe_text(text[:500])}],
            )
        except Exception as ex:
            logger.warning(f"存储情景记忆失败: {ex}")

    async def _get_profile(self, user_id: str) -> Dict[str, Any]:
        """获取用户画像（取最新一条）。"""
        try:
            results = self._profile.get(where={"user_id": user_id}, limit=1)
            if results["documents"]:
                return json.loads(results["documents"][0])
        except Exception:
            pass
        return {}

    @staticmethod
    def _scope_component(user_id: str, conv_id: str) -> str:
        canonical = json.dumps(
            [user_id, conv_id],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @classmethod
    def _profile_doc_id(cls, user_id: str, conv_id: str) -> str:
        return f"profile:{cls._scope_component(user_id, conv_id)}"

    @classmethod
    def _episodic_doc_id(cls, user_id: str, conv_id: str) -> str:
        return (
            f"episodic:{cls._scope_component(user_id, conv_id)}:"
            f"{uuid.uuid4().hex}"
        )

    @classmethod
    def _wm_key(cls, user_id: str, conv_id: str) -> str:
        return f"wm:{cls._scope_component(user_id, conv_id)}"

    @classmethod
    def _revision_key(cls, user_id: str, conv_id: str) -> str:
        return f"wm-rev:{cls._scope_component(user_id, conv_id)}"

    @classmethod
    def _exchange_marker_key(
        cls,
        user_id: str,
        conv_id: str,
        exchange_key: str,
    ) -> str:
        digest = hashlib.sha256(
            json.dumps(
                [cls._scope_component(user_id, conv_id), exchange_key],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return f"wm-exchange:{digest}"

    @classmethod
    def _operation_key(
        cls,
        user_id: str,
        conv_id: str,
        operation_id: str,
    ) -> str:
        digest = hashlib.sha256(
            json.dumps(
                [cls._scope_component(user_id, conv_id), operation_id],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return f"chat-operation:{digest}"

    @classmethod
    def _compression_lease_key(cls, user_id: str, conv_id: str) -> str:
        return f"wm-compress:{cls._scope_component(user_id, conv_id)}"

    @classmethod
    def _summary_key(cls, user_id: str, conv_id: str) -> str:
        return f"summary:{cls._scope_component(user_id, conv_id)}"

    @classmethod
    def _clean_metadata(
        cls,
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            cls._safe_text(key): cls._safe_metadata_value(value)
            for key, value in (metadata or {}).items()
        }

    @staticmethod
    def _encode_message(message: Message) -> str:
        return json.dumps({
            "role": message.role.value,
            "content": message.content,
            "ts": message.timestamp.isoformat(),
            "metadata": message.metadata,
        })

    @staticmethod
    def _validate_exchange_key(exchange_key: Any) -> None:
        if (
            not isinstance(exchange_key, str)
            or not exchange_key
            or len(exchange_key) > 128
            or not exchange_key.isprintable()
            or any(character.isspace() for character in exchange_key)
        ):
            raise ValueError("invalid exchange key")

    @staticmethod
    def _validate_digest(value: Any, name: str) -> None:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"invalid {name}")

    @staticmethod
    def _validate_owner_id(value: Any) -> None:
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 128
            or not value.isprintable()
            or any(character.isspace() for character in value)
        ):
            raise ValueError("invalid operation owner")

    @staticmethod
    def _safe_text(value: Any) -> str:
        """转成 ChromaDB 可接受的普通 UTF-8 字符串。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @classmethod
    def _safe_metadata_value(cls, value: Any) -> Any:
        """递归清洗 metadata，避免 Redis/ChromaDB 后续读写遇到非法 UTF-8。"""
        if isinstance(value, str):
            return cls._safe_text(value)
        if isinstance(value, dict):
            return {cls._safe_text(k): cls._safe_metadata_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._safe_metadata_value(v) for v in value]
        return value
