"""
EchoMind 智能客服系统 — FastAPI 入口

启动时打印小熊饼干图案。
所有核心组件在 lifespan 中初始化，通过环境变量配置。
"""
import asyncio
import hmac
import logging
import math
import os
import pathlib
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional


_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, UploadFile, File
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_serializer,
    model_validator,
)

from api.auth import JWTAuthMiddleware, JWTAuthSettings


load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = r"""
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
   ╔══════════════════════╗
   ║   EchoMind  v2.0     ║
   ║   智能客服 AI 系统    ║
   ╚══════════════════════╝
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
"""

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_orchestrator = None
_memory       = None
_tool_manager = None
_monitor      = None
_evaluator    = None
_skill_manager = None
_intent_recognizer = None
_intent_feedback_store = None
_chat_service = None
_campus_store = None

def _anthropic_cfg() -> Dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")
    cfg: Dict[str, Any] = {
        "api_key":  key,
        "model":    os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip(),
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        cfg["base_url"] = base_url
    return cfg


def _intent_fusion_config(base_url: Optional[str] = None) -> Dict[str, Any]:
    default_enabled = not bool(base_url)
    raw_enabled = os.getenv(
        "INTENT_EMBEDDING_ENABLED",
        "true" if default_enabled else "false",
    ).strip().lower()
    if raw_enabled not in {"true", "false"}:
        raise RuntimeError("INTENT_EMBEDDING_ENABLED must be true or false")
    enabled = raw_enabled == "true"
    default_weights = (0.7, 0.2, 0.1) if enabled else (0.85, 0.0, 0.15)
    return {
        "embedding_enabled": enabled,
        "strategy_weights": {
            "llm": float(os.getenv("INTENT_LLM_WEIGHT", str(default_weights[0]))),
            "embedding": float(os.getenv("INTENT_EMBEDDING_WEIGHT", str(default_weights[1]))),
            "pattern": float(os.getenv("INTENT_PATTERN_WEIGHT", str(default_weights[2]))),
        },
        "confidence_threshold": float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.5")),
        "multi_label_threshold": float(os.getenv("INTENT_MULTI_LABEL_THRESHOLD", "0.6")),
    }


def _build_reranker():
    from mcp.reranker import DEFAULT_RERANK_MODEL, FastEmbedCrossEncoderReranker

    provider = os.getenv("RERANK_PROVIDER", "fastembed").strip().lower()
    if provider in {"disabled", "none", "off"}:
        return None
    if provider != "fastembed":
        raise RuntimeError(f"不支持的 RERANK_PROVIDER: {provider}")
    return FastEmbedCrossEncoderReranker(
        model_name=(
            os.getenv("RERANK_MODEL", DEFAULT_RERANK_MODEL).strip()
            or DEFAULT_RERANK_MODEL
        ),
        cache_dir=(
            os.getenv("RERANK_CACHE_DIR", "./data/reranker-cache").strip()
            or None
        ),
    )


def _campus_database_path() -> pathlib.Path:
    configured = os.getenv("CAMPUS_DB_PATH")
    if configured:
        return pathlib.Path(configured)
    return pathlib.Path(_ROOT) / "data" / "campus" / "campus.db"


def _build_runtime_orchestrator(
    *,
    cfg: Dict[str, Any],
    tool_manager: Any,
    campus_store: Any,
    skill_manager: Any,
    intent_recognizer: Any,
):
    """Wire one shared campus store through tools, runtime, and agents."""
    from agents.agent_orchestrator import AgentOrchestrator
    from agents.agent_runtime import AgentRuntime
    from campus.tools import CampusToolset, register_campus_tools
    from core.evidence_verifier import EvidenceVerifier

    register_campus_tools(tool_manager, CampusToolset(campus_store))
    runtime = AgentRuntime(
        tool_manager._client,
        cfg["model"],
        tool_manager,
        verifier=EvidenceVerifier(),
        max_reflections=int(os.getenv("EVIDENCE_MAX_REFLECTIONS", "1")),
    )
    return AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=skill_manager,
        intent_recognizer=intent_recognizer,
        runtime=runtime,
    )


def _build_end_to_end_evaluator(
    *,
    chat_service,
    recognizer,
    cfg: Dict[str, Any],
    baseline_path: Optional[str],
):
    """Build evaluation on the exact ChatService used by ``/chat``."""
    from anthropic import AsyncAnthropic

    from evaluation.evaluator import EndToEndEvaluator
    from evaluation.ragas_evaluator import RagasEvaluator

    client_kwargs: Dict[str, Any] = {"api_key": cfg["api_key"]}
    if cfg.get("base_url"):
        client_kwargs["base_url"] = cfg["base_url"]
    ragas_llm_options: Dict[str, Any] = {}
    if "api.deepseek.com" in str(cfg.get("base_url", "")).lower():
        # RAGAS uses forced structured output (tool_choice).  DeepSeek V4
        # rejects that combination while thinking mode is enabled, so keep
        # reasoning on for Agents but disable it only for the evaluation Judge.
        ragas_llm_options = {
            "extra_body": {"thinking": {"type": "disabled"}},
        }
    ragas_evaluator = RagasEvaluator(
        client=AsyncAnthropic(**client_kwargs),
        model=cfg["model"],
        embedding_model=os.getenv(
            "RAGAS_EMBEDDING_MODEL",
            "BAAI/bge-small-zh-v1.5",
        ).strip() or "BAAI/bge-small-zh-v1.5",
        embedding_cache_dir=(
            os.getenv("EMBEDDING_CACHE_DIR", "").strip() or None
        ),
        llm_options=ragas_llm_options,
    )

    return EndToEndEvaluator(
        chat_service=chat_service,
        recognizer=recognizer,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        baseline_path=baseline_path,
        ragas_evaluator=ragas_evaluator,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _skill_manager
    global _intent_recognizer, _intent_feedback_store, _chat_service, _campus_store

    print(BANNER, flush=True)

    from core.intent_feedback_store import RedisIntentFeedbackStore
    from core.intent_recognizer import IntentRecognizer
    from core.retrieval_policy import RetrievalPolicy
    from mcp.knowledge_base import KnowledgeBase
    from mcp.tool_manager import MCPToolManager, Tool
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor
    from core.skill_loader import SkillManager
    from services.chat_service import ChatService
    from campus.store import CampusStore

    cfg = _anthropic_cfg()
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")

    _campus_store = CampusStore(_campus_database_path())

    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    _intent_feedback_store = RedisIntentFeedbackStore(redis_url=redis_url)
    _intent_recognizer = IntentRecognizer(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        feedback_store=_intent_feedback_store,
        **_intent_fusion_config(cfg.get("base_url")),
    )

    # Skills：启动时从目录加载业务能力说明，并在 Agent 调用 LLM 时动态注入。
    skills_dir = os.getenv("ECHOMIND_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills"))
    _skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=int(os.getenv("ECHOMIND_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    _skill_manager.load()

    # 记忆管理器（Redis 工作记忆 + ChromaDB 情景记忆/用户画像）
    from mcp.local_embeddings import DEFAULT_EMBEDDING_MODEL, FastEmbedTextModel

    embedding_model = FastEmbedTextModel(
        model_name=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
        or DEFAULT_EMBEDDING_MODEL,
        cache_dir=os.getenv("EMBEDDING_CACHE_DIR", "/app/data/fastembed-cache").strip()
        or None,
    )
    _memory = MemoryManager(
        redis_url=redis_url,
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # MCP 工具管理器 + RAG 知识库（基于 ChromaDB 的真实检索）
    _tool_manager = MCPToolManager(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        reranker=_build_reranker(),
    )
    kb = KnowledgeBase(
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        embedder=embedding_model,
        bootstrap_defaults=False,
    )
    logger.info(f"知识库已加载: {kb.doc_count} 个文档片段")

    def knowledge_fallback(params: Dict[str, Any], context: Optional[Dict[str, Any]], error: str):
        query = params.get("query", "")
        return [{
            "title": "知识库降级结果",
            "content": f"知识库暂时不可用，未能完成对“{query}”的语义检索。请稍后重试，或转人工客服确认。",
            "score": 0.0,
            "fallback": True,
            "error": error,
        }]

    _tool_manager.register(Tool(
        name="knowledge_search",
        description="搜索知识库（基于 ChromaDB 向量检索）",
        handler=kb.search_handler,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        cache_ttl=300.0,
        supports_rerank=True,
        fallback=knowledge_fallback,
    ))

    # Campus tools and AgentRuntime share this manager and the API's one store.
    _orchestrator = _build_runtime_orchestrator(
        cfg=cfg,
        tool_manager=_tool_manager,
        campus_store=_campus_store,
        skill_manager=_skill_manager,
        intent_recognizer=_intent_recognizer,
    )

    # 性能监控（可选启动 Prometheus）
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    _monitor = PerformanceMonitor(
        orchestrator=_orchestrator,
        tool_manager=_tool_manager,
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        prometheus_port=prom_port,
    )
    await _monitor.start()

    _chat_service = ChatService(
        memory=_memory,
        intent_recognizer=_intent_recognizer,
        retrieval_policy=RetrievalPolicy(),
        orchestrator=_orchestrator,
        knowledge_search=_tool_manager,
        monitor=_monitor,
    )

    # 评测器
    _evaluator = _build_end_to_end_evaluator(
        chat_service=_chat_service,
        recognizer=_intent_recognizer,
        cfg=cfg,
        baseline_path=os.getenv("EVAL_BASELINE_PATH", "/app/data/eval/baseline.json"),
    )

    logger.info("EchoMind 已就绪")
    try:
        yield
    finally:
        if _chat_service is not None:
            await _chat_service.aclose()
        await _monitor.stop()
        logger.info("EchoMind 已关闭")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EchoMind 智能客服",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    JWTAuthMiddleware,
    settings=JWTAuthSettings.from_env(),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def safe_ticket_validation_error(
    request: Request,
    error: RequestValidationError,
):
    """Keep rejected ticket payload values out of validation responses."""
    path = request.url.path
    if path.startswith("/tickets/") and path.endswith("/status"):
        return JSONResponse(
            status_code=422,
            content={"detail": "Invalid ticket status payload"},
        )
    return await request_validation_exception_handler(request, error)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:     str = Field(min_length=1, max_length=8000)
    user_id:     str = Field(default="anonymous", min_length=1, max_length=128)
    conv_id:     Optional[str] = Field(default=None, min_length=1, max_length=128)


class ToolTraceOutput(BaseModel):
    """Public, bounded subset of an internal tool execution trace."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    success: bool
    latency_ms: float = Field(default=0.0, ge=0.0, le=3_600_000.0)
    cached: bool = False
    error_type: Optional[str] = Field(default=None, min_length=1, max_length=128)


class EvidenceVerificationOutput(BaseModel):
    """Bounded public summary of deterministic grounding checks."""

    model_config = ConfigDict(extra="forbid")

    checked: bool
    passed: bool
    checked_claims: int = Field(ge=0, le=10_000)
    issue_codes: List[str] = Field(default_factory=list, max_length=16)
    abstained: bool = False
    reflection_count: int = Field(default=0, ge=0, le=6)
    corrected: bool = False
    safe_fallback_used: bool = False


class ChatResponse(BaseModel):
    conv_id:     str
    response:    str
    intent:      str
    agent_type:  str
    escalated:   bool
    latency_ms:  float
    knowledge_used: bool = False
    intent_scores: Dict[str, float] = Field(default_factory=dict)
    matched_intents: List[str] = Field(default_factory=list)
    agent_types: List[str] = Field(default_factory=list)
    trace_id: Optional[str] = None
    tool_calls: List[ToolTraceOutput] = Field(default_factory=list)
    ticket_ids: List[str] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_verification: Dict[str, Any] = Field(default_factory=dict)

    @model_serializer(mode="wrap")
    def _serialize_backward_compatibly(self, handler):
        payload = handler(self)
        for field_name in (
            "tool_calls", "ticket_ids", "citations", "evidence_verification",
        ):
            if not getattr(self, field_name):
                payload.pop(field_name, None)
        if self.trace_id is None:
            payload.pop("trace_id", None)
        return payload


class TicketStatusInput(BaseModel):
    """Strict status-only payload for an administrator transition."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["OPEN", "PROCESSING", "RESOLVED"]


class TicketOwnerOutput(BaseModel):
    """Ticket fields safe to return to the authenticated owner."""

    model_config = ConfigDict(extra="ignore")

    id: str
    user_id: str
    category: str
    title: str
    description: str
    status: Literal["OPEN", "PROCESSING", "RESOLVED"]
    created_at: str
    updated_at: str


class TicketStatusOutput(BaseModel):
    """Transition result without owner-only ticket content."""

    model_config = ConfigDict(extra="ignore")

    id: str
    status: Literal["OPEN", "PROCESSING", "RESOLVED"]
    updated_at: str


class IntentFeedbackInput(BaseModel):
    """Human-reviewed correction submitted through the protected admin API."""

    message: str = Field(min_length=1, max_length=500)
    correct_intent: str


def resolve_chat_principal(request: Request) -> Optional[str]:
    """Read identity only from trusted state populated by server middleware."""
    for attribute in ("principal_id", "user_id"):
        principal_id = getattr(request.state, attribute, None)
        if (
            isinstance(principal_id, str)
            and principal_id
            and principal_id == principal_id.strip()
            and len(principal_id) <= 128
            and principal_id.isprintable()
        ):
            return principal_id
    return None


def _require_admin_token(x_intent_admin_token: Optional[str]) -> None:
    """Apply the shared constant-time admin-token contract."""
    configured_token = os.getenv("INTENT_FEEDBACK_ADMIN_TOKEN", "")
    if not configured_token:
        raise HTTPException(503, "管理接口未启用")
    if not isinstance(x_intent_admin_token, str) or not hmac.compare_digest(
        x_intent_admin_token.encode("utf-8"),
        configured_token.encode("utf-8"),
    ):
        raise HTTPException(403, "管理员令牌无效")


def require_admin_token(
    x_intent_admin_token: Optional[str] = Header(
        default=None,
        alias="X-Intent-Admin-Token",
    ),
) -> str:
    """FastAPI dependency for every protected administrative endpoint."""
    _require_admin_token(x_intent_admin_token)
    return x_intent_admin_token


def _require_campus_store():
    if _campus_store is None:
        raise HTTPException(503, "工单存储未初始化")
    return _campus_store


def _safe_public_text(
    value: Any,
    maximum: int,
    *,
    allow_layout: bool = False,
) -> Optional[str]:
    from agents.agent_runtime import sanitize_text_content

    if not isinstance(value, str):
        return None
    sanitized = sanitize_text_content(value, maximum)
    if (
        not sanitized
        or sanitized != sanitized.strip()
        or (not allow_layout and any(char in sanitized for char in "\n\t"))
    ):
        return None
    return sanitized


def _public_trace_id(value: Any) -> Optional[str]:
    trace_id = _safe_public_text(value, 32)
    if (
        trace_id is None
        or len(trace_id) != 32
        or any(character not in "0123456789abcdef" for character in trace_id)
    ):
        return None
    return trace_id


def _result_attribute(result: Any, name: str, default: Any) -> Any:
    try:
        return getattr(result, name, default)
    except Exception:
        return default


def _public_tool_calls(value: Any) -> List[ToolTraceOutput]:
    if not isinstance(value, (list, tuple)):
        return []
    traces: List[ToolTraceOutput] = []
    for item in value[:64]:
        if not isinstance(item, dict):
            continue
        name = _safe_public_text(item.get("name"), 128)
        success = item.get("success")
        latency = item.get("latency_ms", 0.0)
        cached = item.get("cached", False)
        error_type = item.get("error_type")
        if (
            name is None
            or type(success) is not bool
            or isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or not 0 <= float(latency) <= 3_600_000
            or type(cached) is not bool
        ):
            continue
        if error_type is not None:
            error_type = _safe_public_text(error_type, 128)
            if error_type is None:
                continue
        traces.append(
            ToolTraceOutput(
                name=name,
                success=success,
                latency_ms=float(latency),
                cached=cached,
                error_type=error_type,
            )
        )
    return traces


def _public_ticket_ids(value: Any) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return []
    ticket_ids: List[str] = []
    seen = set()
    for item in value[:128]:
        ticket_id = _safe_public_text(item, 128)
        if ticket_id is not None and ticket_id not in seen:
            seen.add(ticket_id)
            ticket_ids.append(ticket_id)
    return ticket_ids


def _public_citations(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    citations: List[Dict[str, Any]] = []
    for item in value[:16]:
        if not isinstance(item, dict):
            continue
        citation_id = _safe_public_text(item.get("id"), 128)
        title = _safe_public_text(
            item.get("title"),
            500,
            allow_layout=True,
        )
        content = _safe_public_text(
            item.get("content"),
            4_000,
            allow_layout=True,
        )
        score = item.get("score")
        if (
            citation_id is None
            or title is None
            or content is None
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            continue
        citations.append(
            {
                "id": citation_id,
                "title": title,
                "content": content,
                "score": float(score),
            }
        )
    return citations


def _public_evidence_verification(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict) or value.get("checked") is not True:
        return {}
    try:
        model = EvidenceVerificationOutput(**value)
    except Exception:
        return {}
    payload = model.model_dump() if hasattr(model, "model_dump") else model.dict()
    safe_codes = []
    for code in payload.get("issue_codes", []):
        cleaned = _safe_public_text(code, 64)
        if (
            cleaned is not None
            and cleaned.replace("_", "").isalnum()
            and cleaned not in safe_codes
        ):
            safe_codes.append(cleaned)
    payload["issue_codes"] = safe_codes
    return payload


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {"status": "ok", "agents": _orchestrator.get_stats()}


@app.get("/skills", tags=["Skills"])
async def skills_summary():
    """查看当前已加载的 Skills，便于确认热加载结果和排查解析错误。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return _skill_manager.summary()


@app.post("/skills/reload", tags=["Skills"])
async def reload_skills():
    """运行时重新扫描 Skill 目录，不需要重启服务。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    _skill_manager.reload()
    if _orchestrator is not None:
        _orchestrator.set_skill_manager(_skill_manager)
    return _skill_manager.summary()


@app.get("/tickets/{ticket_id}", response_model=TicketOwnerOutput, tags=["Tickets"])
async def get_ticket(
    ticket_id: str,
    request: Request,
    user_id: Optional[str] = None,
):
    """Return an owner-scoped ticket using only server-authenticated identity."""
    principal_id = resolve_chat_principal(request)
    if principal_id is None:
        raise HTTPException(401, "身份认证信息缺失")
    if user_id is not None and user_id != principal_id:
        raise HTTPException(403, "请求身份与认证身份不匹配")

    store = _require_campus_store()
    ticket = await asyncio.to_thread(
        store.get_ticket,
        ticket_id,
        user_id=principal_id,
    )
    if ticket is None:
        raise HTTPException(404, "工单不存在")
    return TicketOwnerOutput.model_validate(ticket)


@app.patch(
    "/tickets/{ticket_id}/status",
    response_model=TicketStatusOutput,
    tags=["Tickets"],
)
async def update_ticket_status(
    ticket_id: str,
    body: TicketStatusInput,
    x_intent_admin_token: Optional[str] = Depends(require_admin_token),
):
    """Apply one protected ticket status transition."""
    _require_admin_token(x_intent_admin_token)
    store = _require_campus_store()
    try:
        ticket = await asyncio.to_thread(
            store.update_ticket_status,
            ticket_id,
            body.status,
        )
    except ValueError:
        raise HTTPException(409, "工单状态转换冲突") from None
    if ticket is None:
        raise HTTPException(404, "工单不存在")
    return TicketStatusOutput.model_validate(ticket)


@app.post("/intent/feedback", tags=["意图识别"])
async def submit_intent_feedback(
    body: IntentFeedbackInput,
    x_intent_admin_token: Optional[str] = Depends(require_admin_token),
):
    """Persist one human-reviewed intent correction and refresh templates."""
    _require_admin_token(x_intent_admin_token)
    if _intent_recognizer is None:
        raise HTTPException(503, "意图识别器未初始化")

    from core.intent_recognizer import IntentCategory

    try:
        correct = IntentCategory(body.correct_intent.strip().lower())
    except ValueError:
        raise HTTPException(400, f"未知意图标签: {body.correct_intent}") from None

    message = body.message.strip()
    if not message:
        raise HTTPException(400, "反馈消息不能为空")
    try:
        changed = _intent_recognizer.learn(message, correct)
    except Exception as ex:
        logger.exception("保存人工意图反馈失败")
        raise HTTPException(503, "意图反馈持久化失败") from ex

    return {
        "message": message,
        "correct_intent": correct.value,
        "changed": changed,
        "template_fingerprint": _intent_recognizer.template_fingerprint,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    request: Request,
    idempotency_key: Optional[str] = Header(
        default=None,
        alias="Idempotency-Key",
    ),
):
    """
    主对话接口。完整流程：
      记忆读取 → 意图识别 → Agent 路由 → 执行 → 记忆写入
    """
    if _chat_service is None:
        raise HTTPException(503, "服务未就绪")

    from services.chat_service import (
        ChatCommand,
        ChatIdempotencyConflict,
        ChatOperationInProgress,
        ChatPipelineError,
    )

    trusted_principal = resolve_chat_principal(request)
    request_id = (
        idempotency_key
        if trusted_principal is not None
        and isinstance(idempotency_key, str)
        else None
    )
    try:
        result = await _chat_service.chat(
            ChatCommand(
                message=req.message,
                user_id=req.user_id,
                conv_id=req.conv_id,
                principal_id=trusted_principal,
                request_id=request_id,
            )
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from None
    except ChatIdempotencyConflict:
        raise HTTPException(409, "聊天请求与幂等键冲突") from None
    except ChatOperationInProgress:
        raise HTTPException(
            409,
            "聊天请求仍在处理中，请稍后重试",
        ) from None
    except ChatPipelineError:
        raise HTTPException(503, "聊天服务暂时不可用") from None

    return ChatResponse(
        conv_id=result.conv_id,
        response=result.response,
        intent=result.intent,
        agent_type=result.agent_type,
        escalated=result.escalated,
        latency_ms=result.latency_ms,
        knowledge_used=result.knowledge_used,
        intent_scores=dict(result.intent_scores),
        matched_intents=list(result.matched_intents),
        agent_types=list(result.agent_types),
        trace_id=_public_trace_id(_result_attribute(result, "trace_id", None)),
        tool_calls=_public_tool_calls(
            _result_attribute(result, "tool_calls", [])
        ),
        ticket_ids=_public_ticket_ids(
            _result_attribute(result, "ticket_ids", [])
        ),
        citations=_public_citations(
            _result_attribute(result, "citations", [])
        ),
        evidence_verification=_public_evidence_verification(
            _result_attribute(result, "evidence_verification", {})
        ),
    )


@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标入口。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search")
async def search(query: str, top_k: int = 5):
    """
    演示检索优化链路：查询改写 → 并行召回 → 重排 → Top-K。
    展示 MCP 工具调用的核心亮点。
    """
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    result = await _tool_manager.search_with_rewrite("knowledge_search", query, top_k=top_k)
    return {"query": query, "results": result.data, "reranked": result.reranked}


class DocInput(BaseModel):
    """单篇文档输入。"""
    title:   str
    content: str


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput]


class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""
    message: str
    expected_intent: Optional[str] = None
    expected_intents: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def require_expected_label(self):
        labels = [
            label
            for label in (self.expected_intents or [])
            if isinstance(label, str) and label.strip()
        ]
        has_primary = (
            isinstance(self.expected_intent, str)
            and bool(self.expected_intent.strip())
        )
        if not has_primary and not labels:
            raise ValueError(
                "expected_intent or expected_intents is required"
            )
        return self


class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮。"""
    question: Optional[str] = None
    turns: Optional[List[str]] = None
    user_id: Optional[str] = None
    conv_id: Optional[str] = None
    expected_intents: Optional[List[str]] = None
    expected_agents: Optional[List[str]] = None
    expected_tools: Optional[List[str]] = None
    expect_ticket: Optional[bool] = None
    expect_knowledge: Optional[bool] = None
    reference_answer: Optional[str] = None
    reference_answers: Optional[List[str]] = None
    reference_context_ids: Optional[List[str]] = None
    reference_context_relevance: Optional[Dict[str, float]] = None


class EvalRunInput(BaseModel):
    """评测请求。为空时使用内置默认用例。"""
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
    """
    批量导入文档到知识库。

    文档会自动切片（每片 500 字）并存入 ChromaDB，ChromaDB 内置 Embedding 模型自动向量化。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "退款政策", "content": "用户在购买后 7 天内可以申请无理由退款..."},
        {"title": "配送说明", "content": "标准配送 3-5 个工作日..."}
      ]
    }
    ```
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    count = kb.add_documents([{"title": d.title, "content": d.content} for d in body.documents])
    return {"message": f"成功导入 {count} 个文档片段", "added_chunks": count, "total_chunks": kb.doc_count}


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...)):
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md`：整个文件作为一篇文档，文件名作为标题
    - `.json`：JSON 数组格式 `[{"title": "...", "content": "..."}, ...]`

    文件大小限制：10MB
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"

    if filename.endswith(".json"):
        import json as _json
        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON 文件应为数组格式: [{title, content}, ...]")
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
    else:
        # txt / md：整个文件作为一篇文档
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        docs = [{"title": title, "content": text}]

    count = kb.add_documents(docs)
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": kb.doc_count,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    return {"total_chunks": kb.doc_count}


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):
    """运行内置评测用例，返回评测报告。"""
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES, IntentTestCase

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=c.message,
                expected_intent=(
                    c.expected_intent
                    or (c.expected_intents or [""])[0]
                ),
                context=c.context,
                expected_intents=c.expected_intents,
            )
            for c in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [
            c.model_dump(exclude_none=True)
            for c in body.dialog_cases
        ]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
    )
    return {
        "pass_rate":       report.pass_rate,
        "total":           report.total,
        "passed":          report.passed,
        "avg_scores":      report.avg_scores,
        "metrics":         report.metrics,
        "regressions":     report.regressions,
        "recommendations": report.recommendations,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }


# ── 交互式 CLI ────────────────────────────────────────────────────────────────
async def _cli():
    print(BANNER)
    print("EchoMind CLI — 输入 quit 退出\n")

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from memory.conversation_memory import MemoryManager, MsgRole
    from core.skill_loader import SkillManager

    cfg = _anthropic_cfg()
    skill_manager = SkillManager(
        root_dir=os.getenv("ECHOMIND_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills")),
        max_prompt_chars=int(os.getenv("ECHOMIND_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    skill_manager.load()
    orch = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=skill_manager,
    )
    mem  = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/tmp/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    user_id, conv_id = "cli_user", str(uuid.uuid4())

    while True:
        try:
            msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 ʕ•ᴥ•ʔ")
            break
        if not msg or msg.lower() in ("quit", "exit", "退出"):
            print("再见 ʕ•ᴥ•ʔ")
            break

        ctx = await mem.get_context(user_id, conv_id, query=msg)
        history = [
            {"role": m.role.value, "content": m.content}
            for m in ctx.recent_messages[-5:]
        ] if ctx.recent_messages else None
        req = Request(message=msg, user_id=user_id, conv_id=conv_id, context=ctx.to_prompt_text(), history=history)
        result = await orch.run(req)

        await mem.add_message(user_id, conv_id, MsgRole.USER, msg)
        await mem.add_message(user_id, conv_id, MsgRole.ASSISTANT, result.response)

        print(f"\nEchoMind [{result.agent_type.value}]: {result.response}\n")


if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(_cli())
    else:
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("APP_ENV") == "development",
        )
