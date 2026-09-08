"""Bounded Anthropic-compatible Agent tool-calling runtime."""

import asyncio
import hashlib
import inspect
import json
import math
import re
import secrets
import unicodedata
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from core.llm_utils import content_blocks, extract_text_content


MAX_STEPS_RESPONSE = (
    "I couldn't complete the request within the tool-call step limit."
)
TIMEOUT_RESPONSE = (
    "I couldn't complete the request before the runtime deadline."
)
PROTOCOL_ERROR_RESPONSE = (
    "The model returned an invalid tool-call protocol response."
)
RUNTIME_ERROR_RESPONSE = (
    "The assistant service is temporarily unavailable."
)
UNGROUNDED_RESPONSE = (
    "我无法根据当前证据可靠确认该信息，请稍后重试或联系人工服务。"
)
FORBIDDEN_TOOL_MESSAGE = "Requested tool is not allowed."
MALFORMED_TOOL_MESSAGE = "Tool request was malformed."
REJECTED_TOOL_MESSAGE = "Tool request was rejected."
FAILED_TOOL_MESSAGE = "Tool execution failed."
UNSAFE_RESULT_MESSAGE = "Tool result could not be safely processed."
_REDACTED = "[REDACTED]"
_MAX_ALLOWED_TOOLS = 32
_MAX_TOOL_NAME_LENGTH = 64
_MAX_TOOL_USE_ID_LENGTH = 128
_MAX_TOOL_USES_PER_STEP = 16
_MAX_ASSISTANT_TEXT_LENGTH = 8192
_MAX_TOOL_INPUT_DEPTH = 6
_MAX_TOOL_INPUT_ITEMS = 50
_MAX_TOOL_INPUT_NODES = 256
_MAX_TOOL_INPUT_STRING_LENGTH = 2048
_MAX_TOOL_INPUT_JSON_BYTES = 16384
_MAX_TRACE_PROPERTIES = 32
_MAX_PROPERTY_NAME_LENGTH = 64
_MAX_RESULT_DEPTH = 6
_MAX_RESULT_ITEMS = 50
_MAX_RESULT_NODES = 256
_MAX_RESULT_STRING_LENGTH = 2048
_MAX_RESULT_JSON_BYTES = 16384
_MAX_TICKET_IDS = 128
# About 1,233 decimal digits: below CPython's default conversion ceiling
# and comfortably inside the serialized result budget.
_MAX_SAFE_COUNT_BITS = 4096
_TOOL_NAME_PATTERN = re.compile(
    rf"^[A-Za-z0-9_-]{{1,{_MAX_TOOL_NAME_LENGTH}}}$"
)
_SENSITIVE_RESULT_KEYS = frozenset({
    "access_key",
    "access_token",
    "api_key",
    "auth_header",
    "auth_token",
    "authorization",
    "backend_error",
    "client_secret",
    "context",
    "credential",
    "credentials",
    "error",
    "error_message",
    "exception",
    "exception_message",
    "id_token",
    "idempotency",
    "idempotency_key",
    "password",
    "passwd",
    "private_key",
    "pwd",
    "raw_error",
    "refresh_token",
    "request_context",
    "secret_key",
    "session_cookie",
    "session_token",
    "set_cookie",
    "stack",
    "stack_trace",
    "secret",
    "token",
    "traceback",
    "trusted_context",
    "user_id",
    "cookie",
})
_SAFE_RESULT_KEYS = frozenset({
    "authorization_status",
    "backend_status",
    "error_code",
    "retry_with_same_idempotency_key",
    "token_count",
})
_SAFE_OPERATIONAL_STATUSES = frozenset({
    "active",
    "available",
    "closed",
    "configured",
    "connected",
    "disabled",
    "disconnected",
    "enabled",
    "error",
    "failed",
    "healthy",
    "inactive",
    "invalid",
    "missing",
    "open",
    "pending",
    "present",
    "processing",
    "required",
    "resolved",
    "success",
    "unavailable",
    "unhealthy",
    "unknown",
    "valid",
})
_AUTHORIZATION_STATUSES = frozenset({
    "allowed",
    "approved",
    "authorized",
    "denied",
    "forbidden",
    "rejected",
    "unauthorized",
})
_SAFE_ERROR_CODES = frozenset({
    "CIRCUIT_OPEN",
    "CONFLICT",
    "FORBIDDEN",
    "INTERNAL_ERROR",
    "NOT_FOUND",
    "PROTOCOL_ERROR",
    "RATE_LIMITED",
    "SERVICE_UNAVAILABLE",
    "TIMEOUT",
    "TOOL_FAILED",
    "TOOL_REJECTED",
    "UNAUTHORIZED",
    "VALIDATION_ERROR",
})
_DEFAULT_IGNORABLE_RANGES = (
    (0x00AD, 0x00AD),
    (0x034F, 0x034F),
    (0x061C, 0x061C),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180F),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x206F),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0),
    (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)


def sanitize_text_content(value: Any, max_chars: int = 16384) -> str:
    """Keep display text Unicode-safe while preserving layout whitespace."""
    if not isinstance(value, str) or max_chars < 1:
        return ""
    safe = []
    for character in value[:max_chars]:
        if character in {"\n", "\t"}:
            safe.append(character)
            continue
        codepoint = ord(character)
        if (
            unicodedata.category(character) in {"Cc", "Cf", "Cs"}
            or any(
                start <= codepoint <= end
                for start, end in _DEFAULT_IGNORABLE_RANGES
            )
        ):
            continue
        safe.append(character)
    return "".join(safe)


class _UnsafeToolResult(ValueError):
    pass


class _UnsafeToolInput(ValueError):
    pass


@dataclass(frozen=True)
class RuntimeToolCall:
    """A safe tool trace that excludes trusted context and raw error details."""

    name: str
    success: bool
    fingerprint: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    cached: bool = False
    error_type: Optional[str] = None


@dataclass
class RuntimeResult:
    """Final runtime output and bounded tool trace."""

    text: str
    tool_calls: List[RuntimeToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    ticket_ids: List[str] = field(default_factory=list)
    verification: Optional[Any] = None
    reflection_count: int = 0
    corrected: bool = False
    safe_fallback_used: bool = False


class AgentRuntime:
    """Run a bounded model/tool loop using stable Anthropic message shapes."""

    def __init__(
        self,
        client: Any,
        model: str,
        manager: Any,
        *,
        max_steps: int = 4,
        total_timeout_s: float = 30.0,
        timeout_s: Optional[float] = None,
        max_tokens: int = 1024,
        verifier: Optional[Any] = None,
        max_reflections: int = 1,
    ):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if timeout_s is not None:
            total_timeout_s = timeout_s
        if total_timeout_s <= 0:
            raise ValueError("total_timeout_s must be positive")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if max_reflections < 0 or max_reflections > 2:
            raise ValueError("max_reflections must be between 0 and 2")

        self._client = client
        self._model = model
        self._manager = manager
        self._max_steps = max_steps
        self._total_timeout_s = total_timeout_s
        self._max_tokens = max_tokens
        self._verifier = verifier
        self._max_reflections = max_reflections

    async def run(
        self,
        *,
        system_prompt: str,
        messages: Sequence[Mapping[str, Any]],
        allowed_tools: Sequence[str],
        context: Optional[Mapping[str, Any]] = None,
        question: Optional[str] = None,
        citations: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> RuntimeResult:
        """Execute allowed tools until text, max steps, or the total deadline."""
        traces: List[RuntimeToolCall] = []
        ticket_ids: List[str] = []
        tool_evidence: List[Dict[str, Any]] = []
        reflection_count = 0
        first_verification_failed = False
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._total_timeout_s

        try:
            fingerprint_scope = secrets.token_bytes(32)
            conversation = deepcopy(list(messages))
            trusted_context = (
                deepcopy(dict(context)) if context is not None else None
            )
            allowed = self._normalize_allowed_tools(allowed_tools)
            allowed_set = set(allowed)
            seen_tool_use_ids = set()

            for _ in range(self._max_steps):
                response, schema_properties = await self._model_response(
                    system_prompt=system_prompt,
                    messages=conversation,
                    allowed_tools=allowed,
                    deadline=deadline,
                )
                raw_content = self._response_content(response)
                try:
                    raw_blocks = list(raw_content or [])
                except TypeError:
                    raw_blocks = []
                    safe_protocol = None
                else:
                    blocks = content_blocks(raw_blocks)
                    safe_protocol = (
                        self._protocol_safe_blocks(
                            blocks,
                            seen_tool_use_ids,
                        )
                        if len(blocks) == len(raw_blocks)
                        else None
                    )
                if safe_protocol is None:
                    traces.append(
                        RuntimeToolCall(
                            name="invalid_tool_use",
                            success=False,
                            error_type="protocol_error",
                        )
                    )
                    return RuntimeResult(
                        text=PROTOCOL_ERROR_RESPONSE,
                        tool_calls=traces,
                        stop_reason="protocol_error",
                        ticket_ids=list(ticket_ids),
                    )
                safe_blocks, tool_uses = safe_protocol
                if not tool_uses:
                    draft = sanitize_text_content(
                        extract_text_content(raw_blocks),
                        _MAX_ASSISTANT_TEXT_LENGTH,
                    )
                    if self._verifier is None:
                        return RuntimeResult(
                            text=draft,
                            tool_calls=traces,
                            ticket_ids=list(ticket_ids),
                        )
                    report = self._verifier.verify(
                        question=question or self._last_user_text(conversation),
                        response=draft,
                        citations=list(citations or ()),
                        tool_evidence=tool_evidence,
                    )
                    if report.passed:
                        return RuntimeResult(
                            text=draft,
                            tool_calls=traces,
                            ticket_ids=list(ticket_ids),
                            verification=report,
                            reflection_count=reflection_count,
                            corrected=(
                                first_verification_failed
                                and reflection_count > 0
                            ),
                        )
                    first_verification_failed = True
                    if reflection_count < self._max_reflections:
                        conversation.append(
                            {"role": "assistant", "content": safe_blocks}
                        )
                        conversation.append({
                            "role": "user",
                            "content": self._reflection_prompt(report),
                        })
                        reflection_count += 1
                        continue
                    return RuntimeResult(
                        text=UNGROUNDED_RESPONSE,
                        tool_calls=traces,
                        ticket_ids=list(ticket_ids),
                        verification=report,
                        reflection_count=reflection_count,
                        safe_fallback_used=True,
                    )

                conversation.append(
                    {"role": "assistant", "content": safe_blocks}
                )
                tool_results = []
                for tool_use in tool_uses:
                    result_block, trace = await self._execute_tool_use(
                        tool_use=tool_use,
                        allowed_tools=allowed_set,
                        context=trusted_context,
                        trace_properties=schema_properties,
                        deadline=deadline,
                        ticket_ids=ticket_ids,
                        fingerprint_scope=fingerprint_scope,
                    )
                    tool_results.append(result_block)
                    traces.append(trace)
                    if trace.success:
                        evidence = self._tool_evidence(trace.name, result_block)
                        if evidence is not None:
                            tool_evidence.append(evidence)
                conversation.append({"role": "user", "content": tool_results})
                seen_tool_use_ids.update(
                    tool_use["id"] for tool_use in tool_uses
                )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            return RuntimeResult(
                text=TIMEOUT_RESPONSE,
                tool_calls=traces,
                stop_reason="timeout",
                ticket_ids=list(ticket_ids),
            )
        except Exception:
            return RuntimeResult(
                text=RUNTIME_ERROR_RESPONSE,
                tool_calls=traces,
                stop_reason="error",
                ticket_ids=list(ticket_ids),
            )

        return RuntimeResult(
            text=MAX_STEPS_RESPONSE,
            tool_calls=traces,
            stop_reason="max_steps",
            ticket_ids=list(ticket_ids),
        )

    @staticmethod
    def _last_user_text(messages: Sequence[Mapping[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content
        return ""

    @staticmethod
    def _reflection_prompt(report: Any) -> str:
        codes = [
            getattr(issue, "code", "unsupported_claim")
            for issue in getattr(report, "issues", ())
        ]
        compact_codes = ", ".join(dict.fromkeys(codes))[:512]
        return (
            "证据校验未通过（"
            + compact_codes
            + "）。请只依据已经提供的知识库引用和成功工具结果重写回答；"
              "不得编造编号、状态、数值或已执行动作，并保留正确的 Citation ID。"
        )

    @staticmethod
    def _tool_evidence(
        name: str,
        result_block: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        content = result_block.get("content")
        if not isinstance(content, str):
            return None
        try:
            payload = json.loads(content)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, Mapping) or payload.get("success") is not True:
            return None
        return {
            "name": name,
            "success": True,
            "data": deepcopy(payload.get("data")),
        }

    async def _model_response(
        self,
        *,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        allowed_tools: List[str],
        deadline: float,
    ) -> tuple[Any, Dict[str, set]]:
        raw_schemas = await self._invoke(
            self._manager.schemas,
            allowed_tools,
            deadline=deadline,
        )
        schemas, schema_properties = self._normalize_schemas(
            raw_schemas,
            allowed_tools,
        )
        response = await self._invoke(
            self._client.messages.create,
            deadline=deadline,
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=deepcopy(messages),
            tools=schemas,
        )
        return response, schema_properties

    async def _execute_tool_use(
        self,
        *,
        tool_use: Dict[str, Any],
        allowed_tools: set,
        context: Optional[Dict[str, Any]],
        trace_properties: Dict[str, set],
        deadline: float,
        ticket_ids: List[str],
        fingerprint_scope: bytes,
    ) -> tuple[Dict[str, Any], RuntimeToolCall]:
        tool_use_id = tool_use["id"]
        name = tool_use.get("name")
        raw_params = tool_use.get("input", {})
        fingerprint = (
            self._tool_call_fingerprint(
                fingerprint_scope,
                tool_use_id,
                name,
                raw_params,
            )
            if self._valid_tool_name(name) and isinstance(raw_params, dict)
            else ""
        )

        if (
            not self._valid_tool_name(name)
        ):
            return self._error_result(
                tool_use_id,
                "invalid_tool",
                {},
                MALFORMED_TOOL_MESSAGE,
                "malformed_tool_use",
                fingerprint,
            )
        if not isinstance(raw_params, dict):
            return self._error_result(
                tool_use_id,
                name if name in allowed_tools else "invalid_tool",
                {},
                MALFORMED_TOOL_MESSAGE,
                "malformed_tool_use",
                fingerprint,
            )
        if name not in allowed_tools:
            return self._error_result(
                tool_use_id,
                "forbidden_tool",
                {},
                FORBIDDEN_TOOL_MESSAGE,
                "forbidden_tool",
                fingerprint,
            )

        trace_params = self._trace_params(
            raw_params,
            trace_properties.get(name, set()),
        )
        try:
            manager_call = self._manager.call
            if not self._is_async_callable(manager_call):
                return self._error_result(
                    tool_use_id,
                    name,
                    trace_params,
                    FAILED_TOOL_MESSAGE,
                    "tool_failure",
                    fingerprint,
                )
            result = await self._invoke(
                manager_call,
                name,
                deepcopy(raw_params),
                self._operation_context(
                    context,
                    name,
                    raw_params,
                ),
                deadline=deadline,
            )
            success = bool(self._result_field(result, "success", False))
            rejected = bool(self._result_field(result, "rejected", False))
            latency_ms = self._safe_float(
                self._result_field(result, "latency_ms", 0.0)
            )
            cached = bool(self._result_field(result, "cached", False))
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            raise
        except Exception:
            return self._error_result(
                tool_use_id,
                name,
                trace_params,
                FAILED_TOOL_MESSAGE,
                "tool_failure",
                fingerprint,
            )

        if rejected:
            return (
                self._tool_result(
                    tool_use_id,
                    REJECTED_TOOL_MESSAGE,
                    is_error=True,
                ),
                RuntimeToolCall(
                    name=name,
                    success=False,
                    fingerprint=fingerprint,
                    params=trace_params,
                    latency_ms=latency_ms,
                    cached=cached,
                    error_type="tool_rejected",
                ),
            )
        if not success:
            return (
                self._tool_result(
                    tool_use_id,
                    FAILED_TOOL_MESSAGE,
                    is_error=True,
                ),
                RuntimeToolCall(
                    name=name,
                    success=False,
                    fingerprint=fingerprint,
                    params=trace_params,
                    latency_ms=latency_ms,
                    cached=cached,
                    error_type="tool_failure",
                ),
            )

        data = self._result_field(result, "data", None)
        try:
            content = self._serialize_tool_data(data)
        except Exception:
            return (
                self._tool_result(
                    tool_use_id,
                    UNSAFE_RESULT_MESSAGE,
                    is_error=True,
                ),
                RuntimeToolCall(
                    name=name,
                    success=False,
                    fingerprint=fingerprint,
                    params=trace_params,
                    latency_ms=latency_ms,
                    cached=cached,
                    error_type="unsafe_tool_result",
                ),
            )

        if not bool(self._result_field(result, "fallback_used", False)):
            for ticket_id in self._structured_ticket_ids(name, data):
                if (
                    len(ticket_ids) < _MAX_TICKET_IDS
                    and ticket_id not in ticket_ids
                ):
                    ticket_ids.append(ticket_id)
                if len(ticket_ids) >= _MAX_TICKET_IDS:
                    break

        return (
            self._tool_result(tool_use_id, content, is_error=False),
            RuntimeToolCall(
                name=name,
                success=True,
                fingerprint=fingerprint,
                params=trace_params,
                latency_ms=latency_ms,
                cached=cached,
            ),
        )

    @staticmethod
    def _structured_ticket_ids(name: str, data: Any) -> List[str]:
        if not isinstance(data, Mapping):
            return []
        candidates = [data.get("ticket_id")]
        if name in {"create_ticket", "get_ticket"}:
            candidates.append(data.get("id"))
        return list(dict.fromkeys(
            ticket_id
            for ticket_id in candidates
            if (
                isinstance(ticket_id, str)
                and ticket_id
                and len(ticket_id) <= 128
                and ticket_id.isprintable()
                and not any(
                    character.isspace() for character in ticket_id
                )
            )
        ))

    @staticmethod
    def _tool_call_fingerprint(
        fingerprint_scope: bytes,
        tool_use_id: str,
        name: str,
        params: Mapping[str, Any],
    ) -> str:
        canonical = json.dumps(
            [tool_use_id, name, params],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(
            fingerprint_scope + b"\x00" + canonical
        ).hexdigest()

    @staticmethod
    def _operation_context(
        context: Optional[Mapping[str, Any]],
        name: str,
        params: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if context is None:
            return None
        forwarded = {
            key: deepcopy(value)
            for key, value in context.items()
            if key not in {"idempotency_key", "idempotency_scope"}
        }
        scope = context.get("idempotency_scope")
        if (
            not isinstance(scope, str)
            or not scope
            or len(scope) > 128
            or not scope.isprintable()
        ):
            return forwarded
        try:
            canonical = json.dumps(
                [scope, name, params],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            return forwarded
        forwarded["idempotency_key"] = (
            f"op_{hashlib.sha256(canonical).hexdigest()}"
        )
        return forwarded

    @classmethod
    async def _invoke(
        cls,
        func: Any,
        *args: Any,
        deadline: float,
        **kwargs: Any,
    ) -> Any:
        """Invoke async or sync callables without blocking the event loop."""
        remaining = cls._remaining(deadline)

        async def invoke() -> Any:
            if cls._is_async_callable(func):
                value = func(*args, **kwargs)
            else:
                value = await asyncio.to_thread(func, *args, **kwargs)
            if inspect.isawaitable(value):
                return await value
            return value

        return await asyncio.wait_for(invoke(), timeout=remaining)

    @staticmethod
    def _is_async_callable(func: Any) -> bool:
        return inspect.iscoroutinefunction(func) or inspect.iscoroutinefunction(
            getattr(func, "__call__", None)
        )

    @staticmethod
    def _response_content(response: Any) -> Any:
        if isinstance(response, Mapping):
            return response.get("content", [])
        return getattr(response, "content", [])

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError
        return remaining

    @classmethod
    def _normalize_allowed_tools(
        cls,
        allowed_tools: Sequence[str],
    ) -> List[str]:
        allowed: List[str] = []
        seen = set()
        for name in allowed_tools:
            if (
                isinstance(name, str)
                and cls._valid_tool_name(name)
                and name not in seen
            ):
                allowed.append(name)
                seen.add(name)
            if len(allowed) >= _MAX_ALLOWED_TOOLS:
                break
        return allowed

    @classmethod
    def _normalize_schemas(
        cls,
        raw_schemas: Any,
        allowed_tools: Sequence[str],
    ) -> tuple[List[Dict[str, Any]], Dict[str, set]]:
        if not isinstance(raw_schemas, (list, tuple)):
            raise ValueError("Invalid tool schema collection")

        allowed = set(allowed_tools)
        schemas: List[Dict[str, Any]] = []
        properties_by_name: Dict[str, set] = {}
        seen = set()
        for raw_schema in raw_schemas:
            if not isinstance(raw_schema, Mapping):
                continue
            name = raw_schema.get("name")
            if (
                not isinstance(name, str)
                or name not in allowed
                or name in seen
            ):
                continue
            input_schema = raw_schema.get("input_schema", {})
            if not isinstance(input_schema, Mapping):
                input_schema = {}
            normalized_input = deepcopy(dict(input_schema))
            description = raw_schema.get("description", "")
            if not isinstance(description, str):
                description = ""
            schemas.append(
                {
                    "name": name,
                    "description": description[:2048],
                    "input_schema": normalized_input,
                }
            )
            raw_properties = normalized_input.get("properties", {})
            properties = set()
            if isinstance(raw_properties, Mapping):
                for key in raw_properties:
                    if (
                        isinstance(key, str)
                        and len(key) <= _MAX_PROPERTY_NAME_LENGTH
                    ):
                        properties.add(key)
                    if len(properties) >= _MAX_TRACE_PROPERTIES:
                        break
            properties_by_name[name] = properties
            seen.add(name)

        return schemas, properties_by_name

    @classmethod
    def _protocol_safe_blocks(
        cls,
        blocks: Sequence[Dict[str, Any]],
        seen_ids: set,
    ) -> Optional[tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
        safe_blocks: List[Dict[str, Any]] = []
        tool_uses: List[Dict[str, Any]] = []
        current_ids = set()
        required_fields = {"type", "id", "name", "input"}

        for block in blocks:
            block_type = block.get("type")
            if block_type == "thinking":
                # DeepSeek's Anthropic-compatible endpoint prepends a signed
                # reasoning block to the final text/tool response.  Reasoning
                # is neither executable protocol data nor user-visible text,
                # so validate its envelope and deliberately do not forward it
                # into the next model turn.
                if (
                    not set(block).issubset(
                        {"type", "text", "thinking", "signature"}
                    )
                    or block.get("text") is not None
                    or not isinstance(block.get("thinking"), str)
                    or len(block["thinking"]) > _MAX_ASSISTANT_TEXT_LENGTH
                    or not isinstance(block.get("signature"), str)
                    or not block["signature"]
                ):
                    return None
                continue
            if block_type == "text":
                text = block.get("text")
                if (
                    set(block) != {"type", "text"}
                    or not isinstance(text, str)
                    or len(text) > _MAX_ASSISTANT_TEXT_LENGTH
                ):
                    return None
                safe_blocks.append({"type": "text", "text": text})
                continue
            if block_type != "tool_use":
                return None
            if (
                len(tool_uses) >= _MAX_TOOL_USES_PER_STEP
                or set(block) != required_fields
            ):
                return None

            tool_use_id = block.get("id")
            name = block.get("name")
            params = block.get("input")
            if (
                not isinstance(tool_use_id, str)
                or not tool_use_id
                or len(tool_use_id) > _MAX_TOOL_USE_ID_LENGTH
                or tool_use_id in seen_ids
                or tool_use_id in current_ids
                or not cls._valid_tool_name(name)
                or not isinstance(params, dict)
            ):
                return None
            try:
                safe_params = cls._reconstruct_tool_input(params)
                safe_tool_use = {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": name,
                    "input": safe_params,
                }
            except Exception:
                return None
            current_ids.add(tool_use_id)
            tool_uses.append(safe_tool_use)
            safe_blocks.append(safe_tool_use)

        return safe_blocks, tool_uses

    @classmethod
    def _reconstruct_tool_input(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        nodes = [0]
        reconstructed = cls._reconstruct_tool_input_value(
            params,
            depth=0,
            nodes=nodes,
            ancestors=set(),
        )
        serialized = json.dumps(
            reconstructed,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        if len(serialized.encode("utf-8")) > _MAX_TOOL_INPUT_JSON_BYTES:
            raise _UnsafeToolInput("Tool input is too large")
        return reconstructed

    @classmethod
    def _reconstruct_tool_input_value(
        cls,
        value: Any,
        *,
        depth: int,
        nodes: List[int],
        ancestors: set,
    ) -> Any:
        nodes[0] += 1
        if (
            nodes[0] > _MAX_TOOL_INPUT_NODES
            or depth > _MAX_TOOL_INPUT_DEPTH
        ):
            raise _UnsafeToolInput("Tool input exceeds structural limits")

        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise _UnsafeToolInput("Tool input contains non-finite data")
            return value
        if isinstance(value, str):
            if len(value) > _MAX_TOOL_INPUT_STRING_LENGTH:
                raise _UnsafeToolInput("Tool input string is too large")
            return value
        if not isinstance(value, (dict, list)):
            raise _UnsafeToolInput("Tool input type is unsupported")
        if len(value) > _MAX_TOOL_INPUT_ITEMS:
            raise _UnsafeToolInput("Tool input container is too large")

        container_id = id(value)
        if container_id in ancestors:
            raise _UnsafeToolInput("Tool input contains a cycle")
        ancestors.add(container_id)
        try:
            if isinstance(value, dict):
                reconstructed: Dict[str, Any] = {}
                for key, nested in value.items():
                    if (
                        not isinstance(key, str)
                        or len(key) > _MAX_TOOL_INPUT_STRING_LENGTH
                    ):
                        raise _UnsafeToolInput("Tool input key is invalid")
                    reconstructed[key] = cls._reconstruct_tool_input_value(
                        nested,
                        depth=depth + 1,
                        nodes=nodes,
                        ancestors=ancestors,
                    )
                return reconstructed
            return [
                cls._reconstruct_tool_input_value(
                    nested,
                    depth=depth + 1,
                    nodes=nodes,
                    ancestors=ancestors,
                )
                for nested in value
            ]
        finally:
            ancestors.remove(container_id)

    @staticmethod
    def _trace_params(
        params: Any,
        schema_properties: set,
    ) -> Dict[str, Any]:
        if not isinstance(params, Mapping):
            return {}
        traced: Dict[str, Any] = {}
        for key in params:
            if (
                isinstance(key, str)
                and key in schema_properties
                and len(key) <= _MAX_PROPERTY_NAME_LENGTH
            ):
                traced[key] = _REDACTED
            if len(traced) >= _MAX_TRACE_PROPERTIES:
                break
        return traced

    @staticmethod
    def _valid_tool_name(name: Any) -> bool:
        return isinstance(name, str) and bool(_TOOL_NAME_PATTERN.fullmatch(name))

    @classmethod
    def _serialize_tool_data(cls, data: Any) -> str:
        node_count = [0]
        sanitized = cls._sanitize_tool_data(data, depth=0, nodes=node_count)
        payload = {"success": True, "data": sanitized}
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        if len(serialized.encode("utf-8")) > _MAX_RESULT_JSON_BYTES:
            raise _UnsafeToolResult("Tool result is too large")
        return serialized

    @classmethod
    def _sanitize_tool_data(
        cls,
        value: Any,
        *,
        depth: int,
        nodes: List[int],
    ) -> Any:
        nodes[0] += 1
        if nodes[0] > _MAX_RESULT_NODES or depth > _MAX_RESULT_DEPTH:
            raise _UnsafeToolResult("Tool result exceeds structural limits")

        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise _UnsafeToolResult("Tool result contains non-finite data")
            return value
        if isinstance(value, str):
            if len(value) > _MAX_RESULT_STRING_LENGTH:
                raise _UnsafeToolResult("Tool result string is too large")
            return value
        if isinstance(value, Mapping):
            if len(value) > _MAX_RESULT_ITEMS:
                raise _UnsafeToolResult("Tool result mapping is too large")
            sanitized: Dict[str, Any] = {}
            for key, nested in value.items():
                if (
                    not isinstance(key, str)
                    or not key
                    or len(key) > _MAX_PROPERTY_NAME_LENGTH
                ):
                    raise _UnsafeToolResult("Tool result key is invalid")
                if cls._is_sensitive_result_key(key, nested):
                    sanitized[key] = _REDACTED
                else:
                    sanitized[key] = cls._sanitize_tool_data(
                        nested,
                        depth=depth + 1,
                        nodes=nodes,
                    )
            return sanitized
        if isinstance(value, (list, tuple)):
            if len(value) > _MAX_RESULT_ITEMS:
                raise _UnsafeToolResult("Tool result sequence is too large")
            return [
                cls._sanitize_tool_data(
                    nested,
                    depth=depth + 1,
                    nodes=nodes,
                )
                for nested in value
            ]
        raise _UnsafeToolResult("Tool result type is unsupported")

    @staticmethod
    def _normalize_result_key(key: str) -> str:
        acronym_separated = re.sub(
            r"([A-Z]+)([A-Z][a-z])",
            r"\1_\2",
            key,
        )
        separated = re.sub(
            r"([a-z0-9])([A-Z])",
            r"\1_\2",
            acronym_separated,
        )
        return re.sub(r"[^A-Za-z0-9]+", "_", separated).strip("_").lower()

    @classmethod
    def _is_sensitive_result_key(cls, key: str, value: Any) -> bool:
        normalized = cls._normalize_result_key(key)
        if normalized in _SAFE_RESULT_KEYS:
            return not cls._is_safe_result_control(normalized, value)
        if normalized in _SENSITIVE_RESULT_KEYS:
            return True

        singular_tokens = {
            "apikeys": "apikey",
            "cookies": "cookie",
            "credentials": "credential",
            "dsns": "dsn",
            "keys": "key",
            "passwords": "password",
            "secrets": "secret",
            "strings": "string",
            "tokens": "token",
            "uris": "uri",
            "urls": "url",
        }
        raw_tokens = [
            singular_tokens.get(token, token)
            for token in normalized.split("_")
            if token
        ]
        acronym_plurals = {
            ("ds", "ns"): "dsn",
            ("ur", "is"): "uri",
            ("ur", "ls"): "url",
        }
        tokens = []
        index = 0
        while index < len(raw_tokens):
            pair = tuple(raw_tokens[index:index + 2])
            if pair in acronym_plurals:
                tokens.append(acronym_plurals[pair])
                index += 2
            else:
                tokens.append(raw_tokens[index])
                index += 1
        token_set = set(tokens)
        suffix = tokens[-1]
        if suffix in {"configured", "enabled", "present", "required"} and (
            type(value) is bool
        ):
            return False
        if suffix == "count" and cls._is_safe_count(value):
            return False
        if suffix == "status" and cls._is_safe_status(value):
            return False
        if token_set & {"password", "passwd", "pwd"}:
            return True
        if token_set & {"secret", "credential", "cookie"}:
            return True
        if "key" in token_set and token_set & {
            "api",
            "access",
            "private",
        }:
            return True
        if "token" in token_set and (
            tokens[-1] == "token"
            or token_set
            & {
                "access",
                "api",
                "auth",
                "authorization",
                "bearer",
                "id",
                "oauth",
                "refresh",
                "session",
            }
        ):
            return True
        if token_set & {"auth", "authorization"} and token_set & {
            "header",
            "value",
        }:
            return True

        database_tokens = {
            "database",
            "db",
            "mongo",
            "mongodb",
            "mysql",
            "postgres",
            "postgresql",
            "redis",
        }
        connection_tokens = {"dsn", "uri", "url"}
        if "dsn" in token_set or (
            "connection" in token_set
            and token_set & {"string", "uri", "url"}
        ):
            return True
        if token_set & database_tokens and (
            token_set & connection_tokens
            or {"connection", "string"}.issubset(token_set)
        ):
            return True
        return False

    @classmethod
    def _is_safe_result_control(cls, key: str, value: Any) -> bool:
        if key == "retry_with_same_idempotency_key":
            return type(value) is bool
        if key == "token_count":
            return cls._is_safe_count(value)
        if key == "authorization_status":
            return cls._is_safe_status(
                value,
                extra=_AUTHORIZATION_STATUSES,
            )
        if key == "backend_status":
            return cls._is_safe_status(value, extra=frozenset({"degraded"}))
        if key == "error_code":
            return isinstance(value, str) and value in _SAFE_ERROR_CODES
        return False

    @staticmethod
    def _is_safe_count(value: Any) -> bool:
        if type(value) is bool:
            return False
        if type(value) is int:
            return value >= 0 and value.bit_length() <= _MAX_SAFE_COUNT_BITS
        if type(value) is float:
            return math.isfinite(value) and value >= 0
        return False

    @staticmethod
    def _is_safe_status(
        value: Any,
        *,
        extra: frozenset = frozenset(),
    ) -> bool:
        return (
            isinstance(value, str)
            and len(value) <= 32
            and value.strip().lower()
            in (_SAFE_OPERATIONAL_STATUSES | extra)
        )

    @staticmethod
    def _result_field(result: Any, key: str, default: Any) -> Any:
        if isinstance(result, Mapping):
            return result.get(key, default)
        return getattr(result, key, default)

    @classmethod
    def _error_result(
        cls,
        tool_use_id: str,
        name: str,
        params: Dict[str, Any],
        message: str,
        error_type: str,
        fingerprint: str,
    ) -> tuple[Dict[str, Any], RuntimeToolCall]:
        return (
            cls._tool_result(tool_use_id, message, is_error=True),
            RuntimeToolCall(
                name=name,
                success=False,
                fingerprint=fingerprint,
                params=params,
                error_type=error_type,
            ),
        )

    @staticmethod
    def _tool_result(
        tool_use_id: str,
        content: str,
        *,
        is_error: bool,
    ) -> Dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
            "is_error": is_error,
        }

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(number) or number < 0:
            return 0.0
        return min(number, 86_400_000.0)
