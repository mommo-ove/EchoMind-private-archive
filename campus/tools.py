"""Authorized Agent-facing adapters for campus-service data."""

import asyncio
from collections.abc import Mapping
from typing import Any

from campus.store import CampusStore
from mcp.tool_manager import MCPToolManager, Tool


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


def require_context(
    context: Mapping[str, Any] | None,
    key: str,
) -> str:
    """Return one non-empty trusted-context value."""
    if not isinstance(context, Mapping):
        raise ValueError(f"Trusted context requires {key}")
    value = context.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Trusted context requires {key}")
    return value.strip()


def _reject_sensitive_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            compact = normalized.replace("_", "")
            if any(
                marker in normalized or marker.replace("_", "") in compact
                for marker in _SENSITIVE_FIELD_MARKERS
            ):
                raise ValueError("Tool parameters contain a forbidden credential field")
            _reject_sensitive_fields(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_sensitive_fields(nested)


def _require_user_context(
    _params: Mapping[str, Any],
    context: Mapping[str, Any] | None,
) -> None:
    require_context(context, "user_id")


def _require_ticket_creation_context(
    _params: Mapping[str, Any],
    context: Mapping[str, Any] | None,
) -> None:
    require_context(context, "user_id")
    require_context(context, "idempotency_key")


class CampusToolset:
    """Bind campus operations to identity supplied only by trusted context."""

    def __init__(self, store: CampusStore):
        self.store = store

    async def query_campus_card(
        self,
        params: dict[str, Any],
        context: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        _reject_sensitive_fields(params)
        user_id = require_context(context, "user_id")
        try:
            return await asyncio.to_thread(
                self.store.query_transactions,
                user_id,
                days=params.get("days", 7),
            )
        except Exception:
            raise RuntimeError("Campus card service unavailable") from None

    async def query_network_status(
        self,
        params: dict[str, Any],
        context: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        del context
        _reject_sensitive_fields(params)
        try:
            return await asyncio.to_thread(
                self.store.get_network_status,
                params["site"],
            )
        except Exception:
            raise RuntimeError("Campus network service unavailable") from None

    async def create_ticket(
        self,
        params: dict[str, Any],
        context: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        _reject_sensitive_fields(params)
        user_id = require_context(context, "user_id")
        idempotency_key = require_context(context, "idempotency_key")
        try:
            return await asyncio.to_thread(
                self.store.create_ticket,
                idempotency_key=idempotency_key,
                user_id=user_id,
                category=params["category"],
                title=params["title"],
                description=params["description"],
            )
        except Exception:
            raise RuntimeError("Ticket service unavailable") from None

    async def get_ticket(
        self,
        params: dict[str, Any],
        context: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        _reject_sensitive_fields(params)
        user_id = require_context(context, "user_id")
        try:
            return await asyncio.to_thread(
                self.store.get_ticket,
                params["ticket_id"],
                user_id=user_id,
            )
        except Exception:
            raise RuntimeError("Ticket service unavailable") from None


def _campus_card_fallback(_params, _context, _error):
    return []


def _network_status_fallback(_params, _context, _error):
    return {
        "site": "campus",
        "status": "UNKNOWN",
        "message": "Campus network status is temporarily unavailable.",
        "updated_at": None,
    }


def _create_ticket_fallback(_params, _context, _error):
    return {
        "created": None,
        "status": "unknown",
        "retry_with_same_idempotency_key": True,
        "message": (
            "Ticket creation outcome is unknown. "
            "Retry with the same idempotency key."
        ),
    }


def _get_ticket_fallback(_params, _context, _error):
    return None


def register_campus_tools(
    manager: MCPToolManager,
    toolset: CampusToolset,
) -> None:
    """Register the four bounded campus tools exposed to end-user Agents."""
    manager.register(
        Tool(
            name="query_campus_card",
            description="Query the current user's synthetic campus-card transactions.",
            handler=toolset.query_campus_card,
            schema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 30,
                    },
                },
                "additionalProperties": False,
            },
            context_validator=_require_user_context,
            timeout_s=3.0,
            fallback=_campus_card_fallback,
        )
    )
    manager.register(
        Tool(
            name="query_network_status",
            description="Query the synthetic campus network status.",
            handler=toolset.query_network_status,
            schema={
                "type": "object",
                "properties": {
                    "site": {
                        "type": "string",
                        "enum": ["campus"],
                    },
                },
                "required": ["site"],
                "additionalProperties": False,
            },
            cache_ttl=30.0,
            timeout_s=3.0,
            fallback=_network_status_fallback,
        )
    )
    manager.register(
        Tool(
            name="create_ticket",
            description="Create an idempotent support ticket for the current user.",
            handler=toolset.create_ticket,
            schema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 64,
                    },
                    "title": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 200,
                    },
                    "description": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 2000,
                    },
                },
                "required": ["category", "title", "description"],
                "additionalProperties": False,
            },
            context_validator=_require_ticket_creation_context,
            timeout_s=3.0,
            fallback=_create_ticket_fallback,
        )
    )
    manager.register(
        Tool(
            name="get_ticket",
            description="Get one support ticket owned by the current user.",
            handler=toolset.get_ticket,
            schema={
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                    },
                },
                "required": ["ticket_id"],
                "additionalProperties": False,
            },
            context_validator=_require_user_context,
            timeout_s=3.0,
            fallback=_get_ticket_fallback,
        )
    )
