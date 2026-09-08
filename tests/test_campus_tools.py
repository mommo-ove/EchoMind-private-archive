import asyncio
import threading
import time

import pytest

from campus.store import CampusStore
from campus.tools import CampusToolset, register_campus_tools
from mcp.tool_manager import AGENT_TOOL_ALLOWLIST, MCPToolManager


@pytest.fixture
def store(tmp_path):
    return CampusStore(tmp_path / "campus.db")


@pytest.fixture
def toolset(store):
    return CampusToolset(store)


@pytest.fixture
def manager(toolset):
    registered = MCPToolManager(api_key="test-key")
    register_campus_tools(registered, toolset)
    return registered


def run(coro):
    return asyncio.run(coro)


def test_agent_tool_allowlist_is_bounded_and_does_not_expose_status_updates():
    assert AGENT_TOOL_ALLOWLIST == {
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
    assert all(
        "update_ticket_status" not in tools
        for tools in AGENT_TOOL_ALLOWLIST.values()
    )


def test_query_campus_card_uses_context_user_not_model_user_id(toolset):
    result = run(
        toolset.query_campus_card(
            {"days": 7, "user_id": "demo_user_02"},
            {"user_id": "demo_user_01"},
        )
    )

    assert result
    assert all(item["user_id"] == "demo_user_01" for item in result)


def test_registered_user_id_parameter_is_rejected_before_store_access(
    manager,
    monkeypatch,
):
    called = False

    def query_transactions(*_args, **_kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(
        manager._tools["query_campus_card"].handler.__self__.store,
        "query_transactions",
        query_transactions,
    )

    result = run(
        manager.call(
            "query_campus_card",
            {"days": 7, "user_id": "demo_user_02"},
            {"user_id": "demo_user_01"},
        )
    )

    assert result.rejected is True
    assert called is False


@pytest.mark.parametrize(
    "credential_field",
    [
        "password",
        "access_token",
        "client-secret",
        "apiKey",
        "credentials",
        "authorization",
    ],
)
def test_credential_like_fields_are_rejected_without_leaking_values(
    manager,
    credential_field,
):
    credential_value = "sensitive-value-that-must-not-escape"

    result = run(
        manager.call(
            "query_network_status",
            {"site": "campus", credential_field: credential_value},
        )
    )

    assert result.rejected is True
    assert credential_value not in (result.error or "")
    assert credential_value not in repr(result.data)


def test_direct_adapter_rejects_nested_credential_fields(toolset):
    with pytest.raises(ValueError, match="forbidden credential field"):
        run(
            toolset.query_network_status(
                {
                    "site": "campus",
                    "metadata": {"refresh_token": "sensitive-value"},
                },
                {},
            )
        )


def test_query_network_status_returns_structured_store_record(toolset):
    result = run(
        toolset.query_network_status(
            {"site": "campus"},
            {},
        )
    )

    assert result["site"] == "campus"
    assert result["status"] == "OPERATIONAL"
    assert result["message"].startswith("Synthetic demo status:")


def test_create_ticket_requires_trusted_context_user_and_idempotency_key(
    toolset,
    store,
):
    with pytest.raises(ValueError, match="idempotency_key"):
        run(
            toolset.create_ticket(
                {
                    "category": "billing",
                    "title": "Synthetic duplicate charge",
                    "description": "Two matching synthetic demo charges.",
                },
                {"user_id": "demo_user_01"},
            )
        )

    assert store.get_ticket("ticket_missing") is None


def test_create_ticket_uses_only_trusted_identity_and_is_idempotent(
    toolset,
):
    params = {
        "user_id": "demo_user_02",
        "category": "billing",
        "title": "Synthetic duplicate charge",
        "description": "Two matching synthetic demo charges.",
    }
    context = {
        "user_id": "demo_user_01",
        "idempotency_key": "conversation-1:tool-call-1",
    }

    first = run(toolset.create_ticket(params, context))
    second = run(toolset.create_ticket(params, context))

    assert first == second
    assert first["user_id"] == "demo_user_01"


def test_get_ticket_cannot_read_another_users_ticket(toolset, store):
    ticket = store.create_ticket(
        idempotency_key="conversation-1:tool-call-1",
        user_id="demo_user_01",
        category="billing",
        title="Synthetic duplicate charge",
        description="Two matching synthetic demo charges.",
    )

    own = run(
        toolset.get_ticket(
            {"ticket_id": ticket["id"], "user_id": "demo_user_02"},
            {"user_id": "demo_user_01"},
        )
    )
    other = run(
        toolset.get_ticket(
            {"ticket_id": ticket["id"], "user_id": "demo_user_01"},
            {"user_id": "demo_user_02"},
        )
    )

    assert own == ticket
    assert other is None


def test_registration_exports_exact_strict_schemas_with_finite_timeouts(manager):
    schemas = manager.schemas()

    assert [schema["name"] for schema in schemas] == [
        "query_campus_card",
        "query_network_status",
        "create_ticket",
        "get_ticket",
    ]
    assert all(
        schema["input_schema"]["type"] == "object"
        and schema["input_schema"]["additionalProperties"] is False
        for schema in schemas
    )
    assert all(0 < tool.timeout_s <= 5 for tool in manager._tools.values())
    assert "user_id" not in repr(schemas)
    assert "update_ticket_status" not in manager._tools


@pytest.mark.parametrize(
    ("tool_name", "params", "context"),
    [
        ("query_campus_card", {"days": 7}, None),
        ("query_campus_card", {"days": 7}, {"user_id": "  "}),
        (
            "create_ticket",
            {
                "category": "billing",
                "title": "Synthetic duplicate charge",
                "description": "Two matching synthetic demo charges.",
            },
            {"user_id": "demo_user_01"},
        ),
        (
            "get_ticket",
            {"ticket_id": "ticket_123"},
            {},
        ),
    ],
)
def test_invalid_trusted_context_is_a_preflight_rejection(
    manager,
    tool_name,
    params,
    context,
):
    tool = manager._tools[tool_name]

    result = run(manager.call(tool_name, params, context))

    assert result.success is False
    assert result.rejected is True
    assert result.fallback_used is False
    assert "demo_user" not in (result.error or "")
    assert tool.stats.executed == 0
    assert tool.stats.failed == 0
    assert tool.breaker.fail_count == 0
    assert tool.breaker.state.value == "closed"


def test_repeated_context_rejections_do_not_block_later_authorized_call(
    manager,
    monkeypatch,
):
    tool = manager._tools["query_campus_card"]
    tool.breaker.threshold = 2
    backend_calls = 0
    real_query = tool.handler.__self__.store.query_transactions

    def counting_query(*args, **kwargs):
        nonlocal backend_calls
        backend_calls += 1
        return real_query(*args, **kwargs)

    monkeypatch.setattr(
        tool.handler.__self__.store,
        "query_transactions",
        counting_query,
    )

    rejected = [
        run(manager.call("query_campus_card", {"days": 7}, None))
        for _ in range(3)
    ]
    authorized = run(
        manager.call(
            "query_campus_card",
            {"days": 7},
            {"user_id": "demo_user_01"},
        )
    )

    assert all(
        result.success is False
        and result.rejected is True
        and result.fallback_used is False
        for result in rejected
    )
    assert authorized.success is True
    assert authorized.data
    assert backend_calls == 1
    assert tool.stats.rejected == 3
    assert tool.stats.executed == 1
    assert tool.stats.failed == 0
    assert tool.breaker.state.value == "closed"


def test_blocking_store_call_keeps_event_loop_responsive_and_times_out():
    release = threading.Event()
    started = threading.Event()

    class BlockingStore:
        def query_transactions(self, _user_id, *, days):
            assert days == 7
            started.set()
            release.wait(timeout=0.5)
            return [{"id": "too-late"}]

    async def scenario():
        manager = MCPToolManager(api_key="test-key")
        register_campus_tools(manager, CampusToolset(BlockingStore()))
        tool = manager._tools["query_campus_card"]
        tool.timeout_s = 0.05
        started_at = time.monotonic()
        task = asyncio.create_task(
            manager.call(
                "query_campus_card",
                {"days": 7},
                {"user_id": "demo_user_01"},
            )
        )
        try:
            await asyncio.sleep(0.02)
            heartbeat_latency = time.monotonic() - started_at
            assert started.is_set()
            result = await task
            return result, heartbeat_latency, tool
        finally:
            release.set()

    result, heartbeat_latency, tool = run(scenario())

    assert heartbeat_latency < 0.2
    assert result.success is True
    assert result.fallback_used is True
    assert result.data == []
    assert "超时" in result.error
    assert tool.stats.executed == 1
    assert tool.stats.failed == 1


def test_late_ticket_write_returns_indeterminate_same_key_retry_guidance(
    tmp_path,
):
    release = threading.Event()
    started = threading.Event()
    completed = threading.Event()
    persisted_results = []
    store = CampusStore(tmp_path / "late-ticket.db")
    idempotency_key = "trusted-key-must-not-be-echoed"

    class DelayedTicketStore:
        def create_ticket(self, **kwargs):
            started.set()
            release.wait(timeout=1)
            ticket = store.create_ticket(**kwargs)
            persisted_results.append(ticket)
            completed.set()
            return ticket

    async def scenario():
        manager = MCPToolManager(api_key="test-key")
        register_campus_tools(manager, CampusToolset(DelayedTicketStore()))
        manager._tools["create_ticket"].timeout_s = 0.05
        params = {
            "category": "billing",
            "title": "Synthetic duplicate charge",
            "description": "Two matching synthetic demo charges.",
        }
        context = {
            "user_id": "demo_user_01",
            "idempotency_key": idempotency_key,
        }

        first = await manager.call("create_ticket", params, context)
        completed_when_manager_returned = completed.is_set()
        release.set()
        late_write_completed = await asyncio.to_thread(completed.wait, 1)
        retry = await manager.call("create_ticket", params, context)
        return first, retry, completed_when_manager_returned, late_write_completed

    first, retry, completed_when_manager_returned, late_write_completed = run(
        scenario()
    )

    assert started.is_set()
    assert completed_when_manager_returned is False
    assert late_write_completed is True
    assert first.success is True
    assert first.fallback_used is True
    assert first.data == {
        "created": None,
        "status": "unknown",
        "retry_with_same_idempotency_key": True,
        "message": (
            "Ticket creation outcome is unknown. "
            "Retry with the same idempotency key."
        ),
    }
    assert idempotency_key not in repr(first.data)
    assert idempotency_key not in (first.error or "")
    assert retry.success is True
    assert retry.fallback_used is False
    assert len(persisted_results) == 2
    assert len({ticket["id"] for ticket in persisted_results}) == 1
    assert retry.data == persisted_results[0]
    assert retry.data["id"] == persisted_results[0]["id"]


@pytest.mark.parametrize(
    ("tool_name", "params", "context", "expected_data"),
    [
        (
            "query_campus_card",
            {"days": 7},
            {"user_id": "demo_user_01"},
            [],
        ),
        (
            "query_network_status",
            {"site": "campus"},
            {},
            {
                "site": "campus",
                "status": "UNKNOWN",
                "message": "Campus network status is temporarily unavailable.",
                "updated_at": None,
            },
        ),
        (
            "create_ticket",
            {
                "category": "billing",
                "title": "Synthetic duplicate charge",
                "description": "Two matching synthetic demo charges.",
            },
            {
                "user_id": "demo_user_01",
                "idempotency_key": "conversation-1:tool-call-1",
            },
            {
                "created": None,
                "status": "unknown",
                "retry_with_same_idempotency_key": True,
                "message": (
                    "Ticket creation outcome is unknown. "
                    "Retry with the same idempotency key."
                ),
            },
        ),
        (
            "get_ticket",
            {"ticket_id": "ticket_123"},
            {"user_id": "demo_user_01"},
            None,
        ),
    ],
)
def test_backend_failures_use_safe_non_leaking_fallbacks(
    tmp_path,
    monkeypatch,
    tool_name,
    params,
    context,
    expected_data,
):
    secret = "database-password-must-not-escape"
    failing_store = CampusStore(tmp_path / f"{tool_name}.db")
    method_name = {
        "query_campus_card": "query_transactions",
        "query_network_status": "get_network_status",
        "create_ticket": "create_ticket",
        "get_ticket": "get_ticket",
    }[tool_name]

    def fail(*_args, **_kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr(failing_store, method_name, fail)
    failing_manager = MCPToolManager(api_key="test-key")
    register_campus_tools(failing_manager, CampusToolset(failing_store))

    result = run(failing_manager.call(tool_name, params, context))

    assert result.success is True
    assert result.fallback_used is True
    assert result.data == expected_data
    assert secret not in (result.error or "")
    assert secret not in repr(result.data)
