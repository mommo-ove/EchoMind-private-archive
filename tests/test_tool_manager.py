import asyncio
import time
from unittest.mock import AsyncMock, Mock

import pytest

from mcp.tool_manager import CircuitBreaker, CircuitState, MCPToolManager, Tool


@pytest.fixture
def manager():
    return MCPToolManager(api_key="test-key")


def run(coro):
    return asyncio.run(coro)


class FakeDedicatedReranker:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    async def score(self, query, documents):
        self.calls.append((query, documents))
        return list(self.scores)


def make_tool(handler=None, **overrides):
    options = {
        "name": "lookup",
        "description": "Look up one record",
        "handler": handler or AsyncMock(return_value={"id": "T1"}),
        "schema": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}},
            "required": ["ticket_id"],
            "additionalProperties": False,
        },
    }
    options.update(overrides)
    return Tool(**options)


def test_rerank_uses_dedicated_model_scores_without_calling_chat_llm():
    reranker = FakeDedicatedReranker([0.1, 0.95, 0.4])
    manager = MCPToolManager(api_key="test-key", reranker=reranker)
    manager._client.messages.create = AsyncMock(
        side_effect=AssertionError("chat LLM must not perform reranking")
    )
    items = [
        {"title": "campus card", "content": "card balance"},
        {"title": "network", "content": "campus network 401"},
        {"title": "library", "content": "library hours"},
    ]

    result = run(manager._rerank("network login failed", items, top_k=2))

    assert result == [items[1], items[2]]
    assert reranker.calls == [
        (
            "network login failed",
            [
                "campus card\ncard balance",
                "network\ncampus network 401",
                "library\nlibrary hours",
            ],
        )
    ]
    manager._client.messages.create.assert_not_awaited()


def test_deepseek_query_rewrite_disables_thinking_for_structured_json():
    manager = MCPToolManager(
        api_key="test-key",
        base_url="https://api.deepseek.com/anthropic",
        model="deepseek-chat",
    )
    manager._client.messages.create = AsyncMock(
        return_value=Mock(
            content=[Mock(type="text", text='["校园网认证失败", "校园网401"]')]
        )
    )

    queries = run(manager.rewrite_query("宿舍网络登不上", n=2))

    assert queries == ["宿舍网络登不上", "校园网认证失败", "校园网401"]
    call_kwargs = manager._client.messages.create.await_args.kwargs
    assert call_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def open_circuit(tool, *, cooldown_elapsed=False):
    tool.breaker.state = CircuitState.OPEN
    elapsed = tool.breaker.recovery_s + 1 if cooldown_elapsed else 0
    tool.breaker.opened_at = time.monotonic() - elapsed


def test_tool_exports_anthropic_schema(manager):
    tool = make_tool(name="get_ticket", description="Get one ticket")
    manager.register(tool)

    assert manager.schemas(["get_ticket"]) == [
        {
            "name": "get_ticket",
            "description": "Get one ticket",
            "input_schema": tool.schema,
        }
    ]


def test_schema_export_filters_unknown_and_preserves_allowed_order(manager):
    first = make_tool(name="first")
    second = make_tool(name="second")
    manager.register(first)
    manager.register(second)

    assert [schema["name"] for schema in manager.schemas(["second", "missing", "first"])] == [
        "second",
        "first",
    ]


def test_unknown_tool_returns_rejected_failure(manager):
    result = run(manager.call("missing", {}))

    assert result.success is False
    assert result.rejected is True
    assert "missing" in result.error


def test_missing_required_parameter_does_not_invoke_handler(manager):
    handler = AsyncMock()
    manager.register(make_tool(handler))

    result = run(manager.call("lookup", {}))

    assert result.success is False
    assert result.rejected is True
    handler.assert_not_awaited()
    assert manager.get_stats()["lookup"]["executed"] == 0
    assert manager.get_stats()["lookup"]["rejected"] == 1


def test_unexpected_parameter_is_rejected_when_schema_is_strict(manager):
    handler = AsyncMock()
    manager.register(make_tool(handler))

    result = run(manager.call("lookup", {"ticket_id": "T1", "token": "secret"}))

    assert result.success is False
    assert result.rejected is True
    handler.assert_not_awaited()


@pytest.mark.parametrize(
    ("property_schema", "value"),
    [
        ({"type": "string", "enum": ["open", "closed"]}, "pending"),
        ({"type": "integer", "minimum": 1}, 0),
        ({"type": "integer", "maximum": 30}, 31),
        ({"type": "string", "minLength": 3}, "ab"),
        ({"type": "string", "maxLength": 5}, "abcdef"),
    ],
)
def test_used_json_schema_constraints_reject_invalid_values(manager, property_schema, value):
    handler = AsyncMock()
    manager.register(
        make_tool(
            handler,
            schema={
                "type": "object",
                "properties": {"value": property_schema},
                "required": ["value"],
                "additionalProperties": False,
            },
        )
    )

    result = run(manager.call("lookup", {"value": value}))

    assert result.success is False
    assert result.rejected is True
    handler.assert_not_awaited()


@pytest.mark.parametrize("schema_type", ["integer", "number"])
def test_json_schema_numeric_types_reject_booleans(manager, schema_type):
    handler = AsyncMock()
    manager.register(
        make_tool(
            handler,
            schema={
                "type": "object",
                "properties": {"value": {"type": schema_type}},
                "required": ["value"],
                "additionalProperties": False,
            },
        )
    )

    result = run(manager.call("lookup", {"value": True}))

    assert result.success is False
    assert result.rejected is True
    handler.assert_not_awaited()


@pytest.mark.parametrize("value", [1, 3])
def test_json_schema_numeric_boundaries_are_inclusive(manager, value):
    handler = AsyncMock(return_value={"accepted": value})
    manager.register(
        make_tool(
            handler,
            schema={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3,
                    }
                },
                "required": ["value"],
                "additionalProperties": False,
            },
        )
    )

    result = run(manager.call("lookup", {"value": value}))

    assert result.success is True
    handler.assert_awaited_once()


def test_timeout_records_backend_failure_and_latency(manager):
    async def slow_handler(_params, _context):
        await asyncio.sleep(0.2)

    # A 1 ms deadline can expire before Windows advances its reported clock,
    # which makes the rounded latency appear as 0.0 despite a real timeout.
    tool = make_tool(slow_handler, timeout_s=0.05)
    tool.breaker.threshold = 1
    manager.register(tool)

    result = run(manager.call("lookup", {"ticket_id": "T1"}))
    stats = manager.get_stats()["lookup"]

    assert result.success is False
    assert result.rejected is False
    assert "超时" in result.error
    assert stats["executed"] == 1
    assert stats["failed"] == 1
    assert stats["execution_failed"] == 1
    assert stats["consecutive_fails"] == 1
    assert stats["avg_latency_ms"] > 0
    assert result.latency_ms > 0
    assert tool.breaker.state is CircuitState.OPEN


def test_cache_hit_does_not_invoke_handler_again(manager):
    handler = AsyncMock(return_value={"id": "T1"})
    manager.register(make_tool(handler, cache_ttl=60))

    first = run(manager.call("lookup", {"ticket_id": "T1"}))
    second = run(manager.call("lookup", {"ticket_id": "T1"}))
    stats = manager.get_stats()["lookup"]

    assert first.cached is False
    assert second.cached is True
    assert handler.await_count == 1
    assert stats["executed"] == 1
    assert stats["cache_hits"] == 1
    assert stats["rejected"] == 0


def test_cached_result_is_not_served_while_circuit_is_open(manager):
    handler = AsyncMock(return_value={"id": "T1"})
    tool = make_tool(handler, cache_ttl=60)
    manager.register(tool)
    run(manager.call("lookup", {"ticket_id": "T1"}))
    open_circuit(tool)

    result = run(manager.call("lookup", {"ticket_id": "T1"}))

    assert result.success is False
    assert result.cached is False
    assert result.rejected is True
    assert handler.await_count == 1


def test_cooldown_probe_bypasses_populated_cache(manager):
    handler = AsyncMock(side_effect=[{"source": "initial"}, {"source": "probe"}])
    tool = make_tool(handler, cache_ttl=60)
    manager.register(tool)
    initial = run(manager.call("lookup", {"ticket_id": "T1"}))
    open_circuit(tool, cooldown_elapsed=True)

    probe = run(manager.call("lookup", {"ticket_id": "T1"}))

    assert initial.data == {"source": "initial"}
    assert probe.success is True
    assert probe.cached is False
    assert probe.data == {"source": "probe"}
    assert handler.await_count == 2
    assert tool.breaker.state is CircuitState.CLOSED


def test_open_circuit_rejects_without_invoking_handler(manager):
    handler = AsyncMock()
    tool = make_tool(handler)
    manager.register(tool)
    open_circuit(tool)

    result = run(manager.call("lookup", {"ticket_id": "T1"}))

    assert result.success is False
    assert result.rejected is True
    handler.assert_not_awaited()
    assert manager.get_stats()["lookup"]["executed"] == 0
    assert manager.get_stats()["lookup"]["rejected"] == 1


def test_fallback_preserves_original_error_and_marks_metadata(manager):
    handler = AsyncMock(side_effect=RuntimeError("primary failed"))
    fallback = AsyncMock(return_value={"source": "fallback"})
    manager.register(make_tool(handler, fallback=fallback))

    result = run(manager.call("lookup", {"ticket_id": "T1"}))

    assert result.success is True
    assert result.data == {"source": "fallback"}
    assert result.error == "primary failed"
    assert result.fallback_used is True
    assert result.rejected is False
    fallback.assert_awaited_once()


def test_successful_fallback_separates_result_from_backend_outcome(manager):
    handler = AsyncMock(side_effect=RuntimeError("primary failed"))
    fallback = AsyncMock(return_value={"source": "fallback"})
    manager.register(make_tool(handler, fallback=fallback))

    result = run(manager.call("lookup", {"ticket_id": "T1"}))
    stats = manager.get_stats()["lookup"]

    assert result.success is True
    assert stats["result_success"] == 1
    assert stats["result_failed"] == 0
    assert stats["execution_success"] == 0
    assert stats["execution_failed"] == 1
    assert stats["fallback_success"] == 1
    assert stats["fallback_failed"] == 0
    assert stats["success_rate"] == 0.0
    assert stats["result_success_rate"] == 1.0


def test_failed_fallback_records_backend_and_final_failures(manager):
    handler = AsyncMock(side_effect=RuntimeError("primary failed"))
    fallback = AsyncMock(side_effect=RuntimeError("fallback failed"))
    manager.register(make_tool(handler, fallback=fallback))

    result = run(manager.call("lookup", {"ticket_id": "T1"}))
    stats = manager.get_stats()["lookup"]

    assert result.success is False
    assert result.error.startswith("primary failed")
    assert stats["result_success"] == 0
    assert stats["result_failed"] == 1
    assert stats["execution_failed"] == 1
    assert stats["fallback_success"] == 0
    assert stats["fallback_failed"] == 1


def test_stats_distinguish_execution_cache_and_rejection(manager):
    handler = AsyncMock(return_value={"id": "T1"})
    manager.register(make_tool(handler, cache_ttl=60))

    run(manager.call("lookup", {"ticket_id": "T1"}))
    run(manager.call("lookup", {"ticket_id": "T1"}))
    run(manager.call("lookup", {}))

    stats = manager.get_stats()["lookup"]
    assert stats["total"] == 3
    assert stats["executed"] == 1
    assert stats["cache_hits"] == 1
    assert stats["rejected"] == 1
    assert stats["execution_success"] == 1
    assert stats["execution_failed"] == 0
    assert stats["result_success"] == 2
    assert stats["result_failed"] == 1
    assert stats["success_rate"] == 1.0
    assert stats["result_success_rate"] == 0.667


def test_rejection_is_a_failed_result_without_backend_execution(manager):
    handler = AsyncMock()
    manager.register(make_tool(handler))

    result = run(manager.call("lookup", {}))
    stats = manager.get_stats()["lookup"]

    assert result.rejected is True
    assert stats["result_failed"] == 1
    assert stats["executed"] == 0
    assert stats["execution_success"] == 0
    assert stats["execution_failed"] == 0
    assert stats["rejected"] == 1


def test_mixed_outcomes_average_latency_over_executed_calls_only(manager):
    async def handler(params, _context):
        if params["ticket_id"] == "timeout":
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.002)
        return {"id": params["ticket_id"]}

    # Keep a wide margin above Windows' scheduler quantum so the successful
    # 2 ms branch is not mistaken for a timeout on a busy CI host.
    tool = make_tool(handler, cache_ttl=60, timeout_s=0.05)
    manager.register(tool)

    success = run(manager.call("lookup", {"ticket_id": "T1"}))
    cached = run(manager.call("lookup", {"ticket_id": "T1"}))
    timeout = run(manager.call("lookup", {"ticket_id": "timeout"}))
    rejected = run(manager.call("lookup", {}))
    stats = manager.get_stats()["lookup"]

    assert success.success is True
    assert cached.cached is True
    assert timeout.success is False
    assert rejected.rejected is True
    assert stats["total"] == 4
    assert stats["executed"] == 2
    assert stats["execution_success"] == 1
    assert stats["execution_failed"] == 1
    assert stats["cache_hits"] == 1
    assert stats["rejected"] == 1
    assert stats["avg_latency_ms"] == round(
        (success.latency_ms + timeout.latency_ms) / 2,
        1,
    )


def test_availability_reports_unknown_and_open_tools(manager):
    available, error = manager.availability("missing")
    assert available is False
    assert "missing" in error

    tool = make_tool()
    manager.register(tool)
    tool.breaker.state = CircuitState.OPEN
    tool.breaker.opened_at = time.monotonic()

    available, error = manager.availability("lookup")
    assert available is False
    assert "lookup" in error


def test_availability_after_cooldown_is_side_effect_free(manager):
    tool = make_tool()
    manager.register(tool)
    open_circuit(tool, cooldown_elapsed=True)

    first = manager.availability("lookup")
    second = manager.availability("lookup")

    assert first == (True, None)
    assert second == (True, None)
    assert tool.breaker.state is CircuitState.OPEN


def test_availability_uses_side_effect_free_check_not_legacy_allow(manager):
    tool = make_tool()
    manager.register(tool)
    tool.breaker.allow = Mock(side_effect=AssertionError("legacy allow must not be called"))

    assert manager.availability("lookup") == (True, None)
    tool.breaker.allow.assert_not_called()


def test_legacy_allow_reserves_one_probe_and_no_token_success_closes_it():
    breaker = CircuitBreaker(failure_threshold=1, recovery_s=0)
    breaker.state = CircuitState.OPEN
    breaker.opened_at = time.monotonic() - 1

    assert breaker.allow() is True
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.allow() is False

    assert breaker.record_success() is True
    assert breaker.state is CircuitState.CLOSED
    assert breaker.record_failure() is False
    assert breaker.state is CircuitState.CLOSED


def test_legacy_no_token_failure_reopens_probe_and_stale_success_is_ignored():
    breaker = CircuitBreaker(failure_threshold=1, recovery_s=0)
    breaker.state = CircuitState.OPEN
    breaker.opened_at = time.monotonic() - 1

    assert breaker.allow() is True
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.allow() is False

    assert breaker.record_failure() is True
    assert breaker.state is CircuitState.OPEN
    assert breaker.record_success() is False
    assert breaker.state is CircuitState.OPEN


def test_only_reserved_probe_can_resolve_half_open_state():
    breaker = CircuitBreaker(failure_threshold=1, recovery_s=0)
    breaker.state = CircuitState.OPEN
    breaker.opened_at = time.monotonic() - 1

    probe = breaker.acquire()
    assert probe is not None
    assert breaker.state is CircuitState.HALF_OPEN

    breaker.record_success()
    assert breaker.state is CircuitState.HALF_OPEN

    breaker.record_success(probe)
    assert breaker.state is CircuitState.CLOSED


def test_conflicting_late_probe_outcomes_cannot_overwrite_new_generation():
    first = CircuitBreaker(failure_threshold=1, recovery_s=0)
    first.state = CircuitState.OPEN
    first.opened_at = time.monotonic() - 1
    failed_probe = first.acquire()
    first.record_failure(failed_probe)
    first.record_success(failed_probe)
    assert first.state is CircuitState.OPEN

    second = CircuitBreaker(failure_threshold=1, recovery_s=0)
    second.state = CircuitState.OPEN
    second.opened_at = time.monotonic() - 1
    successful_probe = second.acquire()
    second.record_success(successful_probe)
    second.record_failure(successful_probe)
    assert second.state is CircuitState.CLOSED


def test_cooldown_allows_exactly_one_concurrent_direct_probe(manager):
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()
        handler_calls = 0

        async def handler(_params, _context):
            nonlocal handler_calls
            handler_calls += 1
            started.set()
            await release.wait()
            return {"id": "T1"}

        tool = make_tool(handler)
        manager.register(tool)
        open_circuit(tool, cooldown_elapsed=True)

        first_task = asyncio.create_task(manager.call("lookup", {"ticket_id": "T1"}))
        await started.wait()
        second_task = asyncio.create_task(manager.call("lookup", {"ticket_id": "T2"}))
        await asyncio.sleep(0)
        release.set()
        first, second = await asyncio.gather(first_task, second_task)
        return tool, first, second, handler_calls

    tool, first, second, handler_calls = run(scenario())

    assert first.success is True
    assert second.success is False
    assert second.rejected is True
    assert handler_calls == 1
    assert tool.stats.executed == 1
    assert tool.breaker.state is CircuitState.CLOSED


def test_cancelled_half_open_probe_is_settled_and_can_recover(manager):
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def handler(_params, _context):
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                await release.wait()
            return {"id": "T1"}

        tool = make_tool(handler)
        tool.breaker.recovery_s = 0
        manager.register(tool)
        open_circuit(tool, cooldown_elapsed=True)

        cancelled_call = asyncio.create_task(
            manager.call("lookup", {"ticket_id": "T1"})
        )
        await started.wait()
        assert tool.breaker.state is CircuitState.HALF_OPEN

        cancelled_call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_call

        state_after_cancel = tool.breaker.state
        recovered = await manager.call("lookup", {"ticket_id": "T1"})
        return tool, recovered, calls, state_after_cancel

    tool, recovered, calls, state_after_cancel = run(scenario())

    assert state_after_cancel is CircuitState.OPEN
    assert recovered.success is True
    assert calls == 2
    assert tool.breaker.state is CircuitState.CLOSED
    assert tool.stats.executed == 2
    assert tool.stats.failed == 1
    assert tool.stats.success == 1
    assert tool.stats.consecutive_fails == 0
    assert tool.stats.result_success == 1
    assert tool.stats.result_failed == 1
    assert tool.stats.result_success + tool.stats.result_failed == tool.stats.total
    assert tool.stats.result_success_rate == 0.5


def test_cancelled_context_validator_counts_one_failed_result(manager):
    async def scenario():
        started = asyncio.Event()

        async def validator(_params, _context):
            started.set()
            await asyncio.Event().wait()

        tool = make_tool(context_validator=validator)
        manager.register(tool)
        task = asyncio.create_task(
            manager.call("lookup", {"ticket_id": "T1"})
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return tool

    tool = run(scenario())

    assert tool.stats.total == 1
    assert tool.stats.executed == 0
    assert tool.stats.result_success == 0
    assert tool.stats.result_failed == 1
    assert tool.stats.result_success + tool.stats.result_failed == tool.stats.total
    assert tool.breaker.state is CircuitState.CLOSED


@pytest.mark.parametrize("registry_mutation", ["replace", "unregister"])
def test_cancelled_call_accounts_on_original_tool_instance(
    manager,
    registry_mutation,
):
    async def scenario():
        started = asyncio.Event()

        async def handler(_params, _context):
            started.set()
            await asyncio.Event().wait()

        original = make_tool(handler)
        replacement = make_tool()
        manager.register(original)
        task = asyncio.create_task(
            manager.call("lookup", {"ticket_id": "T1"})
        )
        await started.wait()
        if registry_mutation == "replace":
            manager.register(replacement)
        else:
            manager.unregister("lookup")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return original, replacement

    original, replacement = run(scenario())

    assert original.stats.total == 1
    assert original.stats.result_success == 0
    assert original.stats.result_failed == 1
    assert (
        original.stats.result_success + original.stats.result_failed
        == original.stats.total
    )
    assert replacement.stats.total == 0
    assert replacement.stats.result_failed == 0


def test_cancelled_fallback_counts_once_and_does_not_leak_probe(manager):
    async def scenario():
        started = asyncio.Event()

        async def fallback(_params, _context, _error):
            started.set()
            await asyncio.Event().wait()

        tool = make_tool(
            AsyncMock(side_effect=RuntimeError("primary failed")),
            fallback=fallback,
        )
        tool.breaker.recovery_s = 0
        manager.register(tool)
        open_circuit(tool, cooldown_elapsed=True)

        task = asyncio.create_task(
            manager.call("lookup", {"ticket_id": "T1"})
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return tool

    tool = run(scenario())

    assert tool.stats.total == 1
    assert tool.stats.executed == 1
    assert tool.stats.failed == 1
    assert tool.stats.result_success == 0
    assert tool.stats.result_failed == 1
    assert tool.stats.result_success + tool.stats.result_failed == tool.stats.total
    assert tool.breaker.state is CircuitState.OPEN


def test_cancelled_rerank_counts_one_failed_result(manager):
    async def scenario():
        started = asyncio.Event()

        async def rerank(_query, _items, _top_k):
            started.set()
            await asyncio.Event().wait()

        tool = make_tool(
            AsyncMock(return_value=[{"id": "T1"}]),
            supports_rerank=True,
        )
        manager.register(tool)
        manager._rerank = rerank

        task = asyncio.create_task(
            manager.call(
                "lookup",
                {"ticket_id": "T1"},
                rerank_top_k=1,
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return tool

    tool = run(scenario())

    assert tool.stats.total == 1
    assert tool.stats.executed == 1
    assert tool.stats.success == 1
    assert tool.stats.result_success == 0
    assert tool.stats.result_failed == 1
    assert tool.stats.result_success + tool.stats.result_failed == tool.stats.total
    assert tool.breaker.state is CircuitState.CLOSED


def test_invalid_parameters_do_not_consume_half_open_probe_reservation(manager):
    handler = AsyncMock(return_value={"id": "T1"})
    tool = make_tool(handler)
    manager.register(tool)
    open_circuit(tool, cooldown_elapsed=True)

    invalid = run(manager.call("lookup", {}))
    assert invalid.rejected is True
    assert tool.breaker.state is CircuitState.OPEN

    valid = run(manager.call("lookup", {"ticket_id": "T1"}))
    assert valid.success is True
    assert tool.breaker.state is CircuitState.CLOSED
    handler.assert_awaited_once()


def test_search_with_rewrite_deduplicates_same_document_across_queries(manager):
    async def handler(params, _context):
        if params["query"] == "query-a":
            return [
                {"id": "doc-a", "content": "same document", "score": 0.9},
            ]
        return [
            {"id": "doc-a", "content": "same document", "score": 0.7},
            {"id": "doc-b", "content": "other document", "score": 0.6},
        ]

    tool = make_tool(
        handler,
        name="knowledge_search",
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    manager.register(tool)
    manager.rewrite_query = AsyncMock(return_value=["query-a", "query-b"])

    result = run(manager.search_with_rewrite("knowledge_search", "original", top_k=2))

    assert [item["id"] for item in result.data] == ["doc-a", "doc-b"]


def test_search_with_rewrite_stops_entire_pipeline_when_circuit_is_open(manager):
    handler = AsyncMock()
    fallback = AsyncMock(return_value=[{"content": "temporary result"}])
    tool = make_tool(
        handler,
        name="knowledge_search",
        fallback=fallback,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    manager.register(tool)
    open_circuit(tool)
    manager.rewrite_query = AsyncMock()
    manager._rerank = AsyncMock()

    result = run(manager.search_with_rewrite("knowledge_search", "campus network", top_k=5))

    assert result.success is True
    assert result.data == [{"content": "temporary result"}]
    assert result.rejected is True
    assert result.fallback_used is True
    assert "knowledge_search" in result.error
    manager.rewrite_query.assert_not_awaited()
    manager._rerank.assert_not_awaited()
    handler.assert_not_awaited()
    fallback.assert_awaited_once()


def test_search_with_rewrite_uses_one_direct_probe_after_cooldown(manager):
    handler = AsyncMock(return_value=[{"content": "recovered"}])
    tool = make_tool(
        handler,
        name="knowledge_search",
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    manager.register(tool)
    open_circuit(tool, cooldown_elapsed=True)
    manager.rewrite_query = AsyncMock()
    manager._rerank = AsyncMock()

    result = run(manager.search_with_rewrite("knowledge_search", "campus network", top_k=5))

    assert result.success is True
    assert result.data == [{"content": "recovered"}]
    assert result.reranked is False
    handler.assert_awaited_once()
    manager.rewrite_query.assert_not_awaited()
    manager._rerank.assert_not_awaited()
    assert tool.breaker.state is CircuitState.CLOSED
