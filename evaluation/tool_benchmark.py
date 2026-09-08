"""Deterministic benchmark for the real ToolManager and SQLite campus adapters."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from campus.store import CampusStore
from campus.tools import CampusToolset, register_campus_tools
from mcp.tool_manager import MCPToolManager


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


async def run_tool_benchmark(database_path: Path) -> dict[str, Any]:
    store = CampusStore(database_path)
    manager = MCPToolManager(api_key="benchmark-key", model="benchmark-no-llm")
    register_campus_tools(manager, CampusToolset(store))
    cases: list[dict[str, Any]] = []
    latencies: list[float] = []

    async def check(
        name: str,
        category: str,
        operation: Callable[[], Awaitable[Any]],
        predicate: Callable[[Any], bool],
    ) -> Any:
        started = time.perf_counter()
        result = await operation()
        latency = (time.perf_counter() - started) * 1000
        passed = bool(predicate(result))
        latencies.append(latency)
        cases.append({
            "name": name,
            "category": category,
            "passed": passed,
            "latency_ms": round(latency, 6),
        })
        return result

    await check(
        "query_card_uses_trusted_user",
        "tool_correctness",
        lambda: manager.call(
            "query_campus_card", {"days": 7}, {"user_id": "demo_user_01"}
        ),
        lambda result: result.success and bool(result.data)
        and all(row["user_id"] == "demo_user_01" for row in result.data),
    )
    await check(
        "query_network_status",
        "tool_correctness",
        lambda: manager.call("query_network_status", {"site": "campus"}, {}),
        lambda result: result.success and result.data.get("status") == "OPERATIONAL",
    )
    ticket_params = {
        "category": "technical",
        "title": "Benchmark network ticket",
        "description": "Synthetic deterministic benchmark case.",
    }
    ticket_context = {
        "user_id": "benchmark_user",
        "idempotency_key": "benchmark-ticket-key",
    }
    created = await check(
        "create_ticket",
        "tool_correctness",
        lambda: manager.call("create_ticket", ticket_params, ticket_context),
        lambda result: result.success and bool(result.data.get("id")),
    )
    repeated = await check(
        "repeat_ticket_is_idempotent",
        "idempotency_correctness",
        lambda: manager.call("create_ticket", ticket_params, ticket_context),
        lambda result: result.success and result.data.get("id") == created.data.get("id"),
    )
    await check(
        "owner_can_read_ticket",
        "trusted_context_security",
        lambda: manager.call(
            "get_ticket", {"ticket_id": repeated.data["id"]}, {"user_id": "benchmark_user"}
        ),
        lambda result: result.success and result.data.get("user_id") == "benchmark_user",
    )
    await check(
        "other_user_cannot_read_ticket",
        "trusted_context_security",
        lambda: manager.call(
            "get_ticket", {"ticket_id": repeated.data["id"]}, {"user_id": "intruder"}
        ),
        lambda result: result.success and result.data is None,
    )
    await check(
        "missing_trusted_context_is_rejected",
        "trusted_context_security",
        lambda: manager.call("get_ticket", {"ticket_id": repeated.data["id"]}, None),
        lambda result: result.rejected and not result.fallback_used,
    )
    await check(
        "invalid_network_parameter_is_rejected",
        "trusted_context_security",
        lambda: manager.call("query_network_status", {"site": "private"}, {}),
        lambda result: result.rejected
        and manager._tools["query_network_status"].stats.executed == 1,
    )

    tool = manager._tools["query_network_status"]
    toolset = tool.handler.__self__
    original_store_method = toolset.store.get_network_status

    def fail(_site):
        raise RuntimeError("private-backend-secret")

    toolset.store.get_network_status = fail
    try:
        await check(
            "backend_failure_uses_safe_fallback",
            "fallback_correctness",
            lambda: manager.call(
                "query_network_status", {"site": "campus"}, {}, use_cache=False
            ),
            lambda result: result.success and result.fallback_used
            and result.data.get("status") == "UNKNOWN"
            and "private-backend-secret" not in (result.error or "")
            and "private-backend-secret" not in repr(result.data),
        )
    finally:
        toolset.store.get_network_status = original_store_method

    categories = {
        category: [case for case in cases if case["category"] == category]
        for category in {
            "tool_correctness", "trusted_context_security",
            "idempotency_correctness", "fallback_correctness",
        }
    }
    metrics = {
        category: round(sum(case["passed"] for case in rows) / len(rows), 6)
        for category, rows in categories.items()
    }
    return {
        "schema_version": 1,
        "evidence": "real_tool_manager_sqlite",
        "case_count": len(cases),
        "metrics": metrics,
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 6),
            "p50": round(_percentile(latencies, 0.50), 6),
            "p95": round(_percentile(latencies, 0.95), 6),
            "max": round(max(latencies), 6),
        },
        "tool_stats": {
            name: {
                "total": tool.stats.total,
                "executed": tool.stats.executed,
                "rejected": tool.stats.rejected,
                "result_success_rate": round(tool.stats.result_success_rate, 6),
                "fallback_success": tool.stats.fallback_success,
            }
            for name, tool in manager._tools.items()
        },
        "cases": cases,
    }
