import asyncio
from copy import deepcopy
from types import SimpleNamespace

from fastapi import Request

import api.main as main
from core.intent_recognizer import IntentCategory, IntentResult, UrgencyLevel
from mcp.tool_manager import AGENT_TOOL_ALLOWLIST, MCPToolManager, Tool
from services.chat_service import ChatService
from campus.store import CampusStore


def model_response(*blocks):
    return SimpleNamespace(content=list(blocks))


def tool_use(tool_id, name, params):
    return {"type": "tool_use", "id": tool_id, "name": name, "input": params}


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        response = self.responses.pop(0)
        return response() if callable(response) else response


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


class FakeMemory:
    async def get_context(self, _user_id, _conv_id, query=""):
        return SimpleNamespace(
            recent_messages=(),
            to_prompt_text=lambda: "",
        )

    async def add_exchange(self, *_args, **_kwargs):
        return True

    async def update_profile(self, *_args, **_kwargs):
        return None


class TechnicalRecognizer:
    async def recognize(self, _message, history=None):
        return IntentResult(
            intent=IntentCategory.TECHNICAL,
            confidence=0.99,
            urgency=UrgencyLevel.LOW,
            entities={},
            reasoning="test",
            latency_ms=0.1,
            intent_scores={IntentCategory.TECHNICAL: 0.99},
            matched_intents=[IntentCategory.TECHNICAL],
        )


class NoKnowledgePolicy:
    def decide(self, _intent, _message):
        return SimpleNamespace(use_knowledge=False, reason="test")


def request_with_principal(principal_id):
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/chat",
            "query_string": b"",
            "headers": [],
        }
    )
    request.state.principal_id = principal_id
    return request


def test_runtime_wiring_drives_chat_ticket_round_trip_through_shared_api_store(
    tmp_path,
    monkeypatch,
):
    store = CampusStore(tmp_path / "campus.db")
    fake_client = FakeClient(
        [
            model_response(
                tool_use(
                    "create-1",
                    "create_ticket",
                    {
                        "category": "network",
                        "title": "Cannot connect",
                        "description": "Synthetic Wi-Fi failure",
                    },
                )
            ),
            model_response({"type": "text", "text": "Ticket created."}),
        ]
    )
    manager = MCPToolManager(api_key="test-key", model="test-model")
    manager._client = fake_client
    manager.register(Tool(
        name="knowledge_search",
        description="Test knowledge search.",
        handler=lambda _params, _context: [],
        schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    ))
    recognizer = TechnicalRecognizer()
    orchestrator = main._build_runtime_orchestrator(
        cfg={"api_key": "test-key", "model": "test-model"},
        tool_manager=manager,
        campus_store=store,
        skill_manager=None,
        intent_recognizer=recognizer,
    )

    assert set(manager._tools) == {
        "knowledge_search",
        "query_campus_card",
        "query_network_status",
        "create_ticket",
        "get_ticket",
    }
    assert "update_ticket_status" not in manager._tools
    assert (
        manager._tools["create_ticket"].handler.__self__.store
        is store
        is manager._tools["get_ticket"].handler.__self__.store
    )
    runtimes = {
        id(agent._runtime): agent._runtime
        for agents in orchestrator._pool.values()
        for agent in agents
    }
    assert len(runtimes) == 1
    runtime = next(iter(runtimes.values()))
    assert runtime._client is fake_client
    assert runtime._manager is manager
    assert runtime._verifier is not None
    assert {
        agent.agent_type.value: list(agent._allowed_tools)
        for agents in orchestrator._pool.values()
        for agent in agents
    } == AGENT_TOOL_ALLOWLIST

    service = ChatService(
        memory=FakeMemory(),
        intent_recognizer=recognizer,
        retrieval_policy=NoKnowledgePolicy(),
        orchestrator=orchestrator,
        knowledge_search=None,
    )
    monkeypatch.setattr(main, "_campus_store", store)
    monkeypatch.setattr(main, "_chat_service", service)

    async def scenario():
        created = await main.chat(
            main.ChatRequest(
                message="Create a network support ticket.",
                user_id="display-only",
                conv_id="conv-create",
            ),
            request_with_principal("owner"),
        )
        ticket_id = created.ticket_ids[0]
        assert store.get_ticket(ticket_id, user_id="owner") is not None

        fake_client.messages.responses.extend(
            [
                model_response(
                    tool_use("get-1", "get_ticket", {"ticket_id": ticket_id})
                ),
                model_response({"type": "text", "text": "Ticket found."}),
            ]
        )
        queried = await main.chat(
            main.ChatRequest(
                message="Find my network support ticket.",
                user_id="display-only",
                conv_id="conv-query",
            ),
            request_with_principal("owner"),
        )
        owner_view = await main.get_ticket(
            ticket_id,
            request_with_principal("owner"),
        )
        await service.aclose()
        return created, queried, owner_view

    created, queried, owner_view = asyncio.run(scenario())

    assert created.ticket_ids == [owner_view.id]
    assert queried.ticket_ids == [owner_view.id]
    assert owner_view.description == "Synthetic Wi-Fi failure"
