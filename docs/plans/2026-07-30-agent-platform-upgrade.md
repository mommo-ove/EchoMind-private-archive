# EchoMind Agent Platform Upgrade Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Make the six EchoMind PDF documents match verifiable behavior while upgrading the project into a campus IT Agent system with real tool calling, persistent tickets, shared chat/evaluation flow, reliable monitoring, and broad automated test coverage.

**Architecture:** Introduce a shared `ChatService` that owns the end-to-end request pipeline. Keep `AgentOrchestrator` responsible for routing, add a bounded `AgentRuntime` for Anthropic-compatible `tool_use` / `tool_result` loops, keep `MCPToolManager` as the internal execution-governance layer, and back demo campus tools with SQLite. Preserve the existing `/chat` response contract and Docker services while adding optional traces, citations, and ticket identifiers.

**Tech Stack:** Python 3.12, FastAPI, Anthropic Python SDK 0.40.0, Redis 7, ChromaDB 0.5.23, SQLite, Pydantic 2, Prometheus, Docker Compose, pytest.

**Implementation rules:** Use @Test-Driven Development for every behavior change, @Verification Before Completion before handoff, and keep commits small. Do not modify or restore unrelated dirty files in the original worktree. Do all implementation in `.worktrees/agent-platform-upgrade`.

---

## Verification command used throughout

Windows Python 3.12 cannot build the pinned `chroma-hnswlib` without Visual C++ Build Tools. Tests must therefore run in the reproducible Docker development image.

After Task 1, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1
```

For a single test:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1 tests/test_tool_manager.py -k circuit -vv
```

Every task must leave the full suite green.

---

### Task 1: Reproducible Docker test runner

**Files:**
- Modify: `Dockerfile:48-57`
- Create: `tools/test.ps1`
- Verify: `requirements-dev.txt`

**Step 1: Add a development-image smoke check**

Run:

```powershell
docker build --target development -t echomind-dev-test .
docker run --rm --entrypoint python echomind-dev-test -m pytest -q
```

Expected before the change: FAIL with `No module named pytest`.

**Step 2: Install development requirements only in the development stage**

Add to the `development` stage:

```dockerfile
FROM dependencies AS development

COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

COPY . .
```

Do not install `pytest` in the production stage.

**Step 3: Add the reusable PowerShell test runner**

Create `tools/test.ps1`:

```powershell
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs = @("-q")
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$image = "echomind-dev-test"

docker build --target development -t $image $repoRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$dockerArgs = @(
    "run", "--rm",
    "--entrypoint", "python",
    "-v", "${repoRoot}:/app",
    "-w", "/app",
    $image,
    "-m", "pytest"
) + $PytestArgs

docker @dockerArgs
exit $LASTEXITCODE
```

**Step 4: Verify the baseline**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1
```

Expected: `28 passed`.

**Step 5: Commit**

```bash
git add Dockerfile tools/test.ps1
git commit -m "test: add reproducible Docker test runner"
```

---

### Task 2: Strengthen ToolManager contracts and full-pipeline fallback

**Files:**
- Modify: `mcp/tool_manager.py:40-444`
- Create: `tests/test_tool_manager.py`

**Step 1: Write failing tests for schema export and validated calls**

Add tests using async handlers and no real model client:

```python
def test_tool_exports_anthropic_schema(manager):
    manager.register(Tool(
        name="get_ticket",
        description="Get one ticket",
        handler=AsyncMock(return_value={"id": "T1"}),
        schema={
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}},
            "required": ["ticket_id"],
            "additionalProperties": False,
        },
    ))

    assert manager.schemas(["get_ticket"]) == [{
        "name": "get_ticket",
        "description": "Get one ticket",
        "input_schema": manager._tools["get_ticket"].schema,
    }]
```

Also cover:

- unknown tool returns `success=False`;
- missing required parameter does not invoke the handler;
- unexpected parameter is rejected when `additionalProperties=False`;
- timeout increments failure state;
- cache hits do not invoke the handler again;
- open circuit does not invoke the handler;
- fallback preserves the original error;
- stats distinguish executed calls, cache hits, and rejected calls.

**Step 2: Run the tests and verify failure**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1 tests/test_tool_manager.py -vv
```

Expected: FAIL because `schemas()` and the stronger validation/stat fields do not exist.

**Step 3: Add explicit execution metadata**

Extend `ToolResult`:

```python
@dataclass
class ToolResult:
    success: bool
    data: Any
    tool_name: str
    error: Optional[str] = None
    cached: bool = False
    latency_ms: float = 0.0
    reranked: bool = False
    fallback_used: bool = False
    rejected: bool = False
```

Add `schemas()`:

```python
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
```

Extend `_validate_params()` to reject undeclared keys when `additionalProperties` is false and validate enum/minimum/maximum/string length for the schema subset used by project tools. Keep the implementation small; do not implement unrelated JSON Schema features.

**Step 4: Stop the whole search pipeline when the tool circuit is open**

Add a public preflight:

```python
def availability(self, name: str) -> tuple[bool, Optional[str]]:
    tool = self._tools.get(name)
    if tool is None:
        return False, f"工具不存在: {name}"
    if not tool.breaker.allow():
        return False, f"工具熔断中: {name}"
    return True, None
```

At the start of `search_with_rewrite()`, return one fallback result without calling `rewrite_query()` or `_rerank()` when availability fails.

**Step 5: Run focused and full tests**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1 tests/test_tool_manager.py -vv
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1
```

Expected: all tests pass.

**Step 6: Commit**

```bash
git add mcp/tool_manager.py tests/test_tool_manager.py
git commit -m "feat: harden tool execution lifecycle"
```

---

### Task 3: Add the persistent campus ticket and demo-data store

**Files:**
- Create: `campus/__init__.py`
- Create: `campus/store.py`
- Create: `tests/test_campus_store.py`

**Step 1: Write failing store tests**

Test a temporary SQLite file:

```python
def test_create_ticket_is_idempotent(tmp_path):
    store = CampusStore(tmp_path / "campus.db")
    first = store.create_ticket(
        idempotency_key="conv-1:billing",
        user_id="demo_user",
        category="billing",
        title="校园卡重复扣费",
        description="同一分钟出现两笔相同扣费",
    )
    second = store.create_ticket(
        idempotency_key="conv-1:billing",
        user_id="demo_user",
        category="billing",
        title="校园卡重复扣费",
        description="同一分钟出现两笔相同扣费",
    )
    assert first["id"] == second["id"]
```

Also test:

- allowed transitions `OPEN -> PROCESSING -> RESOLVED`;
- direct `RESOLVED -> OPEN` is rejected;
- unknown ticket returns `None`;
- query only returns the requesting user's ticket;
- demo campus-card transactions are queryable by user and date;
- network status returns a structured record.

**Step 2: Verify failure**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1 tests/test_campus_store.py -vv
```

Expected: FAIL because `CampusStore` does not exist.

**Step 3: Implement the SQLite schema and state machine**

Use the standard `sqlite3` module. Create these tables:

```sql
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    category TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS campus_card_transactions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    amount_cents INTEGER NOT NULL,
    merchant TEXT NOT NULL,
    occurred_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS network_status (
    site TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    message TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

Use parameterized SQL only. Seed records must use `demo_user_*` identifiers and be clearly marked as demonstration data.

**Step 4: Run tests**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1 tests/test_campus_store.py -vv
```

Expected: PASS.

**Step 5: Commit**

```bash
git add campus tests/test_campus_store.py
git commit -m "feat: add persistent campus ticket store"
```

---

### Task 4: Register executable campus tools with authorization boundaries

**Files:**
- Create: `campus/tools.py`
- Modify: `mcp/tool_manager.py`
- Create: `tests/test_campus_tools.py`

**Step 1: Write failing tool tests**

Cover these tools:

```text
query_campus_card
query_network_status
create_ticket
get_ticket
```

Example:

```python
async def test_query_campus_card_uses_context_user_not_model_user_id(toolset):
    result = await toolset.query_campus_card(
        {"days": 7},
        {"user_id": "demo_user_01"},
    )
    assert all(item["user_id"] == "demo_user_01" for item in result)
```

Security assertions:

- the model cannot override `user_id` in parameters;
- passwords/tokens are rejected fields;
- `create_ticket` requires an idempotency key from trusted context;
- `get_ticket` cannot read another user's ticket.

**Step 2: Verify failure**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1 tests/test_campus_tools.py -vv
```

Expected: FAIL because campus tool adapters do not exist.

**Step 3: Implement tool adapters**

Create `CampusToolset`:

```python
class CampusToolset:
    def __init__(self, store: CampusStore):
        self.store = store

    async def query_campus_card(self, params, context):
        user_id = require_context(context, "user_id")
        return self.store.query_transactions(user_id, days=params.get("days", 7))

    async def create_ticket(self, params, context):
        user_id = require_context(context, "user_id")
        idempotency_key = require_context(context, "idempotency_key")
        return self.store.create_ticket(
            idempotency_key=idempotency_key,
            user_id=user_id,
            category=params["category"],
            title=params["title"],
            description=params["description"],
        )
```

Provide `register_campus_tools(manager, toolset)` that registers all four tools with strict schemas, timeouts, and safe fallbacks.

**Step 4: Add allowed-tool policy**

Add constants:

```python
AGENT_TOOL_ALLOWLIST = {
    "general": ["get_ticket"],
    "technical": ["knowledge_search", "query_network_status", "create_ticket", "get_ticket"],
    "billing": ["knowledge_search", "query_campus_card", "create_ticket", "get_ticket"],
}
```

Do not expose `update_ticket_status` to end-user Agents. Status updates belong to a protected support API in Task 9.

**Step 5: Run tests and commit**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1 tests/test_campus_tools.py tests/test_tool_manager.py -vv
```

Then:

```bash
git add campus/tools.py mcp/tool_manager.py tests/test_campus_tools.py
git commit -m "feat: add authorized campus tools"
```

---

### Task 5: Implement the bounded Agent tool-calling runtime

**Files:**
- Create: `agents/agent_runtime.py`
- Modify: `core/llm_utils.py`
- Create: `tests/test_agent_runtime.py`

**Step 1: Write failing runtime tests with a fake model client**

Use fake responses containing text and tool-use blocks. Test:

```python
async def test_runtime_executes_tool_and_returns_final_text():
    client = FakeClient([
        fake_tool_use("tool-1", "query_network_status", {"site": "campus"}),
        fake_text("校园网认证服务正常，建议检查账号状态。"),
    ])
    runtime = AgentRuntime(client, "model", manager, max_steps=4)

    result = await runtime.run(
        system_prompt="你是技术支持",
        messages=[{"role": "user", "content": "校园网登录不上"}],
        allowed_tools=["query_network_status"],
        context={"user_id": "demo_user", "idempotency_key": "trace:technical"},
    )

    assert result.text.startswith("校园网认证服务正常")
    assert result.tool_calls[0].name == "query_network_status"
```

Also cover:

- forbidden tool returns an error result to the model and is not executed;
- tool parameters are passed to ToolManager;
- several tool calls in one assistant message all receive matching `tool_result` blocks;
- maximum step limit returns a controlled response;
- total timeout returns a controlled response;
- tool failures are visible to the model;
- trace parameters are redacted.

**Step 2: Verify failure**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1 tests/test_agent_runtime.py -vv
```

Expected: FAIL because `AgentRuntime` does not exist.

**Step 3: Add normalized content-block helpers**

In `core/llm_utils.py`, add small helpers that accept SDK objects or dictionaries:

```python
def content_blocks(content: Any) -> List[Dict[str, Any]]:
    blocks = []
    for block in content or []:
        if hasattr(block, "model_dump"):
            blocks.append(block.model_dump())
        elif isinstance(block, dict):
            blocks.append(block)
    return blocks
```

Do not depend on the latest SDK's beta tool runner. The project is pinned to `anthropic==0.40.0`.

**Step 4: Implement the loop**

Core behavior:

```python
for step in range(self._max_steps):
    response = await asyncio.wait_for(
        self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
            tools=self._manager.schemas(allowed_tools),
        ),
        timeout=remaining_timeout,
    )
    blocks = content_blocks(response.content)
    tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
    if not tool_uses:
        return RuntimeResult(text=extract_text_content(response.content), ...)

    messages.append({"role": "assistant", "content": blocks})
    tool_results = await execute_allowed_tools(tool_uses)
    messages.append({"role": "user", "content": tool_results})
```

Each result block must include the matching `tool_use_id`. Set `is_error=True` for rejected or failed calls.

**Step 5: Run tests and commit**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1 tests/test_agent_runtime.py -vv
```

Then:

```bash
git add agents/agent_runtime.py core/llm_utils.py tests/test_agent_runtime.py
git commit -m "feat: add bounded agent tool runtime"
```

---

### Task 6: Add intent-aware RetrievalPolicy

**Files:**
- Create: `core/retrieval_policy.py`
- Create: `tests/test_retrieval_policy.py`

**Step 1: Write the policy table as failing tests**

Required behavior:

```python
@pytest.mark.parametrize(
    ("intent", "message", "expected"),
    [
        (IntentCategory.GREETING, "你好呀", False),
        (IntentCategory.FEEDBACK, "谢谢你的帮助", False),
        (IntentCategory.ESCALATION, "我要人工客服", False),
        (IntentCategory.TECHNICAL, "校园网报401", True),
        (IntentCategory.BILLING, "为什么重复扣款", True),
        (IntentCategory.ACCOUNT, "校园账号被冻结", True),
        (IntentCategory.QUERY, "图书馆今天几点关门", True),
    ],
)
def test_retrieval_policy(intent, message, expected):
    assert RetrievalPolicy().should_retrieve(intent, message) is expected
```

Add an explicit override test for blank input and a configurable deny/allow list.

**Step 2: Run and verify failure**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1 tests/test_retrieval_policy.py -vv
```

Expected: FAIL because `RetrievalPolicy` does not exist.

**Step 3: Implement the policy**

Use an explicit intent set, not `len(message) >= 4`:

```python
DEFAULT_RETRIEVAL_INTENTS = {
    IntentCategory.QUERY,
    IntentCategory.REQUEST,
    IntentCategory.COMPLAINT,
    IntentCategory.TECHNICAL,
    IntentCategory.BILLING,
    IntentCategory.ACCOUNT,
}
```

Return a decision object containing `use_knowledge` and `reason` so traces and tests can explain the choice.

**Step 4: Run tests and commit**

```bash
git add core/retrieval_policy.py tests/test_retrieval_policy.py
git commit -m "feat: add intent-aware retrieval policy"
```

---

### Task 7: Integrate AgentRuntime and add result composition

**Files:**
- Modify: `agents/agent_orchestrator.py:67-459`
- Create: `agents/result_composer.py`
- Modify: `tests/test_agent_orchestrator_routing.py`
- Create: `tests/test_result_composer.py`

**Step 1: Write failing compatibility tests**

Keep existing behavior when no runtime is injected. Add tests that verify:

- TechnicalAgent receives the technical allowlist;
- BillingAgent receives the billing allowlist;
- parallel results preserve both tool traces;
- one failed specialist still falls back to GeneralAgent;
- escalation remains higher priority than tool use.

Write deterministic composer tests:

```python
def test_composer_removes_duplicate_next_steps():
    text = ResultComposer().compose([
        AgentPart("billing", "请保留截图。"),
        AgentPart("technical", "请保留截图。"),
    ])
    assert text.count("请保留截图") == 1
```

**Step 2: Verify failure**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1 tests/test_result_composer.py tests/test_agent_orchestrator_routing.py -vv
```

**Step 3: Extend AgentResponse and OrchestratorResult**

Add optional fields with safe defaults:

```python
@dataclass
class AgentResponse:
    content: str
    agent_type: AgentType
    success: bool = True
    escalate: bool = False
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    ticket_ids: List[str] = field(default_factory=list)

@dataclass
class OrchestratorResult:
    ...
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    ticket_ids: List[str] = field(default_factory=list)
```

**Step 4: Delegate BaseAgent LLM execution to AgentRuntime when configured**

Keep `_call_llm()` as a fallback for tests and deployments with tool calling disabled. Inject `AgentRuntime` and an allowlist into each Agent instance.

**Step 5: Use `ResultComposer` in `run_parallel()`**

Start with deterministic composition. Do not add another paid LLM call unless deterministic composition cannot meet a tested requirement. This keeps latency bounded while still giving the component a clear “summary/coordinator” responsibility.

**Step 6: Run full tests and commit**

```bash
git add agents/agent_orchestrator.py agents/result_composer.py tests
git commit -m "feat: integrate tool runtime with agent orchestration"
```

---

### Task 8: Introduce shared ChatService

**Files:**
- Create: `services/__init__.py`
- Create: `services/chat_service.py`
- Create: `tests/test_chat_service.py`
- Modify: `api/main.py:306-430`

**Step 1: Write an end-to-end service test with fakes**

Test the exact order:

```python
async def test_chat_service_recognizes_before_retrieval():
    events = []
    service = build_fake_service(events)
    result = await service.chat(
        message="校园网报401",
        user_id="demo_user",
        conv_id="conv-1",
    )
    assert events[:4] == [
        "memory.read",
        "intent.recognize",
        "retrieval.decide",
        "knowledge.search",
    ]
    assert result.knowledge_used is True
```

Also verify:

- Greeting skips knowledge search;
- intent recognition is not repeated in Orchestrator;
- memory receives both user and assistant messages;
- background profile failure does not fail the response;
- trace contains policy, route, knowledge, tools, and duration;
- existing `intent`, `agent_type`, `matched_intents`, and `agent_types` fields remain available.

**Step 2: Verify failure**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1 tests/test_chat_service.py -vv
```

**Step 3: Implement service models**

Create:

```python
@dataclass
class ChatCommand:
    message: str
    user_id: str
    conv_id: Optional[str] = None

@dataclass
class ChatResult:
    conv_id: str
    response: str
    intent: str
    agent_type: str
    escalated: bool
    latency_ms: float
    knowledge_used: bool
    citations: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    ticket_ids: List[str]
    trace_id: str
```

**Step 4: Implement the pipeline**

Required order:

```text
memory.get_context
intent_recognizer.recognize
retrieval_policy.decide
optional knowledge search
orchestrator.run with precomputed intent/scores
memory.add_message twice
schedule profile update with exception logging
return ChatResult
```

Build the knowledge prompt in the service, including title, score, content, and a stable citation identifier.

**Step 5: Replace `/chat` internals**

`api.main.chat()` should validate the request, call `_chat_service.chat()`, and map the result to `ChatResponse`. Remove `_should_use_knowledge()` after all callers are migrated.

**Step 6: Run tests and commit**

```bash
git add services api/main.py tests/test_chat_service.py
git commit -m "refactor: centralize the end-to-end chat pipeline"
```

---

### Task 9: Add ticket APIs and backward-compatible response fields

**Files:**
- Modify: `api/main.py:212-621`
- Create: `tests/test_ticket_api.py`
- Modify: `tests/test_chat_response_model.py`

**Step 1: Write failing API tests**

Add:

```text
GET   /tickets/{ticket_id}
PATCH /tickets/{ticket_id}/status
```

Assertions:

- users can only read their own ticket when `user_id` is supplied;
- status update requires the existing admin token dependency;
- invalid status transition returns 409;
- unknown ticket returns 404;
- `/chat` response accepts missing new fields for backward compatibility;
- `/chat` returns `trace_id`, `tool_calls`, `ticket_ids`, and `citations` when present.

**Step 2: Implement Pydantic response models**

Add:

```python
class ToolTraceOutput(BaseModel):
    name: str
    success: bool
    latency_ms: float = 0.0
    cached: bool = False
    error_type: Optional[str] = None

class ChatResponse(BaseModel):
    ...
    trace_id: Optional[str] = None
    tool_calls: List[ToolTraceOutput] = []
    ticket_ids: List[str] = []
    citations: List[Dict[str, Any]] = []
```

Use `Field(default_factory=list)` rather than mutable list defaults.

**Step 3: Implement protected status transitions**

Reuse the same constant-time admin-token validation used by `/intent/feedback`. Do not expose private ticket descriptions in monitoring or traces.

**Step 4: Run API tests and commit**

```bash
git add api/main.py tests/test_ticket_api.py tests/test_chat_response_model.py
git commit -m "feat: expose campus ticket workflow APIs"
```

---

### Task 10: Route end-to-end evaluation through ChatService

**Files:**
- Modify: `evaluation/evaluator.py:213-464`
- Modify: `api/main.py:569-619`
- Create: `tests/test_end_to_end_evaluator.py`

**Step 1: Write failing evaluator tests**

Use a fake `ChatService` that records calls:

```python
async def test_dialog_evaluation_uses_chat_service_and_same_conv_id():
    service = FakeChatService()
    evaluator = EndToEndEvaluator(
        chat_service=service,
        recognizer=recognizer,
        judge=FakeJudge(),
        baseline_path=None,
    )
    await evaluator.run(dialog_cases=[{"turns": ["你好", "校园网报401"]}])
    assert len(service.calls) == 2
    assert service.calls[0].conv_id == service.calls[1].conv_id
```

Also test report fields for:

- route correctness;
- expected tool names;
- expected ticket creation;
- knowledge/citation use;
- task completion;
- latency and model/tool-call counts;
- regression comparison.

**Step 2: Verify failure**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1 tests/test_end_to_end_evaluator.py -vv
```

**Step 3: Replace direct Orchestrator calls**

`_evaluate_dialog_case()` must call `ChatService.chat()` for every turn. Preserve one generated `user_id` and `conv_id` per case and clean test memory/tickets by namespace where practical.

**Step 4: Extend evaluation case schema**

Support optional fields:

```json
{
  "turns": ["校园卡重复扣费"],
  "expected_intents": ["billing"],
  "expected_agents": ["billing"],
  "expected_tools": ["query_campus_card", "create_ticket"],
  "expect_ticket": true,
  "expect_knowledge": true
}
```

Do not require all fields for existing PDF examples.

**Step 5: Run evaluator and full tests**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1 tests/test_end_to_end_evaluator.py -vv
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1
```

**Step 6: Commit**

```bash
git add evaluation/evaluator.py api/main.py tests/test_end_to_end_evaluator.py
git commit -m "feat: evaluate the real chat pipeline"
```

---

### Task 11: Fix monitoring semantics and add trace metrics

**Files:**
- Modify: `monitor/performance_monitor.py:101-317`
- Create: `tests/test_performance_monitor.py`

**Step 1: Write failing monitor tests**

Cover:

```python
async def test_repeated_collection_does_not_duplicate_same_active_alert():
    monitor = build_monitor(avg_ms=8000)
    await monitor._collect()
    await monitor._collect()
    assert len(monitor.summary()["active_alerts"]) == 1
```

Also verify:

- alert resolves when metric returns to normal;
- webhook sends only on state transition;
- Prometheus histogram is not updated by periodic average snapshots;
- request/tool counters increment from actual trace events;
- routing penalty remains a gauge-like current value.

**Step 2: Verify failure**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1 tests/test_performance_monitor.py -vv
```

Expected: duplicate-alert and histogram tests fail.

**Step 3: Store active alerts by metric key**

Replace append-only alert creation with:

```python
self._active_alerts: Dict[str, Alert] = {}
```

On trigger, create only if absent. On recovery, mark resolved and remove from the active map while keeping a bounded history deque.

**Step 4: Correct Prometheus metrics**

Use:

- Gauges for current success rate, current average latency, routing penalty, and circuit state;
- Counters for completed chat/tool calls and failures;
- Histograms observed at actual request completion, not during periodic collection.

Add `record_chat_trace(trace)` or an equivalent event method called by `ChatService`.

**Step 5: Run tests and commit**

```bash
git add monitor/performance_monitor.py tests/test_performance_monitor.py
git commit -m "fix: make monitoring metrics and alerts stateful"
```

---

### Task 12: Merge user profiles safely and test memory boundaries

**Files:**
- Modify: `memory/conversation_memory.py:159-208`
- Create: `tests/test_conversation_memory.py`

**Step 1: Write failing profile-merge tests**

Use fake Redis/Chroma clients and a fake model result:

```python
async def test_profile_update_preserves_existing_preferences():
    memory = build_memory(
        existing={"preferences": ["步骤化回答"], "entities": {"building": "3号楼"}},
        extracted={"preferences": ["简洁"], "entities": {"network": "校园网"}},
    )
    await memory.update_profile("u1", "c1")
    saved = memory.saved_profile
    assert saved["preferences"] == ["步骤化回答", "简洁"]
    assert saved["entities"]["building"] == "3号楼"
```

Also test:

- malformed LLM JSON preserves the previous profile;
- profile writes are upserts, not delete-then-add gaps;
- episodic search filters by `user_id`;
- Redis working-memory ordering and TTL;
- compression preserves five recent messages.

**Step 2: Verify failure**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1 tests/test_conversation_memory.py -vv
```

**Step 3: Implement deterministic merge**

Before model extraction, load the existing profile. Merge:

- list fields by stable de-duplication;
- dictionary fields by key, preferring newly confirmed values;
- scalars only when the new value is non-empty.

Store one current profile document per user, not one competing profile per conversation. Keep metadata with the last source conversation and update time.

**Step 4: Run tests and commit**

```bash
git add memory/conversation_memory.py tests/test_conversation_memory.py
git commit -m "feat: merge durable user profiles"
```

---

### Task 13: Wire production startup, persistence, and demo data

**Files:**
- Modify: `api/main.py:40-210`
- Modify: `docker-compose.yml`
- Modify: `.env.example`
- Create: `data/demo/README.md`
- Create: `tests/test_app_wiring.py`

**Step 1: Write a failing wiring test**

Extract startup construction into a testable factory or dependency container. Assert that:

- ToolManager registers `knowledge_search` plus four campus tools;
- AgentRuntime receives the same ToolManager;
- ChatService receives Memory, IntentRecognizer, RetrievalPolicy, Orchestrator, and Monitor;
- Evaluator receives ChatService;
- CampusStore receives the configured database path.

**Step 2: Verify failure**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1 tests/test_app_wiring.py -vv
```

**Step 3: Reorder application startup**

Construct in this dependency order:

```text
config
→ Redis/Chroma-backed Memory
→ IntentRecognizer and feedback store
→ SkillManager
→ ToolManager and KnowledgeBase
→ CampusStore and campus tools
→ AgentRuntime
→ AgentOrchestrator
→ RetrievalPolicy
→ Monitor
→ ChatService
→ Evaluator
```

Avoid module-level circular imports.

**Step 4: Add campus database persistence**

Use:

```yaml
environment:
  - CAMPUS_DB_PATH=/app/data/campus/campus.db
volumes:
  - campus-data:/app/data/campus
```

Declare:

```yaml
volumes:
  campus-data:
```

Do not remove or rename existing volumes.

**Step 5: Document demo-data boundaries**

`data/demo/README.md` must state that all campus-card, network, and ticket records are synthetic and must not contain real student identifiers.

**Step 6: Run tests and commit**

```bash
git add api/main.py docker-compose.yml .env.example data/demo/README.md tests/test_app_wiring.py
git commit -m "feat: wire campus agent services and persistence"
```

---

### Task 14: Add PDF-claim evidence and final end-to-end verification

**Files:**
- Create: `docs/PDF能力实现证据表.md`
- Modify: `docs/工程学习记录.md`
- Modify: `README.md` only if it exists on this branch at implementation time; do not recreate a user-deleted README in the original dirty worktree.
- Test: full suite and live Docker endpoints

**Step 1: Build the claim-to-evidence table**

For each PDF capability, record:

```text
PDF claim
implementation file/function
automated test
live verification command
known boundary
```

Required rows:

- three-way/multi-label intent recognition;
- multi-Agent routing and parallel collaboration;
- Skills loading;
- Redis/Chroma memory;
- RAG rewrite/recall/rerank;
- ToolManager reliability;
- real tool calling;
- ticket workflow;
- monitoring and routing feedback;
- end-to-end evaluation;
- Docker persistence.

Explicitly state that `MCPToolManager` is an internal tool-governance layer, not a standalone standard MCP Server.

**Step 2: Run the complete test suite**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\test.ps1
```

Expected: all tests pass with no warnings caused by project code.

**Step 3: Compile application modules**

Run:

```powershell
docker run --rm --entrypoint python -v "${PWD}:/app" -w /app echomind-dev-test -m compileall -q agents api campus core evaluation mcp memory monitor services
```

Expected: exit code 0.

**Step 4: Build the production image**

Run:

```powershell
docker compose build echomind
```

Expected: build completes successfully.

**Step 5: Start and inspect services**

Run:

```powershell
docker compose up -d
docker compose ps
curl.exe -sS http://localhost:8000/health
curl.exe -sS http://localhost:8000/skills
curl.exe -sS http://localhost:8000/knowledge/stats
curl.exe -sS http://localhost:8000/monitor
curl.exe -sS http://localhost:9090/api/v1/targets
```

Expected:

- all five existing services remain healthy;
- Prometheus target `echomind:8000` is `up`;
- Tool stats include campus tools;
- Skills count remains at least three.

**Step 6: Verify one complete campus scenario**

Send a request equivalent to:

```text
我的校园卡今天被重复扣费，而且校园网登录不上。
```

Verify:

- matched intents include billing and technical;
- both specialist Agents run;
- `query_campus_card` and `query_network_status` appear in tool traces;
- at least one idempotent ticket can be created;
- the returned ticket is queryable;
- repeating the same request does not create a duplicate ticket;
- the final answer contains both domains without duplicated next steps.

Record only synthetic identifiers in the evidence document.

**Step 7: Verify persistence**

Restart only the application:

```powershell
docker compose restart echomind
```

Query the ticket again. Expected: it still exists in `campus-data`.

**Step 8: Update learning records**

Append the verified architecture, commands, results, and remaining boundaries to `docs/工程学习记录.md`. Follow `AGENTS.md` and synchronize the same concise section to the existing private Notion page after reading it again.

**Step 9: Final commit**

```bash
git add docs
git commit -m "docs: add verified agent platform evidence"
```

**Step 10: Review before completion**

Run:

```powershell
git status --short
git log --oneline --decorate -15
```

Use @Requesting Code Review against this plan, address confirmed issues, and repeat the full verification suite before claiming completion.

