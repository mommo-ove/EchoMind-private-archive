# Intent Feedback and Context-Aware Cache Implementation Plan

> **For Claude:** Use `C:/Users/liushilei/.codex/skills/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Add context-aware intent-result caching and a protected, Redis-persisted human feedback loop that survives EchoMind restarts.

**Architecture:** Keep derived intent results in the existing per-process Python cache, but key them by message, recent history, model, and an intent-template fingerprint. Store only human-confirmed intent samples in a Redis hash with AOF-backed persistence. Share one `IntentRecognizer` between the orchestrator and evaluator, and expose a token-protected admin endpoint for corrections.

**Tech Stack:** Python 3.12, FastAPI, Redis 5.2, Pydantic 2, pytest.

---

### Task 1: Redis feedback store

**Files:**
- Create: `core/intent_feedback_store.py`
- Create: `tests/test_intent_feedback_store.py`

**Step 1: Write the failing tests**

Create a small in-memory Redis double and test:

```python
def test_upsert_and_load_latest_label():
    redis_client = FakeRedis()
    store = RedisIntentFeedbackStore(client=redis_client)

    assert store.upsert("网总是断", "technical") is True
    assert store.upsert("网总是断", "billing") is False
    assert store.load_all() == [IntentFeedbackSample(message="网总是断", intent="billing")]
```

Also test that malformed persisted JSON is skipped rather than crashing startup.

**Step 2: Run tests to verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_feedback_store.py -q
```

Expected: FAIL because `core.intent_feedback_store` does not exist.

**Step 3: Implement the minimal store**

Implement:

```python
@dataclass(frozen=True)
class IntentFeedbackSample:
    message: str
    intent: str

class RedisIntentFeedbackStore:
    HASH_KEY = "intent:feedback:samples:v1"

    def __init__(self, redis_url=None, client=None): ...
    def upsert(self, message: str, intent: str) -> bool: ...
    def load_all(self) -> list[IntentFeedbackSample]: ...
```

Use a SHA-256 digest of normalized message text as the Redis hash field. Store UTF-8 JSON with message, intent, and update timestamp. `upsert()` returns whether a new field was created.

**Step 4: Run tests to verify GREEN**

Run the same pytest command. Expected: PASS.

**Step 5: Commit**

```powershell
git add core/intent_feedback_store.py tests/test_intent_feedback_store.py
git commit -m "feat: persist reviewed intent feedback in redis"
```

### Task 2: Context-aware cache and feedback-aware recognizer

**Files:**
- Modify: `core/intent_recognizer.py`
- Create: `tests/test_intent_recognizer_feedback.py`

**Step 1: Write failing cache tests**

Test the wished-for behavior without calling the remote LLM:

```python
def test_cache_key_changes_with_history(recognizer):
    a = recognizer._cache_key("还是不行", [{"role": "user", "content": "校园网401"}])
    b = recognizer._cache_key("还是不行", [{"role": "user", "content": "退款未到账"}])
    assert a != b

def test_cache_key_changes_after_learning(recognizer):
    before = recognizer._cache_key("网总是断", [])
    recognizer.learn("网总是断", IntentCategory.TECHNICAL)
    after = recognizer._cache_key("网总是断", [])
    assert before != after
```

Test that persisted feedback is loaded on construction, duplicate samples are not added twice, a sample is removed from the wrong category before being added to the corrected category, and learning clears stale result cache.

**Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_recognizer_feedback.py -q
```

Expected: FAIL because cache keys ignore history and the recognizer has no feedback-store integration.

**Step 3: Implement minimal behavior**

- Give each recognizer an instance-local copy of the built-in templates.
- Accept an optional feedback store.
- Load reviewed samples from the store during initialization without re-persisting them.
- Build cache keys from normalized message, last three normalized history items, model name, and SHA-256 template fingerprint.
- Make `learn()` normalize input, remove conflicting old labels, add the corrected label once, clear template embeddings and result cache, and optionally persist.
- Update LLM and embedding template loops to use instance templates.
- Make `_vote()` return final intent plus fused confidence, and expose that fused confidence in `IntentResult`.

**Step 4: Verify GREEN**

Run the new test file, then both test files. Expected: PASS.

**Step 5: Commit**

```powershell
git add core/intent_recognizer.py tests/test_intent_recognizer_feedback.py
git commit -m "fix: make intent cache context and feedback aware"
```

### Task 3: Shared recognizer and protected feedback API

**Files:**
- Modify: `agents/agent_orchestrator.py`
- Modify: `api/main.py`
- Create: `tests/test_intent_feedback_api.py`

**Step 1: Write failing tests**

Test:

```python
def test_orchestrator_uses_injected_recognizer():
    recognizer = object()
    orchestrator = AgentOrchestrator(..., intent_recognizer=recognizer)
    assert orchestrator.intent_recognizer is recognizer
```

Test the feedback endpoint helper:

```python
async def test_feedback_rejects_missing_admin_token(monkeypatch): ...
async def test_feedback_rejects_invalid_intent(monkeypatch): ...
async def test_feedback_persists_reviewed_sample(monkeypatch): ...
```

Expected endpoint request:

```json
{
  "message": "网总是断",
  "correct_intent": "technical"
}
```

Require `X-Intent-Admin-Token`. If `INTENT_FEEDBACK_ADMIN_TOKEN` is empty, return 503; incorrect token returns 403.

**Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_feedback_api.py -q
```

Expected: FAIL because dependency injection and endpoint do not exist.

**Step 3: Implement minimal API integration**

- Add optional `intent_recognizer` injection and a read-only property to `AgentOrchestrator`.
- Build `RedisIntentFeedbackStore` from `REDIS_URL` during FastAPI lifespan startup.
- Construct one recognizer with the store and pass it to both orchestrator and evaluator.
- Add `IntentFeedbackInput`.
- Add `POST /intent/feedback` with constant-time admin-token comparison.
- Return normalized message, corrected intent, whether the Redis field was newly created, and current template fingerprint; never return the admin token.

**Step 4: Verify GREEN**

Run the API test, then all tests. Expected: PASS.

**Step 5: Commit**

```powershell
git add agents/agent_orchestrator.py api/main.py tests/test_intent_feedback_api.py
git commit -m "feat: add protected intent feedback endpoint"
```

### Task 4: Documentation and full verification

**Files:**
- Create: `requirements-dev.txt`
- Modify: `docs/工程学习记录.md` in the main workspace after integration

**Step 1: Add development test dependency**

Create:

```text
-r requirements.txt
pytest==9.1.1
```

Do not add pytest to the production image requirements.

**Step 2: Run complete verification**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q api agents core
```

Expected: all tests pass and compile command exits 0.

**Step 3: Build and run in Docker**

After integrating the worktree commits into the main workspace:

```powershell
docker compose up -d --build echomind
docker compose ps
curl.exe http://localhost:8000/health
```

Configure a temporary local admin token without printing it. Submit one reviewed feedback sample, verify the Redis hash exists, restart only `echomind-app`, and verify the sample is loaded after restart. Remove only the exact demonstration sample afterward.

**Step 4: Record learning**

Append the implementation result, persistence path, API security behavior, test evidence, and remaining limits to `docs/工程学习记录.md`, then safely append the same concise result to the connected Notion page.

**Step 5: Commit**

```powershell
git add requirements-dev.txt docs/plans/2026-07-30-intent-feedback.md
git commit -m "docs: add intent feedback development workflow"
```
