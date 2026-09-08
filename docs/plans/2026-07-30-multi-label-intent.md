# Multi-Label Intent Routing Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Replace the keyword-only second-Agent detector with fused multi-label intent scores while preserving the existing primary intent and escalation behavior.

**Architecture:** LLM, Embedding, and Pattern strategies each return per-intent score maps. `IntentRecognizer` fuses those maps with the existing mode-dependent weights, keeps the highest score as the backwards-compatible primary intent, and exposes every intent above a configurable `0.60` multi-label threshold. `AgentOrchestrator` maps the matched specialist intents to at most Technical and Billing Agents; escalation or critical urgency still overrides normal parallel routing.

**Tech Stack:** Python 3.12, dataclasses, asyncio, FastAPI/Pydantic, pytest, Docker Compose.

---

### Task 1: Define and test fused multi-label intent output

**Files:**
- Modify: `core/intent_recognizer.py`
- Modify: `tests/test_intent_recognizer_feedback.py`

**Step 1: Write the failing tests**

Add tests that require:

```python
intent, confidence, scores, matched = recognizer._vote(
    {
        "scores": {
            IntentCategory.TECHNICAL: 0.91,
            IntentCategory.BILLING: 0.86,
        }
    },
    {"scores": {}},
    {"scores": {}},
)

assert intent == IntentCategory.TECHNICAL
assert scores[IntentCategory.TECHNICAL] > 0.60
assert scores[IntentCategory.BILLING] > 0.60
assert matched == [IntentCategory.TECHNICAL, IntentCategory.BILLING]
```

Also test that an old single-label strategy payload is normalized, and that a failed LLM causes the remaining strategy weights to be renormalized instead of suppressing all labels.

**Step 2: Run tests to verify RED**

Run:

```powershell
docker run --rm --user root -v "${worktree}:/app" -w /app echomind-echomind sh -lc "pip install -q -r requirements-dev.txt && python -m pytest tests/test_intent_recognizer_feedback.py -q"
```

Expected: FAIL because `_vote()` does not return `scores` or `matched`.

**Step 3: Implement the minimal fused score model**

In `IntentResult`, add:

```python
intent_scores: Dict[IntentCategory, float] = field(default_factory=dict)
matched_intents: List[IntentCategory] = field(default_factory=list)
```

Add `multi_label_threshold=0.60` to `IntentRecognizer.__init__`. Normalize each strategy result into a per-intent score map, fuse with `70/20/10` or `85/15`, renormalize around failed strategies, select the primary maximum, and return all non-`OTHER` intents whose fused score reaches the multi-label threshold.

**Step 4: Run focused tests to verify GREEN**

Run the focused command from Step 2.

Expected: all intent recognizer tests pass.

**Step 5: Commit**

```powershell
git add core/intent_recognizer.py tests/test_intent_recognizer_feedback.py
git commit -m "feat: add fused multi-label intent scores"
```

### Task 2: Make all three strategies produce multiple scores

**Files:**
- Modify: `core/intent_recognizer.py`
- Modify: `tests/test_intent_recognizer_feedback.py`

**Step 1: Write failing parser and strategy tests**

Require the LLM parser to accept:

```json
{
  "intents": [
    {"intent": "technical", "confidence": 0.91},
    {"intent": "billing", "confidence": 0.86}
  ],
  "reasoning": "同时包含登录故障和重复扣款"
}
```

Require Pattern to retain both technical and billing scores when both domains match. Require Embedding to retain one score per template category instead of only its maximum category.

**Step 2: Run focused tests to verify RED**

Expected: FAIL because each strategy currently returns one `intent`.

**Step 3: Implement strategy score maps**

- Update the LLM prompt to request an `intents` list and parse it defensively.
- Retain backwards compatibility with the old `intent/confidence` JSON.
- Return all category similarities from `_embedding_recognize()`.
- Return all hit-category scores from `_pattern_recognize()`.
- Keep few-shot examples and the latest three history messages in the LLM prompt.

**Step 4: Run focused tests to verify GREEN**

Expected: all focused tests pass.

**Step 5: Commit**

```powershell
git add core/intent_recognizer.py tests/test_intent_recognizer_feedback.py
git commit -m "feat: emit multi-label scores from intent strategies"
```

### Task 3: Route Agents from matched intents, not a second keyword scan

**Files:**
- Modify: `agents/agent_orchestrator.py`
- Modify: `tests/test_agent_orchestrator_routing.py`

**Step 1: Write failing routing tests**

Test:

```python
request = Request(
    message="换一种说法的复合问题",
    user_id="student-1",
    conv_id="conv-1",
    intent=IntentCategory.TECHNICAL,
    intent_scores={
        IntentCategory.TECHNICAL: 0.91,
        IntentCategory.BILLING: 0.86,
    },
    matched_intents=[
        IntentCategory.TECHNICAL,
        IntentCategory.BILLING,
    ],
    urgency=UrgencyLevel.LOW,
)
```

Require both specialist Agents to run even when the literal message contains none of the legacy domain keywords. Test that a billing score below `0.60` stays single-Agent and that escalation still overrides both.

**Step 2: Run routing tests to verify RED**

Expected: FAIL because `Request` lacks multi-label fields and routing still scans keywords.

**Step 3: Implement minimal score-driven routing**

- Add `intent_scores` and `matched_intents` to `Request`.
- Copy recognizer output into the request inside `run()`.
- Map matched `technical` to TechnicalAgent and matched `billing/account` to BillingAgent.
- Remove the legacy domain keyword scan from `_collaboration_targets()`.
- Keep a maximum of two specialist Agent types.
- Preserve escalation and critical urgency override.

**Step 4: Run routing tests and full suite**

Expected: routing tests and the full suite pass.

**Step 5: Commit**

```powershell
git add agents/agent_orchestrator.py tests/test_agent_orchestrator_routing.py
git commit -m "feat: route agents from multi-label intent scores"
```

### Task 4: Expose observable routing results through `/chat`

**Files:**
- Modify: `agents/agent_orchestrator.py`
- Modify: `api/main.py`
- Create: `tests/test_chat_response_model.py`

**Step 1: Write failing response-model tests**

Require `ChatResponse` to serialize:

```json
{
  "intent": "technical",
  "intent_scores": {"technical": 0.91, "billing": 0.86},
  "matched_intents": ["technical", "billing"],
  "agent_types": ["technical", "billing"]
}
```

**Step 2: Run focused test to verify RED**

Expected: FAIL because the response fields do not exist.

**Step 3: Add backwards-compatible response fields**

Keep `intent` and `agent_type`, and add:

```python
intent_scores: Dict[str, float] = {}
matched_intents: List[str] = []
agent_types: List[str] = []
```

Populate the values from `OrchestratorResult`, converting Enum values to strings.

**Step 4: Run focused and full tests**

Expected: all tests pass.

**Step 5: Commit**

```powershell
git add agents/agent_orchestrator.py api/main.py tests/test_chat_response_model.py
git commit -m "feat: expose multi-label routing in chat responses"
```

### Task 5: Document, rebuild, and verify

**Files:**
- Modify: `docs/工程学习记录.md`
- Update: Notion page `EchoMind 工程学习记录`

**Step 1: Update the learning note**

Document:

```text
Few-shot + recent history → LLM per-label scores
Embedding per-label similarities
Pattern per-label keyword scores
→ weighted fusion
→ primary intent + matched intents
→ one or two specialist Agents
```

Clarify that RAG runs only when `_should_use_knowledge()` allows it, and that the project currently has a knowledge search Tool but no real transaction, campus-card, schedule, repair, refund, or human-handoff APIs.

**Step 2: Sync Notion**

Fetch the existing page first, then append or locally update the related section without replacing unrelated content.

**Step 3: Verify**

Run:

```powershell
python -m compileall -q core agents api tests
docker run --rm --user root -v "${worktree}:/app" -w /app echomind-echomind sh -lc "pip install -q -r requirements-dev.txt && python -m pytest -q"
docker compose up -d --build echomind
docker compose ps
curl.exe -sS http://localhost:8000/health
```

Expected: compile exit `0`, all tests pass, `echomind-app` is healthy, and health endpoint returns `status=ok`.

