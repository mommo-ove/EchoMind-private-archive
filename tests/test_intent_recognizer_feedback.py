import asyncio
from types import SimpleNamespace

import pytest

from core.intent_feedback_store import IntentFeedbackSample
from core.intent_recognizer import IntentCategory, IntentRecognizer


class StubFeedbackStore:
    def __init__(self, samples=None):
        self.samples = list(samples or [])
        self.upserts = []

    def load_all(self):
        return list(self.samples)

    def upsert(self, message, intent):
        self.upserts.append((message, intent))
        return True


class FailingFeedbackStore(StubFeedbackStore):
    def upsert(self, message, intent):
        raise RuntimeError("redis unavailable")


def make_recognizer(feedback_store=None):
    return IntentRecognizer(
        api_key="test-key",
        base_url="https://example.invalid",
        model="test-model",
        feedback_store=feedback_store,
    )


def test_cache_key_changes_with_history():
    recognizer = make_recognizer()

    technical = recognizer._cache_key(
        "还是不行",
        [{"role": "user", "content": "校园网登录报401"}],
    )
    billing = recognizer._cache_key(
        "还是不行",
        [{"role": "user", "content": "退款一直没有到账"}],
    )

    assert technical != billing


def test_cache_key_changes_after_learning_and_stale_cache_is_cleared():
    recognizer = make_recognizer()
    before = recognizer._cache_key("网总是断", [])
    recognizer._cache["stale"] = object()

    recognizer.learn("网总是断", IntentCategory.TECHNICAL)

    after = recognizer._cache_key("网总是断", [])
    assert before != after
    assert recognizer._cache == {}


def test_persisted_feedback_loads_without_writing_back():
    store = StubFeedbackStore(
        [IntentFeedbackSample(message="网总是断", intent="technical")]
    )

    recognizer = make_recognizer(store)

    assert "网总是断" in recognizer._templates[IntentCategory.TECHNICAL]
    assert store.upserts == []


def test_learning_moves_conflicting_sample_and_persists_once():
    store = StubFeedbackStore(
        [IntentFeedbackSample(message="费用怎么扣的", intent="billing")]
    )
    recognizer = make_recognizer(store)

    created = recognizer.learn("费用怎么扣的", IntentCategory.QUERY)

    assert created is True
    assert "费用怎么扣的" in recognizer._templates[IntentCategory.QUERY]
    assert "费用怎么扣的" not in recognizer._templates[IntentCategory.BILLING]
    assert store.upserts == [("费用怎么扣的", "query")]


def test_learning_duplicate_label_is_idempotent():
    store = StubFeedbackStore()
    recognizer = make_recognizer(store)

    first = recognizer.learn("网总是断", IntentCategory.TECHNICAL)
    second = recognizer.learn("网总是断", IntentCategory.TECHNICAL)

    assert first is True
    assert second is False
    assert recognizer._templates[IntentCategory.TECHNICAL].count("网总是断") == 1
    assert store.upserts == [("网总是断", "technical")]


def test_learning_surfaces_persistence_failure():
    recognizer = make_recognizer(FailingFeedbackStore())
    recognizer._cache["stale"] = object()
    before_fingerprint = recognizer.template_fingerprint

    with pytest.raises(RuntimeError, match="redis unavailable"):
        recognizer.learn("网总是断", IntentCategory.TECHNICAL)

    assert recognizer.template_fingerprint == before_fingerprint
    assert "stale" in recognizer._cache


def test_reviewed_feedback_is_used_as_bounded_llm_few_shot():
    recognizer = make_recognizer()
    for index in range(4):
        recognizer.learn(f"校园网反馈样本{index}", IntentCategory.TECHNICAL)

    class CaptureMessages:
        prompt = ""
        temperature = None
        kwargs = {}

        async def create(self, **kwargs):
            self.kwargs = kwargs
            self.prompt = kwargs["messages"][0]["content"]
            self.temperature = kwargs["temperature"]
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        text='{"intent":"technical","confidence":0.9,"reasoning":"test"}'
                    )
                ]
            )

    capture = CaptureMessages()
    recognizer.client = SimpleNamespace(messages=capture)

    asyncio.run(recognizer._llm_recognize("校园网又断了", []))

    assert "校园网反馈样本3" in capture.prompt
    assert "校园网反馈样本2" in capture.prompt
    assert "校园网反馈样本0" not in capture.prompt
    assert capture.prompt.count("意图: technical") == 3


def test_vote_returns_fused_confidence():
    recognizer = make_recognizer()

    intent, confidence, scores, matched = recognizer._vote(
        {"intent": IntentCategory.TECHNICAL, "confidence": 0.8},
        {"intent": IntentCategory.OTHER, "confidence": 0.0},
        {"intent": IntentCategory.TECHNICAL, "confidence": 0.4},
    )

    assert intent == IntentCategory.TECHNICAL
    assert confidence == 0.74
    assert scores[IntentCategory.TECHNICAL] == 0.74
    assert matched == [IntentCategory.TECHNICAL]


def test_vote_uses_calibrated_three_way_weights_when_configured():
    recognizer = IntentRecognizer(
        api_key="test-key",
        base_url="https://example.invalid",
        embedding_enabled=True,
        confidence_threshold=0.3,
        strategy_weights={"llm": 0.3, "embedding": 0.5, "pattern": 0.2},
    )

    intent, confidence, scores, _matched = recognizer._vote(
        {"scores": {IntentCategory.TECHNICAL: 0.6}},
        {"scores": {IntentCategory.BILLING: 1.0}},
        {"scores": {IntentCategory.TECHNICAL: 1.0}},
    )

    assert intent == IntentCategory.BILLING
    assert confidence == pytest.approx(0.5)
    assert scores[IntentCategory.TECHNICAL] == pytest.approx(0.38)


def test_strategy_weights_must_be_non_negative_and_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        IntentRecognizer(
            api_key="test-key",
            strategy_weights={"llm": 0.7, "embedding": 0.7, "pattern": -0.4},
        )


def test_vote_returns_all_intents_above_multi_label_threshold():
    recognizer = make_recognizer()

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
    assert confidence == pytest.approx(0.91)
    assert scores[IntentCategory.TECHNICAL] == pytest.approx(0.91)
    assert scores[IntentCategory.BILLING] == pytest.approx(0.86)
    assert matched == [IntentCategory.TECHNICAL, IntentCategory.BILLING]


def test_vote_renormalizes_remaining_strategies_when_llm_fails():
    recognizer = make_recognizer()

    intent, confidence, scores, matched = recognizer._vote(
        {"scores": {}, "failed": True},
        {"scores": {}},
        {"scores": {IntentCategory.BILLING: 0.8}},
    )

    assert intent == IntentCategory.BILLING
    assert confidence == pytest.approx(0.8)
    assert scores[IntentCategory.BILLING] == pytest.approx(0.8)
    assert matched == [IntentCategory.BILLING]


def test_llm_recognizer_parses_multiple_intent_scores_and_keeps_context():
    recognizer = IntentRecognizer(
        api_key="test-key",
        base_url="https://api.deepseek.com/anthropic",
        model="test-model",
    )

    class CaptureMessages:
        prompt = ""
        temperature = None
        kwargs = {}

        async def create(self, **kwargs):
            self.kwargs = kwargs
            self.prompt = kwargs["messages"][0]["content"]
            self.temperature = kwargs["temperature"]
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        text=(
                            '{"intents":['
                            '{"intent":"technical","confidence":0.91},'
                            '{"intent":"billing","confidence":0.86}'
                            '],"reasoning":"登录故障和重复扣款"}'
                        )
                    )
                ]
            )

    capture = CaptureMessages()
    recognizer.client = SimpleNamespace(messages=capture)

    result = asyncio.run(
        recognizer._llm_recognize(
            "登录失败，而且重复扣款",
            [{"role": "user", "content": "校园网一直报 401"}],
        )
    )

    assert result["scores"] == {
        IntentCategory.TECHNICAL: 0.91,
        IntentCategory.BILLING: 0.86,
    }
    assert "校园网一直报 401" in capture.prompt
    assert '"intents"' in capture.prompt
    assert "领域意图和动作意图不是互斥类别" in capture.prompt
    assert "校园网401应该怎么排查" in capture.prompt
    assert "technical" in capture.prompt
    assert "query" in capture.prompt
    assert "校园卡消费、充值、余额、扣款、退款属于 billing" in capture.prompt
    assert "统一身份认证、密码、绑定邮箱、账号锁定属于 account" in capture.prompt
    assert "明确要求转人工" in capture.prompt
    assert capture.temperature == 0.0
    assert capture.kwargs["extra_body"] == {
        "thinking": {"type": "disabled"},
    }


def test_pattern_recognizer_keeps_all_matching_categories():
    recognizer = make_recognizer()

    result = recognizer._pattern_recognize("应用报错，而且还被重复扣款")

    assert result["scores"][IntentCategory.TECHNICAL] > 0
    assert result["scores"][IntentCategory.BILLING] > 0


def test_pattern_recognizer_uses_binary_strong_domain_evidence():
    recognizer = make_recognizer()

    billing = recognizer._pattern_recognize("查询最近7天校园卡消费")
    escalation = recognizer._pattern_recognize("马上给我转人工")

    assert billing["scores"][IntentCategory.BILLING] == 1.0
    assert escalation["scores"][IntentCategory.ESCALATION] == 1.0


def test_embedding_recognizer_keeps_score_for_each_template_category():
    recognizer = make_recognizer()
    recognizer._tpl_embeddings = {
        IntentCategory.TECHNICAL: [[1.0, 0.0]],
        IntentCategory.BILLING: [[0.8, 0.6]],
        IntentCategory.GREETING: [[0.0, 1.0]],
    }

    async def already_loaded():
        return None

    async def message_vector(_message):
        return [1.0, 0.0]

    recognizer._load_template_embeddings = already_loaded
    recognizer._embed_text = message_vector

    result = asyncio.run(recognizer._embedding_recognize("复合问题"))

    assert result["scores"][IntentCategory.TECHNICAL] == pytest.approx(1.0)
    assert result["scores"][IntentCategory.BILLING] == pytest.approx(0.8)
    assert result["scores"][IntentCategory.GREETING] == pytest.approx(0.0)
