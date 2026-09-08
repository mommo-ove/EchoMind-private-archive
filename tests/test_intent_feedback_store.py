import json

from core.intent_feedback_store import IntentFeedbackSample, RedisIntentFeedbackStore


class FakeRedis:
    def __init__(self):
        self.hashes = {}

    def hset(self, name, key, value):
        bucket = self.hashes.setdefault(name, {})
        created = key not in bucket
        bucket[key] = value
        return 1 if created else 0

    def hvals(self, name):
        return list(self.hashes.get(name, {}).values())


def test_upsert_and_load_latest_label():
    redis_client = FakeRedis()
    store = RedisIntentFeedbackStore(client=redis_client)

    assert store.upsert("网总是断", "technical") is True
    assert store.upsert("网总是断", "billing") is False
    assert store.load_all() == [
        IntentFeedbackSample(message="网总是断", intent="billing")
    ]


def test_load_all_skips_malformed_records():
    redis_client = FakeRedis()
    store = RedisIntentFeedbackStore(client=redis_client)
    redis_client.hset(store.HASH_KEY, "broken-json", "{not-json")
    redis_client.hset(
        store.HASH_KEY,
        "missing-message",
        json.dumps({"intent": "technical"}, ensure_ascii=False),
    )
    store.upsert("登录失败", "technical")

    assert store.load_all() == [
        IntentFeedbackSample(message="登录失败", intent="technical")
    ]
