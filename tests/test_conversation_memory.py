import asyncio
import json
from types import SimpleNamespace

import pytest
from redis.exceptions import WatchError

from memory.conversation_memory import (
    MemoryManager,
    MsgRole,
    OperationStatus,
)


TTL = 86400


def test_operation_statuses_use_public_claim_and_progress_names():
    assert OperationStatus.CLAIMED.value == "claimed"
    assert OperationStatus.IN_PROGRESS.value == "in_progress"


class FakeMemoryEmbedder:
    model_name = "BAAI/bge-small-zh-v1.5"

    def embed(self, texts):
        return [[0.2, 0.8] for _ in texts]

    def embed_query(self, text):
        return [0.7, 0.3]


class FakeSemanticMemoryCollection:
    def __init__(self):
        self.add_kwargs = None
        self.query_kwargs = None

    def add(self, **kwargs):
        self.add_kwargs = kwargs

    def query(self, **kwargs):
        self.query_kwargs = kwargs
        return {"documents": [["上周通过清理旧认证恢复"]]}


def test_episodic_memory_uses_shared_bge_vectors_for_storage_and_retrieval():
    manager = object.__new__(MemoryManager)
    manager._embedder = FakeMemoryEmbedder()
    manager._episodic = FakeSemanticMemoryCollection()

    asyncio.run(manager._store_episodic(
        "student-1", "conv-1", "校园网401", "清理旧认证后恢复"
    ))
    history = asyncio.run(manager._search_episodic("student-1", "又出现401"))

    assert manager._episodic.add_kwargs["embeddings"] == [[0.2, 0.8]]
    assert manager._episodic.query_kwargs["query_embeddings"] == [[0.7, 0.3]]
    assert history == ["上周通过清理旧认证恢复"]
WM_KEY = MemoryManager._wm_key("principal", "conv")
REVISION_KEY = MemoryManager._revision_key("principal", "conv")
SUMMARY_KEY = MemoryManager._summary_key("principal", "conv")


class FakeRedis:
    def __init__(self):
        self.lists = {}
        self.strings = {}
        self.markers = set()
        self.ttls = {}
        self.eval_calls = []
        self.ambiguous_once = False
        self.ambiguous_commit_once = False
        self.fail_rebuild = False
        self.watch_conflict = False
        self.rebuild_attempts = 0
        self.operation_records = {}
        self.set_calls = []
        self.lrange_calls = []

    def pipeline(self, transaction=True):
        return FakePipeline(self, transaction=transaction)

    def eval(self, script, key_count, *args):
        self.eval_calls.append((script, key_count, args))
        if "CLAIM_OPERATION" in script:
            assert key_count == 1
            key, payload_hash, owner_id, ttl = args
            record = self.operation_records.get(key)
            if record is None:
                self.operation_records[key] = {
                    "state": "pending",
                    "payload_hash": payload_hash,
                    "owner": owner_id,
                    "payload": "",
                }
                self.ttls[key] = int(ttl)
                return ["claimed", ""]
            if record["payload_hash"] != payload_hash:
                return ["conflict", ""]
            status = (
                "in_progress"
                if record["state"] == "pending"
                else record["state"]
            )
            return [status, record.get("payload", "")]
        if "COMMIT_OPERATION" in script:
            assert key_count == 3
            (
                operation_key,
                working_key,
                revision_key,
                payload_hash,
                owner_id,
                user_payload,
                assistant_payload,
                payload,
                ttl,
            ) = args
            record = self.operation_records.get(operation_key)
            if (
                record is None
                or record["state"] != "pending"
                or record["payload_hash"] != payload_hash
                or record["owner"] != owner_id
            ):
                return 0
            self.lists.setdefault(working_key, [])[0:0] = [
                assistant_payload,
                user_payload,
            ]
            self.strings[revision_key] = (
                int(self.strings.get(revision_key, 0)) + 1
            )
            record.update(
                state="completed",
                owner="",
                payload=payload,
            )
            self.ttls[operation_key] = int(ttl)
            self.ttls[working_key] = int(ttl)
            self.ttls[revision_key] = int(ttl)
            if self.ambiguous_commit_once:
                self.ambiguous_commit_once = False
                raise ConnectionError("response lost after operation commit")
            return len(self.lists[working_key])
        if "RELEASE_OPERATION" in script:
            assert key_count == 1
            key, payload_hash, owner_id = args
            record = self.operation_records.get(key)
            if (
                record is None
                or record["state"] != "pending"
                or record["payload_hash"] != payload_hash
                or record["owner"] != owner_id
            ):
                return 0
            del self.operation_records[key]
            self.ttls.pop(key, None)
            return 1
        if "RELEASE_COMPRESSION_LEASE" in script:
            assert key_count == 1
            key, owner_id = args
            if self.strings.get(key) != owner_id:
                return 0
            self.strings.pop(key, None)
            self.ttls.pop(key, None)
            return 1
        assert key_count == 3
        (
            working_key,
            marker_key,
            revision_key,
            user_payload,
            assistant_payload,
            ttl,
        ) = args
        if marker_key in self.markers:
            return -1
        staged_lists = {
            key: list(value) for key, value in self.lists.items()
        }
        staged_strings = dict(self.strings)
        staged_markers = set(self.markers)
        staged_ttls = dict(self.ttls)
        staged_lists.setdefault(working_key, [])[0:0] = [
            assistant_payload,
            user_payload,
        ]
        staged_markers.add(marker_key)
        staged_strings[revision_key] = (
            int(staged_strings.get(revision_key, 0)) + 1
        )
        staged_ttls[working_key] = int(ttl)
        staged_ttls[marker_key] = int(ttl)
        staged_ttls[revision_key] = int(ttl)
        self.lists = staged_lists
        self.strings = staged_strings
        self.markers = staged_markers
        self.ttls = staged_ttls
        if self.ambiguous_once:
            self.ambiguous_once = False
            raise ConnectionError("response lost after commit")
        return len(self.lists[working_key])

    def get(self, key):
        value = self.strings.get(key)
        return None if value is None else str(value)

    def lrange(self, key, start, stop):
        self.lrange_calls.append((key, start, stop))
        values = self.lists.get(key, [])
        if stop == -1:
            return list(values[start:])
        return list(values[start:stop + 1])

    def set(self, key, value, *, nx=False, ex=None):
        self.set_calls.append((key, value, nx, ex))
        if nx and key in self.strings:
            return False
        self.strings[key] = value
        if ex is not None:
            self.ttls[key] = ex
        return True

    def append_concurrently(self, working_key, revision_key, payload):
        self.lists.setdefault(working_key, []).insert(0, payload)
        self.strings[revision_key] = (
            int(self.strings.get(revision_key, 0)) + 1
        )
        self.ttls[working_key] = TTL
        self.ttls[revision_key] = TTL


class FakePipeline:
    def __init__(self, redis, *, transaction):
        self.redis = redis
        self.transaction = transaction
        self.commands = []
        self.watching = False
        self.in_multi = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def watch(self, *_keys):
        self.watching = True
        return True

    def unwatch(self):
        self.watching = False
        return True

    def multi(self):
        self.in_multi = True
        return self

    def get(self, key):
        if self.watching and not self.in_multi:
            return self.redis.get(key)
        self.commands.append(("get", key))
        return self

    def lrange(self, key, start, stop):
        self.redis.lrange_calls.append((key, start, stop))
        self.commands.append(("lrange", key, start, stop))
        return self

    def lpush(self, key, *values):
        self.commands.append(("lpush", key, *values))
        return self

    def expire(self, key, seconds):
        self.commands.append(("expire", key, seconds))
        return self

    def incr(self, key):
        self.commands.append(("incr", key))
        return self

    def llen(self, key):
        self.commands.append(("llen", key))
        return self

    def delete(self, key):
        self.commands.append(("delete", key))
        return self

    def setex(self, key, seconds, value):
        self.commands.append(("setex", key, seconds, value))
        return self

    def execute(self):
        is_rebuild = any(command[0] == "delete" for command in self.commands)
        if is_rebuild:
            self.redis.rebuild_attempts += 1
            if self.redis.watch_conflict:
                raise WatchError("revision changed")
            if self.redis.fail_rebuild:
                raise ConnectionError("transaction failed")

        lists = {key: list(value) for key, value in self.redis.lists.items()}
        strings = dict(self.redis.strings)
        ttls = dict(self.redis.ttls)
        results = []
        for command in self.commands:
            operation, *args = command
            if operation == "get":
                value = strings.get(args[0])
                results.append(None if value is None else str(value))
            elif operation == "lrange":
                key, start, stop = args
                values = lists.get(key, [])
                results.append(
                    list(values[start:] if stop == -1 else values[start:stop + 1])
                )
            elif operation == "lpush":
                key, *values = args
                target = lists.setdefault(key, [])
                for value in values:
                    target.insert(0, value)
                results.append(len(target))
            elif operation == "expire":
                key, seconds = args
                ttls[key] = seconds
                results.append(True)
            elif operation == "incr":
                key = args[0]
                strings[key] = int(strings.get(key, 0)) + 1
                results.append(strings[key])
            elif operation == "llen":
                results.append(len(lists.get(args[0], [])))
            elif operation == "delete":
                lists.pop(args[0], None)
                results.append(1)
            elif operation == "setex":
                key, seconds, value = args
                strings[key] = value
                ttls[key] = seconds
                results.append(True)
        self.redis.lists = lists
        self.redis.strings = strings
        self.redis.ttls = ttls
        return results


class FakeMessages:
    def __init__(self, before_response=None):
        self.before_response = before_response
        self.calls = 0

    async def create(self, **_kwargs):
        self.calls += 1
        if self.before_response is not None:
            self.before_response()
        return SimpleNamespace(content=["compressed summary"])


class FakeChromaCollection:
    def __init__(self):
        self.records = {}
        self.deleted = []

    def delete(self, *, ids):
        self.deleted.extend(ids)
        for doc_id in ids:
            self.records.pop(doc_id, None)

    def add(self, *, ids, documents, metadatas, embeddings=None):
        for doc_id, document, metadata in zip(
            ids, documents, metadatas
        ):
            self.records[doc_id] = (document, dict(metadata))

    def get(self, *, where, limit):
        documents = [
            document
            for document, metadata in self.records.values()
            if all(metadata.get(key) == value for key, value in where.items())
        ][:limit]
        return {"documents": documents}


def memory_with(redis, *, before_summary=None):
    manager = object.__new__(MemoryManager)
    manager._redis = redis
    manager._client = SimpleNamespace(
        messages=FakeMessages(before_summary)
    )
    manager._model = "test-model"
    manager._embedder = FakeMemoryEmbedder()

    async def store_episodic(*_args):
        return None

    manager._store_episodic = store_episodic
    return manager


def test_profile_documents_use_stable_collision_safe_scope_ids():
    redis = FakeRedis()
    manager = memory_with(redis)
    manager._profile = FakeChromaCollection()
    left = ("tenant_profile_alice", "support")
    right = ("tenant", "alice_profile_support")
    for scope, content in ((left, "left"), (right, "right")):
        redis.lists[manager._wm_key(*scope)] = [
            encoded(MsgRole.USER, content)
        ]

    class ProfileMessages:
        def __init__(self):
            self.calls = 0

        async def create(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(content=[json.dumps({
                "preferences": [f"profile-{self.calls}"],
                "entities": {},
            })])

    manager._client.messages = ProfileMessages()

    asyncio.run(manager.update_profile(*left))
    asyncio.run(manager.update_profile(*right))

    left_id = manager._profile_doc_id(*left)
    right_id = manager._profile_doc_id(*right)
    assert left_id != right_id
    assert left_id == manager._profile_doc_id(*left)
    assert set(manager._profile.records) == {left_id, right_id}
    assert asyncio.run(manager._get_profile(left[0]))["preferences"] == [
        "profile-1"
    ]
    assert asyncio.run(manager._get_profile(right[0]))["preferences"] == [
        "profile-2"
    ]

    asyncio.run(manager.update_profile(*left))
    assert set(manager._profile.records) == {left_id, right_id}
    assert manager._profile.deleted[-1] == left_id
    assert asyncio.run(manager._get_profile(right[0]))["preferences"] == [
        "profile-2"
    ]


def test_episodic_document_ids_isolate_ambiguous_scopes(monkeypatch):
    import memory.conversation_memory as conversation_memory

    monkeypatch.setattr(conversation_memory.time, "time", lambda: 42.0)
    redis = FakeRedis()
    manager = memory_with(redis)
    manager._episodic = FakeChromaCollection()
    left = ("tenant_profile_alice", "support")
    right = ("tenant_profile_", "alicesupport")

    asyncio.run(MemoryManager._store_episodic(
        manager, *left, "left text", "left summary"
    ))
    asyncio.run(MemoryManager._store_episodic(
        manager, *right, "right text", "right summary"
    ))

    ids = list(manager._episodic.records)
    assert len(ids) == 2
    assert ids[0] != ids[1]
    assert ids[0].startswith(
        f"episodic:{manager._scope_component(*left)}:"
    )
    assert ids[1].startswith(
        f"episodic:{manager._scope_component(*right)}:"
    )
    assert all(len(doc_id) <= 128 for doc_id in ids)


def test_operation_claim_replay_conflict_and_owner_safe_release():
    redis = FakeRedis()
    manager = memory_with(redis)

    first = asyncio.run(manager.claim_operation(
        "principal",
        "conv",
        "a" * 64,
        "b" * 64,
        "owner-1",
    ))
    operation_key = redis.eval_calls[0][2][0]
    assert 30 <= manager.OPERATION_PENDING_TTL_SECONDS <= 120
    assert redis.ttls[operation_key] == manager.OPERATION_PENDING_TTL_SECONDS
    pending = asyncio.run(manager.claim_operation(
        "principal",
        "conv",
        "a" * 64,
        "b" * 64,
        "owner-2",
    ))
    conflict = asyncio.run(manager.claim_operation(
        "principal",
        "conv",
        "a" * 64,
        "c" * 64,
        "owner-2",
    ))

    assert first.status is OperationStatus.CLAIMED
    assert pending.status is OperationStatus.IN_PROGRESS
    assert conflict.status is OperationStatus.CONFLICT
    assert asyncio.run(manager.release_operation(
        "principal", "conv", "a" * 64, "b" * 64, "owner-2"
    )) is False
    assert asyncio.run(manager.commit_operation(
        "principal",
        "conv",
        "a" * 64,
        "b" * 64,
        "owner-1",
        "question",
        "answer",
        '{"version":1,"result":{}}',
    )) is True

    completed = asyncio.run(manager.claim_operation(
        "principal",
        "conv",
        "a" * 64,
        "b" * 64,
        "owner-3",
    ))
    assert completed.status is OperationStatus.COMPLETED
    assert completed.payload == '{"version":1,"result":{}}'
    assert "principal" not in operation_key
    assert "conv" not in operation_key
    assert redis.ttls[operation_key] == manager.OPERATION_COMPLETED_TTL_SECONDS
    assert (
        manager.OPERATION_PENDING_TTL_SECONDS
        < manager.OPERATION_COMPLETED_TTL_SECONDS
    )


def test_operation_claim_uses_redis_32_compatible_single_field_hset_calls():
    redis = FakeRedis()
    manager = memory_with(redis)

    asyncio.run(manager.claim_operation(
        "principal", "conv", "a" * 64, "b" * 64, "owner"
    ))

    script = redis.eval_calls[0][0]
    assert script.count("redis.call('HSET', KEYS[1]") == 4


def test_operation_commit_uses_redis_32_compatible_single_field_hset_calls():
    redis = FakeRedis()
    manager = memory_with(redis)
    operation_id = "a" * 64
    payload_hash = "b" * 64

    asyncio.run(manager.claim_operation(
        "principal", "conv", operation_id, payload_hash, "owner"
    ))
    asyncio.run(manager.commit_operation(
        "principal",
        "conv",
        operation_id,
        payload_hash,
        "owner",
        "question",
        "answer",
        '{"version":1,"result":{}}',
    ))

    script = redis.eval_calls[1][0]
    assert script.count("redis.call('HSET', KEYS[1]") == 3


def test_operation_payload_is_bounded_before_redis_write():
    redis = FakeRedis()
    manager = memory_with(redis)

    asyncio.run(manager.claim_operation(
        "principal", "conv", "a" * 64, "b" * 64, "owner"
    ))
    with pytest.raises(ValueError, match="payload"):
        asyncio.run(manager.commit_operation(
            "principal",
            "conv",
            "a" * 64,
            "b" * 64,
            "owner",
            "question",
            "answer",
            "x" * (manager.OPERATION_PAYLOAD_MAX_BYTES + 1),
        ))

    assert len(redis.eval_calls) == 1


def test_canonical_scope_keys_do_not_collide_on_delimiters_or_cross_read():
    redis = FakeRedis()
    manager = memory_with(redis)
    left = ("tenant:alice", "support")
    right = ("tenant", "alice:support")

    left_keys = {
        manager._wm_key(*left),
        manager._revision_key(*left),
        manager._summary_key(*left),
        manager._exchange_marker_key(*left, "exchange"),
        manager._operation_key(*left, "a" * 64),
        manager._compression_lease_key(*left),
    }
    right_keys = {
        manager._wm_key(*right),
        manager._revision_key(*right),
        manager._summary_key(*right),
        manager._exchange_marker_key(*right, "exchange"),
        manager._operation_key(*right, "a" * 64),
        manager._compression_lease_key(*right),
    }

    assert left_keys.isdisjoint(right_keys)
    assert manager._wm_key(*left) == manager._wm_key(*left)
    assert not any(raw in repr(left_keys | right_keys) for raw in left + right)

    asyncio.run(manager.add_exchange(
        *left, "left question", "left answer", exchange_key="left"
    ))
    asyncio.run(manager.add_exchange(
        *right, "right question", "right answer", exchange_key="right"
    ))
    left_messages = asyncio.run(manager._get_working_memory(*left))
    right_messages = asyncio.run(manager._get_working_memory(*right))
    assert [item.content for item in left_messages] == [
        "left question", "left answer"
    ]
    assert [item.content for item in right_messages] == [
        "right question", "right answer"
    ]


def test_commit_operation_atomically_writes_exchange_and_replay_result():
    redis = FakeRedis()
    manager = memory_with(redis)
    operation_id = "a" * 64
    payload_hash = "b" * 64
    owner_id = "owner"
    asyncio.run(manager.claim_operation(
        "principal", "conv", operation_id, payload_hash, owner_id
    ))
    before = len(redis.eval_calls)

    committed = asyncio.run(manager.commit_operation(
        "principal",
        "conv",
        operation_id,
        payload_hash,
        owner_id,
        "question",
        "answer",
        '{"version":1,"result":{}}',
    ))

    assert committed is True
    assert len(redis.eval_calls) == before + 1
    assert "COMMIT_OPERATION" in redis.eval_calls[-1][0]
    messages = asyncio.run(manager._get_working_memory("principal", "conv"))
    assert [item.content for item in messages] == ["question", "answer"]
    replay = asyncio.run(manager.claim_operation(
        "principal", "conv", operation_id, payload_hash, "other-owner"
    ))
    assert replay.status is OperationStatus.COMPLETED
    assert replay.payload == '{"version":1,"result":{}}'


def test_retry_after_ambiguous_operation_commit_replays_without_duplicate():
    redis = FakeRedis()
    manager = memory_with(redis)
    operation_id = "a" * 64
    payload_hash = "b" * 64
    payload = '{"version":1,"result":{}}'
    asyncio.run(manager.claim_operation(
        "principal", "conv", operation_id, payload_hash, "owner"
    ))
    redis.ambiguous_commit_once = True

    with pytest.raises(ConnectionError, match="response lost"):
        asyncio.run(manager.commit_operation(
            "principal",
            "conv",
            operation_id,
            payload_hash,
            "owner",
            "question",
            "answer",
            payload,
        ))

    replay = asyncio.run(manager.claim_operation(
        "principal", "conv", operation_id, payload_hash, "retry-owner"
    ))
    messages = asyncio.run(
        manager._get_working_memory("principal", "conv")
    )
    assert replay.status is OperationStatus.COMPLETED
    assert replay.payload == payload
    assert [item.content for item in messages] == ["question", "answer"]


def encoded(role, content):
    return json.dumps({
        "role": role.value,
        "content": content,
        "ts": "2026-07-31T12:00:00",
        "metadata": {},
    })


def seed_messages(redis, count=15):
    working_key = WM_KEY
    revision_key = REVISION_KEY
    chronological = [
        encoded(
            MsgRole.USER if index % 2 == 0 else MsgRole.ASSISTANT,
            f"message-{index}",
        )
        for index in range(count)
    ]
    redis.lists[working_key] = list(reversed(chronological))
    redis.strings[revision_key] = count
    redis.ttls[working_key] = TTL
    redis.ttls[revision_key] = TTL
    return chronological


def test_add_message_updates_list_revision_and_ttls_atomically():
    redis = FakeRedis()
    manager = memory_with(redis)

    asyncio.run(manager.add_message(
        "principal", "conv", MsgRole.USER, "question"
    ))

    assert len(redis.lists[WM_KEY]) == 1
    assert redis.strings[REVISION_KEY] == 1
    assert redis.ttls[WM_KEY] == TTL
    assert redis.ttls[REVISION_KEY] == TTL


def test_add_exchange_requires_key_and_commits_once_with_revision():
    redis = FakeRedis()
    manager = memory_with(redis)

    with pytest.raises(TypeError):
        asyncio.run(manager.add_exchange(
            "principal", "conv", "question", "answer"
        ))

    inserted = asyncio.run(manager.add_exchange(
        "principal",
        "conv",
        "question",
        "answer",
        exchange_key="operation-1",
    ))

    assert inserted is True
    assert redis.strings[REVISION_KEY] == 1
    assert len(redis.lists[WM_KEY]) == 2
    marker_key = redis.eval_calls[0][2][1]
    assert "operation-1" not in marker_key
    assert redis.ttls[marker_key] == TTL


def test_add_exchange_same_key_is_idempotent_and_different_key_appends():
    redis = FakeRedis()
    manager = memory_with(redis)

    async def add(key):
        return await manager.add_exchange(
            "principal",
            "conv",
            "question",
            "answer",
            exchange_key=key,
        )

    assert asyncio.run(add("same-key")) is True
    assert asyncio.run(add("same-key")) is False
    assert asyncio.run(add("different-key")) is True
    assert len(redis.lists[WM_KEY]) == 4
    assert redis.strings[REVISION_KEY] == 2


def test_add_exchange_retry_after_ambiguous_commit_does_not_duplicate():
    redis = FakeRedis()
    redis.ambiguous_once = True
    manager = memory_with(redis)

    with pytest.raises(ConnectionError, match="response lost"):
        asyncio.run(manager.add_exchange(
            "principal",
            "conv",
            "question",
            "answer",
            exchange_key="retry-key",
        ))

    duplicate = asyncio.run(manager.add_exchange(
        "principal",
        "conv",
        "question",
        "answer",
        exchange_key="retry-key",
    ))

    assert duplicate is False
    assert len(redis.lists[WM_KEY]) == 2
    assert redis.strings[REVISION_KEY] == 1


def test_compression_aborts_when_writer_changes_revision_during_llm():
    redis = FakeRedis()
    original = seed_messages(redis)
    concurrent = encoded(MsgRole.USER, "concurrent-message")

    def append_while_summarizing():
        redis.append_concurrently(
            WM_KEY,
            REVISION_KEY,
            concurrent,
        )

    manager = memory_with(redis, before_summary=append_while_summarizing)

    asyncio.run(manager._compress("principal", "conv"))

    assert redis.lists[WM_KEY] == [concurrent, *reversed(original)]
    assert redis.rebuild_attempts == 0


def test_compression_transaction_failure_leaves_original_list_intact():
    redis = FakeRedis()
    original = seed_messages(redis)
    redis.fail_rebuild = True
    manager = memory_with(redis)

    asyncio.run(manager._compress("principal", "conv"))

    assert redis.lists[WM_KEY] == list(reversed(original))
    assert redis.strings[REVISION_KEY] == 15
    assert redis.rebuild_attempts == 1


def test_compression_watch_conflict_is_bounded_and_preserves_memory():
    redis = FakeRedis()
    original = seed_messages(redis)
    redis.watch_conflict = True
    manager = memory_with(redis)

    asyncio.run(manager._compress("principal", "conv"))

    assert redis.lists[WM_KEY] == list(reversed(original))
    assert redis.rebuild_attempts == 1


def test_successful_compression_is_atomic_ordered_and_refreshes_ttls():
    redis = FakeRedis()
    original = seed_messages(redis)
    manager = memory_with(redis)

    asyncio.run(manager._compress("principal", "conv"))
    messages = asyncio.run(
        manager._get_working_memory("principal", "conv")
    )

    assert [message.content for message in messages] == [
        "compressed summary",
        *[f"message-{index}" for index in range(10, 15)],
    ]
    assert messages[0].role is MsgRole.SYSTEM
    assert redis.strings[REVISION_KEY] == 16
    assert redis.strings[SUMMARY_KEY] == "compressed summary"
    assert redis.ttls[WM_KEY] == TTL
    assert redis.ttls[REVISION_KEY] == TTL
    assert redis.ttls[SUMMARY_KEY] == TTL


def test_compression_snapshot_is_bounded_and_lease_prevents_duplicate_llm():
    redis = FakeRedis()
    seed_messages(redis, count=MemoryManager.WORKING_MAX + 10)
    manager = memory_with(redis)

    lease_key = manager._compression_lease_key("principal", "conv")
    redis.strings[lease_key] = "another-owner"
    asyncio.run(manager._compress("principal", "conv"))

    assert manager._client.messages.calls == 0
    assert redis.lrange_calls == []

    redis.strings.pop(lease_key)
    asyncio.run(manager._compress("principal", "conv"))

    assert manager._client.messages.calls == 1
    assert (
        WM_KEY,
        0,
        MemoryManager.WORKING_MAX - 1,
    ) in redis.lrange_calls
    assert lease_key not in redis.strings
    assert redis.set_calls[-1][2:] == (
        True,
        manager.COMPRESSION_LEASE_SECONDS,
    )


def test_compression_release_does_not_delete_a_replaced_lease():
    redis = FakeRedis()
    seed_messages(redis)
    manager = memory_with(redis)
    lease_key = manager._compression_lease_key("principal", "conv")

    def replace_lease():
        redis.strings[lease_key] = "new-owner"

    manager._client.messages.before_response = replace_lease
    asyncio.run(manager._compress("principal", "conv"))

    assert redis.strings[lease_key] == "new-owner"


def test_compression_cancellation_releases_owned_lease():
    redis = FakeRedis()
    seed_messages(redis)
    manager = memory_with(redis)
    lease_key = manager._compression_lease_key("principal", "conv")

    async def exercise():
        started = asyncio.Event()

        async def block(**_kwargs):
            started.set()
            await asyncio.Event().wait()

        manager._client.messages.create = block
        task = asyncio.create_task(manager._compress("principal", "conv"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    assert lease_key not in redis.strings
