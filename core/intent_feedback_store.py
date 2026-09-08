"""Redis-backed storage for human-reviewed intent corrections."""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import redis


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntentFeedbackSample:
    """A reviewed message and its corrected intent label."""

    message: str
    intent: str


class RedisIntentFeedbackStore:
    """Persist reviewed intent samples in one Redis hash."""

    HASH_KEY = "intent:feedback:samples:v1"

    def __init__(
        self,
        redis_url: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        if client is None:
            if not redis_url:
                raise ValueError("redis_url or client is required")
            client = redis.from_url(redis_url, decode_responses=True)
        self._redis = client

    def upsert(self, message: str, intent: str) -> bool:
        """Insert or replace one reviewed sample; return True when newly created."""
        normalized_message = str(message or "").strip()
        normalized_intent = str(intent or "").strip().lower()
        if not normalized_message:
            raise ValueError("message must not be empty")
        if not normalized_intent:
            raise ValueError("intent must not be empty")

        field = hashlib.sha256(normalized_message.encode("utf-8")).hexdigest()
        payload = json.dumps(
            {
                "message": normalized_message,
                "intent": normalized_intent,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
        )
        return bool(self._redis.hset(self.HASH_KEY, field, payload))

    def load_all(self) -> list[IntentFeedbackSample]:
        """Load valid reviewed samples, skipping corrupted Redis values."""
        samples: list[IntentFeedbackSample] = []
        for raw in self._redis.hvals(self.HASH_KEY):
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                data = json.loads(raw)
                message = str(data["message"]).strip()
                intent = str(data["intent"]).strip().lower()
                if not message or not intent:
                    raise ValueError("empty feedback field")
                samples.append(IntentFeedbackSample(message=message, intent=intent))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as ex:
                logger.warning("跳过损坏的意图反馈记录: %s", ex)

        return sorted(samples, key=lambda sample: (sample.message, sample.intent))
