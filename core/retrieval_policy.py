"""Intent-aware policy for deciding whether a message needs knowledge retrieval."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional, Union

from core.intent_recognizer import IntentCategory


IntentInput = Union[IntentCategory, str]

DEFAULT_RETRIEVAL_INTENTS = frozenset(
    {
        IntentCategory.QUERY,
        IntentCategory.REQUEST,
        IntentCategory.COMPLAINT,
        IntentCategory.TECHNICAL,
        IntentCategory.BILLING,
        IntentCategory.ACCOUNT,
    }
)


@dataclass(frozen=True)
class RetrievalDecision:
    """Explainable result of applying the retrieval policy."""

    use_knowledge: bool
    reason: str


@dataclass(frozen=True, init=False)
class RetrievalPolicy:
    """Choose knowledge retrieval from intent, with explicit policy overrides."""

    allow_intents: frozenset[IntentCategory]
    deny_intents: frozenset[IntentCategory]
    retrieval_intents: frozenset[IntentCategory]

    def __init__(
        self,
        *,
        allow_intents: Optional[Iterable[IntentInput]] = None,
        deny_intents: Optional[Iterable[IntentInput]] = None,
    ) -> None:
        normalized_allow = self._normalize_config(
            allow_intents,
            "allow_intents",
        )
        normalized_deny = self._normalize_config(
            deny_intents,
            "deny_intents",
        )
        retrieval_intents = frozenset(
            (DEFAULT_RETRIEVAL_INTENTS | normalized_allow)
            - normalized_deny
        )
        object.__setattr__(self, "allow_intents", normalized_allow)
        object.__setattr__(self, "deny_intents", normalized_deny)
        object.__setattr__(self, "retrieval_intents", retrieval_intents)

    def decide(
        self,
        intent: Any,
        message: Optional[str],
    ) -> RetrievalDecision:
        """Return whether to retrieve and a stable reason for the choice."""
        if message is None:
            return RetrievalDecision(False, "blank_message")
        if not isinstance(message, str):
            return RetrievalDecision(False, "invalid_message")
        if not message.strip():
            return RetrievalDecision(False, "blank_message")

        category = self._coerce_intent(intent)
        if category is None:
            return RetrievalDecision(False, "unknown_intent")
        if category in self.deny_intents:
            return RetrievalDecision(False, f"intent_denied:{category.value}")
        if category in self.allow_intents:
            return RetrievalDecision(True, f"intent_allowed:{category.value}")
        if category in DEFAULT_RETRIEVAL_INTENTS:
            return RetrievalDecision(
                True,
                f"intent_default_enabled:{category.value}",
            )
        return RetrievalDecision(
            False,
            f"intent_default_disabled:{category.value}",
        )

    def should_retrieve(
        self,
        intent: Any,
        message: Optional[str],
    ) -> bool:
        """Convenience wrapper returning only the retrieval choice."""
        return self.decide(intent, message).use_knowledge

    @staticmethod
    def _coerce_intent(value: Any) -> Optional[IntentCategory]:
        if isinstance(value, IntentCategory):
            return value
        if isinstance(value, str):
            try:
                return IntentCategory(value.strip().lower())
            except ValueError:
                return None
        return None

    @classmethod
    def _normalize_config(
        cls,
        values: Optional[Iterable[IntentInput]],
        argument_name: str,
    ) -> frozenset[IntentCategory]:
        if values is None:
            return frozenset()
        if isinstance(values, (IntentCategory, str)):
            items = (values,)
        else:
            try:
                items = iter(values)
            except TypeError as exc:
                raise TypeError(
                    f"{argument_name} must be an iterable of intents"
                ) from exc

        normalized = set()
        for value in items:
            category = cls._coerce_intent(value)
            if category is None:
                raise ValueError(
                    f"Unknown intent in {argument_name}: {value!r}"
                )
            normalized.add(category)
        return frozenset(normalized)
