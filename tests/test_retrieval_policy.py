import pytest

from core.intent_recognizer import IntentCategory
from core.retrieval_policy import (
    DEFAULT_RETRIEVAL_INTENTS,
    RetrievalDecision,
    RetrievalPolicy,
)


@pytest.mark.parametrize(
    ("intent", "message", "expected"),
    [
        (IntentCategory.GREETING, "你好呀", False),
        (IntentCategory.FEEDBACK, "谢谢你的帮助", False),
        (IntentCategory.ESCALATION, "我要人工客服", False),
        (IntentCategory.TECHNICAL, "校园网报 401", True),
        (IntentCategory.BILLING, "为什么重复扣款？", True),
        (IntentCategory.ACCOUNT, "校园账号被冻结", True),
        (IntentCategory.QUERY, "图书馆今天几点关门？", True),
        (IntentCategory.REQUEST, "请帮我查询课表", True),
        (IntentCategory.COMPLAINT, "网络一直断线", True),
    ],
)
def test_retrieval_policy_uses_explicit_intent_table(intent, message, expected):
    assert RetrievalPolicy().should_retrieve(intent, message) is expected


@pytest.mark.parametrize("message", ["", "   ", "\t\n", None])
def test_blank_input_explicitly_disables_retrieval(message):
    policy = RetrievalPolicy(allow_intents={IntentCategory.GREETING})

    decision = policy.decide(IntentCategory.GREETING, message)

    assert decision == RetrievalDecision(
        use_knowledge=False,
        reason="blank_message",
    )
    assert policy.should_retrieve(IntentCategory.GREETING, message) is False


def test_decision_explains_default_policy_choice():
    policy = RetrievalPolicy()

    assert policy.decide(" TECHNICAL ", "Campus Wi-Fi returns 401.") == (
        RetrievalDecision(
            use_knowledge=True,
            reason="intent_default_enabled:technical",
        )
    )
    assert policy.decide("feedback", "That was helpful.") == RetrievalDecision(
        use_knowledge=False,
        reason="intent_default_disabled:feedback",
    )


def test_allow_and_deny_lists_override_defaults_for_enum_and_string_inputs():
    policy = RetrievalPolicy(
        allow_intents={" GREETING "},
        deny_intents={IntentCategory.QUERY},
    )

    assert policy.decide(IntentCategory.GREETING, "你好").reason == (
        "intent_allowed:greeting"
    )
    assert policy.should_retrieve("greeting", "Hello") is True
    assert policy.decide("query", "When does the library close?").reason == (
        "intent_denied:query"
    )
    assert policy.should_retrieve(
        IntentCategory.QUERY,
        "图书馆什么时候关门？",
    ) is False


def test_deny_list_has_deterministic_precedence_over_allow_list():
    policy = RetrievalPolicy(
        allow_intents={"billing"},
        deny_intents={IntentCategory.BILLING},
    )

    assert policy.decide("billing", "I was charged twice.") == RetrievalDecision(
        use_knowledge=False,
        reason="intent_denied:billing",
    )


@pytest.mark.parametrize(
    "intent",
    ["not-a-real-intent", "   ", object(), None],
)
def test_unknown_runtime_intent_safely_disables_retrieval(intent):
    policy = RetrievalPolicy()

    decision = policy.decide(intent, "Please look this up.")

    assert decision == RetrievalDecision(
        use_knowledge=False,
        reason="unknown_intent",
    )
    assert policy.should_retrieve(intent, "Please look this up.") is False


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("allow_intents", {"not-a-real-intent"}),
        ("deny_intents", [object()]),
    ],
)
def test_configuration_rejects_unknown_intents(argument, value):
    with pytest.raises(ValueError, match="Unknown intent"):
        RetrievalPolicy(**{argument: value})


def test_configuration_rejects_non_iterable_intent_lists():
    with pytest.raises(TypeError, match="iterable"):
        RetrievalPolicy(allow_intents=42)


class MessageWithExplodingStringConversion:
    def __str__(self):
        raise AssertionError("RetrievalPolicy must not stringify messages")


@pytest.mark.parametrize(
    "message",
    [
        False,
        0,
        [],
        object(),
        MessageWithExplodingStringConversion(),
    ],
)
def test_non_string_messages_fail_closed_without_string_conversion(message):
    policy = RetrievalPolicy(allow_intents={"greeting"})

    decision = policy.decide(IntentCategory.GREETING, message)

    assert decision == RetrievalDecision(
        use_knowledge=False,
        reason="invalid_message",
    )
    assert policy.should_retrieve(IntentCategory.GREETING, message) is False


def test_configuration_rejects_whitespace_only_intent_labels():
    with pytest.raises(ValueError, match="Unknown intent"):
        RetrievalPolicy(allow_intents={"   "})


def test_defaults_are_immutable_and_not_mutated_by_configuration():
    policy = RetrievalPolicy(deny_intents={"query"})

    assert isinstance(DEFAULT_RETRIEVAL_INTENTS, frozenset)
    assert IntentCategory.QUERY in DEFAULT_RETRIEVAL_INTENTS
    assert IntentCategory.QUERY not in policy.retrieval_intents
    assert IntentCategory.QUERY in RetrievalPolicy().retrieval_intents


def test_policy_sets_are_read_only_and_caller_inputs_are_isolated():
    caller_allow = {"greeting"}
    policy = RetrievalPolicy(allow_intents=caller_allow)
    caller_allow.add("feedback")

    assert policy.allow_intents == frozenset({IntentCategory.GREETING})
    assert IntentCategory.FEEDBACK not in policy.retrieval_intents

    for attribute in ("allow_intents", "deny_intents", "retrieval_intents"):
        with pytest.raises(AttributeError):
            setattr(policy, attribute, frozenset())

    assert policy.allow_intents == frozenset({IntentCategory.GREETING})
    assert policy.deny_intents == frozenset()
    assert policy.retrieval_intents == (
        DEFAULT_RETRIEVAL_INTENTS | {IntentCategory.GREETING}
    )
