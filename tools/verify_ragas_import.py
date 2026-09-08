"""Offline import smoke test for the optional RAGAS evaluation adapter."""

from anthropic import AsyncAnthropic
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

from evaluation.ragas_evaluator import RagasEvaluator


client = AsyncAnthropic(api_key="offline-smoke-test-key")
llm = llm_factory(
    "offline-smoke-test-model",
    provider="anthropic",
    client=client,
)
assert Faithfulness(llm=llm)
assert ContextPrecision(llm=llm)
assert ContextRecall(llm=llm)
assert FactualCorrectness(llm=llm)
assert RagasEvaluator(client=client, model="offline-smoke-test-model")
print("ragas-anthropic-adapter-ok")
