"""RAGAS v0.4 adapter for EchoMind's retrieved evidence and answers."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from evaluation.evaluator import RagasScores

logger = logging.getLogger(__name__)


def _fastembed_adapter(model_name: str, cache_dir: Optional[str] = None):
    """Create a RAGAS embedding adapter backed by the existing ONNX runtime."""

    from fastembed import TextEmbedding
    from ragas.embeddings.base import BaseRagasEmbedding

    class FastEmbedRagasEmbedding(BaseRagasEmbedding):
        def __init__(self) -> None:
            super().__init__()
            self._model = TextEmbedding(
                model_name=model_name,
                cache_dir=cache_dir,
            )

        def embed_text(self, text: str, **kwargs: Any) -> List[float]:
            return next(iter(self._model.embed([text]))).tolist()

        async def aembed_text(self, text: str, **kwargs: Any) -> List[float]:
            return await asyncio.to_thread(self.embed_text, text, **kwargs)

        def embed_texts(
            self,
            texts: List[str],
            **kwargs: Any,
        ) -> List[List[float]]:
            return [vector.tolist() for vector in self._model.embed(texts)]

        async def aembed_texts(
            self,
            texts: List[str],
            **kwargs: Any,
        ) -> List[List[float]]:
            return await asyncio.to_thread(self.embed_texts, texts, **kwargs)

    return FastEmbedRagasEmbedding()


class RagasEvaluator:
    """Evaluate RAG answers with official RAGAS collection metrics.

    The heavy RAGAS imports and the local embedding model are initialized only
    when the first knowledge-backed case is evaluated.  This keeps normal chat
    startup independent from the optional evaluation stack.
    """

    def __init__(
        self,
        *,
        client: AsyncAnthropic,
        model: str,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        embedding_cache_dir: Optional[str] = None,
        llm_options: Optional[Dict[str, Any]] = None,
        judge_max_tokens: int = 8192,
    ) -> None:
        self._client = client
        self._model = model
        self._embedding_model = embedding_model
        self._embedding_cache_dir = embedding_cache_dir
        self._llm_options = dict(llm_options or {})
        self._judge_max_tokens = judge_max_tokens
        self._metrics: Optional[Dict[str, Any]] = None
        self._init_lock = asyncio.Lock()

    async def _get_metrics(self) -> Dict[str, Any]:
        if self._metrics is not None:
            return self._metrics
        async with self._init_lock:
            if self._metrics is not None:
                return self._metrics
            from ragas.llms import llm_factory
            from ragas.metrics.collections import (
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                FactualCorrectness,
                Faithfulness,
            )

            llm = llm_factory(
                self._model,
                provider="anthropic",
                client=self._client,
                temperature=0.0,
                max_tokens=self._judge_max_tokens,
                **self._llm_options,
            )
            # RAGAS AnswerRelevancy needs an embedding model.  Use a local
            # Chinese sentence embedding so evaluation does not assume that the
            # chat provider also offers an embeddings endpoint.
            embeddings = await asyncio.to_thread(
                _fastembed_adapter,
                self._embedding_model,
                self._embedding_cache_dir,
            )
            self._metrics = {
                "faithfulness": Faithfulness(llm=llm),
                "answer_relevancy": AnswerRelevancy(
                    llm=llm,
                    embeddings=embeddings,
                ),
                "context_precision": ContextPrecision(llm=llm),
                "context_recall": ContextRecall(llm=llm),
                "answer_correctness": FactualCorrectness(llm=llm),
            }
            return self._metrics

    async def evaluate(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: List[str],
        reference: Optional[str] = None,
    ) -> RagasScores:
        """Run every applicable metric and preserve partial results on failure."""

        if not retrieved_contexts:
            return RagasScores()
        try:
            metrics = await self._get_metrics()
        except Exception as exc:
            logger.warning("RAGAS initialization failed: %s", exc)
            return RagasScores(judge_failed=True, error=str(exc))

        calls = {
            "faithfulness": metrics["faithfulness"].ascore(
                user_input=user_input,
                response=response,
                retrieved_contexts=retrieved_contexts,
            ),
            "answer_relevancy": metrics["answer_relevancy"].ascore(
                user_input=user_input,
                response=response,
            ),
        }
        if reference:
            calls.update({
                "context_precision": metrics["context_precision"].ascore(
                    user_input=user_input,
                    reference=reference,
                    retrieved_contexts=retrieved_contexts,
                ),
                "context_recall": metrics["context_recall"].ascore(
                    user_input=user_input,
                    reference=reference,
                    retrieved_contexts=retrieved_contexts,
                ),
                "answer_correctness": metrics["answer_correctness"].ascore(
                    response=response,
                    reference=reference,
                ),
            })

        names = list(calls)
        results = await asyncio.gather(*calls.values(), return_exceptions=True)
        values: Dict[str, Optional[float]] = {name: None for name in calls}
        errors = []
        for name, result in zip(names, results):
            if isinstance(result, BaseException):
                errors.append(f"{name}: {result}")
                continue
            try:
                value = float(result.value)
                if math.isfinite(value):
                    values[name] = max(0.0, min(1.0, value))
                else:
                    errors.append(f"{name}: non-finite score")
            except (AttributeError, TypeError, ValueError) as exc:
                errors.append(f"{name}: {exc}")

        return RagasScores(
            faithfulness=values.get("faithfulness"),
            answer_relevancy=values.get("answer_relevancy"),
            context_precision=values.get("context_precision"),
            context_recall=values.get("context_recall"),
            answer_correctness=values.get("answer_correctness"),
            judge_failed=bool(errors),
            error="; ".join(errors) if errors else None,
        )
