"""Dedicated retrieval rerankers used after vector recall."""

import asyncio
from threading import Lock
from typing import Any, Callable, Iterable, List, Optional, Protocol, Sequence


# Verified with fastembed 0.8.0 and suitable for the project's Chinese corpus.
# The previously configured BAAI snapshot did not include the ONNX path that
# this fastembed version expects, so it could not be loaded at runtime.
DEFAULT_RERANK_MODEL = "jinaai/jina-reranker-v2-base-multilingual"


class DedicatedReranker(Protocol):
    async def score(self, query: str, documents: Sequence[str]) -> List[float]:
        """Return one relevance score per document; larger is better."""


class FastEmbedCrossEncoderReranker:
    """Lazy CPU cross-encoder backed by FastEmbed's ONNX runtime."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        cache_dir: Optional[str] = None,
        model_factory: Optional[Callable[[str, Optional[str]], Any]] = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model_factory = model_factory or self._create_model
        self._model: Optional[Any] = None
        self._load_lock = Lock()

    async def score(self, query: str, documents: Sequence[str]) -> List[float]:
        if not documents:
            return []
        model = await asyncio.to_thread(self._get_model)
        scores: Iterable[float] = await asyncio.to_thread(
            lambda: list(model.rerank(query, list(documents)))
        )
        return [float(score) for score in scores]

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                self._model = self._model_factory(self.model_name, self.cache_dir)
        return self._model

    @staticmethod
    def _create_model(model_name: str, cache_dir: Optional[str]) -> Any:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)
