"""Shared local text embedding model for Chinese retrieval and memory."""

from __future__ import annotations

from threading import Lock
from typing import Any, Callable, List, Optional, Sequence


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


class FastEmbedTextModel:
    """Lazy CPU embedding model backed by FastEmbed's ONNX runtime."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: Optional[str] = None,
        model_factory: Optional[Callable[[str, Optional[str]], Any]] = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model_factory = model_factory or self._create_model
        self._model: Optional[Any] = None
        self._load_lock = Lock()

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        values = [str(text) for text in texts]
        if not values:
            return []
        return [vector.tolist() if hasattr(vector, "tolist") else list(vector)
                for vector in self._get_model().embed(values)]

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                self._model = self._model_factory(self.model_name, self.cache_dir)
        return self._model

    @staticmethod
    def _create_model(model_name: str, cache_dir: Optional[str]) -> Any:
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=model_name, cache_dir=cache_dir)
