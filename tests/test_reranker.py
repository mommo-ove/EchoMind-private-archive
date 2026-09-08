import asyncio

import api.main as main
from mcp.reranker import DEFAULT_RERANK_MODEL, FastEmbedCrossEncoderReranker


class FakeCrossEncoder:
    def __init__(self):
        self.calls = []

    def rerank(self, query, documents):
        self.calls.append((query, list(documents)))
        return iter([0.2, 0.9])


def test_default_reranker_is_the_verified_multilingual_model():
    assert DEFAULT_RERANK_MODEL == "jinaai/jina-reranker-v2-base-multilingual"


def test_fastembed_reranker_lazily_loads_once_and_returns_scores(tmp_path):
    model = FakeCrossEncoder()
    factory_calls = []

    def model_factory(model_name, cache_dir):
        factory_calls.append((model_name, cache_dir))
        return model

    reranker = FastEmbedCrossEncoderReranker(
        model_name="BAAI/bge-reranker-base",
        cache_dir=str(tmp_path),
        model_factory=model_factory,
    )

    first = asyncio.run(reranker.score("network", ["card", "network 401"]))
    second = asyncio.run(reranker.score("refund", ["network", "refund policy"]))

    assert first == [0.2, 0.9]
    assert second == [0.2, 0.9]
    assert factory_calls == [("BAAI/bge-reranker-base", str(tmp_path))]
    assert model.calls == [
        ("network", ["card", "network 401"]),
        ("refund", ["network", "refund policy"]),
    ]


def test_api_builds_configured_fastembed_reranker(monkeypatch, tmp_path):
    monkeypatch.setenv("RERANK_PROVIDER", "fastembed")
    monkeypatch.setenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    monkeypatch.setenv("RERANK_CACHE_DIR", str(tmp_path))

    reranker = main._build_reranker()

    assert isinstance(reranker, FastEmbedCrossEncoderReranker)
    assert reranker.model_name == "BAAI/bge-reranker-base"
    assert reranker.cache_dir == str(tmp_path)
