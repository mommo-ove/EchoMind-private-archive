from mcp.knowledge_base import KnowledgeBase


class FakeCollection:
    def query(self, **kwargs):
        assert kwargs == {"query_embeddings": [[0.1, 0.9]], "n_results": 1}
        return {
            "ids": [["stable-chunk-id"]],
            "documents": [["Clear stale authentication state."]],
            "metadatas": [[{"title": "Network auth", "chunk_index": 0}]],
            "distances": [[0.1]],
        }


class FakeBgeCollection:
    def __init__(self):
        self.add_kwargs = None

    def add(self, **kwargs):
        self.add_kwargs = kwargs

    def query(self, **kwargs):
        assert kwargs == {"query_embeddings": [[0.1, 0.9]], "n_results": 1}
        return {
            "ids": [["bge-chunk"]],
            "documents": [["清除旧认证状态后重新连接。"]],
            "metadatas": [[{"title": "校园网401", "chunk_index": 0}]],
            "distances": [[0.05]],
        }


class FakeEmbedder:
    model_name = "BAAI/bge-small-zh-v1.5"

    def embed(self, texts):
        return [[float(index), 1.0] for index, _ in enumerate(texts)]

    def embed_query(self, text):
        assert text == "校园网401"
        return [0.1, 0.9]


class FakeClient:
    def __init__(self, collection):
        self.collection = collection

    def get_or_create_collection(self, **_kwargs):
        return self.collection


class EmptyCollection:
    def __init__(self):
        self.add_calls = []

    def count(self):
        return 0

    def add(self, **kwargs):
        self.add_calls.append(kwargs)


def test_search_exposes_chroma_chunk_id_for_retrieval_metrics():
    knowledge = object.__new__(KnowledgeBase)
    knowledge._collection = FakeCollection()
    knowledge._embedder = FakeEmbedder()

    result = knowledge.search("校园网401", top_k=1)

    assert result == [{
        "id": "stable-chunk-id",
        "title": "Network auth",
        "content": "Clear stale authentication state.",
        "score": 0.9,
        "chunk": 0,
    }]


def test_bge_knowledge_base_supplies_embeddings_to_chroma_for_add_and_query():
    collection = FakeBgeCollection()
    knowledge = object.__new__(KnowledgeBase)
    knowledge._collection = collection
    knowledge._embedder = FakeEmbedder()

    count = knowledge.add_documents([
        {"title": "校园网401", "content": "清理旧认证状态。"},
    ])
    result = knowledge.search("校园网401", top_k=1)

    assert count == 1
    assert collection.add_kwargs["embeddings"] == [[0.0, 1.0]]
    assert result[0]["id"] == "bge-chunk"


def test_new_bge_collection_can_skip_legacy_default_documents(monkeypatch, tmp_path):
    collection = EmptyCollection()
    fake_client = FakeClient(collection)
    monkeypatch.setattr(
        "mcp.knowledge_base.chromadb.PersistentClient",
        lambda **_kwargs: fake_client,
    )
    monkeypatch.setattr(
        "mcp.knowledge_base.chromadb.HttpClient",
        lambda **_kwargs: (_ for _ in ()).throw(ConnectionError()),
    )

    knowledge = KnowledgeBase(
        chroma_path=str(tmp_path),
        embedder=FakeEmbedder(),
        bootstrap_defaults=False,
    )

    assert knowledge.doc_count == 0
    assert collection.add_calls == []
