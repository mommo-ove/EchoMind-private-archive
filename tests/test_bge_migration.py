from tools.migrate_bge_collections import migrate_collection


class FakeCollection:
    def __init__(self, rows=None):
        self.rows = rows or {"ids": [], "documents": [], "metadatas": []}
        self.upserts = []

    def get(self, **_kwargs):
        return self.rows

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)


class FakeEmbedder:
    def embed(self, texts):
        return [[float(len(text))] for text in texts]


def test_migrate_collection_reembeds_documents_and_preserves_ids_and_metadata():
    source = FakeCollection({
        "ids": ["memory-1"],
        "documents": ["上周校园网401"],
        "metadatas": [{"user_id": "student-1"}],
    })
    target = FakeCollection()

    migrated = migrate_collection(source, target, FakeEmbedder())

    assert migrated == 1
    assert target.upserts == [{
        "ids": ["memory-1"],
        "documents": ["上周校园网401"],
        "metadatas": [{"user_id": "student-1"}],
        "embeddings": [[8.0]],
    }]
