from mcp.local_embeddings import FastEmbedTextModel


class FakeModel:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        values = list(texts)
        self.calls.append(values)
        return iter([[float(len(text)), 1.0] for text in values])


def test_fastembed_text_model_is_lazy_and_reuses_one_local_model(tmp_path):
    model = FakeModel()
    factory_calls = []

    def factory(model_name, cache_dir):
        factory_calls.append((model_name, cache_dir))
        return model

    embedder = FastEmbedTextModel(
        model_name="BAAI/bge-small-zh-v1.5",
        cache_dir=str(tmp_path),
        model_factory=factory,
    )

    assert embedder.embed(["校园网", "校园卡"]) == [[3.0, 1.0], [3.0, 1.0]]
    assert embedder.embed_query("工单") == [2.0, 1.0]
    assert factory_calls == [("BAAI/bge-small-zh-v1.5", str(tmp_path))]
