import pytest

from api.main import _intent_fusion_config


def test_intent_fusion_config_reads_calibrated_environment(monkeypatch):
    monkeypatch.setenv("INTENT_EMBEDDING_ENABLED", "true")
    monkeypatch.setenv("INTENT_LLM_WEIGHT", "0.5")
    monkeypatch.setenv("INTENT_EMBEDDING_WEIGHT", "0.3")
    monkeypatch.setenv("INTENT_PATTERN_WEIGHT", "0.2")
    monkeypatch.setenv("INTENT_CONFIDENCE_THRESHOLD", "0.45")
    monkeypatch.setenv("INTENT_MULTI_LABEL_THRESHOLD", "0.55")

    config = _intent_fusion_config()

    assert config == {
        "embedding_enabled": True,
        "strategy_weights": {"llm": 0.5, "embedding": 0.3, "pattern": 0.2},
        "confidence_threshold": 0.45,
        "multi_label_threshold": 0.55,
    }


def test_intent_fusion_config_rejects_invalid_boolean(monkeypatch):
    monkeypatch.setenv("INTENT_EMBEDDING_ENABLED", "sometimes")

    with pytest.raises(RuntimeError, match="INTENT_EMBEDDING_ENABLED"):
        _intent_fusion_config()


def test_intent_fusion_defaults_follow_embedding_capability(monkeypatch):
    for name in (
        "INTENT_EMBEDDING_ENABLED", "INTENT_LLM_WEIGHT",
        "INTENT_EMBEDDING_WEIGHT", "INTENT_PATTERN_WEIGHT",
        "INTENT_CONFIDENCE_THRESHOLD", "INTENT_MULTI_LABEL_THRESHOLD",
    ):
        monkeypatch.delenv(name, raising=False)

    official = _intent_fusion_config(base_url=None)
    compatible = _intent_fusion_config(base_url="https://api.deepseek.com/anthropic")

    assert official["embedding_enabled"] is True
    assert official["strategy_weights"] == {
        "llm": 0.7, "embedding": 0.2, "pattern": 0.1,
    }
    assert compatible["embedding_enabled"] is False
    assert compatible["strategy_weights"] == {
        "llm": 0.85, "embedding": 0.0, "pattern": 0.15,
    }
