from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_local_model_caches_are_excluded_from_docker_build_context():
    patterns = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "data/model-cache/" in patterns
    assert "data/reranker-cache/" in patterns
    assert "data/fastembed-cache/" in patterns
