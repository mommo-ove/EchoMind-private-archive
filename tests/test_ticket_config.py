import re
from pathlib import Path

import api.main as main


def test_default_campus_database_is_repo_local_outside_container(monkeypatch, tmp_path):
    repository = tmp_path / "EchoMind"
    monkeypatch.setattr(main, "_ROOT", str(repository))
    monkeypatch.delenv("CAMPUS_DB_PATH", raising=False)

    assert main._campus_database_path() == (
        repository / "data" / "campus" / "campus.db"
    )


def test_explicit_windows_campus_database_path_is_preserved(monkeypatch):
    configured = r"C:\EchoMind\data\campus\campus.db"
    monkeypatch.setenv("CAMPUS_DB_PATH", configured)

    assert str(main._campus_database_path()) == configured


def test_compose_uses_named_persistent_campus_volume():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "CAMPUS_DB_PATH=/app/data/campus/campus.db" in compose
    assert re.search(
        r"(?m)^\s{6}- campus_data:/app/data/campus\s*$",
        compose,
    )
    assert re.search(r"(?m)^\s{2}campus_data:\s*$", compose)
