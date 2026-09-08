"""Non-destructively migrate EchoMind Chroma collections to local BGE vectors."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, List

import chromadb

from mcp.knowledge_base import KnowledgeBase
from mcp.local_embeddings import DEFAULT_EMBEDDING_MODEL, FastEmbedTextModel


def migrate_collection(source: Any, target: Any, embedder: FastEmbedTextModel) -> int:
    """Copy one collection while replacing its embeddings; source is untouched."""

    rows = source.get(include=["documents", "metadatas"])
    ids = list(rows.get("ids") or [])
    documents = list(rows.get("documents") or [])
    metadatas = list(rows.get("metadatas") or [])
    if not ids:
        return 0
    target.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embedder.embed(documents),
    )
    return len(ids)


def migrate(args: argparse.Namespace) -> Dict[str, int]:
    embedder = FastEmbedTextModel(
        model_name=args.embedding_model,
        cache_dir=str(args.cache_dir),
    )
    client = chromadb.HttpClient(
        host=args.chroma_host,
        port=args.chroma_port,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    client.heartbeat()

    knowledge = KnowledgeBase(
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
        embedder=embedder,
        bootstrap_defaults=False,
    )
    docs: List[Dict[str, str]] = json.loads(
        args.knowledge.read_text(encoding="utf-8")
    )
    if knowledge.doc_count == 0:
        knowledge.add_documents(docs)

    counts = {"knowledge_base_bge_v1": knowledge.doc_count}
    for old_name, new_name in (
        ("episodic", "episodic_bge_v1"),
        ("user_profile", "user_profile_bge_v1"),
    ):
        try:
            source = client.get_collection(old_name)
        except Exception:
            counts[new_name] = 0
            continue
        target = client.get_or_create_collection(new_name)
        counts[new_name] = migrate_collection(source, target, embedder)
    return counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroma-host", default="localhost")
    parser.add_argument("--chroma-port", type=int, default=8001)
    parser.add_argument(
        "--knowledge",
        type=pathlib.Path,
        default=pathlib.Path("data/knowledge/campus_knowledge.json"),
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/model-cache/bge-small-zh-complete"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(migrate(_parse_args()), ensure_ascii=False))
