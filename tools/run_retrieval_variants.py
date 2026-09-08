"""Compare EchoMind retrieval variants on one fixed Chinese campus dataset.

Variants:
1. ChromaDB default vector retrieval.
2. BGE Chinese vector retrieval.
3. BGE recall plus multilingual cross-encoder reranking.
4. LLM query rewrite plus BGE recall plus multilingual reranking.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any, Dict, List, Sequence

import chromadb
from dotenv import load_dotenv
from fastembed import TextEmbedding

from evaluation.retrieval_ab import (
    compare_variants,
    cosine_rank,
    merge_unique_rankings,
    rewrite_succeeded,
    write_report,
)
from mcp.reranker import DEFAULT_RERANK_MODEL, FastEmbedCrossEncoderReranker
from mcp.tool_manager import MCPToolManager


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _document_id(title: str, content: str) -> str:
    return hashlib.md5(f"{title}_0_{content[:50]}".encode()).hexdigest()


def _single_turn_cases(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    cases = []
    for index, case in enumerate(payload.get("dialog_cases", []), start=1):
        if not case.get("question") or not case.get("reference_context_relevance"):
            continue
        cases.append(
            {
                "case_id": f"dialog-{index:02d}",
                "question": case["question"],
                "reference_relevance": case["reference_context_relevance"],
            }
        )
    return cases


def _create_default_collection(path: str, documents: Sequence[Dict[str, str]]):
    client = chromadb.PersistentClient(
        path=path,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name="default_vector")
    collection.add(
        ids=[_document_id(doc["title"], doc["content"]) for doc in documents],
        documents=[doc["content"] for doc in documents],
        metadatas=[{"title": doc["title"]} for doc in documents],
    )
    return collection


def _default_rank(collection: Any, query: str, top_k: int) -> List[str]:
    return collection.query(query_texts=[query], n_results=top_k)["ids"][0]


def _make_rewriter() -> MCPToolManager | None:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None
    return MCPToolManager(
        api_key=api_key,
        base_url=os.getenv("ANTHROPIC_BASE_URL", "").strip() or None,
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip(),
    )


async def _run(args: argparse.Namespace) -> pathlib.Path:
    load_dotenv(args.env_file)
    cases = _single_turn_cases(_load_json(args.golden))
    documents: List[Dict[str, str]] = _load_json(args.knowledge)
    ids = [_document_id(doc["title"], doc["content"]) for doc in documents]
    titles = {context_id: doc["title"] for context_id, doc in zip(ids, documents)}

    embedding = TextEmbedding(
        model_name=args.embedding_model,
        cache_dir=str(args.embedding_cache),
    )
    document_vectors = {
        context_id: vector.tolist()
        for context_id, vector in zip(
            ids,
            embedding.embed([doc["content"] for doc in documents]),
        )
    }
    reranker = FastEmbedCrossEncoderReranker(
        model_name=args.rerank_model,
        cache_dir=str(args.rerank_cache),
    )
    rewriter = _make_rewriter()

    variant_names = [
        "default_chroma_vector",
        "bge_zh_vector",
        "bge_zh_plus_reranker",
        "query_rewrite_plus_bge_zh_plus_reranker",
    ]
    rankings: Dict[str, Dict[str, List[str]]] = {
        name: {} for name in variant_names
    }
    rewritten_queries: Dict[str, List[str]] = {}
    rewrite_success: Dict[str, bool] = {}

    with tempfile.TemporaryDirectory(prefix="echomind-default-vector-") as path:
        default_collection = _create_default_collection(path, documents)
        for case in cases:
            case_id = case["case_id"]
            question = case["question"]
            rankings[variant_names[0]][case_id] = _default_rank(
                default_collection, question, args.top_k
            )

            query_vector = next(iter(embedding.embed([question]))).tolist()
            bge_ranking = cosine_rank(query_vector, document_vectors)
            rankings[variant_names[1]][case_id] = bge_ranking[: args.top_k]

            bge_candidates = bge_ranking[: args.recall_k]
            bge_scores = await reranker.score(
                question,
                [documents[ids.index(context_id)]["content"] for context_id in bge_candidates],
            )
            bge_reranked = [
                context_id
                for _, context_id in sorted(
                    zip(bge_scores, bge_candidates),
                    reverse=True,
                )
            ]
            rankings[variant_names[2]][case_id] = bge_reranked[: args.top_k]

            queries = (
                await rewriter.rewrite_query(question, n=args.rewrite_count)
                if rewriter is not None
                else [question]
            )
            rewritten_queries[case_id] = queries
            rewrite_success[case_id] = rewrite_succeeded(question, queries)
            query_rankings = []
            for rewritten in queries:
                vector = next(iter(embedding.embed([rewritten]))).tolist()
                query_rankings.append(
                    cosine_rank(vector, document_vectors)[: args.per_query_k]
                )
            merged = merge_unique_rankings(query_rankings)[: args.recall_k]
            rewrite_scores = await reranker.score(
                question,
                [documents[ids.index(context_id)]["content"] for context_id in merged],
            )
            rewritten_reranked = [
                context_id
                for _, context_id in sorted(
                    zip(rewrite_scores, merged),
                    reverse=True,
                )
            ]
            rankings[variant_names[3]][case_id] = rewritten_reranked[: args.top_k]

    report = compare_variants(
        cases,
        rankings,
        top_k=args.top_k,
        baseline_name=variant_names[0],
    )
    report["configuration"] = {
        "corpus_document_count": len(documents),
        "embedding_model": args.embedding_model,
        "rerank_model": args.rerank_model,
        "recall_k": args.recall_k,
        "per_query_k": args.per_query_k,
        "rewrite_count": args.rewrite_count,
        "rewrite_provider_configured": rewriter is not None,
        "rewrite_llm_attempts": len(cases) if rewriter is not None else 0,
        "rewrite_success_count": sum(rewrite_success.values()),
        "rewrite_fallback_count": len(cases) - sum(rewrite_success.values()),
        "rewrite_variant_valid": all(rewrite_success.values()),
    }
    for row in report["cases"]:
        case_id = row["case_id"]
        row["rewritten_queries"] = rewritten_queries[case_id]
        row["rewrite_success"] = rewrite_success[case_id]
        row["titles"] = {
            name: [titles.get(value, "") for value in row["rankings"][name]]
            for name in variant_names
        }
    return write_report(report, args.output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=pathlib.Path, default=pathlib.Path(".env"))
    parser.add_argument("--golden", type=pathlib.Path, default=pathlib.Path("data/eval/campus_golden.json"))
    parser.add_argument("--knowledge", type=pathlib.Path, default=pathlib.Path("data/knowledge/campus_knowledge.json"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("data/eval/results/retrieval_variants_latest.json"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-cache", type=pathlib.Path, default=pathlib.Path("data/model-cache/bge-small-zh"))
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--rerank-cache", type=pathlib.Path, default=pathlib.Path("data/model-cache/jina-v2"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--recall-k", type=int, default=10)
    parser.add_argument("--per-query-k", type=int, default=5)
    parser.add_argument("--rewrite-count", type=int, default=3)
    args = parser.parse_args()
    if args.top_k <= 0 or args.recall_k < args.top_k:
        parser.error("require recall-k >= top-k > 0")
    return args


if __name__ == "__main__":
    print(asyncio.run(_run(_parse_args())))
