"""Benchmark the migrated production BGE Chroma collection and reranker."""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import time
from typing import Any, Dict, List

import chromadb

from evaluation.evaluator import retrieval_ranking_metrics
from evaluation.retrieval_ab import categorize_retrieval_case, latency_summary, write_report
from evaluation.retrieval_dataset import load_cases, split_cases
from mcp.local_embeddings import DEFAULT_EMBEDDING_MODEL, FastEmbedTextModel
from mcp.reranker import DEFAULT_RERANK_MODEL, FastEmbedCrossEncoderReranker


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    names = (
        "retrieval_hit_rate",
        "retrieval_recall_at_k",
        "retrieval_mrr",
        "retrieval_ndcg_at_k",
    )
    return {
        name: round(sum(row[name] for row in rows) / len(rows), 6)
        for name in names
    }


async def run(args: argparse.Namespace) -> pathlib.Path:
    all_cases = load_cases(args.golden)
    cases, unanswerable_cases = split_cases(all_cases)
    client = chromadb.HttpClient(
        host=args.chroma_host,
        port=args.chroma_port,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(args.collection)
    embedder = FastEmbedTextModel(
        model_name=args.embedding_model,
        cache_dir=str(args.embedding_cache),
    )
    reranker = FastEmbedCrossEncoderReranker(
        model_name=args.rerank_model,
        cache_dir=str(args.rerank_cache),
    )

    # Warm local models so cold-start download/load time does not contaminate
    # per-query online latency.  Cold start remains a separately visible fact.
    embedder.embed_query("模型预热")
    await reranker.score("模型预热", ["模型预热文档", "其他文档"])

    variants = []
    for top_k in args.top_k:
        variants.append({"name": f"bge_vector_top_{top_k}", "top_k": top_k, "recall_k": top_k, "rerank": False})
    for recall_k in args.recall_k:
        for top_k in args.top_k:
            if recall_k >= top_k:
                variants.append({
                    "name": f"bge_recall_{recall_k}_jina_top_{top_k}",
                    "top_k": top_k,
                    "recall_k": recall_k,
                    "rerank": True,
                })

    aggregate_metrics: Dict[str, Dict[str, float]] = {}
    category_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    aggregate_latency: Dict[str, Dict[str, Any]] = {}
    per_case: List[Dict[str, Any]] = [
        {**case, "variants": {}} for case in cases
    ]

    for variant in variants:
        metrics_rows = []
        embedding_ms, chroma_ms, rerank_ms, total_ms = [], [], [], []
        for index, case in enumerate(cases):
            total_start = time.perf_counter()
            start = time.perf_counter()
            query_vector = embedder.embed_query(case["question"])
            embedding_ms.append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            result = collection.query(
                query_embeddings=[query_vector],
                n_results=variant["recall_k"],
                include=["documents", "metadatas", "distances"],
            )
            chroma_ms.append((time.perf_counter() - start) * 1000)
            candidates = [
                {
                    "id": context_id,
                    "document": document,
                    "title": metadata.get("title", ""),
                }
                for context_id, document, metadata in zip(
                    result["ids"][0],
                    result["documents"][0],
                    result["metadatas"][0],
                )
            ]

            rerank_elapsed = 0.0
            if variant["rerank"]:
                start = time.perf_counter()
                scores = await reranker.score(
                    case["question"],
                    [f"{item['title']}\n{item['document']}" for item in candidates],
                )
                rerank_elapsed = (time.perf_counter() - start) * 1000
                candidates = [
                    item for _, item in sorted(
                        zip(scores, candidates), reverse=True
                    )
                ]
            rerank_ms.append(rerank_elapsed)
            selected = candidates[: variant["top_k"]]
            ids = [item["id"] for item in selected]
            metrics = retrieval_ranking_metrics(ids, case["reference_relevance"])
            metrics_rows.append(metrics)
            total_ms.append((time.perf_counter() - total_start) * 1000)
            per_case[index]["variants"][variant["name"]] = {
                "ids": ids,
                "titles": [item["title"] for item in selected],
                "metrics": metrics,
            }

        aggregate_metrics[variant["name"]] = _aggregate(metrics_rows)
        by_category: Dict[str, List[Dict[str, float]]] = {}
        for case, metrics in zip(cases, metrics_rows):
            by_category.setdefault(categorize_retrieval_case(case), []).append(metrics)
        category_metrics[variant["name"]] = {
            category: {"case_count": len(rows), **_aggregate(rows)}
            for category, rows in by_category.items()
        }
        aggregate_latency[variant["name"]] = {
            "embedding": latency_summary(embedding_ms),
            "chroma_query": latency_summary(chroma_ms),
            "rerank": latency_summary(rerank_ms),
            "total": latency_summary(total_ms),
        }

    report = {
        "experiment": "production_bge_retrieval_benchmark",
        "case_count": len(cases),
        "dataset_case_count": len(all_cases),
        "answerable_case_count": len(cases),
        "unanswerable_case_count": len(unanswerable_cases),
        "corpus_document_count": collection.count(),
        "collection": args.collection,
        "embedding_model": args.embedding_model,
        "rerank_model": args.rerank_model,
        "metrics": aggregate_metrics,
        "metrics_by_category": category_metrics,
        "latency_ms": aggregate_latency,
        "query_rewrite": {
            "status": "not_run",
            "reason": "configured DeepSeek-compatible API returned HTTP 401",
        },
        "ragas_generation": {
            "status": "not_run",
            "reason": "requires a valid generation/judge API key",
        },
        "unanswerable_evaluation": {
            "status": "not_run",
            "reason": "retriever rejection threshold is not implemented; these cases are reserved for abstention and faithfulness evaluation",
            "case_ids": [case["case_id"] for case in unanswerable_cases],
        },
        "cases": per_case,
    }
    return write_report(report, args.output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroma-host", default="localhost")
    parser.add_argument("--chroma-port", type=int, default=8001)
    parser.add_argument("--collection", default="knowledge_base_bge_v1")
    parser.add_argument("--golden", type=pathlib.Path, default=pathlib.Path("data/eval/campus_retrieval_golden.json"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("data/eval/results/production_retrieval_benchmark_latest.json"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-cache", type=pathlib.Path, default=pathlib.Path("data/model-cache/bge-small-zh-complete"))
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--rerank-cache", type=pathlib.Path, default=pathlib.Path("data/model-cache/jina-v2"))
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--recall-k", type=int, nargs="+", default=[5, 10, 20])
    return parser.parse_args()


if __name__ == "__main__":
    print(asyncio.run(run(_parse_args())))
