"""Run the expanded BGE benchmark through ChromaDB's HTTP API."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluator import retrieval_ranking_metrics
from evaluation.retrieval_ab import categorize_retrieval_case, latency_summary, write_report
from evaluation.retrieval_dataset import load_cases, split_cases
from mcp.local_embeddings import DEFAULT_EMBEDDING_MODEL, FastEmbedTextModel


METRICS = (
    "retrieval_hit_rate",
    "retrieval_recall_at_k",
    "retrieval_mrr",
    "retrieval_ndcg_at_k",
)


def _request_json(url: str, *, body: Dict[str, Any] | None = None) -> Any:
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _mean(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        name: round(sum(row[name] for row in rows) / len(rows), 6)
        for name in METRICS
    }


def run(args: argparse.Namespace) -> Path:
    all_cases = load_cases(args.golden)
    cases, unanswerable = split_cases(all_cases)
    base_url = f"http://{args.chroma_host}:{args.chroma_port}/api/v1"
    collection = _request_json(f"{base_url}/collections/{args.collection}")
    corpus_document_count = _request_json(
        f"{base_url}/collections/{collection['id']}/count"
    )
    embedder = FastEmbedTextModel(
        model_name=args.embedding_model,
        cache_dir=str(args.embedding_cache),
    )
    embedder.embed_query("模型预热")
    query_vectors = embedder.embed([case["question"] for case in cases])

    variants: Dict[str, Dict[str, Any]] = {}
    per_case = [{**case, "variants": {}} for case in cases]
    for top_k in args.top_k:
        name = f"bge_vector_top_{top_k}"
        rows: List[Dict[str, float]] = []
        latency: List[float] = []
        category_rows: Dict[str, List[Dict[str, float]]] = {}
        for index, (case, vector) in enumerate(zip(cases, query_vectors)):
            started = time.perf_counter()
            result = _request_json(
                f"{base_url}/collections/{collection['id']}/query",
                body={
                    "query_embeddings": [vector],
                    "n_results": top_k,
                    "include": ["documents", "metadatas", "distances"],
                },
            )
            latency.append((time.perf_counter() - started) * 1000)
            ids = result["ids"][0]
            metrics = retrieval_ranking_metrics(ids, case["reference_relevance"])
            rows.append(metrics)
            category_rows.setdefault(categorize_retrieval_case(case), []).append(metrics)
            per_case[index]["variants"][name] = {
                "ids": ids,
                "titles": [item["title"] for item in result["metadatas"][0]],
                "metrics": metrics,
            }
        variants[name] = {
            "metrics": _mean(rows),
            "metrics_by_category": {
                category: {"case_count": len(group), **_mean(group)}
                for category, group in category_rows.items()
            },
            "chroma_query_latency_ms": latency_summary(latency),
        }

    report = {
        "experiment": "expanded_bge_http_baseline",
        "dataset_case_count": len(all_cases),
        "answerable_case_count": len(cases),
        "unanswerable_case_count": len(unanswerable),
        "corpus_document_count": corpus_document_count,
        "collection": args.collection,
        "embedding_model": args.embedding_model,
        "variants": variants,
        "unanswerable_evaluation": {
            "status": "reserved",
            "reason": "requires calibrated rejection threshold or generation faithfulness evaluation",
            "case_ids": [case["case_id"] for case in unanswerable],
        },
        "query_rewrite": {
            "status": "not_run",
            "reason": "baseline isolates the dense retriever; valid generation API key is still unavailable",
        },
        "cases": per_case,
    }
    return write_report(report, args.output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroma-host", default="localhost")
    parser.add_argument("--chroma-port", type=int, default=8001)
    parser.add_argument("--collection", default="knowledge_base_bge_v1")
    parser.add_argument("--golden", type=Path, default=Path("data/eval/campus_retrieval_golden.json"))
    parser.add_argument("--output", type=Path, default=Path("data/eval/results/expanded_bge_baseline_latest.json"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-cache", type=Path, default=Path("data/model-cache/bge-small-zh-windows"))
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    return parser.parse_args()


if __name__ == "__main__":
    print(run(_parse_args()))
