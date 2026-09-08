"""Run a token-free retrieval A/B test on the fixed campus gold set.

Baseline: ChromaDB vector retrieval, Top-K directly.
Candidate: ChromaDB vector recall, then a dedicated cross-encoder reranker.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import pathlib
import tempfile
from typing import Any, Dict, List, Sequence

import chromadb

from evaluation.retrieval_ab import compare_rankings, write_report
from mcp.reranker import DEFAULT_RERANK_MODEL, FastEmbedCrossEncoderReranker


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _document_id(title: str, content: str) -> str:
    return hashlib.md5(f"{title}_0_{content[:50]}".encode()).hexdigest()


def _single_turn_cases(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for index, case in enumerate(payload.get("dialog_cases", []), start=1):
        question = case.get("question")
        relevance = case.get("reference_context_relevance")
        if not isinstance(question, str) or not isinstance(relevance, dict) or not relevance:
            continue
        cases.append(
            {
                "case_id": f"dialog-{index:02d}",
                "question": question,
                "reference_relevance": relevance,
            }
        )
    return cases


def _create_isolated_collection(
    chroma_path: str,
    documents: Sequence[Dict[str, str]],
):
    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name="campus_retrieval_ab")
    ids = [_document_id(doc["title"], doc["content"]) for doc in documents]
    collection.add(
        ids=ids,
        documents=[doc["content"] for doc in documents],
        metadatas=[{"title": doc["title"], "chunk_index": 0} for doc in documents],
    )
    return collection


def _vector_recall(collection: Any, question: str, recall_k: int) -> List[Dict[str, str]]:
    result = collection.query(query_texts=[question], n_results=recall_k)
    return [
        {
            "id": context_id,
            "content": content,
            "title": metadata.get("title", ""),
        }
        for context_id, content, metadata in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
        )
    ]


async def _run(args: argparse.Namespace) -> pathlib.Path:
    golden = _load_json(args.golden)
    documents = _load_json(args.knowledge)
    cases = _single_turn_cases(golden)
    if not cases:
        raise RuntimeError("golden set has no single-turn retrieval cases")

    reranker = FastEmbedCrossEncoderReranker(
        model_name=args.rerank_model,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    baseline_rankings: Dict[str, List[str]] = {}
    candidate_rankings: Dict[str, List[str]] = {}
    titles_by_case: Dict[str, Dict[str, str]] = {}

    with tempfile.TemporaryDirectory(prefix="echomind-retrieval-ab-") as chroma_path:
        collection = _create_isolated_collection(chroma_path, documents)
        for case in cases:
            recalled = _vector_recall(collection, case["question"], args.recall_k)
            baseline_rankings[case["case_id"]] = [item["id"] for item in recalled[: args.top_k]]
            scores = await reranker.score(
                case["question"],
                [item["content"] for item in recalled],
            )
            reranked = [
                item
                for _, item in sorted(
                    zip(scores, recalled),
                    key=lambda pair: pair[0],
                    reverse=True,
                )
            ]
            candidate_rankings[case["case_id"]] = [item["id"] for item in reranked[: args.top_k]]
            titles_by_case[case["case_id"]] = {
                item["id"]: item["title"] for item in recalled
            }

    report = compare_rankings(
        cases,
        baseline_rankings,
        candidate_rankings,
        top_k=args.top_k,
        baseline_name=f"chroma_vector_top_{args.top_k}",
        candidate_name=(
            f"chroma_vector_recall_{args.recall_k}_plus_"
            f"{args.rerank_model}_top_{args.top_k}"
        ),
    )
    report["configuration"] = {
        "corpus_document_count": len(documents),
        "embedding": "ChromaDB default embedding function",
        "rerank_model": args.rerank_model,
        "recall_k": args.recall_k,
        "top_k": args.top_k,
        "llm_calls": 0,
    }
    for row in report["cases"]:
        titles = titles_by_case[row["case_id"]]
        row["baseline_titles"] = [titles.get(value, "") for value in row["baseline_ids"]]
        row["candidate_titles"] = [titles.get(value, "") for value in row["candidate_ids"]]
    return write_report(report, args.output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=pathlib.Path, default=pathlib.Path("data/eval/campus_golden.json"))
    parser.add_argument("--knowledge", type=pathlib.Path, default=pathlib.Path("data/knowledge/campus_knowledge.json"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("data/eval/results/retrieval_ab_latest.json"))
    parser.add_argument("--cache-dir", type=pathlib.Path)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--recall-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.top_k <= 0 or args.recall_k < args.top_k:
        parser.error("require recall-k >= top-k > 0")
    return args


if __name__ == "__main__":
    output = asyncio.run(_run(_parse_args()))
    print(output)
