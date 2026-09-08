"""Deterministic A/B comparison helpers for retrieval rankings.

The runner supplies two rankings produced from the same fixed cases.  This
module deliberately contains no model calls so metric calculation is cheap,
repeatable, and easy to unit test.
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Sequence

from evaluation.evaluator import retrieval_ranking_metrics


_METRIC_NAMES = (
    "retrieval_hit_rate",
    "retrieval_recall_at_k",
    "retrieval_mrr",
    "retrieval_ndcg_at_k",
)


def cosine_rank(
    query_vector: Sequence[float],
    document_vectors: Mapping[str, Sequence[float]],
) -> List[str]:
    """Return document IDs ordered by cosine similarity, highest first."""

    query_norm = math.sqrt(sum(float(value) ** 2 for value in query_vector))

    def similarity(vector: Sequence[float]) -> float:
        doc_norm = math.sqrt(sum(float(value) ** 2 for value in vector))
        if query_norm == 0.0 or doc_norm == 0.0:
            return 0.0
        return sum(
            float(left) * float(right)
            for left, right in zip(query_vector, vector)
        ) / (query_norm * doc_norm)

    return sorted(document_vectors, key=lambda key: similarity(document_vectors[key]), reverse=True)


def merge_unique_rankings(rankings: Sequence[Sequence[str]]) -> List[str]:
    """Merge multiple recall lists without returning duplicate chunk IDs."""

    merged: List[str] = []
    seen = set()
    for ranking in rankings:
        for context_id in ranking:
            value = str(context_id)
            if value not in seen:
                seen.add(value)
                merged.append(value)
    return merged


def rewrite_succeeded(original: str, queries: Sequence[str]) -> bool:
    """Return true only when rewriting produced a distinct usable query."""

    normalized_original = original.strip()
    return any(
        isinstance(query, str)
        and query.strip()
        and query.strip() != normalized_original
        for query in queries
    )


def latency_summary(values_ms: Sequence[float]) -> Dict[str, float | int]:
    """Summarize measured stage latency using linear percentile interpolation."""

    if not values_ms:
        return {"count": 0}
    values = sorted(float(value) for value in values_ms)

    def percentile(fraction: float) -> float:
        position = (len(values) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return values[lower]
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    return {
        "count": len(values),
        "mean_ms": round(statistics.mean(values), 3),
        "p50_ms": round(percentile(0.5), 3),
        "p95_ms": round(percentile(0.95), 3),
        "max_ms": round(max(values), 3),
    }


def categorize_retrieval_case(case: Mapping[str, Any]) -> str:
    """Separate pure knowledge, tool-assisted, and composite retrieval cases."""

    explicit = case.get("category")
    if explicit:
        return str(explicit)

    tools = case.get("expected_tools") or []
    if len(tools) > 1:
        return "composite"
    if len(tools) == 1:
        return "tool_assisted"
    return "knowledge_only"


def _mean_metrics(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    return {
        name: round(statistics.mean(float(row.get(name, 0.0)) for row in rows), 6)
        for name in _METRIC_NAMES
    }


def compare_rankings(
    cases: Sequence[Mapping[str, Any]],
    baseline_rankings: Mapping[str, Sequence[str]],
    candidate_rankings: Mapping[str, Sequence[str]],
    *,
    top_k: int,
    baseline_name: str,
    candidate_name: str,
) -> Dict[str, Any]:
    """Compare two retrieval pipelines against the same gold chunk IDs."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if not cases:
        raise ValueError("at least one retrieval case is required")

    case_rows: List[Dict[str, Any]] = []
    baseline_metrics: List[Dict[str, float]] = []
    candidate_metrics: List[Dict[str, float]] = []

    for case in cases:
        case_id = str(case["case_id"])
        relevance = dict(case["reference_relevance"])
        baseline_ids = [str(value) for value in baseline_rankings.get(case_id, ())][:top_k]
        candidate_ids = [str(value) for value in candidate_rankings.get(case_id, ())][:top_k]
        before = retrieval_ranking_metrics(baseline_ids, relevance)
        after = retrieval_ranking_metrics(candidate_ids, relevance)
        baseline_metrics.append(before)
        candidate_metrics.append(after)
        case_rows.append(
            {
                "case_id": case_id,
                "question": str(case["question"]),
                "reference_relevance": relevance,
                "baseline_ids": baseline_ids,
                "candidate_ids": candidate_ids,
                "baseline_metrics": before,
                "candidate_metrics": after,
            }
        )

    before_mean = _mean_metrics(baseline_metrics)
    after_mean = _mean_metrics(candidate_metrics)
    delta = {
        name: round(after_mean[name] - before_mean[name], 6)
        for name in _METRIC_NAMES
    }
    return {
        "experiment": "retrieval_ab",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(case_rows),
        "top_k": top_k,
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "baseline": before_mean,
        "candidate": after_mean,
        "delta": delta,
        "cases": case_rows,
    }


def compare_variants(
    cases: Sequence[Mapping[str, Any]],
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    top_k: int,
    baseline_name: str,
) -> Dict[str, Any]:
    """Compare several retrieval pipelines against one fixed baseline."""

    if baseline_name not in rankings:
        raise ValueError("baseline_name must identify one rankings variant")
    variant_order = list(rankings)
    aggregate_rows: Dict[str, List[Dict[str, float]]] = {
        name: [] for name in variant_order
    }
    case_rows: List[Dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        relevance = dict(case["reference_relevance"])
        case_rankings: Dict[str, List[str]] = {}
        case_metrics: Dict[str, Dict[str, float]] = {}
        for name in variant_order:
            ids = [str(value) for value in rankings[name].get(case_id, ())][:top_k]
            metrics = retrieval_ranking_metrics(ids, relevance)
            case_rankings[name] = ids
            case_metrics[name] = metrics
            aggregate_rows[name].append(metrics)
        case_rows.append(
            {
                "case_id": case_id,
                "question": str(case["question"]),
                "reference_relevance": relevance,
                "rankings": case_rankings,
                "metrics": case_metrics,
            }
        )

    aggregates = {
        name: _mean_metrics(aggregate_rows[name]) for name in variant_order
    }
    baseline = aggregates[baseline_name]
    deltas = {
        name: {
            metric: round(aggregates[name][metric] - baseline[metric], 6)
            for metric in _METRIC_NAMES
        }
        for name in variant_order
    }
    return {
        "experiment": "retrieval_variant_ab",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(case_rows),
        "top_k": top_k,
        "baseline_name": baseline_name,
        "variant_order": variant_order,
        "variants": aggregates,
        "delta_vs_baseline": deltas,
        "cases": case_rows,
    }


def write_report(report: Mapping[str, Any], output_path: pathlib.Path | str) -> pathlib.Path:
    """Persist an A/B report as UTF-8 JSON and return its resolved path."""

    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
