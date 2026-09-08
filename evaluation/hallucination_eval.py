"""Deterministic adversarial evaluation for EchoMind evidence verification."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.evidence_verifier import EvidenceVerifier


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _metrics(counts: Mapping[str, int]) -> dict[str, float]:
    tp, fp = counts["tp"], counts["fp"]
    tn, fn = counts["tn"], counts["fn"]
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "false_negative_rate": _ratio(fn, tp + fn),
        "accuracy": _ratio(tp + tn, tp + fp + tn + fn),
    }


class HallucinationEvaluator:
    """Evaluate a binary hallucination detector without inventing labels."""

    REQUIRED_FIELDS = {
        "id", "type", "question", "response", "citations",
        "tool_evidence", "has_hallucination",
    }

    def __init__(self, verifier: EvidenceVerifier):
        self._verifier = verifier

    def evaluate_file(
        self,
        path: str | Path,
        *,
        split: str | None = None,
    ) -> dict[str, Any]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
            raise ValueError("hallucination dataset must contain a cases list")
        cases = payload["cases"]
        if split is not None:
            cases = [case for case in cases if case.get("split") == split]
        report = self.evaluate(cases)
        report["metadata"] = dict(payload.get("metadata") or {})
        report["split"] = split
        return report

    def evaluate(self, cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not cases:
            raise ValueError("at least one hallucination case is required")
        counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        type_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        )
        results = []

        for raw_case in cases:
            case = self._validated_case(raw_case)
            verification = self._verifier.verify(
                question=case["question"],
                response=case["response"],
                citations=case["citations"],
                tool_evidence=case["tool_evidence"],
            )
            actual = case["has_hallucination"]
            predicted = not verification.passed
            bucket = (
                "tp" if actual and predicted
                else "fn" if actual
                else "fp" if predicted
                else "tn"
            )
            counts[bucket] += 1
            type_counts[case["type"]][bucket] += 1
            results.append({
                "id": case["id"],
                "type": case["type"],
                "split": case.get("split"),
                "actual_hallucination": actual,
                "predicted_hallucination": predicted,
                "outcome": bucket,
                "issue_codes": [issue.code for issue in verification.issues],
                "checked_claims": verification.checked_claims,
            })

        by_type = {}
        for case_type, case_counts in sorted(type_counts.items()):
            by_type[case_type] = {
                **case_counts,
                **_metrics(case_counts),
                "total": sum(case_counts.values()),
            }
        return {
            "total": len(results),
            "confusion_matrix": counts,
            "metrics": _metrics(counts),
            "by_type": by_type,
            "results": results,
        }

    @classmethod
    def _validated_case(cls, case: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(case, Mapping) or not cls.REQUIRED_FIELDS <= set(case):
            raise ValueError("hallucination case is missing required fields")
        if (
            not isinstance(case["id"], str)
            or not case["id"]
            or not isinstance(case["type"], str)
            or not case["type"]
            or not isinstance(case["question"], str)
            or not isinstance(case["response"], str)
            or not isinstance(case["citations"], list)
            or not isinstance(case["tool_evidence"], list)
            or type(case["has_hallucination"]) is not bool
        ):
            raise ValueError("hallucination case contains invalid field types")
        return dict(case)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = HallucinationEvaluator(EvidenceVerifier()).evaluate_file(
        args.dataset,
        split=args.split,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
