"""Schema adapters for token-free retrieval benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def load_cases(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "cases" in payload:
        return [
            {
                "case_id": raw["case_id"],
                "question": raw["question"],
                "category": raw["category"],
                "difficulty": raw.get("difficulty", "unknown"),
                "answerable": bool(raw["answerable"]),
                "reference_relevance": raw["reference_context_relevance"],
                "expected_tools": [],
            }
            for raw in payload["cases"]
        ]

    cases = []
    for index, raw in enumerate(payload.get("dialog_cases", []), start=1):
        if not raw.get("question") or not raw.get("reference_context_relevance"):
            continue
        cases.append({
            "case_id": f"dialog-{index:02d}",
            "question": raw["question"],
            "category": "legacy_dialog",
            "difficulty": "unknown",
            "answerable": True,
            "reference_relevance": raw["reference_context_relevance"],
            "expected_tools": raw.get("expected_tools") or [],
        })
    return cases


def split_cases(
    cases: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return (
        [case for case in cases if case["answerable"]],
        [case for case in cases if not case["answerable"]],
    )
