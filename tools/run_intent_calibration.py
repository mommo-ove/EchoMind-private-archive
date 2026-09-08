"""Collect intent strategy scores once, then calibrate without more LLM calls."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from core.intent_recognizer import IntentCategory, IntentRecognizer
from evaluation.intent_calibration import (
    CalibrationConstraints,
    FusionConfig,
    calibrate_and_evaluate,
)


def _scores(result: Dict[str, Any]) -> Dict[str, float]:
    return {
        (label.value if isinstance(label, IntentCategory) else str(label)): float(score)
        for label, score in IntentRecognizer._strategy_scores(result).items()
    }


def _write_checkpoint(path: Path, cases: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"schema_version": 1, "cases": cases}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


async def collect(
    args: argparse.Namespace,
    *,
    recognizer: IntentRecognizer | None = None,
) -> Path:
    payload = json.loads(args.golden.read_text(encoding="utf-8"))
    recognizer = recognizer or IntentRecognizer(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        embedding_enabled=True,
    )
    collected = []
    if args.scores.exists():
        checkpoint = json.loads(args.scores.read_text(encoding="utf-8"))
        collected = list(checkpoint.get("cases", []))
    completed = {case["case_id"] for case in collected}
    for case in payload["cases"]:
        if case["case_id"] in completed:
            continue
        llm = None
        embedding = None
        for attempt in range(1, args.max_attempts + 1):
            llm, embedding = await asyncio.gather(
                recognizer._llm_recognize(case["message"], None),
                recognizer._embedding_recognize(case["message"]),
            )
            if not llm.get("failed"):
                break
            if attempt < args.max_attempts:
                await asyncio.sleep(min(attempt, 2))
        if llm is None or llm.get("failed"):
            raise RuntimeError(
                f"LLM score collection failed at {case['case_id']} after "
                f"{args.max_attempts} attempts; checkpoint contains {len(collected)} cases"
            )
        pattern = recognizer._pattern_recognize(case["message"])
        collected.append({
            **case,
            "scores": {
                "llm": _scores(llm),
                "embedding": _scores(embedding),
                "pattern": _scores(pattern),
            },
        })
        _write_checkpoint(args.scores, collected)
    return args.scores


def calibrate(args: argparse.Namespace) -> Path:
    cases = json.loads(args.scores.read_text(encoding="utf-8"))["cases"]
    validation = [case for case in cases if case["split"] == "validation"]
    test = [case for case in cases if case["split"] == "test"]
    report = calibrate_and_evaluate(
        validation_cases=validation,
        test_cases=test,
        constraints=CalibrationConstraints(
            minimum_recall={
                "escalation": 0.95,
                "technical": 0.80,
                "billing": 0.80,
            },
            minimum_precision={"escalation": 0.80},
        ),
        baseline=FusionConfig(0.7, 0.2, 0.1, 0.5, 0.6),
        confidence_thresholds=(0.2, 0.3, 0.4, 0.5, 0.6),
        multi_label_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return args.output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("collect", "calibrate", "all"))
    parser.add_argument("--golden", type=Path, default=Path("data/eval/intent_calibration_golden.json"))
    parser.add_argument("--scores", type=Path, default=Path("data/eval/results/intent_strategy_scores.json"))
    parser.add_argument("--output", type=Path, default=Path("data/eval/results/intent_calibration_latest.json"))
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    if args.action in {"collect", "all"}:
        print(await collect(args))
    if args.action in {"calibrate", "all"}:
        print(calibrate(args))


if __name__ == "__main__":
    asyncio.run(main())
