"""Evaluate evidence-grounded reflection with a real chat model.

This benchmark deliberately measures a conditional rate:
given a draft already rejected by :class:`EvidenceVerifier`, can the model
rewrite it so that the same verifier accepts the answer?  It is not a natural
hallucination-rate benchmark and the report keeps that distinction explicit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from agents.agent_runtime import AgentRuntime
from core.evidence_verifier import EvidenceVerifier, VerificationReport
from core.llm_utils import extract_text_content


Corrector = Callable[
    [Mapping[str, Any], VerificationReport],
    Awaitable[str],
]


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


async def evaluate_reflections(
    cases: Sequence[Mapping[str, Any]],
    *,
    verifier: EvidenceVerifier,
    corrector: Corrector,
) -> dict[str, Any]:
    """Run one correction attempt for every verifier-detected draft."""
    rows: list[dict[str, Any]] = []
    detected = corrected = fallbacks = 0

    for case in cases:
        initial = verifier.verify(
            question=str(case.get("question", "")),
            response=str(case.get("response", "")),
            citations=case.get("citations", ()),
            tool_evidence=case.get("tool_evidence", ()),
        )
        row: dict[str, Any] = {
            "id": case.get("id"),
            "type": case.get("type"),
            "initial_issue_codes": [issue.code for issue in initial.issues],
            "detected": not initial.passed,
        }
        if initial.passed:
            row.update({"corrected": False, "safe_fallback_used": False})
            rows.append(row)
            continue

        detected += 1
        try:
            revised = await corrector(case, initial)
            final = verifier.verify(
                question=str(case.get("question", "")),
                response=revised,
                citations=case.get("citations", ()),
                tool_evidence=case.get("tool_evidence", ()),
            )
            passed = final.passed
            row.update({
                "revised_response": revised,
                "final_issue_codes": [issue.code for issue in final.issues],
                "corrected": passed,
                "safe_fallback_used": not passed,
            })
        except Exception as error:  # API failures are benchmark outcomes.
            passed = False
            row.update({
                "corrected": False,
                "safe_fallback_used": True,
                "error": f"{type(error).__name__}: {error}",
            })
        if passed:
            corrected += 1
        else:
            fallbacks += 1
        rows.append(row)

    return {
        "methodology": {
            "name": "detected-error reflection correction",
            "scope": (
                "Conditional correction rate after deterministic evidence "
                "verification; not a natural hallucination rate."
            ),
            "max_reflections": 1,
        },
        "summary": {
            "total_cases": len(cases),
            "detected_errors": detected,
            "corrected": corrected,
            "safe_fallbacks": fallbacks,
            "correction_rate": _ratio(corrected, detected),
        },
        "cases": rows,
    }


class DeepSeekReflectionCorrector:
    """Use the configured Anthropic-compatible DeepSeek endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        model: str,
    ) -> None:
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model = model

    async def __call__(
        self,
        case: Mapping[str, Any],
        report: VerificationReport,
    ) -> str:
        evidence_packet = {
            "question": case.get("question", ""),
            "citations": case.get("citations", []),
            "successful_tool_results": case.get("tool_evidence", []),
        }
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=600,
            temperature=0.0,
            system=(
                "你是 EchoMind 的证据约束回答器。只能依据给出的 Citation "
                "和成功 Tool 结果陈述事实；证据不足时明确拒答。"
            ),
            messages=[
                {
                    "role": "user",
                    "content": "权威证据包：\n"
                    + json.dumps(evidence_packet, ensure_ascii=False),
                },
                {"role": "assistant", "content": str(case.get("response", ""))},
                {
                    "role": "user",
                    "content": AgentRuntime._reflection_prompt(report),
                },
            ],
        )
        return extract_text_content(response.content).strip()


def select_detected_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    verifier: EvidenceVerifier,
    split: str,
    limit: int,
) -> list[Mapping[str, Any]]:
    """Select a deterministic, type-balanced set of detectable failures."""
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        if split != "all" and case.get("split") != split:
            continue
        report = verifier.verify(
            question=str(case.get("question", "")),
            response=str(case.get("response", "")),
            citations=case.get("citations", ()),
            tool_evidence=case.get("tool_evidence", ()),
        )
        if report.passed:
            continue
        buckets.setdefault(str(case.get("type", "unknown")), []).append(case)

    selected: list[Mapping[str, Any]] = []
    kinds = sorted(buckets)
    offset = 0
    while kinds and len(selected) < limit:
        remaining: list[str] = []
        for kind in kinds:
            bucket = buckets[kind]
            if offset < len(bucket) and len(selected) < limit:
                selected.append(bucket[offset])
            if offset + 1 < len(bucket):
                remaining.append(kind)
        kinds = remaining
        offset += 1
    return selected


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


async def _main(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    verifier = EvidenceVerifier()
    cases = select_detected_cases(
        dataset["cases"],
        verifier=verifier,
        split=args.split,
        limit=args.limit,
    )
    if not cases:
        raise RuntimeError("no verifier-detected cases matched the selection")
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip()
    corrector = DeepSeekReflectionCorrector(
        api_key=api_key,
        base_url=os.getenv("ANTHROPIC_BASE_URL", "").strip() or None,
        model=model,
    )
    result = await evaluate_reflections(
        cases,
        verifier=verifier,
        corrector=corrector,
    )
    result["methodology"].update({
        "model": model,
        "dataset": str(args.dataset),
        "split": args.split,
    })
    _write_json(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False))
    print(f"report={args.output}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "data" / "eval" / "hallucination_adversarial.json",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "data"
            / "eval"
            / "results"
            / "live_deepseek_reflection_latest.json"
        ),
    )
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
