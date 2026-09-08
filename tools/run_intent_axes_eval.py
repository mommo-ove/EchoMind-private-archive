"""Collect batched DeepSeek predictions and compare pure LLM with safety rules."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content
from evaluation.intent_axes import (
    apply_safety_rules,
    audit_business_gates,
    evaluate_predictions,
    evaluate_routing_intents,
    normalize_prediction,
)

AXIS_PROMPT_VERSION = "domain-action-report-escalation-v2"


class SingleRunLock(AbstractContextManager):
    """Small cross-process guard preventing two collectors from overwriting a checkpoint."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    def __enter__(self) -> "SingleRunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(f"intent axis evaluation is already running: {self.path}") from error
        os.write(self._fd, str(os.getpid()).encode("ascii"))
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self.path.unlink(missing_ok=True)


class BatchAxisClassifier:
    def __init__(self, *, api_key: str, base_url: str | None, model: str) -> None:
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = AsyncAnthropic(**kwargs)
        self.model = model

    async def classify_batch(self, cases: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        items = [{"case_id": case["case_id"], "message": case["message"]} for case in cases]
        prompt = f"""你是校园服务意图分类器。请把每条消息拆成三个互相独立的维度。

domain（可以多选）：
- technical：校园网、认证、连接、报错等技术问题
- billing：校园卡、充值、消费、扣款、退款等资金问题
- account：账号、密码、身份认证、个人资料等账户问题
- general：不属于以上专业领域；不能和其他domain同时出现

action（只能选一个）：
- greeting：只有寒暄，没有业务诉求
- query：询问信息、原因、状态或流程
- request：要求系统执行操作、创建工单或协助处理
- report：陈述正在发生的故障或异常，但没有明确提问、操作要求或不满表达
- complaint：表达不满、投诉或强调问题长期未解决
- feedback：提出产品建议或正面评价
- other：校园服务范围外，或无法归入以上动作

escalated：只有明确要求人工客服、真人、负责人、主管、老师介入或明确要求升级处理时才为true；仅仅说“紧急”“马上解决”仍为false。

边界示例：
- “退款没到，立刻给我接真人” => billing, complaint, true
- “网络坏了，有点着急” => technical, complaint, false
- “建议增加余额提醒” => billing, feedback, false
- “附近有什么好吃的” => general, other, false

返回严格JSON，不要Markdown：
{{"predictions":[{{"case_id":"...","domains":["..."],"action":"...","escalated":false}}]}}

待分类消息：
{json.dumps(items, ensure_ascii=False)}"""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max(2048, len(cases) * 256),
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = extract_text_content(response.content)
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("model returned no JSON object")
        payload = json.loads(raw[start:end])
        rows = payload.get("predictions")
        if not isinstance(rows, list):
            raise ValueError("model response is missing predictions")
        predictions = {}
        expected_ids = {case["case_id"] for case in cases}
        for row in rows:
            case_id = row.get("case_id") if isinstance(row, Mapping) else None
            if case_id in expected_ids:
                predictions[case_id] = normalize_prediction(row)
        missing = expected_ids - predictions.keys()
        if missing:
            raise ValueError(f"model omitted cases: {sorted(missing)}")
        return predictions


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


async def collect_predictions(args: argparse.Namespace, *, classifier: Any | None = None) -> Path:
    cases = json.loads(args.dataset.read_text(encoding="utf-8"))["cases"]
    collect_split = getattr(args, "collect_split", "all")
    target_cases = cases if collect_split == "all" else [case for case in cases if case["split"] == collect_split]
    classifier = classifier or BatchAxisClassifier(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
    )
    predictions: dict[str, dict[str, Any]] = {}
    if args.predictions.exists():
        checkpoint = json.loads(args.predictions.read_text(encoding="utf-8"))
        if checkpoint.get("prompt_version") == AXIS_PROMPT_VERSION:
            predictions.update(checkpoint.get("predictions", {}))
    async def collect_batch(batch: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        result = None
        for attempt in range(1, args.max_attempts + 1):
            try:
                result = await classifier.classify_batch(batch)
                break
            except Exception:
                if attempt == args.max_attempts and len(batch) > 1:
                    midpoint = len(batch) // 2
                    left = await collect_batch(batch[:midpoint])
                    right = await collect_batch(batch[midpoint:])
                    return {**left, **right}
                if attempt == args.max_attempts:
                    raise
                await asyncio.sleep(min(attempt, 2))
        return result or {}

    pending = [case for case in target_cases if case["case_id"] not in predictions]
    batches = [pending[offset:offset + args.batch_size] for offset in range(0, len(pending), args.batch_size)]
    concurrency = max(1, int(getattr(args, "concurrency", 1)))
    for offset in range(0, len(batches), concurrency):
        window = batches[offset:offset + concurrency]
        results = await asyncio.gather(*(collect_batch(batch) for batch in window))
        for result in results:
            predictions.update(result)
        _atomic_write(args.predictions, {
            "schema_version": 2,
            "prompt_version": AXIS_PROMPT_VERSION,
            "predictions": predictions,
        })
        completed_targets = sum(case["case_id"] in predictions for case in target_cases)
        print(f"checkpoint {completed_targets}/{len(target_cases)}")
    return args.predictions


def evaluate(args: argparse.Namespace) -> Path:
    cases = json.loads(args.dataset.read_text(encoding="utf-8"))["cases"]
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))["predictions"]
    evaluation_split = getattr(args, "evaluation_split", "all")
    evaluated_cases = cases if evaluation_split == "all" else [case for case in cases if case["split"] == evaluation_split]
    missing = {case["case_id"] for case in evaluated_cases} - predictions.keys()
    if missing:
        raise ValueError(f"predictions are incomplete: {len(missing)} missing")
    variants = {
        "llm_only": predictions,
        "llm_plus_safety_rules": {
            case["case_id"]: apply_safety_rules(case["message"], predictions[case["case_id"]])
            for case in evaluated_cases
        },
    }
    report: dict[str, Any] = {
        "schema_version": 2,
        "methodology": {
            "selection_split": "validation",
            "final_split": "held-out legacy challenge test",
            "variants": ["llm_only", "llm_plus_safety_rules"],
        },
        "splits": {},
    }
    splits = ("validation", "test") if evaluation_split == "all" else (evaluation_split,)
    for split in splits:
        split_cases = [case for case in evaluated_cases if case["split"] == split]
        report["splits"][split] = {
            name: evaluate_predictions(split_cases, values)
            for name, values in variants.items()
        }
        for name, values in variants.items():
            report["splits"][split][name]["routing_intents"] = (
                evaluate_routing_intents(split_cases, values)
            )
    final_split = "test" if "test" in report["splits"] else splits[-1]
    report["variants"] = report["splits"][final_split]
    report["gate_audits"] = {
        name: audit_business_gates(metrics)
        for name, metrics in report["variants"].items()
    }
    _atomic_write(args.output, report)
    return args.output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("collect", "evaluate", "all"))
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/intent_axes_golden.json"))
    parser.add_argument("--predictions", type=Path, default=Path("data/eval/results/intent_axes_predictions.json"))
    parser.add_argument("--output", type=Path, default=Path("data/eval/results/intent_axes_ab_latest.json"))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--collect-split", choices=("all", "validation", "test"), default="all")
    parser.add_argument("--evaluation-split", choices=("all", "validation", "test"), default="all")
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    lock_path = args.predictions.with_suffix(args.predictions.suffix + ".lock")
    with SingleRunLock(lock_path):
        if args.action in {"collect", "all"}:
            print(await collect_predictions(args))
        if args.action in {"evaluate", "all"}:
            print(evaluate(args))


if __name__ == "__main__":
    asyncio.run(main())
