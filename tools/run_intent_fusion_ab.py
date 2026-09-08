"""Collect DeepSeek domain confidence once and run a fair three-way fusion A/B test."""

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
from evaluation.intent_fusion_ab import ROUTING_LABELS, compare_fusion_variants
from evaluation.intent_robustness import IntentPrototypeRouter
from mcp.local_embeddings import DEFAULT_EMBEDDING_MODEL, FastEmbedTextModel


PROMPT_VERSION = "routing-domain-scores-v1"


class SingleRunLock(AbstractContextManager):
    """Prevent concurrent collectors from overwriting the same checkpoint."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    def __enter__(self) -> "SingleRunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(f"intent fusion score collection is already running: {self.path}") from error
        os.write(self._fd, str(os.getpid()).encode("ascii"))
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self.path.unlink(missing_ok=True)


def parse_domain_score_response(raw: str, *, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("model returned no JSON object")
    payload = json.loads(raw[start:end])
    rows = payload.get("predictions")
    if not isinstance(rows, list):
        raise ValueError("model response is missing predictions")
    parsed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("case_id") not in expected_ids:
            continue
        case_id = str(row["case_id"])
        raw_domains = row.get("domains", [])
        domains = [label for label in ROUTING_LABELS if label in raw_domains]
        raw_scores = row.get("domain_scores")
        if not isinstance(raw_scores, Mapping):
            raise ValueError(f"model omitted domain_scores for {case_id}")
        scores = {}
        for label in ROUTING_LABELS:
            if label not in raw_scores:
                raise ValueError(f"model omitted {label} score for {case_id}")
            scores[label] = min(max(float(raw_scores[label]), 0.0), 1.0)
        parsed[case_id] = {
            "llm_domains": domains,
            "llm": scores,
        }
    missing = expected_ids - parsed.keys()
    if missing:
        raise ValueError(f"model omitted cases: {sorted(missing)}")
    return parsed


class BatchDomainScoreClassifier:
    def __init__(self, *, api_key: str, base_url: str | None, model: str) -> None:
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = AsyncAnthropic(**kwargs)
        self.model = model

    async def classify_batch(self, cases: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        items = [{"case_id": case["case_id"], "message": case["message"]} for case in cases]
        prompt = f"""你是校园服务多标签路由分类器。只判断三个可路由业务领域：
- technical：校园网、网络认证、连接、报错、技术故障
- billing：校园卡、充值、消费、扣款、退款、资金异常
- account：账号、密码、登录身份、设备限制、账户安全

同一句话可以命中多个领域。与三个领域都无关时 domains 返回空数组。
对每个领域都给出 0 到 1 的置信度；不要把礼貌用语、动作类型或转人工当成业务领域。

严格返回 JSON，不要 Markdown，不要解释：
{{"predictions":[{{"case_id":"...","domains":["technical"],"domain_scores":{{"technical":0.95,"billing":0.02,"account":0.03}}}}]}}

待分类消息：
{json.dumps(items, ensure_ascii=False)}"""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max(4096, len(cases) * 384),
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return parse_domain_score_response(
            extract_text_content(response.content),
            expected_ids={str(case["case_id"]) for case in cases},
        )


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


async def _collect_unlocked(args: argparse.Namespace, *, classifier: Any | None = None) -> Path:
    cases = json.loads(args.dataset.read_text(encoding="utf-8"))["cases"]
    classifier = classifier or BatchDomainScoreClassifier(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
    )
    predictions: dict[str, dict[str, Any]] = {}
    if args.llm_scores.exists():
        checkpoint = json.loads(args.llm_scores.read_text(encoding="utf-8"))
        if checkpoint.get("prompt_version") == PROMPT_VERSION:
            predictions.update(checkpoint.get("predictions", {}))

    async def classify_with_retry(batch: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        for attempt in range(1, args.max_attempts + 1):
            try:
                return await asyncio.wait_for(
                    classifier.classify_batch(batch),
                    timeout=float(args.request_timeout),
                )
            except Exception:
                if attempt == args.max_attempts and len(batch) > 1:
                    midpoint = len(batch) // 2
                    left = await classify_with_retry(batch[:midpoint])
                    right = await classify_with_retry(batch[midpoint:])
                    return {**left, **right}
                if attempt == args.max_attempts:
                    raise
                await asyncio.sleep(min(attempt, 2))
        return {}

    pending = [case for case in cases if case["case_id"] not in predictions]
    batches = [pending[offset:offset + args.batch_size] for offset in range(0, len(pending), args.batch_size)]
    for offset in range(0, len(batches), args.concurrency):
        window = batches[offset:offset + args.concurrency]
        results = await asyncio.gather(*(classify_with_retry(batch) for batch in window))
        for result in results:
            predictions.update(result)
        _atomic_write(args.llm_scores, {
            "schema_version": 1,
            "prompt_version": PROMPT_VERSION,
            "model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            "predictions": predictions,
        })
        print(f"checkpoint {len(predictions)}/{len(cases)}", flush=True)
    return args.llm_scores


async def collect(args: argparse.Namespace, *, classifier: Any | None = None) -> Path:
    lock_path = args.llm_scores.with_suffix(args.llm_scores.suffix + ".lock")
    with SingleRunLock(lock_path):
        return await _collect_unlocked(args, classifier=classifier)


def evaluate(args: argparse.Namespace) -> Path:
    cases = json.loads(args.dataset.read_text(encoding="utf-8"))["cases"]
    llm = json.loads(args.llm_scores.read_text(encoding="utf-8"))["predictions"]
    missing = {case["case_id"] for case in cases} - llm.keys()
    if missing:
        raise ValueError(f"LLM score checkpoint is missing {len(missing)} cases")

    embedder = FastEmbedTextModel(model_name=args.embedding_model, cache_dir=args.model_cache)
    router = IntentPrototypeRouter(embedder=embedder)
    local_rows = router.component_scores([case["message"] for case in cases])
    score_rows = {
        case["case_id"]: {
            **llm[case["case_id"]],
            "embedding": local["embedding"],
            "pattern": local["pattern"],
        }
        for case, local in zip(cases, local_rows)
    }
    report = compare_fusion_variants(cases, score_rows)
    report.update({
        "schema_version": 1,
        "methodology": {
            "dataset": str(args.dataset),
            "llm_model": os.getenv("ANTHROPIC_MODEL", "unknown"),
            "embedding_model": args.embedding_model,
            "selection_rule": "weights and per-label thresholds selected on validation only",
            "test_rule": "100 held-out cases evaluated once for every variant",
        },
    })
    _atomic_write(args.output, report)
    _write_markdown(args.markdown, report)
    return args.output


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# EchoMind 多标签意图三路融合公平对比",
        "",
        "- 参数选择：仅使用 validation。",
        f"- 最终测试：同一批 {len(report['test_case_ids'])} 条 held-out 数据。",
        "- 指标仅覆盖 technical / billing / account 多标签 Agent 路由。",
        "",
        "| 方案 | LLM/BGE/Pattern | Macro-F1 | 严格命中率 | 相对LLM Macro-F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for name, row in report["variants"].items():
        config = row["config"]
        if "llm_weight" in config:
            weights = f"{config['llm_weight']:.2f}/{config['embedding_weight']:.2f}/{config['pattern_weight']:.2f}"
        else:
            weights = "直接输出"
        metrics = row["metrics"]
        lines.append(
            f"| {name} | {weights} | {metrics['macro_f1']:.2%} | "
            f"{metrics['exact_match']:.2%} | {row['delta_vs_deepseek']['macro_f1']:+.2%} |"
        )
    lines.extend([
        "",
        "> 若最佳网格权重退化为单路或未超过 DeepSeek，只能说明三路融合的正常精度优势未成立；本地分支的价值应由故障注入实验说明。",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("collect", "evaluate", "all"))
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/intent_axes_golden.json"))
    parser.add_argument("--llm-scores", type=Path, default=Path("data/eval/results/intent_fusion_llm_scores.json"))
    parser.add_argument("--output", type=Path, default=Path("data/eval/results/intent_fusion_ab_latest.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/意图三路融合公平对比.md"))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--model-cache", default=None)
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    if args.action in {"collect", "all"}:
        print(await collect(args))
    if args.action in {"evaluate", "all"}:
        print(evaluate(args))


if __name__ == "__main__":
    asyncio.run(main())
