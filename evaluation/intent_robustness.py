"""Local intent fallback calibration and reproducible fault-injection evaluation."""

from __future__ import annotations

import hashlib
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from evaluation.intent_axes import evaluate_routing_intents, normalize_prediction


ROUTING_LABELS = ("technical", "billing", "account")

INTENT_PROTOTYPES = {
    "technical": (
        "校园网断网、掉线、延迟、无法连接或网页打不开",
        "WiFi认证失败、认证页面报401或403、代理关闭后仍不能联网",
        "宿舍与教学楼网络故障、连接超时、服务状态异常",
    ),
    "billing": (
        "校园卡扣款、重复支付、消费流水、余额或账单异常",
        "充值未到账、退款未到账、退款审核、支付订单问题",
        "校园卡挂失后的资金安全、陌生消费与盗刷",
    ),
    "account": (
        "校园账号或统一认证账号被锁定、冻结、停用",
        "忘记密码、密码重置、验证码、登录失败",
        "绑定手机号、身份资料、陌生设备、设备数量与账号权限",
    ),
}

PATTERN_KEYWORDS = {
    "technical": (
        "校园网", "网络", "wifi", "wi-fi", "断网", "掉线", "延迟", "网速", "联网",
        "401", "403", "认证页", "连接超时", "网页打不开",
    ),
    "billing": (
        "校园卡", "扣款", "退款", "充值", "余额", "账单", "消费", "支付", "流水",
        "到账", "盗刷", "金额",
    ),
    "account": (
        "账号", "账户", "密码", "登录", "验证码", "手机号", "设备数量", "陌生设备",
        "锁定", "冻结", "身份资料", "权限", "统一认证",
    ),
}

NEUTRAL_VARIATIONS = (
    "关于这个情况：{message}",
    "我这边的情况是：{message}",
    "想确认一下，{message}",
    "具体问题如下：{message}",
    "最近遇到了这个情况：{message}",
    "请看一下这个问题：{message}",
    "补充说明一下：{message}",
    "咨询一个校园服务问题：{message}",
)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _routing_domains(prediction: Mapping[str, Any]) -> set[str]:
    return {
        domain
        for domain in normalize_prediction(prediction)["domains"]
        if domain in ROUTING_LABELS
    }


def _prediction(domains: Sequence[str]) -> dict[str, Any]:
    selected = list(dict.fromkeys(domain for domain in domains if domain in ROUTING_LABELS))
    return {
        "domains": selected or ["general"],
        "action": "other",
        "escalated": False,
    }


def expand_calibration_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    target_validation_count: int,
) -> list[dict[str, Any]]:
    """Expand only the calibration split; held-out test rows stay byte-for-byte equivalent."""
    copied = [deepcopy(dict(case)) for case in cases]
    validation = [case for case in copied if case["split"] == "validation"]
    if target_validation_count <= len(validation):
        return copied
    if not validation:
        raise ValueError("at least one validation case is required")

    existing_messages = {case["message"] for case in copied}
    generated: list[dict[str, Any]] = []
    round_index = 0
    while len(validation) + len(generated) < target_validation_count:
        source = validation[round_index % len(validation)]
        template = NEUTRAL_VARIATIONS[(round_index // len(validation)) % len(NEUTRAL_VARIATIONS)]
        message = template.format(message=source["message"])
        if message not in existing_messages:
            row = deepcopy(source)
            row["case_id"] = f"axis-aug-{len(generated) + 1:04d}"
            row["message"] = message
            row["source"] = "routing_paraphrase_augmentation"
            row["augmentation"] = {
                "source_case_id": source["case_id"],
                "method": "neutral_discourse_wrapper",
            }
            generated.append(row)
            existing_messages.add(message)
        round_index += 1
        if round_index > len(validation) * len(NEUTRAL_VARIATIONS) * 2:
            raise ValueError("target exceeds deterministic augmentation capacity")
    return [*validation, *generated, *[case for case in copied if case["split"] != "validation"]]


@dataclass(frozen=True)
class LocalRouterConfig:
    embedding_weight: float
    pattern_weight: float
    thresholds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TuningResult:
    config: LocalRouterConfig
    selection_case_ids: list[str]
    predictions: dict[str, dict[str, Any]]
    validation_macro_f1: float


class IntentPrototypeRouter:
    """Multi-label BGE prototype similarity plus deterministic keyword evidence."""

    def __init__(self, *, embedder: Any) -> None:
        self.embedder = embedder
        flattened = [text for label in ROUTING_LABELS for text in INTENT_PROTOTYPES[label]]
        vectors = self.embedder.embed(flattened)
        self.prototype_vectors: dict[str, list[list[float]]] = {}
        cursor = 0
        for label in ROUTING_LABELS:
            count = len(INTENT_PROTOTYPES[label])
            self.prototype_vectors[label] = vectors[cursor:cursor + count]
            cursor += count

    def component_scores(self, messages: Sequence[str]) -> list[dict[str, dict[str, float]]]:
        vectors = self.embedder.embed(messages)
        rows = []
        for message, vector in zip(messages, vectors):
            normalized = message.lower()
            embedding = {
                label: max(_cosine(vector, prototype) for prototype in self.prototype_vectors[label])
                for label in ROUTING_LABELS
            }
            pattern = {
                label: float(any(keyword in normalized for keyword in PATTERN_KEYWORDS[label]))
                for label in ROUTING_LABELS
            }
            rows.append({"embedding": embedding, "pattern": pattern})
        return rows

    @staticmethod
    def predict(
        component_scores: Sequence[Mapping[str, Mapping[str, float]]],
        config: LocalRouterConfig,
    ) -> list[dict[str, Any]]:
        predictions = []
        for row in component_scores:
            selected = []
            for label in ROUTING_LABELS:
                score = (
                    config.embedding_weight * float(row["embedding"][label])
                    + config.pattern_weight * float(row["pattern"][label])
                )
                if score >= config.thresholds[label]:
                    selected.append(label)
            predictions.append(_prediction(selected))
        return predictions


def _binary_f1(expected: Sequence[bool], predicted: Sequence[bool]) -> float:
    tp = sum(e and p for e, p in zip(expected, predicted))
    fp = sum(not e and p for e, p in zip(expected, predicted))
    fn = sum(e and not p for e, p in zip(expected, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def tune_local_router(
    cases: Sequence[Mapping[str, Any]],
    *,
    router: IntentPrototypeRouter,
    candidate_embedding_weights: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> TuningResult:
    """Select local weights/thresholds on validation only, then predict every row."""
    selection = [case for case in cases if case["split"] == "validation"]
    if not selection:
        raise ValueError("validation cases are required for calibration")
    all_scores = router.component_scores([case["message"] for case in cases])
    score_by_id = {case["case_id"]: row for case, row in zip(cases, all_scores)}
    thresholds = [round(value / 20, 2) for value in range(1, 20)]
    best: tuple[float, float, dict[str, float]] | None = None

    for embedding_weight in candidate_embedding_weights:
        if not 0 <= embedding_weight <= 1:
            raise ValueError("embedding weights must be between zero and one")
        pattern_weight = 1.0 - embedding_weight
        selected_thresholds = {}
        label_f1s = []
        for label in ROUTING_LABELS:
            expected = [label in case["expected"]["domains"] for case in selection]
            scored = [
                embedding_weight * score_by_id[case["case_id"]]["embedding"][label]
                + pattern_weight * score_by_id[case["case_id"]]["pattern"][label]
                for case in selection
            ]
            candidates = [
                (_binary_f1(expected, [score >= threshold for score in scored]), threshold)
                for threshold in thresholds
            ]
            label_f1, threshold = max(candidates, key=lambda item: (item[0], item[1]))
            selected_thresholds[label] = threshold
            label_f1s.append(label_f1)
        macro = sum(label_f1s) / len(label_f1s)
        candidate = (macro, embedding_weight, selected_thresholds)
        if best is None or (candidate[0], candidate[1]) > (best[0], best[1]):
            best = candidate

    assert best is not None
    config = LocalRouterConfig(
        embedding_weight=best[1],
        pattern_weight=1.0 - best[1],
        thresholds=best[2],
    )
    predictions = {
        case["case_id"]: prediction
        for case, prediction in zip(cases, router.predict(all_scores, config))
    }
    return TuningResult(
        config=config,
        selection_case_ids=[case["case_id"] for case in selection],
        predictions=predictions,
        validation_macro_f1=round(best[0], 6),
    )


@dataclass(frozen=True)
class FaultScenario:
    failure_rate: float
    trials: int = 100
    seed: int = 20260813


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _available_exact_match(
    cases: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
) -> float:
    """Treat transport/model failures as failed routes, including GeneralAgent cases."""
    correct = 0
    for case in cases:
        case_id = case["case_id"]
        if case_id not in predictions:
            continue
        expected = {
            domain for domain in case["expected"]["domains"] if domain in ROUTING_LABELS
        }
        correct += _routing_domains(predictions[case_id]) == expected
    return correct / len(cases) if cases else 0.0


def evaluate_fault_scenarios(
    cases: Sequence[Mapping[str, Any]],
    llm_predictions: Mapping[str, Mapping[str, Any]],
    local_predictions: Mapping[str, Mapping[str, Any]],
    *,
    scenarios: Sequence[FaultScenario],
) -> dict[str, Any]:
    """Inject deterministic LLM failures and compare no-fallback with local fallback."""
    test_cases = [case for case in cases if case["split"] == "test"]
    case_ids = [case["case_id"] for case in test_cases]
    report = {}
    for scenario in scenarios:
        if not 0 <= scenario.failure_rate <= 1:
            raise ValueError("failure_rate must be between zero and one")
        failed_count = round(len(case_ids) * scenario.failure_rate)
        baseline_exact, baseline_f1, baseline_available = [], [], []
        hybrid_exact, hybrid_f1, hybrid_available, recovered = [], [], [], []
        for trial in range(scenario.trials):
            rng = random.Random(scenario.seed + trial)
            failed = set(rng.sample(case_ids, failed_count))
            baseline = {case_id: llm_predictions[case_id] for case_id in case_ids if case_id not in failed}
            hybrid = {
                case_id: (local_predictions[case_id] if case_id in failed else llm_predictions[case_id])
                for case_id in case_ids
            }
            baseline_metrics = evaluate_routing_intents(test_cases, baseline)
            hybrid_metrics = evaluate_routing_intents(test_cases, hybrid)
            baseline_exact.append(_available_exact_match(test_cases, baseline))
            baseline_f1.append(baseline_metrics["macro_f1"])
            baseline_available.append((len(case_ids) - failed_count) / len(case_ids))
            hybrid_exact.append(_available_exact_match(test_cases, hybrid))
            hybrid_f1.append(hybrid_metrics["macro_f1"])
            hybrid_available.append(1.0)
            if failed:
                correct = 0
                for case in test_cases:
                    if case["case_id"] in failed:
                        correct += _routing_domains(local_predictions[case["case_id"]]) == {
                            domain for domain in case["expected"]["domains"] if domain in ROUTING_LABELS
                        }
                recovered.append(correct / len(failed))
            else:
                recovered.append(1.0)
        name = f"failure_rate_{round(scenario.failure_rate * 100)}"
        report[name] = {
            "failure_rate": scenario.failure_rate,
            "trials": scenario.trials,
            "llm_without_fallback": {
                "response_availability": _mean(baseline_available),
                "routing_exact_match_mean": _mean(baseline_exact),
                "routing_macro_f1_mean": _mean(baseline_f1),
            },
            "local_fallback": {
                "response_availability": _mean(hybrid_available),
                "routing_exact_match_mean": _mean(hybrid_exact),
                "routing_macro_f1_mean": _mean(hybrid_f1),
                "failed_request_recovery_rate": _mean(recovered),
            },
        }
    return report


def dataset_fingerprint(cases: Sequence[Mapping[str, Any]]) -> str:
    values = "\n".join(f"{case['case_id']}\t{case['message']}" for case in cases)
    return hashlib.sha256(values.encode("utf-8")).hexdigest()
