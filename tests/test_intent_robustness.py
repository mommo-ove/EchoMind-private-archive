import pytest

from evaluation.intent_robustness import (
    FaultScenario,
    IntentPrototypeRouter,
    evaluate_fault_scenarios,
    expand_calibration_cases,
    tune_local_router,
)


class KeywordEmbedder:
    """Small deterministic semantic stand-in; production evaluation uses BGE."""

    def embed(self, texts):
        vectors = []
        for text in texts:
            vectors.append([
                float(any(word in text for word in ("校园网", "网络", "断网", "认证"))),
                float(any(word in text for word in ("扣款", "退款", "校园卡"))),
                float(any(word in text for word in ("密码", "账号", "登录"))),
            ])
        return vectors


def _case(case_id, message, domains, *, split="validation"):
    return {
        "case_id": case_id,
        "split": split,
        "message": message,
        "expected": {"domains": domains, "action": "report", "escalated": False},
    }


def test_expansion_preserves_heldout_test_and_produces_unique_calibration_rows():
    cases = [
        _case("val-1", "校园网断了", ["technical"]),
        _case("val-2", "校园卡重复扣款", ["billing"]),
        _case("test-1", "账号无法登录", ["account"], split="test"),
    ]

    expanded = expand_calibration_cases(cases, target_validation_count=6)

    assert len([case for case in expanded if case["split"] == "validation"]) == 6
    assert [case for case in expanded if case["split"] == "test"] == [cases[-1]]
    assert len({case["message"] for case in expanded}) == len(expanded)


def test_bge_pattern_tuning_selects_thresholds_without_using_test_cases():
    cases = [
        _case("v1", "校园网认证失败", ["technical"]),
        _case("v2", "校园卡重复扣款", ["billing"]),
        _case("v3", "账号密码错误", ["account"]),
        _case("v4", "今天食堂开门吗", ["general"]),
        _case("t1", "校园网断了", ["technical"], split="test"),
    ]
    router = IntentPrototypeRouter(embedder=KeywordEmbedder())

    result = tune_local_router(cases, router=router)

    assert set(result.config.thresholds) == {"technical", "billing", "account"}
    assert result.selection_case_ids == ["v1", "v2", "v3", "v4"]
    assert "t1" not in result.selection_case_ids
    assert result.predictions["t1"]["domains"] == ["technical"]

    embedding_only = tune_local_router(
        cases,
        router=router,
        candidate_embedding_weights=(1.0,),
    )
    assert embedding_only.config.embedding_weight == 1.0
    assert embedding_only.config.pattern_weight == 0.0


def test_fault_ablation_reports_availability_and_correct_recovery():
    cases = [
        _case("t1", "校园网断了", ["technical"], split="test"),
        _case("t2", "校园卡重复扣款", ["billing"], split="test"),
        _case("t3", "账号密码错误", ["account"], split="test"),
        _case("t4", "今天食堂开门吗", ["general"], split="test"),
    ]
    llm = {
        case["case_id"]: {
            "domains": case["expected"]["domains"],
            "action": "report",
            "escalated": False,
        }
        for case in cases
    }
    local = dict(llm)
    scenarios = [
        FaultScenario(failure_rate=0.5, trials=20, seed=7),
        FaultScenario(failure_rate=1.0, trials=20, seed=7),
    ]

    report = evaluate_fault_scenarios(cases, llm, local, scenarios=scenarios)

    baseline = report["failure_rate_50"]["llm_without_fallback"]
    hybrid = report["failure_rate_50"]["local_fallback"]
    assert baseline["response_availability"] == pytest.approx(0.5)
    assert hybrid["response_availability"] == 1.0
    assert hybrid["failed_request_recovery_rate"] == 1.0
    assert hybrid["routing_exact_match_mean"] == 1.0
    assert report["failure_rate_100"]["llm_without_fallback"]["routing_exact_match_mean"] == 0.0
