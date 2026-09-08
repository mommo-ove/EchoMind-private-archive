import json

import pytest

from core.evidence_verifier import EvidenceVerifier


def _subject():
    try:
        from evaluation.hallucination_eval import HallucinationEvaluator
    except ModuleNotFoundError as exc:
        pytest.fail(f"HallucinationEvaluator is not implemented: {exc}")
    return HallucinationEvaluator(EvidenceVerifier())


def test_hallucination_metrics_use_binary_confusion_matrix():
    cases = [
        {
            "id": "caught",
            "type": "entity_fabrication",
            "question": "创建工单",
            "response": "已创建工单 ticket_wrong",
            "citations": [],
            "tool_evidence": [{
                "name": "create_ticket",
                "success": True,
                "data": {"id": "ticket_real", "status": "OPEN"},
            }],
            "has_hallucination": True,
        },
        {
            "id": "clean",
            "type": "supported",
            "question": "工单状态？",
            "response": "工单 ticket_real 当前待处理",
            "citations": [],
            "tool_evidence": [{
                "name": "get_ticket",
                "success": True,
                "data": {"id": "ticket_real", "status": "OPEN"},
            }],
            "has_hallucination": False,
        },
        {
            "id": "missed",
            "type": "unsupported_causal_claim",
            "question": "为什么401？",
            "response": "根因一定是服务器损坏 [Citation kb-1]",
            "citations": [{
                "id": "kb-1",
                "title": "401",
                "content": "401时可以清理旧认证状态。",
            }],
            "tool_evidence": [],
            "has_hallucination": True,
        },
        {
            "id": "clean-abstain",
            "type": "supported",
            "question": "根因？",
            "response": "证据不足，无法确认根因。",
            "citations": [{
                "id": "kb-1",
                "title": "401",
                "content": "401时可以清理旧认证状态。",
            }],
            "tool_evidence": [],
            "has_hallucination": False,
        },
    ]

    report = _subject().evaluate(cases)

    assert report["confusion_matrix"] == {
        "tp": 1, "fp": 0, "tn": 2, "fn": 1,
    }
    assert report["metrics"]["precision"] == 1.0
    assert report["metrics"]["recall"] == 0.5
    assert report["metrics"]["f1"] == pytest.approx(2 / 3, abs=1e-6)
    assert report["metrics"]["false_negative_rate"] == 0.5
    assert report["metrics"]["accuracy"] == 0.75
    assert report["by_type"]["unsupported_causal_claim"]["fn"] == 1


def test_hallucination_evaluator_loads_json_cases(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({
        "metadata": {"kind": "synthetic"},
        "cases": [{
            "id": "one",
            "type": "supported",
            "split": "test",
            "question": "状态？",
            "response": "证据不足，无法确认。",
            "citations": [],
            "tool_evidence": [],
            "has_hallucination": False,
        }],
    }, ensure_ascii=False), encoding="utf-8")

    report = _subject().evaluate_file(path, split="test")

    assert report["total"] == 1
    assert report["metadata"]["kind"] == "synthetic"


def test_hallucination_evaluator_rejects_empty_or_invalid_cases():
    with pytest.raises(ValueError):
        _subject().evaluate([])
    with pytest.raises(ValueError):
        _subject().evaluate([{"id": "broken"}])
