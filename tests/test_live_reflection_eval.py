import asyncio

from core.evidence_verifier import EvidenceVerifier
from evaluation.live_reflection_eval import evaluate_reflections


def test_live_reflection_eval_reports_correction_rate_over_detected_errors():
    cases = [
        {
            "id": "ticket-case",
            "type": "entity_fabrication",
            "question": "请创建工单",
            "response": "已创建工单 ticket_fake。",
            "citations": [],
            "tool_evidence": [
                {
                    "name": "create_ticket",
                    "success": True,
                    "data": {"id": "ticket_real", "status": "OPEN"},
                }
            ],
        },
        {
            "id": "citation-case",
            "type": "missing_citation",
            "question": "退款多久",
            "response": "退款需要3个工作日。",
            "citations": [
                {
                    "id": "refund-policy",
                    "content": "退款需要3个工作日。",
                }
            ],
            "tool_evidence": [],
        },
    ]

    async def correct(case, _report):
        if case["id"] == "ticket-case":
            return "已创建工单 ticket_real，当前状态为待处理。"
        return "退款需要5个工作日。"

    report = asyncio.run(
        evaluate_reflections(
            cases,
            verifier=EvidenceVerifier(),
            corrector=correct,
        )
    )

    assert report["summary"] == {
        "total_cases": 2,
        "detected_errors": 2,
        "corrected": 1,
        "safe_fallbacks": 1,
        "correction_rate": 0.5,
    }
    assert report["cases"][0]["initial_issue_codes"] == [
        "unsupported_ticket_id"
    ]
    assert report["cases"][0]["corrected"] is True
    assert report["cases"][1]["corrected"] is False

