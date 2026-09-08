import pytest


def _verifier():
    try:
        from core.evidence_verifier import EvidenceVerifier
    except ModuleNotFoundError as exc:
        pytest.fail(f"EvidenceVerifier is not implemented: {exc}")
    return EvidenceVerifier()


def _issue_codes(report):
    return {issue.code for issue in report.issues}


def test_empty_answer_never_passes_evidence_verification():
    report = _verifier().verify(
        question="工单状态是什么？",
        response="   ",
        citations=[],
        tool_evidence=[{
            "name": "get_ticket",
            "success": True,
            "data": {"id": "ticket_cafebabe", "status": "OPEN"},
        }],
    )

    assert report.passed is False
    assert "empty_response" in _issue_codes(report)


def test_knowledge_answer_requires_a_stable_citation():
    report = _verifier().verify(
        question="校园网401怎么处理？",
        response="清除旧认证状态后重新登录。",
        citations=[{
            "id": "kb-network-401",
            "title": "校园网认证故障",
            "content": "出现401时清除旧认证状态后重新登录。",
        }],
        tool_evidence=[],
    )

    assert report.passed is False
    assert "missing_citation" in _issue_codes(report)


def test_knowledge_answer_with_matching_citation_passes():
    report = _verifier().verify(
        question="校园网401怎么处理？",
        response="请清除旧认证状态后重新登录 [Citation kb-network-401]。",
        citations=[{
            "id": "kb-network-401",
            "title": "校园网认证故障",
            "content": "出现401时清除旧认证状态后重新登录。",
        }],
        tool_evidence=[],
    )

    assert report.passed is True
    assert report.issues == ()


def test_created_ticket_claim_must_match_successful_tool_result():
    report = _verifier().verify(
        question="帮我创建网络报修工单",
        response="已创建工单 ticket_deadbeef。",
        citations=[],
        tool_evidence=[{
            "name": "create_ticket",
            "success": True,
            "data": {"id": "ticket_cafebabe", "status": "OPEN"},
        }],
    )

    assert report.passed is False
    assert "unsupported_ticket_id" in _issue_codes(report)


def test_structured_status_claim_must_match_tool_result():
    report = _verifier().verify(
        question="工单处理完了吗？",
        response="工单 ticket_cafebabe 已处理完成。",
        citations=[],
        tool_evidence=[{
            "name": "get_ticket",
            "success": True,
            "data": {"id": "ticket_cafebabe", "status": "OPEN"},
        }],
    )

    assert report.passed is False
    assert "conflicting_status" in _issue_codes(report)


def test_supported_structured_claim_passes():
    report = _verifier().verify(
        question="工单处理完了吗？",
        response="工单 ticket_cafebabe 当前仍待处理。",
        citations=[],
        tool_evidence=[{
            "name": "get_ticket",
            "success": True,
            "data": {"id": "ticket_cafebabe", "status": "OPEN"},
        }],
    )

    assert report.passed is True
    assert report.checked_claims >= 2


def test_safe_abstention_does_not_require_a_citation():
    report = _verifier().verify(
        question="认证失败的根因是什么？",
        response="当前证据不足，无法确认具体根因。",
        citations=[{
            "id": "kb-network-401",
            "title": "校园网认证故障",
            "content": "出现401时可以清除旧认证状态。",
        }],
        tool_evidence=[],
    )

    assert report.passed is True
    assert report.abstained is True


def test_numeric_duration_must_match_cited_evidence():
    report = _verifier().verify(
        question="退款多久到账？",
        response="退款预计10个工作日到账 [Citation kb-refund]。",
        citations=[{
            "id": "kb-refund",
            "title": "退款时限",
            "content": "退款审核通过后预计3个工作日到账。",
        }],
        tool_evidence=[],
    )

    assert report.passed is False
    assert "conflicting_numeric_claim" in _issue_codes(report)


def test_matching_numeric_duration_passes():
    report = _verifier().verify(
        question="退款多久到账？",
        response="退款预计3个工作日到账 [Citation kb-refund]。",
        citations=[{
            "id": "kb-refund",
            "title": "退款时限",
            "content": "退款审核通过后预计3个工作日到账。",
        }],
        tool_evidence=[],
    )

    assert report.passed is True


def test_amount_cents_supports_equivalent_yuan_claim():
    supported = _verifier().verify(
        question="这笔消费多少钱？",
        response="本次消费12.50元。",
        citations=[],
        tool_evidence=[{
            "name": "query_campus_card",
            "success": True,
            "data": {"amount_cents": 1250, "merchant": "第一食堂"},
        }],
    )
    unsupported = _verifier().verify(
        question="这笔消费多少钱？",
        response="本次消费125元。",
        citations=[],
        tool_evidence=[{
            "name": "query_campus_card",
            "success": True,
            "data": {"amount_cents": 1250, "merchant": "第一食堂"},
        }],
    )

    assert supported.passed is True
    assert unsupported.passed is False
    assert "conflicting_numeric_claim" in _issue_codes(unsupported)
