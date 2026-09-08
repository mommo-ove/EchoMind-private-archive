"""Build a reproducible synthetic adversarial grounding dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _split(index: int) -> str:
    position = index % 5
    if position == 0:
        return "test"
    if position == 1:
        return "validation"
    return "train"


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def add(case_type: str, payload: dict[str, Any]) -> None:
        index = len(cases)
        cases.append({
            "id": f"hallucination-{index + 1:03d}",
            "type": case_type,
            "split": _split(index),
            **payload,
        })

    for index in range(30):
        add("entity_fabrication", {
            "question": "请创建校园网报修工单",
            "response": f"已创建工单 ticket_fake_{index:03d}。",
            "citations": [],
            "tool_evidence": [{
                "name": "create_ticket",
                "success": True,
                "data": {"id": f"ticket_real_{index:03d}", "status": "OPEN"},
            }],
            "has_hallucination": True,
        })

    statuses = [
        ("OPEN", "已处理完成"),
        ("PROCESSING", "当前仍待处理"),
        ("RESOLVED", "目前仍在处理中"),
    ]
    for index in range(30):
        actual, wrong_text = statuses[index % len(statuses)]
        ticket_id = f"ticket_status_{index:03d}"
        add("status_mismatch", {
            "question": f"工单 {ticket_id} 现在是什么状态？",
            "response": f"工单 {ticket_id} {wrong_text}。",
            "citations": [],
            "tool_evidence": [{
                "name": "get_ticket",
                "success": True,
                "data": {"id": ticket_id, "status": actual},
            }],
            "has_hallucination": True,
        })

    for index in range(25):
        citation_id = f"kb-network-{index:03d}"
        add("missing_citation", {
            "question": "校园网出现401应该怎么处理？",
            "response": "请清除旧认证状态后重新登录。",
            "citations": [{
                "id": citation_id,
                "title": "校园网401认证处理",
                "content": "出现401时清除旧认证状态后重新登录。",
            }],
            "tool_evidence": [],
            "has_hallucination": True,
        })

    for index in range(30):
        citation_id = f"kb-refund-{index:03d}"
        expected_days = 3 + index % 3
        claimed_days = expected_days + 7
        add("numeric_mismatch", {
            "question": "退款需要多久到账？",
            "response": (
                f"退款会在{claimed_days}个工作日到账 "
                f"[Citation {citation_id}]。"
            ),
            "citations": [{
                "id": citation_id,
                "title": "退款到账时限",
                "content": f"退款审核通过后预计{expected_days}个工作日到账。",
            }],
            "tool_evidence": [],
            "has_hallucination": True,
        })

    for index in range(20):
        citation_id = f"kb-cause-{index:03d}"
        add("unsupported_causal_claim", {
            "question": "校园网为什么会出现401？",
            "response": (
                "根因一定是学校服务器硬件损坏 "
                f"[Citation {citation_id}]。"
            ),
            "citations": [{
                "id": citation_id,
                "title": "401处理建议",
                "content": "出现401时可以清除旧认证状态后重新登录。",
            }],
            "tool_evidence": [],
            "has_hallucination": True,
        })

    supported_statuses = [
        ("OPEN", "当前仍待处理"),
        ("PROCESSING", "目前正在处理中"),
        ("RESOLVED", "已经处理完成"),
    ]
    for index in range(65):
        variant = index % 5
        if variant == 0:
            status, status_text = supported_statuses[index % 3]
            ticket_id = f"ticket_supported_{index:03d}"
            payload = {
                "question": f"工单 {ticket_id} 的状态？",
                "response": f"工单 {ticket_id} {status_text}。",
                "citations": [],
                "tool_evidence": [{
                    "name": "get_ticket",
                    "success": True,
                    "data": {"id": ticket_id, "status": status},
                }],
            }
        elif variant == 1:
            citation_id = f"kb-supported-{index:03d}"
            payload = {
                "question": "校园网401怎么处理？",
                "response": (
                    "清除旧认证状态后重新登录 "
                    f"[Citation {citation_id}]。"
                ),
                "citations": [{
                    "id": citation_id,
                    "title": "401处理",
                    "content": "清除旧认证状态后重新登录。",
                }],
                "tool_evidence": [],
            }
        elif variant == 2:
            payload = {
                "question": "401的根因是什么？",
                "response": "当前证据不足，无法确认具体根因。",
                "citations": [{
                    "id": f"kb-abstain-{index:03d}",
                    "title": "401处理",
                    "content": "文档只提供处理步骤，未说明根因。",
                }],
                "tool_evidence": [],
            }
        elif variant == 3:
            ticket_id = f"ticket_created_{index:03d}"
            payload = {
                "question": "创建网络报修工单",
                "response": f"已创建工单 {ticket_id}。",
                "citations": [],
                "tool_evidence": [{
                    "name": "create_ticket",
                    "success": True,
                    "data": {"id": ticket_id, "status": "OPEN"},
                }],
            }
        else:
            citation_id = f"kb-days-{index:03d}"
            payload = {
                "question": "退款多久到账？",
                "response": f"退款预计3个工作日到账 [Citation {citation_id}]。",
                "citations": [{
                    "id": citation_id,
                    "title": "退款到账时限",
                    "content": "退款审核通过后预计3个工作日到账。",
                }],
                "tool_evidence": [],
            }
        add("supported", {**payload, "has_hallucination": False})

    return cases


def main() -> None:
    output = Path("data/eval/hallucination_adversarial.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "name": "EchoMind synthetic adversarial grounding benchmark",
            "synthetic": True,
            "scope": (
                "Rule-scope benchmark for citation, ticket, status, numeric, "
                "and unsupported causal claims; not real production traffic."
            ),
            "count": 200,
            "split": "60/20/20 deterministic stratification",
        },
        "cases": build_cases(),
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
