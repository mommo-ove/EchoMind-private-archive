"""Deterministic grounding checks for knowledge and tool-backed answers.

The verifier deliberately handles only claims that can be checked without a
second model: stable citation identifiers, ticket identifiers, successful
business actions, and normalized status values.  Ambiguous semantic claims
remain the responsibility of offline RAGAS/LLM-as-Judge evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping, Sequence


_TICKET_ID = re.compile(r"\bticket_[A-Za-z0-9_-]+\b", re.IGNORECASE)
_EXPLICIT_CITATION = re.compile(
    r"\[\s*Citation\s+([^\]\r\n]+)\s*\]",
    re.IGNORECASE,
)
_ABSTENTION_MARKERS = (
    "无法确认",
    "不能确认",
    "证据不足",
    "暂无证据",
    "无法根据当前证据",
    "cannot confirm",
    "insufficient evidence",
)
_CREATE_CLAIM_PATTERNS = (
    re.compile(r"(?:已|已经).{0,6}(?:创建|提交).{0,8}(?:工单|报修)"),
    re.compile(r"(?:工单|报修).{0,6}(?:已|已经).{0,6}(?:创建|提交)"),
    re.compile(r"\b(?:created|submitted)\b.{0,24}\b(?:ticket|request)\b", re.I),
)
_STATUS_MARKERS = {
    "OPEN": (
        "待处理", "未处理", "尚未处理", "open", "pending",
    ),
    "PROCESSING": (
        "处理中", "正在处理", "processing", "in progress",
    ),
    "RESOLVED": (
        "已解决", "已处理完成", "处理完成", "已完成", "resolved", "closed",
    ),
    "OPERATIONAL": (
        "运行正常", "网络正常", "服务正常", "operational", "healthy",
    ),
    "DEGRADED": (
        "服务降级", "响应较慢", "degraded",
    ),
    "OUTAGE": (
        "服务中断", "网络故障", "outage", "down",
    ),
}
_NUMERIC_CLAIM = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>个?工作日|天|小时|分钟|元|块|%|GB|MB)",
    re.IGNORECASE,
)
_UNIT_ALIASES = {
    "个工作日": "workday",
    "工作日": "workday",
    "天": "day",
    "小时": "hour",
    "分钟": "minute",
    "元": "yuan",
    "块": "yuan",
    "%": "percent",
    "gb": "gb",
    "mb": "mb",
}


@dataclass(frozen=True)
class EvidenceIssue:
    code: str
    detail: str


@dataclass(frozen=True)
class VerificationReport:
    passed: bool
    issues: tuple[EvidenceIssue, ...] = ()
    checked_claims: int = 0
    abstained: bool = False


class EvidenceVerifier:
    """Check bounded, high-risk claims against authoritative evidence."""

    def verify(
        self,
        *,
        question: str,
        response: str,
        citations: Sequence[Mapping[str, Any]] = (),
        tool_evidence: Sequence[Mapping[str, Any]] = (),
    ) -> VerificationReport:
        question_text = question if isinstance(question, str) else ""
        response_text = response if isinstance(response, str) else ""
        abstained = self._is_abstention(response_text)
        issues: list[EvidenceIssue] = []
        checked = 0

        if not response_text.strip():
            issues.append(EvidenceIssue(
                "empty_response",
                "An empty answer cannot satisfy evidence verification.",
            ))

        citation_ids = {
            value
            for item in citations
            if isinstance(item, Mapping)
            for value in [item.get("id")]
            if isinstance(value, str) and value.strip()
        }
        if citation_ids:
            checked += 1
            explicit = {
                match.strip()
                for match in _EXPLICIT_CITATION.findall(response_text)
                if match.strip()
            }
            unknown = explicit - citation_ids
            if unknown:
                issues.append(EvidenceIssue(
                    "unknown_citation",
                    "The answer cites evidence identifiers that were not retrieved.",
                ))
            if not abstained and not any(
                citation_id in response_text for citation_id in citation_ids
            ):
                issues.append(EvidenceIssue(
                    "missing_citation",
                    "A knowledge-backed answer must cite at least one retrieved identifier.",
                ))

        successful_tools = [
            item
            for item in tool_evidence
            if isinstance(item, Mapping) and item.get("success") is True
        ]
        supported_ticket_ids = set(self._ticket_ids(successful_tools))
        claimed_ticket_ids = {
            value.lower() for value in _TICKET_ID.findall(response_text)
        }
        if claimed_ticket_ids:
            checked += len(claimed_ticket_ids)
            unsupported = claimed_ticket_ids - {
                value.lower() for value in supported_ticket_ids
            }
            if unsupported:
                issues.append(EvidenceIssue(
                    "unsupported_ticket_id",
                    "The answer contains a ticket identifier absent from tool results.",
                ))

        if self._claims_ticket_creation(response_text):
            checked += 1
            if not any(
                item.get("name") in {"create_ticket", "create_repair_ticket"}
                for item in successful_tools
            ):
                issues.append(EvidenceIssue(
                    "unsupported_action",
                    "The answer claims ticket creation without a successful creation tool.",
                ))

        claimed_statuses = self._statuses(response_text)
        evidence_statuses = self._evidence_statuses(successful_tools)
        if claimed_statuses:
            checked += len(claimed_statuses)
            if evidence_statuses and claimed_statuses.isdisjoint(evidence_statuses):
                issues.append(EvidenceIssue(
                    "conflicting_status",
                    "The stated status conflicts with the authoritative tool result.",
                ))

        numeric_claims = self._numeric_claims(response_text)
        if numeric_claims and not abstained and (citations or successful_tools):
            checked += len(numeric_claims)
            supported_numbers = self._supported_numbers(
                citations,
                successful_tools,
            )
            unsupported_numbers = numeric_claims - supported_numbers
            if unsupported_numbers:
                issues.append(EvidenceIssue(
                    "conflicting_numeric_claim",
                    "A numeric claim is absent from or conflicts with authoritative evidence.",
                ))

        # Repeating a ticket id supplied by the user is not independently
        # grounded.  It is tolerated only when no business result is claimed.
        if (
            claimed_ticket_ids
            and not supported_ticket_ids
            and all(value.lower() in question_text.lower() for value in claimed_ticket_ids)
            and not claimed_statuses
            and not self._claims_ticket_creation(response_text)
        ):
            issues = [
                issue for issue in issues if issue.code != "unsupported_ticket_id"
            ]

        return VerificationReport(
            passed=not issues,
            issues=tuple(issues),
            checked_claims=checked,
            abstained=abstained,
        )

    @staticmethod
    def _is_abstention(text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in _ABSTENTION_MARKERS)

    @staticmethod
    def _claims_ticket_creation(text: str) -> bool:
        return any(pattern.search(text) for pattern in _CREATE_CLAIM_PATTERNS)

    @staticmethod
    def _walk(value: Any) -> Iterable[tuple[str | None, Any]]:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                normalized_key = str(key).strip().lower() if isinstance(key, str) else None
                yield normalized_key, nested
                yield from EvidenceVerifier._walk(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                yield None, nested
                yield from EvidenceVerifier._walk(nested)

    @classmethod
    def _ticket_ids(cls, evidence: Sequence[Mapping[str, Any]]) -> Iterable[str]:
        for item in evidence:
            for key, value in cls._walk(item.get("data")):
                if key in {"id", "ticket_id"} and isinstance(value, str):
                    if _TICKET_ID.fullmatch(value):
                        yield value

    @staticmethod
    def _normalize_status(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "PENDING": "OPEN",
            "IN_PROGRESS": "PROCESSING",
            "CLOSED": "RESOLVED",
            "HEALTHY": "OPERATIONAL",
            "NORMAL": "OPERATIONAL",
            "DOWN": "OUTAGE",
        }
        return aliases.get(normalized, normalized if normalized in _STATUS_MARKERS else None)

    @classmethod
    def _evidence_statuses(cls, evidence: Sequence[Mapping[str, Any]]) -> set[str]:
        statuses = set()
        for item in evidence:
            for key, value in cls._walk(item.get("data")):
                if key == "status":
                    normalized = cls._normalize_status(value)
                    if normalized:
                        statuses.add(normalized)
        return statuses

    @staticmethod
    def _statuses(text: str) -> set[str]:
        lowered = text.lower()
        return {
            status
            for status, markers in _STATUS_MARKERS.items()
            if any(marker.lower() in lowered for marker in markers)
        }

    @staticmethod
    def _numeric_claims(text: str) -> set[tuple[Decimal, str]]:
        claims: set[tuple[Decimal, str]] = set()
        for match in _NUMERIC_CLAIM.finditer(text):
            try:
                value = Decimal(match.group("value")).normalize()
            except InvalidOperation:
                continue
            unit = _UNIT_ALIASES.get(match.group("unit").lower())
            if unit:
                claims.add((value, unit))
        return claims

    @classmethod
    def _supported_numbers(
        cls,
        citations: Sequence[Mapping[str, Any]],
        evidence: Sequence[Mapping[str, Any]],
    ) -> set[tuple[Decimal, str]]:
        supported: set[tuple[Decimal, str]] = set()
        for citation in citations:
            if not isinstance(citation, Mapping):
                continue
            for field in ("title", "content"):
                value = citation.get(field)
                if isinstance(value, str):
                    supported.update(cls._numeric_claims(value))
        for item in evidence:
            data = item.get("data")
            for key, value in cls._walk(data):
                if isinstance(value, str):
                    supported.update(cls._numeric_claims(value))
                if (
                    key == "amount_cents"
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    supported.add((
                        (Decimal(str(value)) / Decimal("100")).normalize(),
                        "yuan",
                    ))
                elif (
                    key in {"amount_yuan", "balance_yuan", "price_yuan"}
                    and isinstance(value, (int, float, str))
                    and not isinstance(value, bool)
                ):
                    try:
                        supported.add((Decimal(str(value)).normalize(), "yuan"))
                    except InvalidOperation:
                        pass
        return supported
