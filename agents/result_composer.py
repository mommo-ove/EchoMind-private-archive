"""Deterministic composition for bounded multi-Agent responses."""

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from agents.agent_runtime import sanitize_text_content


_SAFE_LABEL = re.compile(r"[^A-Za-z0-9_-]+")
_ATTRIBUTION_MARKER = re.compile(r"^\[[^\]\n]{1,64}\]$")
_CJK_TERMINATORS = frozenset("。！？")
_ASCII_TERMINATORS = frozenset(".!?")
_CANONICAL_TERMINATORS = "。！？.!?"


@dataclass(frozen=True)
class AgentPart:
    """One specialist response considered for deterministic composition."""

    agent: str
    content: str
    success: bool = True


@dataclass(frozen=True)
class _Fragment:
    text: str
    separator: str = ""


@dataclass
class _Section:
    label: str
    fragments: list[_Fragment]
    included: int = 0

    def render(self) -> str:
        body = ""
        for index, fragment in enumerate(self.fragments[: self.included]):
            if index:
                body += self.fragments[index - 1].separator
            body += fragment.text
        return f"[{self.label}]\n{body}"


class ResultComposer:
    """Merge useful Agent parts without another model call."""

    EMPTY_RESPONSE = "抱歉，所有 Agent 均处理失败。"
    _MAX_PARTS = 8

    def __init__(
        self,
        *,
        max_part_chars: int = 8192,
        max_output_chars: int = 16384,
    ):
        if max_part_chars < 1 or max_output_chars < 1:
            raise ValueError("composition limits must be positive")
        self._max_part_chars = min(max_part_chars, 16384)
        self._max_output_chars = min(max_output_chars, 32768)

    def compose(self, parts: Iterable[AgentPart]) -> str:
        """Return an attributed, stable de-duplication of successful parts."""
        candidates: list[tuple[str, list[_Fragment]]] = []

        for index, part in enumerate(parts):
            if index >= self._MAX_PARTS:
                break
            if not isinstance(part, AgentPart) or not part.success:
                continue

            fragments = self._fragments(part.content)
            if fragments:
                candidates.append(
                    (self._safe_label(part.agent), fragments)
                )

        if not candidates:
            return (
                self.EMPTY_RESPONSE
                if len(self.EMPTY_RESPONSE) <= self._max_output_chars
                else ""
            )

        seen = set()
        included_sections: list[_Section] = []
        pending: list[tuple[_Section, list[_Fragment]]] = []
        for label, fragments in candidates:
            for fragment_index, fragment in enumerate(fragments):
                key = self._canonical(fragment.text)
                if not key or key in seen:
                    continue
                section = _Section(label, [fragment], included=1)
                candidate_output = [*included_sections, section]
                if (
                    self._render_length(candidate_output)
                    > self._max_output_chars
                ):
                    continue
                included_sections.append(section)
                pending.append((section, fragments[fragment_index + 1:]))
                seen.add(key)
                break

        if not included_sections:
            return ""

        made_progress = True
        while made_progress:
            made_progress = False
            for section, fragments in pending:
                while fragments:
                    fragment = fragments.pop(0)
                    key = self._canonical(fragment.text)
                    if not key or key in seen:
                        continue
                    section.fragments.append(fragment)
                    section.included += 1
                    if (
                        self._render_length(included_sections)
                        <= self._max_output_chars
                    ):
                        seen.add(key)
                        made_progress = True
                        break
                    section.fragments.pop()
                    section.included -= 1

        return "\n\n".join(
            section.render() for section in included_sections
        )

    def _fragments(self, content: object) -> list[_Fragment]:
        if not isinstance(content, str):
            return []
        sanitized = self._remove_attribution_markers(
            sanitize_text_content(content, self._max_part_chars + 1)
        )
        truncated = len(sanitized) > self._max_part_chars
        bounded = sanitized[: self._max_part_chars].rstrip()
        if not bounded.strip():
            return []

        fragments: list[_Fragment] = []
        start = 0
        index = 0
        while index < len(bounded):
            character = bounded[index]
            boundary = character in _CJK_TERMINATORS
            if character in _ASCII_TERMINATORS:
                boundary = (
                    index + 1 == len(bounded)
                    or bounded[index + 1].isspace()
                )
                if (
                    character == "."
                    and bounded[start:index].strip().isdigit()
                ):
                    boundary = False
            if character == "\n":
                self._append_fragment(
                    fragments,
                    bounded[start:index],
                    "\n",
                )
                start = index + 1
                index += 1
                continue
            if not boundary:
                index += 1
                continue

            end = index + 1
            separator_end = end
            while (
                separator_end < len(bounded)
                and bounded[separator_end].isspace()
            ):
                separator_end += 1
            self._append_fragment(
                fragments,
                bounded[start:end],
                bounded[end:separator_end],
            )
            start = separator_end
            index = separator_end

        remainder = bounded[start:]
        if remainder and not truncated:
            self._append_fragment(fragments, remainder, "")
        return fragments

    @staticmethod
    def _append_fragment(
        fragments: list[_Fragment],
        text: str,
        separator: str,
    ) -> None:
        text = text.rstrip()
        if text.strip():
            fragments.append(_Fragment(text, separator))
        elif separator and fragments:
            previous = fragments[-1]
            fragments[-1] = _Fragment(
                previous.text,
                previous.separator + separator,
            )

    @staticmethod
    def _canonical(text: str) -> str:
        normalized = " ".join(
            unicodedata.normalize("NFKC", text).split()
        ).casefold()
        return normalized.rstrip(_CANONICAL_TERMINATORS).rstrip()

    @staticmethod
    def _safe_label(label: object) -> str:
        if not isinstance(label, str):
            return "agent"
        safe = _SAFE_LABEL.sub(
            "",
            sanitize_text_content(label, 64),
        )[:32]
        return safe or "agent"

    @staticmethod
    def _remove_attribution_markers(content: str) -> str:
        safe_lines = []
        for line in content.splitlines(keepends=True):
            marker = unicodedata.normalize(
                "NFKC",
                line.rstrip("\n").strip(),
            )
            if _ATTRIBUTION_MARKER.fullmatch(marker):
                continue
            safe_lines.append(line)
        return "".join(safe_lines)

    @staticmethod
    def _render_length(sections: list[_Section]) -> int:
        if not sections:
            return 0
        return sum(len(section.render()) for section in sections) + (
            2 * (len(sections) - 1)
        )
