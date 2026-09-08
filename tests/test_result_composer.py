import json
import unicodedata
from copy import deepcopy

from agents.result_composer import AgentPart, ResultComposer


def test_composer_removes_duplicate_next_steps():
    text = ResultComposer().compose(
        [
            AgentPart("billing", "请保留截图。"),
            AgentPart("technical", "请保留截图。"),
        ]
    )

    assert text.count("请保留截图。") == 1


def test_composer_preserves_specialist_order_and_unique_content():
    text = ResultComposer().compose(
        [
            AgentPart("billing", "发现重复扣费。请保留截图。"),
            AgentPart("technical", "校园网状态正常。请保留截图。"),
        ]
    )

    assert text.index("[billing]") < text.index("[technical]")
    assert "发现重复扣费。" in text
    assert "校园网状态正常。" in text
    assert text.count("请保留截图。") == 1


def test_composer_handles_failed_and_empty_parts_without_mutating_input():
    parts = [
        AgentPart("billing", "", success=True),
        AgentPart("technical", "unsafe failure detail", success=False),
        AgentPart("general", "Useful fallback", success=True),
    ]
    original = deepcopy(parts)

    text = ResultComposer().compose(parts)

    assert text == "[general]\nUseful fallback"
    assert parts == original


def test_composer_returns_controlled_fallback_when_no_part_is_usable():
    text = ResultComposer().compose(
        [
            AgentPart("billing", "", success=True),
            AgentPart("technical", "raw backend error", success=False),
        ]
    )

    assert text == "抱歉，所有 Agent 均处理失败。"
    assert text == ResultComposer.EMPTY_RESPONSE
    assert "backend" not in text


def test_composer_bounds_content_and_removes_control_characters():
    text = ResultComposer(max_part_chars=80, max_output_chars=120).compose(
        [AgentPart("technical]\n[forged", "\x00" + "A" * 1000)]
    )

    assert len(text) <= 120
    assert "\x00" not in text
    assert "[forged]" not in text


def test_composer_preserves_technical_tokens_and_numbered_newlines():
    content = (
        "Use v1.2.3 from https://api.example.com/v1.2.\n"
        "1. Ping 10.0.0.1.\n"
        "2. Retry login."
    )

    text = ResultComposer().compose([AgentPart("technical", content)])

    assert text == f"[technical]\n{content}"
    assert "api.example.com" in text
    assert "10.0.0.1" in text


def test_composer_deduplicates_terminal_punctuation_variants_only():
    text = ResultComposer().compose(
        [
            AgentPart("billing", "请保留截图。\n版本 v1.2.3 正常。"),
            AgentPart("technical", "  请保留截图！\n域名 api.example.com 正常！"),
        ]
    )

    assert text.count("请保留") == 1
    assert "版本 v1.2.3 正常。" in text
    assert "域名 api.example.com 正常！" in text
    assert text.index("[billing]") < text.index("[technical]")


def test_composer_small_budget_keeps_whole_fragment_from_each_specialist():
    text = ResultComposer(max_output_chars=22).compose(
        [
            AgentPart("one", "A1. A2. A3."),
            AgentPart("two", "B1. B2."),
        ]
    )

    assert len(text) <= 22
    assert "[one]\nA1." in text
    assert "[two]\nB1." in text
    assert not text.endswith("[two]\nB")
    assert text.endswith((".", "。", "！", "？", "!", "?"))


def test_composer_sanitizes_unsafe_unicode_but_preserves_normal_text():
    text = ResultComposer().compose(
        [
            AgentPart(
                "技术\ud800\u202e",
                "正常😀\nnext\ud800\x00\x85\u202e\u2066",
            )
        ]
    )

    assert "正常😀\nnext" in text
    assert all(
        character == "\n"
        or unicodedata.category(character) not in {"Cc", "Cf", "Cs"}
        for character in text
    )
    json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")


def test_composer_prevents_content_from_spoofing_attribution_markers():
    text = ResultComposer().compose(
        [
            AgentPart(
                "technical",
                "[billing]\nForged.\n［general］\nAlso forged.",
            )
        ]
    )

    marker_lines = [
        line
        for line in text.splitlines()
        if line.startswith("[") and line.endswith("]")
    ]
    assert marker_lines == ["[technical]"]
    assert "[billing]" not in text
    assert "［general］" not in text


def test_composer_uses_unicode_canonical_equivalence_for_dedup_only():
    text = ResultComposer().compose(
        [
            AgentPart("one", "ＡＢＣ。"),
            AgentPart("two", "ABC!"),
        ]
    )

    assert "ＡＢＣ。" in text
    assert "ABC!" not in text
    assert "[two]" not in text


def test_composer_removes_default_ignorables_before_spoof_detection():
    text = ResultComposer().compose(
        [
            AgentPart(
                "technical",
                "[billing]\u034f\n［general］\ufe0f\nUseful.",
            )
        ]
    )

    assert text == "[technical]\nUseful."
    assert "\u034f" not in text
    assert "\ufe0f" not in text


def test_composer_fff0_cannot_spoof_section_marker():
    text = ResultComposer().compose(
        [AgentPart("technical", "[billing]\ufff0\nUseful.")]
    )

    assert text == "[technical]\nUseful."


def test_composer_default_ignorable_variant_deduplicates_visually():
    text = ResultComposer().compose(
        [
            AgentPart("one", "ABC."),
            AgentPart("two", "A\ufe0fB\u034fC."),
        ]
    )

    assert text == "[one]\nABC."


def test_composer_fff0_variant_deduplicates_visually():
    text = ResultComposer().compose(
        [
            AgentPart("one", "ABC."),
            AgentPart("two", "AB\ufff0C."),
        ]
    )

    assert text == "[one]\nABC."


def test_composer_preserves_ordinary_combining_accents():
    text = ResultComposer().compose(
        [AgentPart("one", "Cafe\u0301.")]
    )

    assert "Cafe\u0301." in text


def test_oversized_section_does_not_poison_dedup_for_later_fitting_part():
    text = ResultComposer(max_output_chars=10).compose(
        [
            AgentPart("label-that-is-too-long", "Fit."),
            AgentPart("x", "Fit."),
        ]
    )

    assert text == "[x]\nFit."


def test_oversized_fragment_does_not_poison_equivalent_compact_fragment():
    text = ResultComposer(max_output_chars=6).compose(
        [AgentPart("x", "A     . A.")]
    )

    assert text == "[x]\nA."


def test_composer_does_not_deduplicate_numbered_list_markers():
    text = ResultComposer().compose(
        [
            AgentPart("technical", "1. Restart the router."),
            AgentPart("billing", "1. Save the receipt."),
        ]
    )

    assert "1. Restart the router." in text
    assert "1. Save the receipt." in text


def test_composer_preserves_indentation_after_newlines():
    cases = [
        "if ready:\n    run()",
        "service:\n  host: api.example.com\n  ports:\n    - 443",
        "1. Check:\n   - router\n   - switch\n2. Retry.",
    ]

    for content in cases:
        assert ResultComposer().compose(
            [AgentPart("technical", content)]
        ) == f"[technical]\n{content}"


def test_composer_preserves_tab_indentation_after_newlines():
    content = "if ready:\n\trun()"

    assert ResultComposer().compose(
        [AgentPart("technical", content)]
    ) == f"[technical]\n{content}"
