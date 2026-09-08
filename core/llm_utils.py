"""LLM response helpers shared by Anthropic-compatible providers."""
from copy import deepcopy
from typing import Any, Dict, Iterable, List


def content_blocks(content: Any) -> List[Dict[str, Any]]:
    """Normalize Anthropic SDK content blocks and dictionaries."""
    blocks: List[Dict[str, Any]] = []
    for block in content or []:
        normalized: Any = None
        if hasattr(block, "model_dump"):
            normalized = block.model_dump()
        elif isinstance(block, dict):
            normalized = dict(block)
        elif hasattr(block, "dict"):
            normalized = block.dict()
        else:
            block_type = getattr(block, "type", None)
            if block_type is not None:
                normalized = {
                    key: value
                    for key in ("type", "text", "id", "name", "input")
                    if (value := getattr(block, key, None)) is not None
                }

        if isinstance(normalized, dict):
            blocks.append(deepcopy(normalized))
    return blocks


def extract_text_content(content: Iterable[Any]) -> str:
    """Return text blocks from Anthropic-style response content."""
    texts: List[str] = []
    for block in content or []:
        if isinstance(block, str):
            texts.append(block)
            continue

        block_type = getattr(block, "type", None)
        text = getattr(block, "text", None)
        normalized = content_blocks([block])
        if normalized:
            block_type = normalized[0].get("type", block_type)
            text = normalized[0].get("text", text)

        if isinstance(text, str) and (block_type in (None, "text")):
            texts.append(text)

    return "\n".join(t for t in texts if t)
