"""Tool result truncation for the archipelago agent.

Applies head-tail truncation to oversized text blocks in tool result messages,
preserving images and other non-text content unchanged.

Truncation is purely char-based:
  - If len(text) <= head_chars + tail_chars: skip immediately (O(1)).
    Covers ~95% of tool calls with zero overhead.
  - If len(text) > head_chars + tail_chars: truncate. len(text) // 4 is
    used as a fast token estimate for the log/marker message only.
    No tiktoken call — it runs at ~5-10 MB/s and would take minutes on
    a 42M char string.

Default threshold: 48K chars (32K head + 16K tail), derived from the
per-call fair share of the 220K token context window across a mean of
16.3 tool calls per trial (~43K chars/call, rounded up to 48K).

Modeled after archipelago's react_toolbelt_agent tool-result handling.
"""

from __future__ import annotations

import logging
from typing import Any


def _truncate_text(
    text: str,
    head_chars: int,
    tail_chars: int,
    logger: logging.Logger,
) -> str | None:
    """Truncate text if its char length exceeds head_chars + tail_chars.

    Returns the truncated string, or None if no truncation was needed.
    """
    if len(text) <= head_chars + tail_chars:
        return None

    estimated_tokens = len(text) // 4
    head = text[:head_chars]
    tail = text[-tail_chars:] if tail_chars else ""
    logger.warning(
        f"Tool result truncated: {len(text):,} chars (~{estimated_tokens:,} tokens). "
        f"Keeping first {head_chars:,} and last {tail_chars:,} chars."
    )
    return (
        f"{head}\n\n"
        f"[TRUNCATED: tool returned {len(text):,} chars (~{estimated_tokens:,} tokens), "
        f"showing first {head_chars / 1000}K and last {tail_chars / 1000}K chars. "
        "Please use more specific queries to access full data. ...]\n\n"
        f"{tail}"
    )


def _truncate_content_list(
    content: list[Any],
    head_chars: int,
    tail_chars: int,
    logger: logging.Logger,
) -> None:
    """Truncate text blocks within a content list. Mutates in place. Skips non-text blocks."""
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        text = item.get("text", "")
        if not isinstance(text, str):
            continue
        truncated = _truncate_text(text, head_chars, tail_chars, logger)
        if truncated is not None:
            item["text"] = truncated


def truncate_tool_result_messages(
    messages: list[dict[str, Any]],
    logger: logging.Logger,
    head_chars: int,
    tail_chars: int,
) -> None:
    """Truncate oversized text blocks in tool result messages. Mutates in place.

    Handles both list-content and string-content message formats.
    Images and other non-text blocks are preserved unchanged.

    Args:
        messages: Tool result messages to truncate.
        logger: Logger for truncation warnings.
        head_chars: Characters to keep from the start of each result.
        tail_chars: Characters to keep from the end of each result.
    """
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            _truncate_content_list(content, head_chars, tail_chars, logger)
        elif isinstance(content, str):
            truncated = _truncate_text(content, head_chars, tail_chars, logger)
            if truncated is not None:
                msg["content"] = truncated
