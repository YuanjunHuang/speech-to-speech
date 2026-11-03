"""Deterministic tool routing for small voice models.

With a ``[style:...]`` reply prefix some models (MiniCPM-V) answer "let me check" in words
and never emit the tool call. When the newest turn is a user message that clearly asks for a
tool's data, narrow the request to that tool and force the call (``tool_choice="required"``).
Requests that follow a tool result are untouched, so the model still answers in words after.
"""

from __future__ import annotations

import re
from typing import Any

INTENT_PATTERNS: dict[str, re.Pattern[str]] = {
    "get_weather": re.compile(
        r"天气|气温|温度|下雨|下雪|降雨|降水|冷不冷|热不热|带伞|穿什么|"
        r"\b(?:weather|forecast|temperature)\b",
        re.IGNORECASE,
    ),
}


def _user_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return ""


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        fn = tool.get("function")
        return str((fn or {}).get("name") or tool.get("name") or "")
    return ""


def force_tool_for_intent(api_input: Any, optional_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return ``optional_kwargs`` with a forced tool call when the newest user turn asks for one."""
    tools = optional_kwargs.get("tools")
    if not tools or optional_kwargs.get("tool_choice") not in (None, "auto"):
        return optional_kwargs
    if not isinstance(api_input, list) or not api_input:
        return optional_kwargs
    last = api_input[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return optional_kwargs
    text = _user_text(last)
    matched = [
        t for t in tools if (p := INTENT_PATTERNS.get(_tool_name(t))) is not None and p.search(text)
    ]
    if not matched:
        return optional_kwargs
    return {**optional_kwargs, "tools": matched, "tool_choice": "required"}
