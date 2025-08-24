"""Utilities for robust JSON parsing with optional LLM repair."""
from __future__ import annotations

from typing import Any, Optional

import json


def try_parse_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_json_block(text: str) -> Optional[str]:
    # naive fence extraction for common cases
    start = text.find("{")
    if start == -1:
        return None
    # try to find matching closing brace by simple balance
    bal = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == '{':
            bal += 1
        elif ch == '}':
            bal -= 1
            if bal == 0:
                return text[start:i+1]
    return None


def robust_parse_json(text: str) -> Optional[Any]:
    data = try_parse_json(text)
    if data is not None:
        return data
    chunk = extract_json_block(text)
    if chunk:
        return try_parse_json(chunk)
    return None
