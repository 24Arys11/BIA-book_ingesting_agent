"""Section indexer agent: builds a lightweight section outline from chunks."""
from __future__ import annotations

from typing import TypedDict, Any
import logging
import re

from agents.common import read_prompt


logger = logging.getLogger(__name__)


class SectionIndexState(TypedDict, total=False):
    chunks: list[str]
    sections: list[dict[str, Any]]


_HEAD_RE = re.compile(r"^(chapter|capitol|section|sectiunea|sectiune|cap\.|chap\.|\d+\.|[ivxlcdm]+\.|appendix)\b", re.I)


def section_indexer(state: SectionIndexState) -> SectionIndexState:
    _ = read_prompt("ingestor.txt")  # reuse context prompt if needed
    chunks = state.get("chunks", []) or []
    if not chunks:
        return state
    sections: list[dict[str, Any]] = []
    current = {"title": "Introduction", "chunk_indices": []}
    for idx, ch in enumerate(chunks):
        # split first line as candidate title
        first_line = (ch.strip().splitlines() or [""])[0]
        if _HEAD_RE.search(first_line):
            if current["chunk_indices"]:
                sections.append(current)
            current = {"title": first_line.strip()[:120], "chunk_indices": [idx]}
        else:
            current["chunk_indices"].append(idx)
    if current["chunk_indices"]:
        sections.append(current)
    logger.info(f"Section indexer created {len(sections)} sections.")
    return {**state, "sections": sections}
