"""Markdown generator agent: produces component entries from graph."""
from __future__ import annotations

from typing import TypedDict, Any
import logging

from agents.common import read_prompt
from config import settings
from tools.file_io import write_text


logger = logging.getLogger(__name__)


class MarkdownState(TypedDict, total=False):
    graph: dict[str, Any]
    markdown_md: str


def _to_md(graph: dict[str, Any]) -> str:
    lines: list[str] = []
    for node in graph.get("nodes", []):
        name = node.get("name") or node.get("id")
        ro = node.get("romanian", "")
        # Always title in English (name), put Romanian in parentheses only if different
        title = f"## {name} ({ro})" if ro and ro != name else f"## {name}"
        lines.append(title)
        lines.append(f"**Responsibility**: {node.get('function','')}")
        ins = ", ".join(node.get("inputs", []) or [])
        outs = ", ".join(node.get("outputs", []) or [])
        lines.append(f"**Inputs**: {ins if ins else '—'}")
        lines.append(f"**Outputs**: {outs if outs else '—'}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def markdown_generator(state: MarkdownState) -> MarkdownState:
    _ = read_prompt("markdown_generator.txt")
    graph = state.get("graph", {"nodes": [], "edges": []})
    md = _to_md(graph)
    out_path = settings.output_dir / "cognitive_architecture.md"
    write_text(out_path, md)
    logger.info(f"Wrote Markdown to {out_path}")
    return {**state, "markdown_md": md}
