"""Validator agent: checks consistency of outputs and emits a report."""
from __future__ import annotations

from typing import TypedDict, Any, Set
import logging

from agents.common import read_prompt
from config import settings
from tools.file_io import write_text


logger = logging.getLogger(__name__)


class ValidatorState(TypedDict, total=False):
    graph: dict[str, Any]
    diagram_puml: str
    markdown_md: str
    validation_report: str


def _check_graph_consistency(graph: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    ids: Set[str] = {n.get("id") for n in graph.get("nodes", []) if n.get("id")}
    for edge in graph.get("edges", []):
        if edge.get("from") not in ids:
            errors.append(f"Edge from '{edge.get('from')}' not in nodes")
        if edge.get("to") not in ids:
            errors.append(f"Edge to '{edge.get('to')}' not in nodes")
    # Optional: no circular dependency check placeholder
    return errors


def validator(state: ValidatorState) -> ValidatorState:
    _ = read_prompt("validator.txt")
    graph = state.get("graph", {"nodes": [], "edges": []})
    errors = _check_graph_consistency(graph)
    report_lines = ["Validation Report", "==================", ""]
    if errors:
        report_lines.append("Errors:")
        report_lines.extend(f"- {e}" for e in errors)
    else:
        report_lines.append("No issues detected.")
    report = "\n".join(report_lines) + "\n"
    out_path = settings.output_dir / "validation_report.txt"
    write_text(out_path, report)
    logger.info(f"Wrote validation report to {out_path}")
    return {**state, "validation_report": report}
