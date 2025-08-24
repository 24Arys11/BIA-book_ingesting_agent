"""Diagram generator agent: produces PlantUML component diagram from graph."""
from __future__ import annotations

from typing import TypedDict, Any
import logging

from agents.common import read_prompt
from config import settings
from tools.file_io import write_text
import subprocess


logger = logging.getLogger(__name__)


class DiagramState(TypedDict, total=False):
    graph: dict[str, Any]
    diagram_puml: str


def _to_puml(graph: dict[str, Any]) -> str:
    lines: list[str] = ["@startuml"]
    for node in graph.get("nodes", []):
        label = node.get("name", node.get("id", "component"))
        ro = node.get("romanian", "")
        if ro and ro != label:
            lines.append(f"[" + label + "] as \"" + label + " (" + ro.replace('"','\"') + ")\"")
        else:
            lines.append(f"[" + label + "]")
    for edge in graph.get("edges", []):
        frm = edge.get("from")
        to = edge.get("to")
        lbl = edge.get("label", "")
        if lbl:
            lines.append(f"[" + str(frm) + "] -->> [" + str(to) + "] : " + lbl)
        else:
            lines.append(f"[" + str(frm) + "] -->> [" + str(to) + "]")
    lines.append("@enduml")
    return "\n".join(lines)


def _to_dot(graph: dict[str, Any]) -> str:
    lines: list[str] = ["digraph G {"]
    lines.append("  rankdir=LR;")
    lines.append("  node [shape=box];")
    # Nodes
    for node in graph.get("nodes", []):
        node_id = str(node.get("id", "component"))
        name = node.get("name", node_id)
        ro = node.get("romanian", "")
        label = name
        if ro and ro != name:
            label = f"{name} ({ro})"
        # escape quotes
        label = label.replace('"', '\\"')
        lines.append(f"  \"{node_id}\" [label=\"{label}\"];")
    # Edges
    for edge in graph.get("edges", []):
        frm = str(edge.get("from"))
        to = str(edge.get("to"))
        lbl = edge.get("label", "")
        if lbl:
            lbl = lbl.replace('"', '\\"')
            lines.append(f"  \"{frm}\" -> \"{to}\" [label=\"{lbl}\"];")
        else:
            lines.append(f"  \"{frm}\" -> \"{to}\";")
    lines.append("}")
    return "\n".join(lines)


def diagram_generator(state: DiagramState) -> DiagramState:
    _ = read_prompt("diagram_generator.txt")
    graph = state.get("graph", {"nodes": [], "edges": []})
    fmt = (settings.diagram_format or "plantuml").lower()
    if fmt == "dot":
        dot = _to_dot(graph)
        dot_path = settings.output_dir / "architecture.dot"
        write_text(dot_path, dot)
        logger.info(f"Wrote DOT diagram to {dot_path}")
        if settings.graphviz_render:
            try:
                engine = settings.graphviz_engine or "dot"
                png_path = settings.output_dir / "architecture.png"
                subprocess.run([engine, "-Tpng", str(dot_path), "-o", str(png_path)], check=True)
                logger.info(f"Rendered DOT to image via Graphviz -> {png_path}")
            except Exception as e:
                logger.warning(f"Graphviz rendering failed: {e}")
        # Maintain compatibility: still return diagram text under diagram_puml key for downstream
        return {**state, "diagram_puml": dot}
    else:
        puml = _to_puml(graph)
        # Save side-effect
        out_path = settings.output_dir / "architecture.puml"
        write_text(out_path, puml)
        logger.info(f"Wrote PlantUML diagram to {out_path}")
        # Optional: Render PNG if PlantUML CLI configured
        if settings.plantuml_cli:
            try:
                subprocess.run([settings.plantuml_cli, str(out_path)], check=True)
                logger.info("Rendered PlantUML to image via CLI.")
            except Exception as e:
                logger.warning(f"PlantUML rendering failed: {e}")
        return {**state, "diagram_puml": puml}
