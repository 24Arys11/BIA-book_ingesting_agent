"""Minimal evaluation harness for entities and edges.

Place a gold file at input/gold.json with keys:
{
  "entities": [{"id": "..."}],
  "edges": [{"from": "a", "to": "b", "label": "..."}]
}
"""
from __future__ import annotations

from typing import Dict, Any, Tuple, Set, TypeVar
from pathlib import Path
import json

from config import settings


def _load_gold() -> Dict[str, Any] | None:
    p = settings.input_dir / "gold.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


T = TypeVar("T")


def _pr(retrieved: Set[T], gold: Set[T]) -> Tuple[float, float]:
    if not retrieved:
        return 0.0, 0.0
    tp = len(retrieved & gold)
    prec = tp / len(retrieved) if retrieved else 0.0
    rec = tp / len(gold) if gold else 0.0
    return prec, rec


def evaluate(graph: Dict[str, Any], normalized_entities: list[Dict[str, Any]]) -> Dict[str, Any] | None:
    gold = _load_gold()
    if not gold:
        return None
    ent_ids = {str(e.get("id")) for e in normalized_entities if e.get("id")}
    gold_ent_ids = {str(e.get("id")) for e in gold.get("entities", [])}
    p_e, r_e = _pr(ent_ids, gold_ent_ids)

    def _edge_key(e: Dict[str, Any]) -> Tuple[str, str, str]:
        return (str(e.get("from")), str(e.get("to")), str(e.get("label", "")))

    edges = {_edge_key(e) for e in graph.get("edges", [])}
    gold_edges = {_edge_key(e) for e in gold.get("edges", [])}
    p_g, r_g = _pr(edges, gold_edges)

    return {
        "entities": {"precision": p_e, "recall": r_e, "retrieved": len(ent_ids), "gold": len(gold_ent_ids)},
        "edges": {"precision": p_g, "recall": r_g, "retrieved": len(edges), "gold": len(gold_edges)},
    }
