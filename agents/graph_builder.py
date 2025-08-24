"""Graph builder agent: constructs knowledge graph JSON from entities."""
from __future__ import annotations

from typing import TypedDict, Any
import logging

from agents.common import read_prompt, get_llm_client
from config import settings
from tools.json_utils import robust_parse_json


logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    normalized_entities: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    graph: dict[str, Any]


def graph_builder(state: GraphState) -> GraphState:
    system_prompt = read_prompt("graph_builder.txt")
    ents = state.get("normalized_entities", [])
    nodes = []
    edges = []
    for e in ents:
        nodes.append({
            "id": e.get("id"),
            "name": e.get("id"),
            "romanian": e.get("romanian", e.get("name")),
            "function": e.get("description", ""),
            "inputs": e.get("inputs", []),
            "outputs": e.get("outputs", []),
            "type": e.get("type", "component"),
        })
    # If relations already distilled, use them directly
    distilled = state.get("relations") or []
    if distilled:
        edges = [{"from": r.get("from"), "to": r.get("to"), "label": r.get("label", "")} for r in distilled]
        graph = {"nodes": nodes, "edges": edges}
        logger.info(f"Graph (distilled) contains {len(nodes)} nodes and {len(edges)} edges.")
        return {**state, "graph": graph}

    # Deterministic IO-based linking first
    def _tok(s: str) -> set[str]:
        import re as _re
        return {t for t in _re.split(r"[^a-z0-9]+", s.lower()) if t}

    def _sim(a: str, b: str) -> float:
        ta, tb = _tok(a), _tok(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    import heapq
    cand: list[tuple[float, tuple[str, str, str]]] = []
    thr = float(getattr(settings, "io_link_similarity_threshold", 0.62))
    max_c = int(getattr(settings, "max_candidate_edges", 200))
    for src in nodes:
        for outv in src.get("outputs", []) or []:
            for dst in nodes:
                if dst is src:
                    continue
                for inv in dst.get("inputs", []) or []:
                    s = _sim(str(outv), str(inv))
                    if s >= thr:
                        heapq.heappush(cand, (-s, (src["id"], dst["id"], str(outv))))
    edges = []
    for _ in range(min(max_c, len(cand))):
        _, (f, t, lbl) = heapq.heappop(cand)
        edges.append({"from": f, "to": t, "label": lbl})

    # Try LLM to refine edges; fallback keeps deterministic edges
    client = get_llm_client()
    try:
        import json
        user_prompt = (
            "Given these nodes (with function/inputs/outputs), infer likely data/control flows as edges. "
            "Return ONLY JSON with 'edges': [{from, to, label}].\n\n" + json.dumps({"nodes": nodes}, ensure_ascii=False)
        )
        n = max(1, int(getattr(settings, "self_consistency_samples", 3)))
        temps = float(getattr(settings, "llm_temperature", 0.2))
        samples: list[list[dict[str, Any]]] = []
        if n > 1:
            answers = client.query_n(user_prompt=user_prompt, system_prompt=system_prompt, temperature=temps, n=n)
            for ans in answers:
                data = robust_parse_json(ans) or {}
                if isinstance(data, dict):
                    samples.append(data.get("edges", []) or [])
        else:
            ans = client.query(user_prompt=user_prompt, system_prompt=system_prompt, temperature=temps)
            data = robust_parse_json(ans) or {}
            if isinstance(data, dict):
                samples.append(data.get("edges", []) or [])
        # Majority vote over edges by (from,to,label) tuples
        from collections import Counter
        counter: Counter[tuple] = Counter()
        for s in samples:
            for e in s:
                key = (e.get("from"), e.get("to"), e.get("label", ""))
                counter[key] += 1
        if counter:
            # keep edges that appear in > n/2 of samples; or top-k fallback
            thresh = (n // 2) + 1
            voted = [
                {"from": k[0], "to": k[1], "label": k[2]} for k, c in counter.items() if c >= thresh
            ]
            if not voted:  # pick top 5
                for k, _ in counter.most_common(5):
                    voted.append({"from": k[0], "to": k[1], "label": k[2]})
            # merge deterministic + voted unique
            seen = {(e["from"], e["to"], e.get("label", "")) for e in edges}
            for e in voted:
                key = (e["from"], e["to"], e.get("label", ""))
                if key not in seen:
                    edges.append(e)
    except Exception as e:
        logger.warning(f"LLM edge inference failed, using sequential fallback. Error: {e}"); edges = []
    if not edges:
        # As last resort only, avoid linear chain: keep none rather than misleading chain
        edges = []
    graph = {"nodes": nodes, "edges": edges}
    logger.info(f"Graph contains {len(nodes)} nodes and {len(edges)} edges.")
    return {**state, "graph": graph}
