"""Logical Distiller agent: produce atomic relations from normalized entities.

Uses instructions to focus only on high-value flows; outputs 'relations' for the graph builder.
"""
from __future__ import annotations

from typing import TypedDict, Any, List
import logging

from agents.common import read_prompt, get_llm_client
from config import settings
from tools.json_utils import robust_parse_json


logger = logging.getLogger(__name__)


class LogicalDistillState(TypedDict, total=False):
    instructions: str
    normalized_entities: List[dict[str, Any]]
    relations: List[dict[str, Any]]


def _io_similarity(a: str, b: str) -> float:
    import re
    ta = {t for t in re.split(r"[^a-z0-9]+", str(a).lower()) if t}
    tb = {t for t in re.split(r"[^a-z0-9]+", str(b).lower()) if t}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _deterministic_relations(ents: List[dict[str, Any]], thr: float = 0.62, cap: int = 200) -> List[dict[str, Any]]:
    import heapq
    cand: list[tuple[float, tuple[str, str, str]]] = []
    for src in ents:
        for outv in (src.get("outputs") or []):
            for dst in ents:
                if dst is src:
                    continue
                for inv in (dst.get("inputs") or []):
                    s = _io_similarity(outv, inv)
                    if s >= thr:
                        heapq.heappush(cand, (-s, (str(src.get("id")), str(dst.get("id")), str(outv))))
    rels: List[dict[str, Any]] = []
    for _ in range(min(cap, len(cand))):
        _, (f, t, lbl) = cand.pop(0) if False else heapq.heappop(cand)
        rels.append({"from": f, "to": t, "label": lbl})
    return rels


def logical_distiller(state: LogicalDistillState) -> LogicalDistillState:
    ents = state.get("normalized_entities", []) or []
    if not ents:
        return state
    system_prompt = read_prompt("logical_distiller.txt")
    client = get_llm_client()
    instr = state.get("instructions") or ""

    # Deterministic proposal first
    thr = float(getattr(settings, "io_link_similarity_threshold", 0.62))
    cap = int(getattr(settings, "max_candidate_edges", 200))
    rels = _deterministic_relations(ents, thr=thr, cap=cap)

    # LLM refinement (optional self-consistency)
    try:
        import json
        user_prompt = (
            "Instructions (goal):\n" + instr + "\n\n" +
            "Entities (normalized):\n" + json.dumps({"entities": ents}, ensure_ascii=False) + "\n\n" +
            "Candidate relations (from deterministic IO matching):\n" + json.dumps({"relations": rels}, ensure_ascii=False) + "\n\n" +
            "Refine: keep only relations that are clearly supported by entity IO and are relevant to the instructions. "
            "Return ONLY JSON {\"relations\": [{\"from\": id, \"to\": id, \"label\": string}]}"
        )
        n = max(1, int(getattr(settings, "self_consistency_samples", 3)))
        temps = float(getattr(settings, "llm_temperature", 0.2))
        answers = client.query_n(user_prompt=user_prompt, system_prompt=system_prompt, temperature=temps, n=n)
        from collections import Counter
        counter: Counter[tuple] = Counter()
        for a in answers:
            data = robust_parse_json(a) or {}
            if isinstance(data, dict):
                for e in (data.get("relations") or []):
                    k = (e.get("from"), e.get("to"), e.get("label", ""))
                    counter[k] += 1
        if counter:
            thresh = (n // 2) + 1
            voted = [{"from": k[0], "to": k[1], "label": k[2]} for k, c in counter.items() if c >= thresh]
            if voted:
                rels = voted
    except Exception as e:
        logger.warning(f"Logical distiller LLM refinement failed; using deterministic relations. Error: {e}")

    logger.info(f"LogicalDistiller produced {len(rels)} relations.")
    return {**state, "relations": rels}
