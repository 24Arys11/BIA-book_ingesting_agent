"""Task filter agent: filters normalized entities to those relevant to the user's instructions.

Keeps the pipeline generic; tailoring happens via instructions only.
"""
from __future__ import annotations

from typing import TypedDict, Any, List
import logging

from agents.common import read_prompt, get_llm_client
from config import settings


logger = logging.getLogger(__name__)


class TaskFilterState(TypedDict, total=False):
    instructions: str
    normalized_entities: List[dict[str, Any]]


def _deterministic_filter(ents: List[dict[str, Any]]) -> List[dict[str, Any]]:
    allowed = {"component", "module", "system", "process", "circuit", "unit", "network", "layer", "function", "agent", "model"}
    bad = ["author", "autori", "bibliography", "preface", "introduc", "chapter", "capitol", "appendix", "isbn", "copyright"]
    out: List[dict[str, Any]] = []
    for e in ents:
        nm = str(e.get("name") or e.get("id") or "").lower()
        desc = str(e.get("description") or "").lower()
        typ = str(e.get("type") or "").lower()
        if any(b in nm or b in desc for b in bad):
            continue
        if typ and typ not in allowed:
            continue
        # Prefer entities with some IO or clear functional description
        ins = e.get("inputs") or []
        outs = e.get("outputs") or []
        if (len(ins) + len(outs)) == 0 and len(desc) < 12:
            continue
        out.append(e)
    return out


def task_filter(state: TaskFilterState) -> TaskFilterState:
    system_prompt = read_prompt("task_filter.txt")
    ents = state.get("normalized_entities", []) or []
    if not ents:
        return state
    # First pass deterministic sieve
    base = _deterministic_filter(ents)
    if not base:
        logger.info("TaskFilter: deterministic sieve removed all; keeping originals to avoid empty output.")
        base = ents

    # Optional LLM refinement for relevance based on instructions
    instr = state.get("instructions") or ""
    client = get_llm_client()
    try:
        import json
        user_prompt = (
            "Instructions (goal):\n" + instr + "\n\n" +
            "From the provided entity list, keep ONLY those directly relevant to accomplishing the instructions. "
            "Focus on cognitive components/models with functional IO. Return ONLY JSON with 'entities' filtered.\n\n" +
            json.dumps({"entities": base}, ensure_ascii=False)
        )
        temps = float(getattr(settings, "llm_temperature", 0.1))
        answer = client.query(user_prompt=user_prompt, system_prompt=system_prompt, temperature=temps)
        data = json.loads(answer)
        filtered = data.get("entities", []) if isinstance(data, dict) else []
        if filtered:
            logger.info(f"TaskFilter: LLM kept {len(filtered)} of {len(base)} entities.")
            return {**state, "normalized_entities": filtered}
    except Exception as e:
        logger.warning(f"TaskFilter LLM refinement failed; using deterministic results. Error: {e}")
    logger.info(f"TaskFilter: deterministic kept {len(base)} of {len(ents)} entities.")
    return {**state, "normalized_entities": base}
