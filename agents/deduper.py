"""Deduper agent: merges duplicate/near-duplicate normalized entities."""
from __future__ import annotations

from typing import TypedDict, Any
import logging
import re
from collections import defaultdict

from config import settings
from agents.common import get_llm_client
from tools.file_io import write_json


logger = logging.getLogger(__name__)


class DeduperState(TypedDict, total=False):
    normalized_entities: list[dict[str, Any]]


def _to_key(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def _jaccard(a: str, b: str) -> float:
    sa = set(a.split("_"))
    sb = set(b.split("_"))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _merge_entities(group: list[dict[str, Any]]) -> dict[str, Any]:
    # Base: the longest description or first entity
    base = max(group, key=lambda e: len((e.get("description") or ""))) if group else {}
    if not base:
        return {}
    merged = dict(base)
    # Union inputs/outputs
    def _uniq(seq: list[str]) -> list[str]:
        seen = set()
        out: list[str] = []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out
    all_inputs: list[str] = []
    all_outputs: list[str] = []
    names: list[str] = []
    for e in group:
        all_inputs.extend(e.get("inputs", []) or [])
        all_outputs.extend(e.get("outputs", []) or [])
        nm = e.get("name") or e.get("id")
        if nm:
            names.append(str(nm))
    merged["inputs"] = _uniq(all_inputs)
    merged["outputs"] = _uniq(all_outputs)
    # Preserve a canonical id: take base.id if exists, else key from base name
    if not merged.get("id") and merged.get("name"):
        merged["id"] = _to_key(str(merged["name"]))
    # Add alt_names for traceability
    merged["alt_names"] = _uniq(names)
    return merged


def _llm_should_merge(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Ask LLM only for borderline cases to decide merge vs separate.

    Returns True if LLM judges the two entities are the same concept.
    """
    try:
        client = get_llm_client()
        import json
        sys_prompt = (
            "You are a precise ontology normalizer. Decide if two entities refer to the SAME component. "
            "Answer with ONLY 'merge' or 'separate'. Consider names, descriptions, inputs, outputs, and roles."
        )
        user_prompt = json.dumps({"a": a, "b": b}, ensure_ascii=False)
        ans = client.query(user_prompt=user_prompt, system_prompt=sys_prompt).strip().lower()
        return ans.startswith("merge")
    except Exception:
        return False


def deduper(state: DeduperState) -> DeduperState:
    ents = state.get("normalized_entities", []) or []
    if not ents:
        return state
    # Group by exact id first
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in ents:
        k = _to_key(str(e.get("id") or e.get("name") or ""))
        buckets[k].append(e)

    # Within each bucket, also merge close variants by Jaccard similarity on token sets
    threshold = float(settings.name_similarity_threshold)
    use_llm = bool(getattr(settings, "use_llm_deduper", False))
    band = float(getattr(settings, "llm_similarity_band", 0.1))
    max_checks = int(getattr(settings, "llm_deduper_max_checks", 25))
    keys = list(buckets.keys())
    used = set()
    merged_groups: list[list[dict[str, Any]]] = []
    llm_checks = 0
    for i, ki in enumerate(keys):
        if ki in used:
            continue
        group = list(buckets[ki])
        used.add(ki)
        for j in range(i + 1, len(keys)):
            kj = keys[j]
            if kj in used:
                continue
            sim = _jaccard(ki, kj)
            if sim >= threshold:
                group.extend(buckets[kj])
                used.add(kj)
            elif use_llm and (threshold - band) <= sim < threshold and llm_checks < max_checks:
                # borderline: ask LLM once
                a = _merge_entities(buckets[ki])
                b = _merge_entities(buckets[kj])
                if _llm_should_merge(a, b):
                    group.extend(buckets[kj])
                    used.add(kj)
                llm_checks += 1
        merged_groups.append(group)

    merged_entities = [_merge_entities(g) for g in merged_groups]
    merged_entities = [e for e in merged_entities if e]
    logger.info(f"Deduper merged {len(ents)} -> {len(merged_entities)} entities (threshold={threshold}).")
    # Emit audit report
    try:
        report = []
        for grp in merged_groups:
            canon = _merge_entities(grp)
            report.append({
                "canonical": canon.get("id") or canon.get("name"),
                "alt_names": canon.get("alt_names", []),
                "members": [e.get("id") or e.get("name") for e in grp],
            })
        write_json(settings.output_dir / "deduper_report.json", {"groups": report})
    except Exception:
        pass
    return {**state, "normalized_entities": merged_entities}
