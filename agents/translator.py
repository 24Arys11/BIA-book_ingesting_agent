"""Translator agent: normalizes names and adds Romanian notes."""
from __future__ import annotations

from typing import TypedDict, Any
import logging
import re

from agents.common import read_prompt, get_llm_client


logger = logging.getLogger(__name__)


class TranslatorState(TypedDict, total=False):
    entities: list[dict[str, Any]]
    normalized_entities: list[dict[str, Any]]
    target_language: str


def _to_snake(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^0-9A-Za-zăâîșțĂÂÎȘȚ]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def translator(state: TranslatorState) -> TranslatorState:
    system_prompt = read_prompt("translator.txt")
    ents = state.get("entities", []) or []
    if not ents:
        return {**state, "normalized_entities": []}
    target_lang = (state.get("target_language") or "en").lower()

    # Try LLM normalization; fallback to deterministic
    client = get_llm_client()
    try:
        import json
        user_prompt = (
            ("Translate names/descriptions/IO to English and " if target_lang == "en" else "") +
            "Normalize the following entities to snake_case IDs, keep original in 'romanian' when applicable, "
            "and return JSON with same fields plus 'id' and 'romanian'. Return ONLY JSON.\n\n"
            + json.dumps({"entities": ents}, ensure_ascii=False)
        )
        answer = client.query(user_prompt=user_prompt, system_prompt=system_prompt, temperature=0.1)
        data = json.loads(answer)
        out = data.get("entities", []) if isinstance(data, dict) else []
        if out:
            # enforce English names; add romanian in field if different
            cleaned: list[dict[str, Any]] = []
            for e in out:
                nm = str(e.get("name") or e.get("id") or "").strip()
                ro = str(e.get("romanian") or "").strip()
                # if identical or empty, keep as-is; otherwise keep both
                if ro and ro.lower() == nm.lower():
                    e["romanian"] = ro
                elif ro and ro.lower() != nm.lower():
                    e["romanian"] = ro
                # normalize id
                e["id"] = _to_snake(nm)
                cleaned.append(e)
            out = cleaned
            logger.info(f"Translator normalized {len(out)} entities via LLM.")
            return {**state, "normalized_entities": out}
    except Exception as e:
        logger.warning(f"LLM translator failed, using fallback. Error: {e}")

    out: list[dict[str, Any]] = []
    for idx, e in enumerate(ents):
        name = e.get("name", f"entity_{idx}")
        out.append({
            **e,
            "id": _to_snake(name),
            "romanian": name,
        })
    logger.info(f"Translator normalized {len(out)} entities (fallback).")
    return {**state, "normalized_entities": out}
