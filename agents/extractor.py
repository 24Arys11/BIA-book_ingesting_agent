"""Extractor agent: identifies cognitive components from text chunks.

This implementation uses a lightweight heuristic with retrieval for demonstration.
In production, replace the heuristic with an LLM call guided by the system prompt.
"""
from __future__ import annotations

from typing import TypedDict, Any
import logging
import re

from agents.common import read_prompt, get_llm_client
from config import settings
from tools.file_io import write_json, read_json
from tools.json_utils import robust_parse_json
from agents.ingestor import get_retriever


logger = logging.getLogger(__name__)


class ExtractorState(TypedDict, total=False):
    chunks: list[str]
    retriever_key: str
    instructions: str
    sections: list[dict[str, Any]]
    entities: list[dict[str, Any]]
    full_ingestion: bool
    target_language: str


def _heuristic_extract_entities(text: str) -> list[dict[str, Any]]:
    # Heuristic: look for patterns like "X is a Y that ..." or bullet lists
    entities: list[dict[str, Any]] = []
    # Simple sentence split
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for s in sentences:
        m = re.search(r"([A-Za-zăâîșțĂÂÎȘȚ\-\s]+)\s+is\s+(a|an|the)\s+([A-Za-z\-\s]+)\s+that\s+(.*)", s, flags=re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            typ = m.group(3).strip().lower()
            desc = m.group(4).strip().rstrip(".")
            if 3 <= len(name) <= 120 and 3 <= len(desc):
                entities.append({
                    "name": name,
                    "type": typ if typ in {"component","module","system","process","circuit","unit","network","layer","function","agent","entity"} else "other",
                    "description": desc,
                    "inputs": [],
                    "outputs": [],
                })
    return entities


def extractor(state: ExtractorState) -> ExtractorState:
    system_prompt = read_prompt("extractor.txt")
    chunks = state.get("chunks", [])
    if not chunks:
        logger.warning("No chunks found; extractor returning empty entities.")
        return {**state, "entities": []}
    retriever = get_retriever(state["retriever_key"]) if "retriever_key" in state else None
    full_ingestion = state.get("full_ingestion", False)

    if full_ingestion:
        # Process all chunks through the LLM in batches to avoid overlong prompts
        client = get_llm_client()
        system_prompt = system_prompt
        all_entities: list[dict[str, Any]] = []
        import json
        cache_path = settings.output_dir / "full_ingestion_entities.json"
        if settings.reuse_full_ingestion_cache and cache_path.exists():
            try:
                cached = read_json(cache_path)
                cached_entities = cached.get("entities", []) if isinstance(cached, dict) else []
                if cached_entities:
                    logger.info(f"Extractor (full_ingestion) reusing cached entities: {len(cached_entities)}")
                    return {**state, "entities": cached_entities}
            except Exception:
                pass
        batch_size = settings.full_ingestion_batch_size
        for i in range(0, len(chunks), batch_size):
            batch = "\n\n".join(chunks[i:i+batch_size])
            lang_hint = "Provide fields in English. " if (state.get("target_language") or "en").lower() == "en" else ""
            user_prompt = (
                f"{lang_hint}Extract ONLY cognitive components/models with clear responsibilities and IO. "
                "Ignore metadata (authors, years, prefaces, chapters). Return ONLY JSON with 'entities' each having name, type, description, inputs, outputs.\n\nText:\n"
                + batch
            )
            try:
                n = max(1, int(getattr(settings, "self_consistency_samples", 3)))
                temps = float(getattr(settings, "llm_temperature", 0.2))
                ents_batch: list[dict[str, Any]] = []
                if n > 1:
                    answers = client.query_n(user_prompt=user_prompt, system_prompt=system_prompt, temperature=temps, n=n)
                    for a in answers:
                        data = robust_parse_json(a) or {}
                        if isinstance(data, dict):
                            ents_batch.extend(data.get("entities", []) or [])
                else:
                    answer = client.query(user_prompt=user_prompt, system_prompt=system_prompt, temperature=temps)
                    data = robust_parse_json(answer) or {}
                    if isinstance(data, dict):
                        ents_batch.extend(data.get("entities", []) or [])
                # simple dedupe within batch by name+type
                seen = set()
                uniq = []
                for e in ents_batch:
                    key = (e.get("name"), e.get("type"))
                    if key not in seen:
                        uniq.append(e)
                        seen.add(key)
                if uniq:
                    all_entities.extend(uniq)
            except Exception as e:
                logger.warning(f"LLM batch extraction failed at chunks {i}-{i+batch_size}: {e}")
                # Heuristic fallback for the batch
                all_entities.extend(_heuristic_extract_entities(batch))
        # Persist an intermediate corpus of all extracted entities for downstream reasoning
        try:
            write_json(settings.output_dir / "full_ingestion_entities.json", {"entities": all_entities})
        except Exception:
            pass
        def _is_relevant(e: dict[str, Any]) -> bool:
            name = (e.get("name") or e.get("id") or "").lower()
            desc = (e.get("description") or "").lower()
            bad = ["author", "autori", "bibliography", "preface", "introduc", "chapter", "capitol", "appendix", "isbn", "copyright"]
            if any(b in name or b in desc for b in bad):
                return False
            typ = (e.get("type") or "").lower()
            allowed = {"component", "module", "system", "process", "circuit", "unit", "network", "layer", "function", "agent", "model"}
            if typ and typ not in allowed:
                return False
            # keep if at least name length reasonable
            return 2 <= len(name) <= 60
        filtered = [e for e in all_entities if _is_relevant(e)]
        logger.info(f"Extractor (full_ingestion) found {len(filtered)} relevant entities across all chunks.")
        return {**state, "entities": filtered}
    else:
        # Section-aware sampled extraction with optional RAG enrichment
        client = get_llm_client()
        sections = state.get("sections", []) or []
        if not sections:
            sections = [{"title": "all", "chunk_indices": list(range(len(chunks)))}]
        aggregated: list[dict[str, Any]] = []
        n = max(1, int(getattr(settings, "self_consistency_samples", 3)))
        temps = float(getattr(settings, "llm_temperature", 0.2))
        for sec in sections[:8]:  # cap sections to keep calls bounded
            idxs = sec.get("chunk_indices", []) or []
            reps: list[str] = []
            if idxs:
                reps.append(chunks[idxs[0]])
                if len(idxs) > 2:
                    reps.append(chunks[idxs[len(idxs)//2]])
                    reps.append(chunks[idxs[-1]])
            # RAG enrichment
            rag_parts: list[str] = []
            if retriever:
                rag_parts.extend(["\n\n".join(retriever.query(q, k=4)) for q in [
                    "list of model components and their roles",
                    "data/control flows between components; inputs and outputs",
                ]])
            context = "\n\n".join([p for p in (reps + rag_parts) if p])
            lang_hint = "Provide fields in English. " if (state.get("target_language") or "en").lower() == "en" else ""
            user_prompt = (
                f"Section: {sec.get('title','')}. {lang_hint}"
                "Extract entities from the following text and return ONLY JSON with 'entities'.\n\nText:\n" + context
            )
            try:
                if n > 1:
                    answers = client.query_n(user_prompt=user_prompt, system_prompt=system_prompt, temperature=temps, n=n)
                    for a in answers:
                        data = robust_parse_json(a) or {}
                        if isinstance(data, dict):
                            aggregated.extend(data.get("entities", []) or [])
                else:
                    ans = client.query(user_prompt=user_prompt, system_prompt=system_prompt, temperature=temps)
                    data = robust_parse_json(ans) or {}
                    if isinstance(data, dict):
                        aggregated.extend(data.get("entities", []) or [])
            except Exception:
                # heuristic fallback per section
                aggregated.extend(_heuristic_extract_entities(context))
        # schema enforcement / coercion
        def _coerce_entity(e: dict[str, Any]) -> dict[str, Any]:
            out = dict(e)
            out["name"] = str(out.get("name") or out.get("id") or "")
            out["type"] = str(out.get("type") or "component")
            if not isinstance(out.get("inputs"), list):
                out["inputs"] = []
            if not isinstance(out.get("outputs"), list):
                out["outputs"] = []
            if not isinstance(out.get("description"), str):
                out["description"] = ""
            return out
        coerced = [_coerce_entity(e) for e in aggregated if e]
        # filter irrelevant
        def _is_relevant(e: dict[str, Any]) -> bool:
            name = (e.get("name") or e.get("id") or "").lower()
            desc = (e.get("description") or "").lower()
            bad = ["author", "autori", "bibliography", "preface", "introduc", "chapter", "capitol", "appendix", "isbn", "copyright"]
            if any(b in name or b in desc for b in bad):
                return False
            typ = (e.get("type") or "").lower()
            allowed = {"component", "module", "system", "process", "circuit", "unit", "network", "layer", "function", "agent", "model"}
            if typ and typ not in allowed:
                return False
            return 2 <= len(name) <= 60
        coerced = [e for e in coerced if _is_relevant(e)]
        # dedupe across sections by (name,type)
        seen = set()
        uniq: list[dict[str, Any]] = []
        for e in coerced:
            key = (e.get("name"), e.get("type"))
            if key not in seen:
                uniq.append(e)
                seen.add(key)
        logger.info(f"Extractor (section-aware) found {len(uniq)} unique entities.")
        return {**state, "entities": uniq}
