"""Entry point for the cognitive architecture ingestion pipeline.

Reads inputs, configures the pipeline, runs agents using LangGraph, and writes outputs.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
import sys

from config import settings
from tools.file_io import read_book, read_text_file, ensure_dir, content_hash
from agents.ingestor import ingestor
from agents.section_indexer import section_indexer
from agents.extractor import extractor
from agents.translator import translator
from agents.graph_builder import graph_builder
from agents.diagram_generator import diagram_generator
from agents.markdown_generator import markdown_generator
from agents.validator import validator
from agents.deduper import deduper
from agents.task_filter import task_filter
from agents.logical_distiller import logical_distiller
from tools.eval import evaluate


class PipelineState(TypedDict, total=False):
    book_text: str
    instructions: str
    chunks: List[str]
    entities: List[Dict[str, Any]]  # extracted raw entities
    normalized_entities: List[Dict[str, Any]]
    graph: Dict[str, Any]
    diagram_puml: str
    markdown_md: str
    validation_report: str
    target_language: str
    full_ingestion: bool


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _load_inputs() -> tuple[str, str]:
    # Preferred: book.txt
    book_candidates = [settings.input_dir / "book.txt"]
    # Fallbacks
    book_candidates.append(settings.input_dir / "book.pdf")
    book_candidates.append(settings.input_dir / "book.docx")

    last_err: Exception | None = None
    book_text = ""
    found_any = False
    for p in book_candidates:
        if not p.exists():
            continue
        found_any = True
        try:
            logging.info(f"Reading book from {p}")
            book_text = read_book(p)
            break
        except Exception as e:  # capture but continue fallbacks
            logging.warning(f"Failed to read book from {p}: {e}")
            last_err = e
    if not book_text:
        if last_err:
            raise last_err
        if not found_any:
            raise ValueError("No input book found. Expected input/book.txt (or input/book.pdf or input/book.docx).")
        # Fallback generic
        raise ValueError("Could not read the input book; see warnings above for details.")

    instr_path = settings.input_dir / "instructions.txt"
    try:
        logging.info(f"Reading instructions from {instr_path}")
        instructions = read_text_file(instr_path)
    except Exception as e:
        logging.error(f"Missing or unreadable instructions: {e}")
        sys.exit(1)
    return book_text, instructions


def _detect_language(instructions: str) -> str:
    text = instructions.lower()
    # crude detection rules; can be improved or replaced by LLM later
    if "english" in text or "in english" in text or "engleza" in text:
        return "en"
    if "romanian" in text or "romana" in text or "în română" in text or "in romana" in text:
        return "ro"
    return "en"  # default to English


def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("ingestor", ingestor)
    graph.add_node("section_indexer", section_indexer)
    graph.add_node("extractor", extractor)
    graph.add_node("translator", translator)
    graph.add_node("deduper", deduper)
    graph.add_node("task_filter", task_filter)
    graph.add_node("logical_distiller", logical_distiller)
    graph.add_node("graph_builder", graph_builder)
    graph.add_node("diagram_generator", diagram_generator)
    graph.add_node("markdown_generator", markdown_generator)
    graph.add_node("validator", validator)

    graph.set_entry_point("ingestor")
    graph.add_edge("ingestor", "section_indexer")
    graph.add_edge("section_indexer", "extractor")
    graph.add_edge("extractor", "translator")
    graph.add_edge("translator", "deduper")
    graph.add_edge("deduper", "task_filter")
    graph.add_edge("task_filter", "logical_distiller")
    graph.add_edge("logical_distiller", "graph_builder")
    graph.add_edge("graph_builder", "diagram_generator")
    graph.add_edge("diagram_generator", "markdown_generator")
    graph.add_edge("markdown_generator", "validator")
    graph.add_edge("validator", END)

    return graph


def main() -> None:
    setup_logging()
    ensure_dir(settings.output_dir)

    book_text, instructions = _load_inputs()
    book_hash = content_hash(book_text)

    # Fast resume: if cached outputs exist and match hash, skip pipeline
    if settings.resume_from_cache:
        try:
            kg_path = settings.output_dir / "knowledge_graph.json"
            md_path = settings.output_dir / "cognitive_architecture.md"
            diag_puml = settings.output_dir / "architecture.puml"
            meta_path = settings.output_dir / "ingestion_meta.json"
            if kg_path.exists() and md_path.exists() and diag_puml.exists() and meta_path.exists():
                import json as _json
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = _json.load(f)
                ok_hash = (meta.get("book_hash") == book_hash) or settings.resume_allow_hash_mismatch
                ok_full = (meta.get("full_ingestion") is True) or (not settings.resume_require_full_ingestion)
                if ok_hash and ok_full and settings.resume_skip_pipeline_if_complete:
                    logging.info("Resuming from cached outputs; skipping pipeline execution.")
                    return
        except Exception:
            pass

    graph = build_graph()
    app = graph.compile()

    initial_state: PipelineState = {
        "book_text": book_text,
        "instructions": instructions,
    "target_language": _detect_language(instructions),
    "full_ingestion": settings.full_ingestion,
    }

    logging.info("Running agent pipeline...")
    final_state = app.invoke(initial_state)
    logging.info("Pipeline completed.")

    # Write outputs to files
    (settings.output_dir / "cognitive_architecture.md").write_text(
        final_state.get("markdown_md", ""), encoding="utf-8"
    )
    (settings.output_dir / "architecture.puml").write_text(
        final_state.get("diagram_puml", ""), encoding="utf-8"
    )
    import json as _json

    if "graph" in final_state:
        with (settings.output_dir / "knowledge_graph.json").open("w", encoding="utf-8") as f:
            _json.dump(final_state["graph"], f, ensure_ascii=False, indent=2)
    # persist meta for resume
    with (settings.output_dir / "ingestion_meta.json").open("w", encoding="utf-8") as f:
        _json.dump({
            "book_hash": book_hash,
            "full_ingestion": bool(settings.full_ingestion),
            "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        }, f, ensure_ascii=False, indent=2)

    # Optional evaluation if gold exists
    try:
        report = evaluate(final_state.get("graph", {}), final_state.get("normalized_entities", []))
        if report:
            with (settings.output_dir / "eval_report.json").open("w", encoding="utf-8") as f:
                _json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    (settings.output_dir / "validation_report.txt").write_text(
        final_state.get("validation_report", ""), encoding="utf-8"
    )

    logging.info(f"Outputs written to {settings.output_dir}")


if __name__ == "__main__":
    main()
