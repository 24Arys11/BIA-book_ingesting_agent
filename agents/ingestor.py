"""Ingestor agent: splits the book into semantic chunks and prepares retrieval."""
from __future__ import annotations

from typing import TypedDict, Any
import logging

from agents.common import read_prompt, split_text_semantic, SimpleRetriever


logger = logging.getLogger(__name__)


_RETRIEVER_CACHE: dict[str, SimpleRetriever] = {}


class IngestorState(TypedDict, total=False):
    book_text: str
    chunks: list[str]
    retriever_key: str
    full_ingestion: bool


def ingestor(state: IngestorState) -> IngestorState:
    # Load prompt even if not used directly here to conform to architecture
    _ = read_prompt("ingestor.txt")
    book_text = state.get("book_text", "")
    if not book_text or not book_text.strip():
        raise ValueError("Book text is empty after reading.")

    chunks = split_text_semantic(book_text)
    retriever = SimpleRetriever(chunks)

    key = str(id(retriever))
    _RETRIEVER_CACHE[key] = retriever

    logger.info(f"Ingested book into {len(chunks)} chunks.")
    return {
        **state,
        "chunks": chunks,
        "retriever_key": key,
    }


def get_retriever(key: str) -> SimpleRetriever:
    return _RETRIEVER_CACHE[key]
