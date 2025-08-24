"""Common utilities for agents: prompt loading and LLM helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import settings
from tools.llm_client import OpenaiResponseGenerator, IResponseGenerator


logger = logging.getLogger(__name__)


def read_prompt(name: str) -> str:
    """Load a prompt file from system_prompts directory.

    Args:
        name: File name like "ingestor.txt".
    """
    path = settings.prompts_dir / name
    return path.read_text(encoding="utf-8")


_LLM_CLIENT: Optional[IResponseGenerator] = None


def get_llm_client() -> IResponseGenerator:
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        _LLM_CLIENT = OpenaiResponseGenerator(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )
    return _LLM_CLIENT


def split_text_semantic(text: str) -> list[str]:
    """Split text into chunks using tokens when possible; fallback to characters.

    Note: chunk_size and chunk_overlap are interpreted as TOKEN counts when the
    token splitter is available (default). If tokenization isn't available,
    the function falls back to character-based splitting with similar values.
    """
    # Try token-based splitter first (uses tiktoken under the hood)
    try:
        splitter = TokenTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            encoding_name="cl100k_base",
        )
        chunks = [c for c in splitter.split_text(text) if c.strip()]
        return chunks
    except Exception as e:
        logger.warning(f"TokenTextSplitter unavailable, falling back to character splitter: {e}")

    # Fallback: character-based with semantic-friendly separators
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", ".", " "]
    )
    chunks = [c for c in splitter.split_text(text) if c.strip()]
    return chunks


class SimpleRetriever:
    """FAISS-backed retriever with optional on-disk persistence.

    Files stored under OUTPUT_DIR/vector_index/:
    - index.faiss, index.pkl (vector index and metadata)
    """

    def __init__(self, chunks: list[str]):
        self._emb = HuggingFaceEmbeddings(model_name=settings.embeddings_model)
        index_dir = settings.output_dir / "vector_index"
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / "index"

        loaded = False
        if settings.reuse_vector_index and (index_path.with_suffix(".faiss").exists() and index_path.with_suffix(".pkl").exists()):
            try:
                self._vs = FAISS.load_local(
                    str(index_path),
                    self._emb,
                    allow_dangerous_deserialization=True,
                )
                loaded = True
                logger.info("Loaded persisted FAISS index from disk.")
            except Exception as e:
                logger.warning(f"Failed to load persisted FAISS index: {e}")

        if not loaded:
            docs = [Document(page_content=c) for c in chunks]
            self._vs = FAISS.from_documents(docs, self._emb)
            if settings.persist_vector_index:
                try:
                    self._vs.save_local(str(index_path))
                    logger.info("Saved FAISS index to disk.")
                except Exception as e:
                    logger.warning(f"Failed to persist FAISS index: {e}")

    def query(self, q: str, k: int = 4) -> list[str]:
        res = self._vs.similarity_search(q, k=k)
        return [d.page_content for d in res]
