"""File I/O utilities for reading inputs and writing outputs.

All path handling uses pathlib for cross-platform correctness.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any
import re

import json
import hashlib

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pdfplumber = None  # type: ignore

try:
    import docx  # python-docx
except Exception:  # pragma: no cover - optional dependency
    docx = None  # type: ignore


from config import settings

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_text(text: str) -> str:
    if settings.transliterate_text:
        try:
            from unidecode import unidecode
            text = unidecode(text)
        except Exception:
            pass
    if settings.normalize_whitespace:
        # Collapse 3+ newlines to 2, 2+ spaces to 1
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def read_text_file(path: Path) -> str:
    if not path.exists():
        raise ValueError(f"Missing required file: {path}")
    content = path.read_text(encoding="utf-8", errors="ignore")
    content = _normalize_text(content)
    if not content.strip():
        raise ValueError(f"File is empty: {path}")
    return content


def read_book(path: Path) -> str:
    """Read a book file (.txt, .pdf, .docx) and return plain text.

    Raises ValueError on missing/empty or unsupported formats.
    """
    if not path.exists():
        raise ValueError(f"Missing required book file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return read_text_file(path)
    if suffix == ".pdf":
        if pdfplumber is None:
            raise ValueError("pdfplumber is required to read PDF files. Install it in your environment.")
        texts: list[str] = []
        with pdfplumber.open(str(path)) as pdf:  # type: ignore[arg-type]
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
        content = "\n".join(texts)
        content = _normalize_text(content)
        if not content.strip():
            raise ValueError(f"No extractable text in PDF: {path}")
        return content
    if suffix in {".docx", ".doc"}:  # doc is best-effort if python-docx can open
        if docx is None:
            raise ValueError("python-docx is required to read DOCX files. Install it in your environment.")
        document = docx.Document(str(path))  # type: ignore[arg-type]
        content = "\n".join(p.text for p in document.paragraphs)
        content = _normalize_text(content)
        if not content.strip():
            raise ValueError(f"No extractable text in DOCX: {path}")
        return content
    raise ValueError(f"Unsupported book file format: {path.suffix}")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def content_hash(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()
