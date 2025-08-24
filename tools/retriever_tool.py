"""Retriever tool wrapper to be used by agents if needed.

In this simplified implementation, agents use SimpleRetriever directly.
This module provides a placeholder for future expansion.
"""
from __future__ import annotations

from typing import Protocol, List


class Retriever(Protocol):
    def query(self, q: str, k: int = 4) -> List[str]:
        ...
