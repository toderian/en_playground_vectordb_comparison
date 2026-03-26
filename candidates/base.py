"""
Abstract base interface for vector database candidates.

Every candidate must implement this interface so that benchmarks and tests
are comparable.  The API surface mirrors what edge_node actually uses today
(HNSWVectorDB from Jina's vectordb package):

    - create / open a named context (workspace)
    - index a batch of (text, embedding, idx) records
    - search by embedding vector, returning matches + scores
    - count stored documents
    - delete a context
"""

from __future__ import annotations

import abc
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Data containers ──────────────────────────────────────────────────────────

@dataclass
class Document:
    """Minimal document record that mirrors NaeuralDoc in edge_node."""
    text: str
    embedding: list[float]
    idx: int = -1


@dataclass
class SearchResult:
    """One search hit."""
    document: Document
    score: float


@dataclass
class SearchResponse:
    """Return type of a single query search."""
    matches: list[SearchResult] = field(default_factory=list)


# ── Candidate metadata ──────────────────────────────────────────────────────

@dataclass
class CandidateInfo:
    """Metadata about a candidate, used in reports."""
    name: str
    version: str
    deployment: str          # "in-process" or "sidecar"
    license: str = ""
    notes: str = ""


# ── Abstract base class ─────────────────────────────────────────────────────

class BaseVectorDB(abc.ABC):
    """Interface that every candidate adapter must implement."""

    def __init__(self, workspace: str | Path, embedding_size: int = 1024):
        self.workspace = Path(workspace)
        self.embedding_size = embedding_size

    # -- lifecycle -------------------------------------------------------------

    @abc.abstractmethod
    def open(self) -> None:
        """Open or create the database / collection at *workspace*."""

    @abc.abstractmethod
    def close(self) -> None:
        """Flush and release resources."""

    def destroy(self) -> None:
        """Delete all persisted data for this workspace."""
        if self.workspace.exists():
            shutil.rmtree(self.workspace)

    # -- CRUD ------------------------------------------------------------------

    @abc.abstractmethod
    def index(self, documents: list[Document]) -> None:
        """Insert or upsert a batch of documents."""

    @abc.abstractmethod
    def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
    ) -> SearchResponse:
        """Return the *limit* nearest neighbours for *query_embedding*."""

    @abc.abstractmethod
    def num_docs(self) -> int:
        """Return the total number of indexed documents."""

    # -- info ------------------------------------------------------------------

    @abc.abstractmethod
    def info(self) -> CandidateInfo:
        """Return metadata about this candidate."""

    # -- context manager -------------------------------------------------------

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
