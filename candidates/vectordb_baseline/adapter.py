"""
Baseline adapter — current Jina vectordb (HNSWVectorDB).

Deployment: in-process
This is what edge_node uses today.  Kept here as the reference baseline
so every benchmark comparison is apples-to-apples.
"""

from __future__ import annotations

from pathlib import Path

from candidates.base import (
    BaseVectorDB,
    CandidateInfo,
    Document,
    SearchResponse,
    SearchResult,
)


class VectorDBAdapter(BaseVectorDB):
    """Thin wrapper around ``vectordb.HNSWVectorDB``."""

    def __init__(self, workspace: str | Path, embedding_size: int = 1024):
        super().__init__(workspace, embedding_size)
        self._db = None
        self._doc_cls = None

    # -- helpers ---------------------------------------------------------------

    def _make_doc_class(self):
        """Build a dynamic NaeuralDoc-style docarray class."""
        from docarray import BaseDoc
        from docarray.typing import NdArray

        size = self.embedding_size

        class _Doc(BaseDoc):
            text: str = ""
            embedding: NdArray[size]
            idx: int = -1

        # pydantic v1 needs forward refs resolved explicitly
        if hasattr(_Doc, "update_forward_refs"):
            _Doc.update_forward_refs()

        return _Doc

    # -- lifecycle -------------------------------------------------------------

    def open(self) -> None:
        from vectordb import HNSWVectorDB

        self._doc_cls = self._make_doc_class()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._db = HNSWVectorDB[self._doc_cls](workspace=str(self.workspace))

    def close(self) -> None:
        self._db = None

    # -- CRUD ------------------------------------------------------------------

    def index(self, documents: list[Document]) -> None:
        from docarray import DocList

        docs = [
            self._doc_cls(text=d.text, embedding=d.embedding, idx=d.idx)
            for d in documents
        ]
        self._db.index(inputs=DocList[self._doc_cls](docs))

    def search(self, query_embedding: list[float], limit: int = 10) -> SearchResponse:
        from docarray import DocList

        query = self._doc_cls(text="", embedding=query_embedding, idx=-1)
        raw = self._db.search(inputs=DocList[self._doc_cls]([query]), limit=limit)[0]
        results = [
            SearchResult(
                document=Document(text=m.text, embedding=list(m.embedding), idx=m.idx),
                score=s,
            )
            for m, s in zip(raw.matches, raw.scores)
        ]
        return SearchResponse(matches=results)

    def num_docs(self) -> int:
        return self._db.num_docs()["num_docs"]

    # -- info ------------------------------------------------------------------

    def info(self) -> CandidateInfo:
        try:
            from importlib.metadata import version
            ver = version("vectordb")
        except Exception:
            ver = "unknown"
        return CandidateInfo(
            name="vectordb (Jina HNSWVectorDB)",
            version=ver,
            deployment="in-process",
            license="Apache-2.0",
            notes="Current baseline. Wraps docarray + hnswlib. No longer maintained.",
        )
