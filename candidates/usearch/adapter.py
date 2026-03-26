"""
USearch adapter.

Deployment: in-process only
Single-file HNSW vector search engine from Unum Cloud.
Faster and lighter than FAISS/hnswlib, with mmap persistence.
"""

from __future__ import annotations

import json
from pathlib import Path

from candidates.base import (
    BaseVectorDB,
    CandidateInfo,
    Document,
    SearchResponse,
    SearchResult,
)


class USearchAdapter(BaseVectorDB):
    """Adapter for USearch (single-file HNSW index)."""

    INDEX_FILE = "index.usearch"
    META_FILE = "meta.json"

    def __init__(self, workspace: str | Path, embedding_size: int = 1024):
        super().__init__(workspace, embedding_size)
        self._index = None
        self._meta: list[dict] = []  # [{text, idx}, ...]

    # -- lifecycle -------------------------------------------------------------

    def open(self) -> None:
        from usearch.index import Index

        self.workspace.mkdir(parents=True, exist_ok=True)
        index_path = self.workspace / self.INDEX_FILE
        meta_path = self.workspace / self.META_FILE

        if index_path.exists() and meta_path.exists():
            self._index = Index.restore(str(index_path), view=False)
            with open(meta_path) as f:
                self._meta = json.load(f)
        else:
            self._index = Index(
                ndim=self.embedding_size,
                metric="cos",
                dtype="f32",
            )
            self._meta = []

    def close(self) -> None:
        self._save()
        self._index = None
        self._meta = []

    def _save(self) -> None:
        if self._index is not None:
            self._index.save(str(self.workspace / self.INDEX_FILE))
            with open(self.workspace / self.META_FILE, "w") as f:
                json.dump(self._meta, f)

    # -- CRUD ------------------------------------------------------------------

    def index(self, documents: list[Document]) -> None:
        import numpy as np

        keys = np.array([d.idx for d in documents], dtype=np.int64)
        vectors = np.array([d.embedding for d in documents], dtype=np.float32)
        self._index.add(keys, vectors)
        self._meta.extend({"text": d.text, "idx": d.idx} for d in documents)

    def search(self, query_embedding: list[float], limit: int = 10) -> SearchResponse:
        import numpy as np

        if len(self._index) == 0:
            return SearchResponse()

        query = np.array(query_embedding, dtype=np.float32)
        results = self._index.search(query, min(limit, len(self._index)))

        # Build a lookup from idx -> meta for the results
        meta_by_idx = {m["idx"]: m for m in self._meta}

        matches = []
        for key, distance in zip(results.keys, results.distances):
            meta = meta_by_idx.get(int(key), {"text": "", "idx": int(key)})
            matches.append(
                SearchResult(
                    document=Document(text=meta["text"], embedding=[], idx=meta["idx"]),
                    score=float(distance),
                )
            )
        return SearchResponse(matches=matches)

    def num_docs(self) -> int:
        return len(self._index) if self._index is not None else 0

    # -- info ------------------------------------------------------------------

    def info(self) -> CandidateInfo:
        try:
            from importlib.metadata import version
            ver = version("usearch")
        except Exception:
            ver = "unknown"
        return CandidateInfo(
            name="USearch",
            version=ver,
            deployment="in-process",
            license="Apache-2.0",
            notes="Single-file HNSW. Faster than FAISS/hnswlib. mmap support via view().",
        )
