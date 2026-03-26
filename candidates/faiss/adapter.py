"""
FAISS adapter.

Deployment: in-process only
Facebook AI Similarity Search — highly optimised C++ library with Python bindings.
No built-in persistence beyond save/load to file; no server mode.
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


class FAISSAdapter(BaseVectorDB):
    """Adapter for FAISS (flat index with inner-product / cosine similarity)."""

    INDEX_FILE = "index.faiss"
    META_FILE = "meta.json"

    def __init__(self, workspace: str | Path, embedding_size: int = 1024):
        super().__init__(workspace, embedding_size)
        self._index = None
        self._meta: list[dict] = []   # [{text, idx}, ...]

    # -- lifecycle -------------------------------------------------------------

    def open(self) -> None:
        import faiss

        self.workspace.mkdir(parents=True, exist_ok=True)
        index_path = self.workspace / self.INDEX_FILE
        meta_path = self.workspace / self.META_FILE

        if index_path.exists() and meta_path.exists():
            self._index = faiss.read_index(str(index_path))
            with open(meta_path) as f:
                self._meta = json.load(f)
        else:
            self._index = faiss.IndexFlatIP(self.embedding_size)  # inner-product (cosine after normalisation)
            self._meta = []

    def close(self) -> None:
        self._save()
        self._index = None
        self._meta = []

    def _save(self) -> None:
        import faiss

        if self._index is not None:
            faiss.write_index(self._index, str(self.workspace / self.INDEX_FILE))
            with open(self.workspace / self.META_FILE, "w") as f:
                json.dump(self._meta, f)

    # -- CRUD ------------------------------------------------------------------

    def index(self, documents: list[Document]) -> None:
        import numpy as np

        vectors = np.array([d.embedding for d in documents], dtype=np.float32)
        # Normalise for cosine similarity via inner product
        faiss = __import__("faiss")
        faiss.normalize_L2(vectors)
        self._index.add(vectors)
        self._meta.extend({"text": d.text, "idx": d.idx} for d in documents)

    def search(self, query_embedding: list[float], limit: int = 10) -> SearchResponse:
        import numpy as np

        if self._index.ntotal == 0:
            return SearchResponse()

        query = np.array([query_embedding], dtype=np.float32)
        faiss = __import__("faiss")
        faiss.normalize_L2(query)
        scores, indices = self._index.search(query, min(limit, self._index.ntotal))

        matches = []
        for score, i in zip(scores[0], indices[0]):
            if i < 0:
                continue
            meta = self._meta[i]
            matches.append(
                SearchResult(
                    document=Document(text=meta["text"], embedding=[], idx=meta["idx"]),
                    score=float(score),
                )
            )
        return SearchResponse(matches=matches)

    def num_docs(self) -> int:
        return self._index.ntotal if self._index else 0

    # -- info ------------------------------------------------------------------

    def info(self) -> CandidateInfo:
        try:
            from importlib.metadata import version
            ver = version("faiss-cpu")
        except Exception:
            ver = "unknown"
        return CandidateInfo(
            name="FAISS",
            version=ver,
            deployment="in-process",
            license="MIT",
            notes="C++ core, Python bindings. No server mode. Manual persistence (save/load).",
        )
