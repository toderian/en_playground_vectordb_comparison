"""
Zvec adapter.

Deployment: in-process only
Alibaba's embedded vector database built on the Proxima engine.
Designed explicitly for edge / on-device RAG.
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


class ZvecAdapter(BaseVectorDB):
    """Adapter for Zvec (Alibaba Proxima)."""

    COLLECTION_NAME = "documents"
    META_FILE = "meta.json"

    def __init__(self, workspace: str | Path, embedding_size: int = 1024):
        super().__init__(workspace, embedding_size)
        self._collection = None
        self._meta: list[dict] = []  # [{text, idx}, ...]

    # -- lifecycle -------------------------------------------------------------

    def open(self) -> None:
        import zvec

        schema = zvec.CollectionSchema(
            name=self.COLLECTION_NAME,
            vectors=zvec.VectorSchema(
                "embedding", zvec.DataType.VECTOR_FP32, self.embedding_size
            ),
        )

        meta_path = self.workspace / self.META_FILE

        if self.workspace.exists():
            self._collection = zvec.open(path=str(self.workspace))
            if meta_path.exists():
                with open(meta_path) as f:
                    self._meta = json.load(f)
        else:
            self._collection = zvec.create_and_open(
                path=str(self.workspace), schema=schema
            )
            self._meta = []

    def close(self) -> None:
        self._save_meta()
        self._collection = None
        self._meta = []

    def _save_meta(self) -> None:
        if self._collection is not None:
            self.workspace.mkdir(parents=True, exist_ok=True)
            with open(self.workspace / self.META_FILE, "w") as f:
                json.dump(self._meta, f)

    # -- CRUD ------------------------------------------------------------------

    def index(self, documents: list[Document]) -> None:
        import zvec

        docs = [
            zvec.Doc(
                id=str(d.idx),
                vectors={"embedding": d.embedding},
            )
            for d in documents
        ]
        self._collection.insert(docs)
        self._meta.extend({"text": d.text, "idx": d.idx} for d in documents)

    def search(self, query_embedding: list[float], limit: int = 10) -> SearchResponse:
        import zvec

        doc_count = self._collection.stats.doc_count
        if doc_count == 0:
            return SearchResponse()

        results = self._collection.query(
            zvec.VectorQuery("embedding", vector=query_embedding),
            topk=min(limit, doc_count),
        )

        meta_by_idx = {m["idx"]: m for m in self._meta}

        matches = [
            SearchResult(
                document=Document(
                    text=meta_by_idx.get(int(hit.id), {}).get("text", ""),
                    embedding=[],
                    idx=int(hit.id) if hit.id.isdigit() else -1,
                ),
                score=hit.score,
            )
            for hit in results
        ]
        return SearchResponse(matches=matches)

    def num_docs(self) -> int:
        return self._collection.stats.doc_count

    # -- info ------------------------------------------------------------------

    def info(self) -> CandidateInfo:
        try:
            from importlib.metadata import version
            ver = version("zvec")
        except Exception:
            ver = "unknown"
        return CandidateInfo(
            name="Zvec",
            version=ver,
            deployment="in-process",
            license="Apache-2.0",
            notes="Edge-first design. Built on Alibaba Proxima engine. Auto-persists to disk.",
        )
