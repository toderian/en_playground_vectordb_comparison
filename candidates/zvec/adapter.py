"""
Zvec adapter.

Deployment: in-process only
Alibaba's embedded vector database built on the Proxima engine.
Designed explicitly for edge / on-device RAG.
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


class ZvecAdapter(BaseVectorDB):
    """Adapter for Zvec (Alibaba Proxima)."""

    COLLECTION_NAME = "documents"

    def __init__(self, workspace: str | Path, embedding_size: int = 1024):
        super().__init__(workspace, embedding_size)
        self._collection = None

    # -- lifecycle -------------------------------------------------------------

    def open(self) -> None:
        import zvec

        self.workspace.mkdir(parents=True, exist_ok=True)

        schema = zvec.CollectionSchema(
            name=self.COLLECTION_NAME,
            vectors=zvec.VectorSchema(
                "embedding", zvec.DataType.VECTOR_FP32, self.embedding_size
            ),
        )

        try:
            self._collection = zvec.open(path=str(self.workspace), schema=schema)
        except Exception:
            self._collection = zvec.create_and_open(
                path=str(self.workspace), schema=schema
            )

    def close(self) -> None:
        self._collection = None

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

    def search(self, query_embedding: list[float], limit: int = 10) -> SearchResponse:
        import zvec

        results = self._collection.query(
            zvec.VectorQuery("embedding", vector=query_embedding),
            topk=limit,
        )

        matches = [
            SearchResult(
                document=Document(
                    text="",
                    embedding=[],
                    idx=int(hit.id) if hit.id.isdigit() else -1,
                ),
                score=hit.score if hasattr(hit, "score") else 0.0,
            )
            for hit in results
        ]
        return SearchResponse(matches=matches)

    def num_docs(self) -> int:
        try:
            return self._collection.count()
        except AttributeError:
            # Fallback if count() is not available in this version
            return 0

    # -- info ------------------------------------------------------------------

    def info(self) -> CandidateInfo:
        try:
            from importlib.metadata import version
            ver = version("zvec")
        except Exception:
            ver = "unknown"
        return CandidateInfo(
            name="Zvec (Alibaba Proxima)",
            version=ver,
            deployment="in-process",
            license="Apache-2.0",
            notes="Edge-first design. Built on Alibaba Proxima engine. Auto-persists to disk.",
        )
