"""
LanceDB adapter.

Deployment: in-process (embedded)
LanceDB is a serverless vector database built on Lance columnar format.
Zero-config, no server needed, persists to disk.
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


class LanceDBAdapter(BaseVectorDB):
    """Adapter for LanceDB (embedded mode)."""

    TABLE_NAME = "documents"

    def __init__(self, workspace: str | Path, embedding_size: int = 1024):
        super().__init__(workspace, embedding_size)
        self._db = None
        self._table = None

    # -- lifecycle -------------------------------------------------------------

    def open(self) -> None:
        import lancedb

        self.workspace.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.workspace))

        if self.TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(self.TABLE_NAME)
        else:
            self._table = None  # created lazily on first index()

    def close(self) -> None:
        self._db = None
        self._table = None

    # -- CRUD ------------------------------------------------------------------

    def index(self, documents: list[Document]) -> None:
        rows = [
            {
                "text": d.text,
                "vector": d.embedding,
                "idx": d.idx,
            }
            for d in documents
        ]

        if self._table is None:
            self._table = self._db.create_table(self.TABLE_NAME, data=rows)
        else:
            self._table.add(rows)

    def search(self, query_embedding: list[float], limit: int = 10) -> SearchResponse:
        if self._table is None:
            return SearchResponse()

        results = (
            self._table
            .search(query_embedding)
            .limit(limit)
            .to_list()
        )

        matches = [
            SearchResult(
                document=Document(
                    text=r["text"],
                    embedding=r["vector"],
                    idx=r["idx"],
                ),
                score=r["_distance"],
            )
            for r in results
        ]
        return SearchResponse(matches=matches)

    def num_docs(self) -> int:
        if self._table is None:
            return 0
        return self._table.count_rows()

    # -- info ------------------------------------------------------------------

    def info(self) -> CandidateInfo:
        try:
            from importlib.metadata import version
            ver = version("lancedb")
        except Exception:
            ver = "unknown"
        return CandidateInfo(
            name="LanceDB",
            version=ver,
            deployment="in-process",
            license="Apache-2.0",
            notes="Serverless, columnar (Lance format). Zero-config persistence.",
        )
