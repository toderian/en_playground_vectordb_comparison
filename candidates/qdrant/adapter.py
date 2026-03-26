"""
Qdrant adapter.

Deployment: in-process (in-memory / on-disk) or sidecar (client-server)
Qdrant supports an embedded local mode (no server) and full client-server.
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


class QdrantAdapter(BaseVectorDB):
    """Adapter for Qdrant."""

    COLLECTION_NAME = "documents"

    def __init__(
        self,
        workspace: str | Path,
        embedding_size: int = 1024,
        *,
        mode: str = "in-process",
        host: str = "localhost",
        port: int = 6333,
    ):
        super().__init__(workspace, embedding_size)
        self._mode = mode
        self._host = host
        self._port = port
        self._client = None

    # -- lifecycle -------------------------------------------------------------

    def open(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        if self._mode == "sidecar":
            self._client = QdrantClient(host=self._host, port=self._port)
        else:
            self.workspace.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self.workspace))

        collections = [c.name for c in self._client.get_collections().collections]
        if self.COLLECTION_NAME not in collections:
            self._client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.embedding_size,
                    distance=Distance.COSINE,
                ),
            )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # -- CRUD ------------------------------------------------------------------

    def index(self, documents: list[Document]) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=d.idx,
                vector=d.embedding,
                payload={"text": d.text, "idx": d.idx},
            )
            for d in documents
        ]
        self._client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
        )

    def search(self, query_embedding: list[float], limit: int = 10) -> SearchResponse:
        hits = self._client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_embedding,
            limit=limit,
        ).points

        matches = [
            SearchResult(
                document=Document(
                    text=h.payload.get("text", ""),
                    embedding=list(h.vector) if h.vector else [],
                    idx=h.payload.get("idx", -1),
                ),
                score=h.score,
            )
            for h in hits
        ]
        return SearchResponse(matches=matches)

    def num_docs(self) -> int:
        info = self._client.get_collection(self.COLLECTION_NAME)
        return info.points_count

    # -- info ------------------------------------------------------------------

    def info(self) -> CandidateInfo:
        try:
            from importlib.metadata import version
            ver = version("qdrant-client")
        except Exception:
            ver = "unknown"
        return CandidateInfo(
            name=f"Qdrant ({self._mode})",
            version=ver,
            deployment=self._mode,
            license="Apache-2.0",
            notes="Rust-based. Supports embedded (no server) and client-server modes.",
        )
