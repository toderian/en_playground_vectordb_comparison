"""
Milvus Lite / Milvus adapter.

Deployment: in-process (Milvus Lite) or sidecar (full Milvus server)
Milvus Lite runs embedded via pymilvus — same API as the full server.
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


class MilvusAdapter(BaseVectorDB):
    """Adapter for Milvus (Lite or server)."""

    COLLECTION_NAME = "documents"

    def __init__(
        self,
        workspace: str | Path,
        embedding_size: int = 1024,
        *,
        mode: str = "in-process",
        uri: str | None = None,
    ):
        super().__init__(workspace, embedding_size)
        self._mode = mode
        self._uri = uri
        self._client = None

    # -- lifecycle -------------------------------------------------------------

    def open(self) -> None:
        from pymilvus import MilvusClient

        if self._mode == "sidecar" and self._uri:
            self._client = MilvusClient(uri=self._uri)
        else:
            # Milvus Lite: embedded, file-based
            self.workspace.mkdir(parents=True, exist_ok=True)
            db_path = str(self.workspace / "milvus.db")
            self._client = MilvusClient(uri=db_path)

        if not self._client.has_collection(self.COLLECTION_NAME):
            from pymilvus import CollectionSchema, DataType, FieldSchema

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="idx", dtype=DataType.INT64),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_size),
            ]
            schema = CollectionSchema(fields=fields)
            index_params = self._client.prepare_index_params()
            index_params.add_index(field_name="vector", metric_type="COSINE")
            self._client.create_collection(
                collection_name=self.COLLECTION_NAME,
                schema=schema,
                index_params=index_params,
            )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # -- CRUD ------------------------------------------------------------------

    def index(self, documents: list[Document]) -> None:
        data = [
            {
                "id": d.idx,
                "text": d.text,
                "idx": d.idx,
                "vector": d.embedding,
            }
            for d in documents
        ]
        self._client.insert(collection_name=self.COLLECTION_NAME, data=data)

    def search(self, query_embedding: list[float], limit: int = 10) -> SearchResponse:
        raw = self._client.search(
            collection_name=self.COLLECTION_NAME,
            data=[query_embedding],
            limit=limit,
            output_fields=["text", "idx"],
        )

        if not raw or not raw[0]:
            return SearchResponse()

        matches = [
            SearchResult(
                document=Document(
                    text=hit["entity"].get("text", ""),
                    embedding=[],
                    idx=hit["entity"].get("idx", -1),
                ),
                score=hit["distance"],
            )
            for hit in raw[0]
        ]
        return SearchResponse(matches=matches)

    def num_docs(self) -> int:
        stats = self._client.get_collection_stats(self.COLLECTION_NAME)
        return stats.get("row_count", 0)

    # -- info ------------------------------------------------------------------

    def info(self) -> CandidateInfo:
        try:
            from importlib.metadata import version
            ver = version("pymilvus")
        except Exception:
            ver = "unknown"
        return CandidateInfo(
            name=f"Milvus ({self._mode})",
            version=ver,
            deployment=self._mode,
            license="Apache-2.0",
            notes="Milvus Lite for embedded, full Milvus for server. Same API for both.",
        )
