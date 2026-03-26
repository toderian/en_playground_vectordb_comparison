"""
ChromaDB adapter.

Deployment: in-process (embedded) or sidecar (client-server)
ChromaDB supports both modes. This adapter handles both via a flag.
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


class ChromaDBAdapter(BaseVectorDB):
    """Adapter for ChromaDB."""

    COLLECTION_NAME = "documents"

    def __init__(
        self,
        workspace: str | Path,
        embedding_size: int = 1024,
        *,
        mode: str = "in-process",
        host: str = "localhost",
        port: int = 8000,
    ):
        super().__init__(workspace, embedding_size)
        self._mode = mode
        self._host = host
        self._port = port
        self._client = None
        self._collection = None

    # -- lifecycle -------------------------------------------------------------

    def open(self) -> None:
        import chromadb

        if self._mode == "sidecar":
            self._client = chromadb.HttpClient(host=self._host, port=self._port)
        else:
            self.workspace.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.workspace))

        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def close(self) -> None:
        self._client = None
        self._collection = None

    # -- CRUD ------------------------------------------------------------------

    def index(self, documents: list[Document]) -> None:
        ids = [str(d.idx) for d in documents]
        embeddings = [d.embedding for d in documents]
        docs = [d.text for d in documents]
        metadatas = [{"idx": d.idx} for d in documents]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=docs,
            metadatas=metadatas,
        )

    def search(self, query_embedding: list[float], limit: int = 10) -> SearchResponse:
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, self._collection.count() or 1),
            include=["documents", "embeddings", "metadatas", "distances"],
        )

        if not raw["ids"] or not raw["ids"][0]:
            return SearchResponse()

        matches = []
        for i, doc_id in enumerate(raw["ids"][0]):
            emb = raw["embeddings"][0][i] if raw.get("embeddings") else []
            matches.append(
                SearchResult(
                    document=Document(
                        text=raw["documents"][0][i],
                        embedding=emb,
                        idx=raw["metadatas"][0][i].get("idx", -1),
                    ),
                    score=raw["distances"][0][i],
                )
            )
        return SearchResponse(matches=matches)

    def num_docs(self) -> int:
        return self._collection.count()

    # -- info ------------------------------------------------------------------

    def info(self) -> CandidateInfo:
        try:
            from importlib.metadata import version
            ver = version("chromadb")
        except Exception:
            ver = "unknown"
        return CandidateInfo(
            name=f"ChromaDB ({self._mode})",
            version=ver,
            deployment=self._mode,
            license="Apache-2.0",
            notes="Supports both embedded and client-server modes. Built-in persistence.",
        )
