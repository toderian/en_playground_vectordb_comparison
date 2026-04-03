"""
Shared fixtures for integration tests.

Reproduces the edge_node environment: 1024-dim normalised embeddings,
overlapping document segments, multiple isolated contexts, and persistence.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from candidates.base import BaseVectorDB, Document

# ── Constants matching edge_node ─────────────────────────────────────────────

DOC_EMBEDDING_SIZE = 1024
MAX_SEGMENT_SIZE = 1000  # words
MAX_SEGMENT_OVERLAP = 50  # words
WORD_FIND_REGEX = (
    r"\b(?:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    r"|[a-zA-Z]+(?:'[a-z]+)?|[0-9]+(?:\.[0-9]+)?|[^\s\w])\b"
)


# ── DocSplitter — ported from edge_node ──────────────────────────────────────


class DocSplitter:
    """Port of edge_node's DocSplitter for realistic segmentation tests."""

    def __init__(
        self,
        max_segment_size: int = MAX_SEGMENT_SIZE,
        max_segment_overlap: int = MAX_SEGMENT_OVERLAP,
    ):
        self._max_segment_size = max_segment_size
        self._max_segment_overlap = max_segment_overlap

    def document_atomizing(self, document: str) -> list[str]:
        return re.findall(WORD_FIND_REGEX, document)

    def compute_best_overlap(self, text_size: int) -> int:
        if text_size <= self._max_segment_size:
            return 0
        best_overlap, best_last = 0, 0
        max_ovl = min(self._max_segment_overlap, self._max_segment_size - 1)
        min_ovl = int(max_ovl // 2)
        for overlap in range(min_ovl, max_ovl + 1):
            last = (text_size - self._max_segment_size) % (self._max_segment_size - overlap)
            if last > best_last:
                best_overlap, best_last = overlap, last
        return best_overlap

    def split_document(self, document: str) -> list[str]:
        words = self.document_atomizing(document)
        overlap = self.compute_best_overlap(len(words))
        step = max(1, self._max_segment_size - overlap)
        return [
            " ".join(words[i : i + self._max_segment_size])
            for i in range(0, max(1, len(words) - overlap), step)
        ]

    def split_documents(self, documents: list[str]) -> list[str]:
        segments = []
        for doc in documents:
            segments.extend(self.split_document(doc))
        return segments


# ── Embedding helpers ────────────────────────────────────────────────────────


def make_embedding(seed: int, dim: int = DOC_EMBEDDING_SIZE) -> list[float]:
    """Create a deterministic normalised embedding from a seed."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(dim).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb.tolist()


def make_similar_embedding(
    base: list[float], noise: float = 0.05, seed: int = 0
) -> list[float]:
    """Create an embedding close to *base* (high cosine similarity)."""
    rng = np.random.default_rng(seed)
    arr = np.array(base, dtype=np.float32)
    arr += rng.standard_normal(len(arr)).astype(np.float32) * noise
    arr /= np.linalg.norm(arr)
    return arr.tolist()


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_workspace(tmp_path):
    """Provide a temporary workspace directory, cleaned up after test."""
    ws = tmp_path / "test_workspace"
    ws.mkdir()
    yield ws
    if ws.exists():
        shutil.rmtree(ws)


@pytest.fixture()
def doc_splitter():
    return DocSplitter()


# ── Adapter registry ─────────────────────────────────────────────────────────

# Each entry: (id, module_path, class_name, extra kwargs)
ADAPTER_SPECS = [
    ("lancedb", "candidates.lancedb", "LanceDBAdapter", {}),
    ("chromadb", "candidates.chromadb", "ChromaDBAdapter", {"mode": "in-process"}),
    ("qdrant", "candidates.qdrant", "QdrantAdapter", {"mode": "in-process"}),
    ("faiss", "candidates.faiss", "FAISSAdapter", {}),
    ("milvus", "candidates.milvus", "MilvusAdapter", {"mode": "in-process"}),
    ("zvec", "candidates.zvec", "ZvecAdapter", {}),
    ("usearch", "candidates.usearch", "USearchAdapter", {}),
]


def _make_adapter(spec, workspace) -> BaseVectorDB:
    """Instantiate an adapter from its spec."""
    import importlib

    adapter_id, module_path, class_name, kwargs = spec
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(workspace=workspace, embedding_size=DOC_EMBEDDING_SIZE, **kwargs)


# Map adapter id → the actual third-party library to probe
_LIB_PROBES = {
    "lancedb": "lancedb",
    "chromadb": "chromadb",
    "zvec": "zvec",
    "qdrant": "qdrant_client",
    "faiss": "faiss",
    "milvus": "pymilvus",
    "usearch": "usearch",
}


def _adapter_available(spec) -> bool:
    """Check if the adapter's third-party library is actually installed."""
    try:
        import importlib

        lib = _LIB_PROBES.get(spec[0], spec[1])
        importlib.import_module(lib)
        return True
    except ImportError:
        return False


def pytest_generate_tests(metafunc):
    """Parametrise any test that requests the 'adapter_factory' fixture."""
    if "adapter_factory" in metafunc.fixturenames:
        available = [
            (spec[0], spec) for spec in ADAPTER_SPECS if _adapter_available(spec)
        ]
        if not available:
            pytest.skip("No adapters available")
        metafunc.parametrize(
            "adapter_factory",
            [spec for _, spec in available],
            ids=[name for name, _ in available],
            indirect=True,
        )


@pytest.fixture()
def adapter_factory(request, tmp_path):
    """Yield a factory function that creates a fresh adapter for each call.

    Usage in tests::
        db = adapter_factory()   # creates adapter with unique workspace
        db.open()
    """
    spec = request.param
    counter = [0]

    def factory(workspace=None):
        counter[0] += 1
        ws = workspace or (tmp_path / f"adapter_{spec[0]}_{counter[0]}")
        return _make_adapter(spec, ws)

    yield factory
