#!/usr/bin/env python3
"""
Run a single candidate benchmark in isolation and print JSON results to stdout.

This script is meant to be called by benchmark.py via subprocess, each time
in its own uv environment with only the candidate's extra installed.

Usage:
    uv run --extra faiss bench_one.py faiss --num-docs 1000 --search-iters 100
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from candidates.base import BaseVectorDB, Document

# ── Constants ────────────────────────────────────────────────────────────────

WORKSPACE_ROOT = Path(__file__).parent / "_bench_workspaces"
EMBEDDING_SIZE = 1024
SEARCH_LIMIT = 10

# ── Synthetic data generation ────────────────────────────────────────────────


def generate_documents(n: int, embedding_size: int = EMBEDDING_SIZE) -> list[Document]:
    rng = np.random.default_rng(42)
    docs = []
    for i in range(n):
        emb = rng.standard_normal(embedding_size).astype(np.float32)
        emb /= np.linalg.norm(emb)
        docs.append(
            Document(
                text=f"Document {i}: synthetic benchmark text for measuring indexing and search performance.",
                embedding=emb.tolist(),
                idx=i,
            )
        )
    return docs


def generate_queries(n: int, embedding_size: int = EMBEDDING_SIZE) -> list[list[float]]:
    rng = np.random.default_rng(123)
    queries = []
    for _ in range(n):
        emb = rng.standard_normal(embedding_size).astype(np.float32)
        emb /= np.linalg.norm(emb)
        queries.append(emb.tolist())
    return queries


# ── Benchmark harness ────────────────────────────────────────────────────────


def bench_candidate(
    db: BaseVectorDB,
    docs: list[Document],
    queries: list[list[float]],
    batch_size: int = 200,
) -> dict:
    info = db.info()
    workspace = db.workspace
    if workspace.exists():
        shutil.rmtree(workspace)

    results: dict = {
        "candidate_name": info.name,
        "candidate_version": info.version,
        "candidate_deployment": info.deployment,
        "candidate_license": info.license,
        "candidate_notes": info.notes,
    }

    # ── Index ────────────────────────────────────────────────────────────
    db.open()

    t0 = time.perf_counter()
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        db.index(batch)
    index_time = time.perf_counter() - t0

    results["index_time_s"] = round(index_time, 4)
    results["index_docs_per_s"] = round(len(docs) / index_time, 1)
    results["num_docs"] = db.num_docs()

    # ── Search ───────────────────────────────────────────────────────────
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        db.search(q, limit=SEARCH_LIMIT)
        latencies.append(time.perf_counter() - t0)

    results["search_avg_ms"] = round(statistics.mean(latencies) * 1000, 3)
    results["search_p50_ms"] = round(sorted(latencies)[len(latencies) // 2] * 1000, 3)
    results["search_p99_ms"] = round(sorted(latencies)[int(len(latencies) * 0.99)] * 1000, 3)
    results["search_queries_per_s"] = round(len(queries) / sum(latencies), 1)

    # ── Persistence (close & reopen) ─────────────────────────────────────
    db.close()
    t0 = time.perf_counter()
    db.open()
    reopen_time = time.perf_counter() - t0
    results["reopen_time_s"] = round(reopen_time, 4)
    results["docs_after_reopen"] = db.num_docs()

    # ── Workspace size ───────────────────────────────────────────────────
    total_bytes = sum(f.stat().st_size for f in workspace.rglob("*") if f.is_file())
    results["disk_mb"] = round(total_bytes / (1024 * 1024), 2)

    db.close()
    return results


# ── Candidate factories ──────────────────────────────────────────────────────

CANDIDATE_FACTORIES = {
    "baseline": lambda: _make("candidates.vectordb_baseline", "VectorDBAdapter", "vectordb"),
    "lancedb": lambda: _make("candidates.lancedb", "LanceDBAdapter", "lancedb"),
    "chromadb": lambda: _make("candidates.chromadb", "ChromaDBAdapter", "chromadb_emb", mode="in-process"),
    "chromadb-sidecar": lambda: _make("candidates.chromadb", "ChromaDBAdapter", "chromadb_sc", mode="sidecar"),
    "qdrant": lambda: _make("candidates.qdrant", "QdrantAdapter", "qdrant_emb", mode="in-process"),
    "qdrant-sidecar": lambda: _make("candidates.qdrant", "QdrantAdapter", "qdrant_sc", mode="sidecar"),
    "faiss": lambda: _make("candidates.faiss", "FAISSAdapter", "faiss"),
    "milvus": lambda: _make("candidates.milvus", "MilvusAdapter", "milvus_lite", mode="in-process"),
    "milvus-sidecar": lambda: _make("candidates.milvus", "MilvusAdapter", "milvus_sc", mode="sidecar", uri="http://localhost:19530"),
    "zvec": lambda: _make("candidates.zvec", "ZvecAdapter", "zvec"),
    "usearch": lambda: _make("candidates.usearch", "USearchAdapter", "usearch"),
}


def _make(module: str, cls_name: str, workspace_name: str, **kwargs):
    import importlib
    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)
    return cls(workspace=WORKSPACE_ROOT / workspace_name, embedding_size=EMBEDDING_SIZE, **kwargs)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Run a single candidate benchmark")
    parser.add_argument("candidate", choices=list(CANDIDATE_FACTORIES.keys()),
                        help="Candidate to benchmark")
    parser.add_argument("--num-docs", type=int, default=1000)
    parser.add_argument("--search-iters", type=int, default=100)
    args = parser.parse_args()

    docs = generate_documents(args.num_docs)
    queries = generate_queries(args.search_iters)

    db = CANDIDATE_FACTORIES[args.candidate]()
    result = bench_candidate(db, docs, queries)

    # Output JSON to stdout (the orchestrator reads this)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
