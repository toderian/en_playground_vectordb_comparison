#!/usr/bin/env python3
"""
Vector DB benchmark suite.

Runs a standardised workload against every candidate adapter and produces a
comparison table.  Both in-process and sidecar candidates are tested (sidecar
only when their containers are reachable).

Usage:
    python benchmark.py                       # run all available candidates
    python benchmark.py --only lancedb faiss  # run specific candidates
    python benchmark.py --sidecar             # also test sidecar (docker) candidates
    python benchmark.py --num-docs 5000       # override dataset size
"""

from __future__ import annotations

import argparse
import gc
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import psutil
from tabulate import tabulate

from candidates.base import BaseVectorDB, CandidateInfo, Document

# ── Constants ────────────────────────────────────────────────────────────────

WORKSPACE_ROOT = Path(__file__).parent / "_bench_workspaces"
EMBEDDING_SIZE = 1024
DEFAULT_NUM_DOCS = 1000
SEARCH_ITERATIONS = 100
SEARCH_LIMIT = 10

# ── Synthetic data generation ────────────────────────────────────────────────


def generate_documents(n: int, embedding_size: int = EMBEDDING_SIZE) -> list[Document]:
    """Create *n* documents with random embeddings (normalised)."""
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
    """Create *n* random query embeddings (normalised)."""
    rng = np.random.default_rng(123)
    queries = []
    for _ in range(n):
        emb = rng.standard_normal(embedding_size).astype(np.float32)
        emb /= np.linalg.norm(emb)
        queries.append(emb.tolist())
    return queries


# ── Benchmark harness ────────────────────────────────────────────────────────


def measure_memory(func: Callable) -> tuple[float, float]:
    """Run *func* and return (result_of_func, peak_rss_delta_mb)."""
    gc.collect()
    before = psutil.Process().memory_info().rss
    result = func()
    after = psutil.Process().memory_info().rss
    delta_mb = (after - before) / (1024 * 1024)
    return result, max(0.0, delta_mb)


def bench_candidate(
    db: BaseVectorDB,
    docs: list[Document],
    queries: list[list[float]],
    batch_size: int = 200,
) -> dict:
    """Run the full benchmark against one candidate and return metrics."""
    info = db.info()
    workspace = WORKSPACE_ROOT / info.name.replace(" ", "_").replace("/", "_")
    if workspace.exists():
        shutil.rmtree(workspace)
    db.workspace = workspace

    results: dict = {"candidate": info}

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
        resp = db.search(q, limit=SEARCH_LIMIT)
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


# ── Candidate registry ───────────────────────────────────────────────────────


def get_inprocess_candidates() -> list[tuple[str, Callable[[], BaseVectorDB]]]:
    """Return (name, factory) for all in-process candidates."""
    candidates = []

    def _try(name: str, factory: Callable):
        try:
            factory()  # import check
            candidates.append((name, factory))
        except ImportError as e:
            print(f"  SKIP {name}: {e}")

    _try("vectordb (baseline)", lambda: __import_vectordb())
    _try("lancedb", lambda: __import_lancedb())
    _try("chromadb (embedded)", lambda: __import_chromadb_embedded())
    _try("qdrant (embedded)", lambda: __import_qdrant_embedded())
    _try("faiss", lambda: __import_faiss())
    _try("milvus-lite", lambda: __import_milvus_lite())
    _try("zvec", lambda: __import_zvec())
    _try("usearch", lambda: __import_usearch())

    return candidates


def get_sidecar_candidates() -> list[tuple[str, Callable[[], BaseVectorDB]]]:
    """Return (name, factory) for sidecar candidates (require docker containers)."""
    candidates = []

    def _try(name: str, factory: Callable):
        try:
            db = factory()
            db.open()
            db.close()
            candidates.append((name, factory))
        except Exception as e:
            print(f"  SKIP {name} (not reachable): {e}")

    _try("chromadb (sidecar)", lambda: __import_chromadb_sidecar())
    _try("qdrant (sidecar)", lambda: __import_qdrant_sidecar())
    _try("milvus (sidecar)", lambda: __import_milvus_sidecar())

    return candidates


# ── Factory functions ────────────────────────────────────────────────────────

def __import_vectordb():
    from candidates.vectordb_baseline import VectorDBAdapter
    return VectorDBAdapter(workspace=WORKSPACE_ROOT / "vectordb", embedding_size=EMBEDDING_SIZE)

def __import_lancedb():
    from candidates.lancedb import LanceDBAdapter
    return LanceDBAdapter(workspace=WORKSPACE_ROOT / "lancedb", embedding_size=EMBEDDING_SIZE)

def __import_chromadb_embedded():
    from candidates.chromadb import ChromaDBAdapter
    return ChromaDBAdapter(workspace=WORKSPACE_ROOT / "chromadb_emb", embedding_size=EMBEDDING_SIZE, mode="in-process")

def __import_chromadb_sidecar():
    from candidates.chromadb import ChromaDBAdapter
    return ChromaDBAdapter(workspace=WORKSPACE_ROOT / "chromadb_sc", embedding_size=EMBEDDING_SIZE, mode="sidecar")

def __import_qdrant_embedded():
    from candidates.qdrant import QdrantAdapter
    return QdrantAdapter(workspace=WORKSPACE_ROOT / "qdrant_emb", embedding_size=EMBEDDING_SIZE, mode="in-process")

def __import_qdrant_sidecar():
    from candidates.qdrant import QdrantAdapter
    return QdrantAdapter(workspace=WORKSPACE_ROOT / "qdrant_sc", embedding_size=EMBEDDING_SIZE, mode="sidecar")

def __import_faiss():
    from candidates.faiss import FAISSAdapter
    return FAISSAdapter(workspace=WORKSPACE_ROOT / "faiss", embedding_size=EMBEDDING_SIZE)

def __import_milvus_lite():
    from candidates.milvus import MilvusAdapter
    return MilvusAdapter(workspace=WORKSPACE_ROOT / "milvus_lite", embedding_size=EMBEDDING_SIZE, mode="in-process")

def __import_milvus_sidecar():
    from candidates.milvus import MilvusAdapter
    return MilvusAdapter(workspace=WORKSPACE_ROOT / "milvus_sc", embedding_size=EMBEDDING_SIZE, mode="sidecar", uri="http://localhost:19530")

def __import_zvec():
    from candidates.zvec import ZvecAdapter
    return ZvecAdapter(workspace=WORKSPACE_ROOT / "zvec", embedding_size=EMBEDDING_SIZE)

def __import_usearch():
    from candidates.usearch import USearchAdapter
    return USearchAdapter(workspace=WORKSPACE_ROOT / "usearch", embedding_size=EMBEDDING_SIZE)


# ── Main ─────────────────────────────────────────────────────────────────────

CANDIDATE_ALIASES = {
    "vectordb": "vectordb (baseline)",
    "lancedb": "lancedb",
    "chromadb": "chromadb (embedded)",
    "qdrant": "qdrant (embedded)",
    "faiss": "faiss",
    "milvus": "milvus-lite",
    "zvec": "zvec",
    "usearch": "usearch",
}


def main():
    parser = argparse.ArgumentParser(description="VectorDB replacement benchmark")
    parser.add_argument("--num-docs", type=int, default=DEFAULT_NUM_DOCS, help="Number of documents to index")
    parser.add_argument("--search-iters", type=int, default=SEARCH_ITERATIONS, help="Number of search queries")
    parser.add_argument("--sidecar", action="store_true", help="Also benchmark sidecar (docker) candidates")
    parser.add_argument("--only", nargs="*", help="Only run these candidates (by alias)")
    parser.add_argument("--clean", action="store_true", help="Remove benchmark workspaces and exit")
    args = parser.parse_args()

    if args.clean:
        if WORKSPACE_ROOT.exists():
            shutil.rmtree(WORKSPACE_ROOT)
            print("Cleaned up benchmark workspaces.")
        return

    print(f"Generating {args.num_docs} synthetic documents (dim={EMBEDDING_SIZE})...")
    docs = generate_documents(args.num_docs)
    queries = generate_queries(args.search_iters)

    print("\nDiscovering in-process candidates...")
    candidates = get_inprocess_candidates()

    if args.sidecar:
        print("\nDiscovering sidecar candidates (need docker compose up)...")
        candidates.extend(get_sidecar_candidates())

    if args.only:
        allowed = set()
        for alias in args.only:
            alias_lower = alias.lower()
            if alias_lower in CANDIDATE_ALIASES:
                allowed.add(CANDIDATE_ALIASES[alias_lower])
            else:
                allowed.add(alias_lower)
        candidates = [(n, f) for n, f in candidates if n.lower() in allowed or n in allowed]

    if not candidates:
        print("No candidates available. Install at least one: pip install lancedb chromadb qdrant-client faiss-cpu pymilvus")
        sys.exit(1)

    print(f"\nRunning benchmarks ({len(candidates)} candidate(s))...\n")

    all_results = []
    for name, factory in candidates:
        print(f"  [{name}]")
        try:
            db = factory()
            result = bench_candidate(db, docs, queries)
            all_results.append(result)
            print(f"    index: {result['index_time_s']}s | search avg: {result['search_avg_ms']}ms | disk: {result['disk_mb']}MB")
        except Exception as e:
            print(f"    ERROR: {e}")

    # ── Report ───────────────────────────────────────────────────────────
    if not all_results:
        print("\nNo results to report.")
        return

    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90)

    headers = [
        "Candidate", "Deploy", "Index (s)", "Docs/s",
        "Search avg (ms)", "Search p99 (ms)", "QPS",
        "Reopen (s)", "Persist OK", "Disk (MB)",
    ]
    rows = []
    for r in all_results:
        info: CandidateInfo = r["candidate"]
        rows.append([
            info.name,
            info.deployment,
            r["index_time_s"],
            r["index_docs_per_s"],
            r["search_avg_ms"],
            r["search_p99_ms"],
            r["search_queries_per_s"],
            r["reopen_time_s"],
            "yes" if r["docs_after_reopen"] == r["num_docs"] else "NO",
            r["disk_mb"],
        ])

    print(tabulate(rows, headers=headers, tablefmt="github"))

    print(f"\nConfig: {args.num_docs} docs, {EMBEDDING_SIZE}-dim embeddings, "
          f"{args.search_iters} search queries, limit={SEARCH_LIMIT}")

    # ── Notes ────────────────────────────────────────────────────────────
    print("\nCandidate notes:")
    for r in all_results:
        info: CandidateInfo = r["candidate"]
        print(f"  {info.name} (v{info.version}) — {info.notes}")


if __name__ == "__main__":
    main()
