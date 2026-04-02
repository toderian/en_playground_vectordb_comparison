#!/usr/bin/env python3
"""
Run a single candidate benchmark in isolation and print JSON results to stdout.

This script is meant to be called by benchmark.py / benchmark_extended.py
via subprocess, each time in its own uv environment with only the
candidate's extra installed.

Usage:
    uv run --extra faiss bench_one.py faiss
    uv run --extra faiss bench_one.py faiss --suite memory
    uv run --extra faiss bench_one.py faiss --suite scaling
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import psutil

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


def _fresh_db(candidate_key: str, workspace_suffix: str = "") -> BaseVectorDB:
    """Create a fresh adapter, cleaning any leftover workspace."""
    db = CANDIDATE_FACTORIES[candidate_key]()
    if workspace_suffix:
        db.workspace = db.workspace.parent / (db.workspace.name + workspace_suffix)
    if db.workspace.exists():
        shutil.rmtree(db.workspace)
    return db


def _candidate_info(candidate_key: str) -> dict:
    """Return candidate metadata as a flat dict."""
    db = CANDIDATE_FACTORIES[candidate_key]()
    info = db.info()
    return {
        "candidate_name": info.name,
        "candidate_version": info.version,
        "candidate_deployment": info.deployment,
        "candidate_license": info.license,
        "candidate_notes": info.notes,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SUITE: standard — the original benchmark
# ═════════════════════════════════════════════════════════════════════════════


def suite_standard(candidate_key: str, num_docs: int, search_iters: int) -> dict:
    docs = generate_documents(num_docs)
    queries = generate_queries(search_iters)

    db = _fresh_db(candidate_key)
    info = _candidate_info(candidate_key)
    workspace = db.workspace

    results = {**info}

    # Index
    db.open()
    t0 = time.perf_counter()
    for start in range(0, len(docs), 200):
        db.index(docs[start : start + 200])
    index_time = time.perf_counter() - t0

    results["index_time_s"] = round(index_time, 4)
    results["index_docs_per_s"] = round(len(docs) / index_time, 1)
    results["num_docs"] = db.num_docs()

    # Search
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        db.search(q, limit=SEARCH_LIMIT)
        latencies.append(time.perf_counter() - t0)

    results["search_avg_ms"] = round(statistics.mean(latencies) * 1000, 3)
    results["search_p50_ms"] = round(sorted(latencies)[len(latencies) // 2] * 1000, 3)
    results["search_p99_ms"] = round(sorted(latencies)[int(len(latencies) * 0.99)] * 1000, 3)
    results["search_queries_per_s"] = round(len(queries) / sum(latencies), 1)

    # Persistence
    db.close()
    t0 = time.perf_counter()
    db.open()
    reopen_time = time.perf_counter() - t0
    results["reopen_time_s"] = round(reopen_time, 4)
    results["docs_after_reopen"] = db.num_docs()

    # Disk
    total_bytes = sum(f.stat().st_size for f in workspace.rglob("*") if f.is_file())
    results["disk_mb"] = round(total_bytes / (1024 * 1024), 2)

    db.close()
    return results


# ═════════════════════════════════════════════════════════════════════════════
# SUITE: memory — RSS footprint after indexing
# ═════════════════════════════════════════════════════════════════════════════


def suite_memory(candidate_key: str, **_kw) -> dict:
    results = _candidate_info(candidate_key)
    process = psutil.Process()

    for n in [1000, 5000, 10000]:
        docs = generate_documents(n)
        db = _fresh_db(candidate_key, workspace_suffix=f"_mem_{n}")

        gc.collect()
        rss_before = process.memory_info().rss

        db.open()
        for start in range(0, len(docs), 200):
            db.index(docs[start : start + 200])

        gc.collect()
        rss_after_index = process.memory_info().rss

        # Run some searches to measure search memory
        queries = generate_queries(50)
        for q in queries:
            db.search(q, limit=SEARCH_LIMIT)

        gc.collect()
        rss_after_search = process.memory_info().rss

        results[f"rss_before_{n}"] = round(rss_before / (1024 * 1024), 1)
        results[f"rss_after_index_{n}"] = round(rss_after_index / (1024 * 1024), 1)
        results[f"rss_delta_mb_{n}"] = round((rss_after_index - rss_before) / (1024 * 1024), 1)
        results[f"rss_after_search_{n}"] = round(rss_after_search / (1024 * 1024), 1)

        db.close()

    return results


# ═════════════════════════════════════════════════════════════════════════════
# SUITE: scaling — performance at different dataset sizes
# ═════════════════════════════════════════════════════════════════════════════


def suite_scaling(candidate_key: str, **_kw) -> dict:
    results = _candidate_info(candidate_key)
    queries = generate_queries(100)

    for n in [1000, 5000, 10000, 50000]:
        docs = generate_documents(n)
        db = _fresh_db(candidate_key, workspace_suffix=f"_scale_{n}")

        db.open()

        # Index
        t0 = time.perf_counter()
        for start in range(0, len(docs), 500):
            db.index(docs[start : start + 500])
        index_time = time.perf_counter() - t0

        # Search
        latencies = []
        for q in queries:
            t0 = time.perf_counter()
            db.search(q, limit=SEARCH_LIMIT)
            latencies.append(time.perf_counter() - t0)

        # Disk
        workspace = db.workspace
        total_bytes = sum(f.stat().st_size for f in workspace.rglob("*") if f.is_file())

        results[f"index_time_s_{n}"] = round(index_time, 4)
        results[f"index_docs_per_s_{n}"] = round(n / index_time, 1)
        results[f"search_avg_ms_{n}"] = round(statistics.mean(latencies) * 1000, 3)
        results[f"search_p99_ms_{n}"] = round(sorted(latencies)[int(len(latencies) * 0.99)] * 1000, 3)
        results[f"qps_{n}"] = round(len(queries) / sum(latencies), 1)
        results[f"disk_mb_{n}"] = round(total_bytes / (1024 * 1024), 2)

        db.close()

    return results


# ═════════════════════════════════════════════════════════════════════════════
# SUITE: incremental — cost of adding batches to a growing DB
# ═════════════════════════════════════════════════════════════════════════════


def suite_incremental(candidate_key: str, **_kw) -> dict:
    results = _candidate_info(candidate_key)
    db = _fresh_db(candidate_key, workspace_suffix="_incr")
    db.open()

    all_docs = generate_documents(5000)
    batch_size = 1000
    batch_times = []

    for batch_num in range(5):
        start = batch_num * batch_size
        batch = all_docs[start : start + batch_size]

        t0 = time.perf_counter()
        db.index(batch)
        elapsed = time.perf_counter() - t0

        batch_times.append(elapsed)
        results[f"batch_{batch_num + 1}_time_s"] = round(elapsed, 4)
        results[f"batch_{batch_num + 1}_docs_per_s"] = round(batch_size / elapsed, 1)
        results[f"total_docs_after_batch_{batch_num + 1}"] = db.num_docs()

    # Search performance after all batches
    queries = generate_queries(100)
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        db.search(q, limit=SEARCH_LIMIT)
        latencies.append(time.perf_counter() - t0)

    results["search_avg_ms_after_5k"] = round(statistics.mean(latencies) * 1000, 3)
    results["slowdown_ratio"] = round(batch_times[-1] / batch_times[0], 2) if batch_times[0] > 0 else 0

    db.close()
    return results


# ═════════════════════════════════════════════════════════════════════════════
# SUITE: recall — accuracy vs brute-force ground truth
# ═════════════════════════════════════════════════════════════════════════════


def _brute_force_knn(docs: list[Document], query: list[float], k: int) -> list[int]:
    """Compute exact k-NN by brute force (cosine similarity)."""
    doc_matrix = np.array([d.embedding for d in docs], dtype=np.float32)
    q = np.array(query, dtype=np.float32)
    # Normalise
    doc_norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True)
    doc_matrix = doc_matrix / np.where(doc_norms == 0, 1, doc_norms)
    q = q / np.linalg.norm(q)
    # Cosine similarity
    sims = doc_matrix @ q
    top_k = np.argsort(-sims)[:k]
    return [docs[i].idx for i in top_k]


def suite_recall(candidate_key: str, **_kw) -> dict:
    results = _candidate_info(candidate_key)

    for n in [1000, 10000]:
        docs = generate_documents(n)
        queries = generate_queries(100)
        db = _fresh_db(candidate_key, workspace_suffix=f"_recall_{n}")

        db.open()
        for start in range(0, len(docs), 500):
            db.index(docs[start : start + 500])

        recalls_at_10 = []
        recalls_at_1 = []

        for q in queries:
            # Ground truth
            gt_10 = set(_brute_force_knn(docs, q, 10))
            gt_1 = set(_brute_force_knn(docs, q, 1))

            # Candidate results
            resp = db.search(q, limit=10)
            candidate_idxs = [m.document.idx for m in resp.matches]
            candidate_set_10 = set(candidate_idxs[:10])
            candidate_set_1 = set(candidate_idxs[:1])

            recalls_at_10.append(len(gt_10 & candidate_set_10) / len(gt_10))
            recalls_at_1.append(len(gt_1 & candidate_set_1) / len(gt_1))

        results[f"recall_at_10_{n}"] = round(statistics.mean(recalls_at_10), 4)
        results[f"recall_at_1_{n}"] = round(statistics.mean(recalls_at_1), 4)

        db.close()

    return results


# ═════════════════════════════════════════════════════════════════════════════
# SUITE: coldstart — import + open time
# ═════════════════════════════════════════════════════════════════════════════


def suite_coldstart(candidate_key: str, **_kw) -> dict:
    results = _candidate_info(candidate_key)
    db = _fresh_db(candidate_key, workspace_suffix="_cold")

    # Time to open an empty DB
    t0 = time.perf_counter()
    db.open()
    open_empty_time = time.perf_counter() - t0
    results["open_empty_s"] = round(open_empty_time, 4)

    # Index some docs, close, then time reopen
    docs = generate_documents(5000)
    for start in range(0, len(docs), 500):
        db.index(docs[start : start + 500])
    db.close()

    t0 = time.perf_counter()
    db.open()
    open_5k_time = time.perf_counter() - t0
    results["open_5k_docs_s"] = round(open_5k_time, 4)

    # Time to first search after reopen
    q = generate_queries(1)[0]
    t0 = time.perf_counter()
    db.search(q, limit=10)
    first_search = time.perf_counter() - t0
    results["first_search_ms"] = round(first_search * 1000, 3)

    db.close()
    return results


# ═════════════════════════════════════════════════════════════════════════════
# SUITE: deps — dependency footprint
# ═════════════════════════════════════════════════════════════════════════════


def suite_deps(candidate_key: str, **_kw) -> dict:
    results = _candidate_info(candidate_key)

    import importlib.metadata

    # Root packages for each candidate
    root_pkgs = {
        "baseline": ["vectordb", "docarray"],
        "lancedb": ["lancedb"],
        "chromadb": ["chromadb"],
        "qdrant": ["qdrant-client"],
        "faiss": ["faiss-cpu"],
        "milvus": ["pymilvus"],
        "usearch": ["usearch"],
    }

    roots = root_pkgs.get(candidate_key, [])

    # Walk the dependency tree from each root
    def _walk_deps(pkg_name: str, seen: set):
        norm = pkg_name.lower().replace("-", "_").replace(".", "_")
        if norm in seen:
            return
        try:
            dist = importlib.metadata.distribution(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            return
        seen.add(norm)
        for req_str in (dist.requires or []):
            # Skip extras-only deps (e.g. "foo ; extra == 'test'")
            if "extra ==" in req_str or "extra==" in req_str:
                continue
            dep_name = req_str.split(";")[0].split(">")[0].split("<")[0].split("=")[0].split("[")[0].split("!")[0].strip()
            if dep_name:
                _walk_deps(dep_name, seen)

    all_deps: set[str] = set()
    for root in roots:
        _walk_deps(root, all_deps)

    results["candidate_dep_count"] = len(all_deps)

    # Measure disk size of candidate's dependency tree
    total_size = 0
    for dist in importlib.metadata.distributions():
        dist_name = (dist.metadata["Name"] or "").lower().replace("-", "_").replace(".", "_")
        if dist_name in all_deps:
            # Sum the size of all files in this distribution
            if dist.files:
                for f in dist.files:
                    try:
                        full_path = dist.locate_file(f)
                        if full_path.exists():
                            total_size += full_path.stat().st_size
                    except Exception:
                        pass
    results["candidate_deps_mb"] = round(total_size / (1024 * 1024), 1)
    results["candidate_deps_list"] = sorted(all_deps)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# SUITE: contexts — overhead of multiple open contexts
# ═════════════════════════════════════════════════════════════════════════════


def suite_contexts(candidate_key: str, **_kw) -> dict:
    results = _candidate_info(candidate_key)
    process = psutil.Process()

    docs_per_ctx = 500
    docs = generate_documents(docs_per_ctx)
    queries = generate_queries(50)

    for num_ctx in [1, 5, 10, 20]:
        gc.collect()
        rss_before = process.memory_info().rss

        dbs = []
        for i in range(num_ctx):
            db = _fresh_db(candidate_key, workspace_suffix=f"_ctx{num_ctx}_{i}")
            db.open()
            db.index(docs)
            dbs.append(db)

        gc.collect()
        rss_after = process.memory_info().rss

        # Search across all contexts
        latencies = []
        for db in dbs:
            for q in queries:
                t0 = time.perf_counter()
                db.search(q, limit=SEARCH_LIMIT)
                latencies.append(time.perf_counter() - t0)

        results[f"rss_delta_mb_{num_ctx}ctx"] = round((rss_after - rss_before) / (1024 * 1024), 1)
        results[f"search_avg_ms_{num_ctx}ctx"] = round(statistics.mean(latencies) * 1000, 3)
        results[f"total_docs_{num_ctx}ctx"] = sum(db.num_docs() for db in dbs)

        for db in dbs:
            db.close()

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

SUITES = {
    "standard": suite_standard,
    "memory": suite_memory,
    "scaling": suite_scaling,
    "incremental": suite_incremental,
    "recall": suite_recall,
    "coldstart": suite_coldstart,
    "deps": suite_deps,
    "contexts": suite_contexts,
}


def main():
    parser = argparse.ArgumentParser(description="Run a single candidate benchmark")
    parser.add_argument("candidate", choices=list(CANDIDATE_FACTORIES.keys()),
                        help="Candidate to benchmark")
    parser.add_argument("--suite", choices=list(SUITES.keys()), default="standard",
                        help="Benchmark suite to run (default: standard)")
    parser.add_argument("--num-docs", type=int, default=1000)
    parser.add_argument("--search-iters", type=int, default=100)
    args = parser.parse_args()

    suite_fn = SUITES[args.suite]
    result = suite_fn(
        args.candidate,
        num_docs=args.num_docs,
        search_iters=args.search_iters,
    )

    print(json.dumps(result))


if __name__ == "__main__":
    main()
