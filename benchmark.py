#!/usr/bin/env python3
"""
Vector DB benchmark orchestrator.

Runs each candidate in its own isolated uv environment (separate process with
only that candidate's extra installed), collects JSON results, and prints a
combined comparison table.

Usage:
    python benchmark.py                       # run all in-process candidates
    python benchmark.py --only lancedb faiss  # run specific candidates
    python benchmark.py --sidecar             # also test sidecar (docker) candidates
    python benchmark.py --num-docs 5000       # override dataset size
    python benchmark.py --clean               # remove benchmark workspace data
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from tabulate import tabulate

# ── Constants ────────────────────────────────────────────────────────────────

WORKSPACE_ROOT = Path(__file__).parent / "_bench_workspaces"
EMBEDDING_SIZE = 1024
DEFAULT_NUM_DOCS = 1000
SEARCH_ITERATIONS = 100

# Map: candidate alias → (uv extra, bench_one.py key, extra --with flags)
# The uv extra is what gets passed to `uv run --extra <extra>`
# The optional with_deps list adds `--with <dep>` flags for compatibility pins
CANDIDATES: dict[str, tuple[str, str, list[str]]] = {
    "baseline":  ("baseline", "baseline", []),
    "lancedb":   ("lancedb",  "lancedb",  []),
    "chromadb":  ("chromadb", "chromadb",  []),
    "qdrant":    ("qdrant",   "qdrant",   []),
    "faiss":     ("faiss",    "faiss",    []),
    "milvus":    ("milvus",   "milvus",   []),
    "usearch":   ("usearch",  "usearch",  []),
}

SIDECAR_CANDIDATES: dict[str, tuple[str, str, list[str]]] = {
    "chromadb-sidecar": ("chromadb", "chromadb-sidecar", []),
    "qdrant-sidecar":   ("qdrant",   "qdrant-sidecar",  []),
    "milvus-sidecar":   ("milvus",   "milvus-sidecar",  []),
}


def run_candidate(
    extra: str,
    candidate_key: str,
    num_docs: int,
    search_iters: int,
    with_deps: list[str] | None = None,
) -> dict | None:
    """Spawn bench_one.py in an isolated uv environment and return parsed JSON results.

    Candidates with conflicting deps (via with_deps) are run last because
    ``uv run --with`` modifies the shared project venv. These candidates
    are sorted to the end of the run order by the orchestrator.
    """
    cmd = ["uv", "run", "--extra", extra]
    for dep in (with_deps or []):
        cmd.extend(["--with", dep])
    cmd.extend([
        "bench_one.py", candidate_key,
        "--num-docs", str(num_docs),
        "--search-iters", str(search_iters),
    ])
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            # Find the most useful error line from stderr
            err_lines = result.stderr.strip().splitlines()
            err_msg = err_lines[-1] if err_lines else "unknown error"
            print(f"    ERROR: {err_msg}")
            return None

        # bench_one.py prints JSON as the last line of stdout
        stdout_lines = result.stdout.strip().splitlines()
        json_line = stdout_lines[-1] if stdout_lines else ""
        return json.loads(json_line)

    except subprocess.TimeoutExpired:
        print("    ERROR: timed out (300s)")
        return None
    except (json.JSONDecodeError, IndexError) as e:
        print(f"    ERROR: failed to parse output: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="VectorDB replacement benchmark")
    parser.add_argument("--num-docs", type=int, default=DEFAULT_NUM_DOCS,
                        help="Number of documents to index")
    parser.add_argument("--search-iters", type=int, default=SEARCH_ITERATIONS,
                        help="Number of search queries")
    parser.add_argument("--sidecar", action="store_true",
                        help="Also benchmark sidecar (docker) candidates")
    parser.add_argument("--only", nargs="*",
                        help="Only run these candidates (by alias)")
    parser.add_argument("--clean", action="store_true",
                        help="Remove benchmark workspaces and exit")
    args = parser.parse_args()

    if args.clean:
        if WORKSPACE_ROOT.exists():
            shutil.rmtree(WORKSPACE_ROOT)
            print("Cleaned up benchmark workspaces.")
        return

    # Build the list of candidates to run
    to_run = dict(CANDIDATES)
    if args.sidecar:
        to_run.update(SIDECAR_CANDIDATES)

    if args.only:
        allowed = {a.lower() for a in args.only}
        to_run = {k: v for k, v in to_run.items() if k in allowed}

    if not to_run:
        print("No candidates selected. Available:", ", ".join(sorted(CANDIDATES)))
        sys.exit(1)

    print(f"Running benchmarks for {len(to_run)} candidate(s), "
          f"{args.num_docs} docs, {EMBEDDING_SIZE}-dim embeddings, "
          f"{args.search_iters} search queries\n")
    print("Each candidate runs in its own isolated uv environment.\n")

    # Sort so candidates with conflicting deps (with_deps) run last,
    # since --with can modify the shared venv
    run_order = sorted(to_run.items(), key=lambda kv: (len(kv[1][2]) > 0, kv[0]))

    all_results = []
    for alias, (extra, candidate_key, with_deps) in run_order:
        print(f"  [{alias}]")
        result = run_candidate(extra, candidate_key, args.num_docs, args.search_iters, with_deps)
        if result:
            all_results.append(result)
            print(f"    index: {result['index_time_s']}s | "
                  f"search avg: {result['search_avg_ms']}ms | "
                  f"disk: {result['disk_mb']}MB")

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
        rows.append([
            r["candidate_name"],
            r["candidate_deployment"],
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
          f"{args.search_iters} search queries, limit=10")

    # ── Notes ────────────────────────────────────────────────────────────
    print("\nCandidate notes:")
    for r in all_results:
        print(f"  {r['candidate_name']} (v{r['candidate_version']}) — {r['candidate_notes']}")

    # ── Save results to JSON ─────────────────────────────────────────────
    results_file = Path(__file__).parent / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
