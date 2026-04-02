#!/usr/bin/env python3
"""
Extended benchmark orchestrator — runs all additional benchmark suites.

Suites: memory, scaling, incremental, recall, coldstart, deps, contexts

Usage:
    python benchmark_extended.py                          # all suites, all candidates
    python benchmark_extended.py --suite memory scaling   # specific suites
    python benchmark_extended.py --only faiss usearch     # specific candidates
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tabulate import tabulate

CANDIDATES = ["lancedb", "chromadb", "qdrant", "faiss", "milvus", "usearch"]
SUITES = ["memory", "scaling", "incremental", "recall", "coldstart", "deps", "contexts"]


def run_suite(extra: str, candidate_key: str, suite: str, timeout: int = 600) -> dict | None:
    cmd = [
        "uv", "run", "--extra", extra,
        "bench_one.py", candidate_key, "--suite", suite,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            err_lines = result.stderr.strip().splitlines()
            err_msg = err_lines[-1] if err_lines else "unknown error"
            print(f"      ERROR: {err_msg}")
            return None
        stdout_lines = result.stdout.strip().splitlines()
        json_line = stdout_lines[-1] if stdout_lines else ""
        return json.loads(json_line)
    except subprocess.TimeoutExpired:
        print(f"      ERROR: timed out ({timeout}s)")
        return None
    except (json.JSONDecodeError, IndexError) as e:
        print(f"      ERROR: parse failed: {e}")
        return None


# ── Report formatters ────────────────────────────────────────────────────────


def report_memory(results: list[dict]):
    print("\n" + "=" * 80)
    print("MEMORY FOOTPRINT (RSS delta in MB after indexing)")
    print("=" * 80)
    headers = ["Candidate", "1K docs", "5K docs", "10K docs"]
    rows = []
    for r in results:
        rows.append([
            r["candidate_name"],
            r.get("rss_delta_mb_1000", "—"),
            r.get("rss_delta_mb_5000", "—"),
            r.get("rss_delta_mb_10000", "—"),
        ])
    print(tabulate(rows, headers=headers, tablefmt="github"))


def report_scaling(results: list[dict]):
    print("\n" + "=" * 80)
    print("SCALING BEHAVIOR")
    print("=" * 80)

    for metric, label, unit in [
        ("index_docs_per_s", "Index throughput", "docs/s"),
        ("search_avg_ms", "Search latency", "ms"),
        ("qps", "Search QPS", "q/s"),
        ("disk_mb", "Disk usage", "MB"),
    ]:
        headers = ["Candidate", "1K", "5K", "10K", "50K"]
        rows = []
        for r in results:
            rows.append([
                r["candidate_name"],
                r.get(f"{metric}_1000", "—"),
                r.get(f"{metric}_5000", "—"),
                r.get(f"{metric}_10000", "—"),
                r.get(f"{metric}_50000", "—"),
            ])
        print(f"\n  {label} ({unit}):")
        print(tabulate(rows, headers=headers, tablefmt="github"))


def report_incremental(results: list[dict]):
    print("\n" + "=" * 80)
    print("INCREMENTAL INDEXING (5 batches of 1000 docs)")
    print("=" * 80)
    headers = [
        "Candidate",
        "Batch 1 (docs/s)", "Batch 2", "Batch 3", "Batch 4", "Batch 5",
        "Slowdown",
    ]
    rows = []
    for r in results:
        rows.append([
            r["candidate_name"],
            r.get("batch_1_docs_per_s", "—"),
            r.get("batch_2_docs_per_s", "—"),
            r.get("batch_3_docs_per_s", "—"),
            r.get("batch_4_docs_per_s", "—"),
            r.get("batch_5_docs_per_s", "—"),
            f"{r.get('slowdown_ratio', '—')}x",
        ])
    print(tabulate(rows, headers=headers, tablefmt="github"))


def report_recall(results: list[dict]):
    print("\n" + "=" * 80)
    print("SEARCH RECALL (vs brute-force ground truth)")
    print("=" * 80)
    headers = [
        "Candidate",
        "Recall@1 (1K)", "Recall@10 (1K)",
        "Recall@1 (10K)", "Recall@10 (10K)",
    ]
    rows = []
    for r in results:
        rows.append([
            r["candidate_name"],
            r.get("recall_at_1_1000", "—"),
            r.get("recall_at_10_1000", "—"),
            r.get("recall_at_1_10000", "—"),
            r.get("recall_at_10_10000", "—"),
        ])
    print(tabulate(rows, headers=headers, tablefmt="github"))


def report_coldstart(results: list[dict]):
    print("\n" + "=" * 80)
    print("COLD START (time to open + first query)")
    print("=" * 80)
    headers = [
        "Candidate",
        "Open empty (s)", "Open 5K docs (s)", "First search (ms)",
    ]
    rows = []
    for r in results:
        rows.append([
            r["candidate_name"],
            r.get("open_empty_s", "—"),
            r.get("open_5k_docs_s", "—"),
            r.get("first_search_ms", "—"),
        ])
    print(tabulate(rows, headers=headers, tablefmt="github"))


def report_deps(results: list[dict]):
    print("\n" + "=" * 80)
    print("DEPENDENCY FOOTPRINT (candidate dependency tree only)")
    print("=" * 80)
    headers = ["Candidate", "Transitive deps", "Deps size (MB)"]
    rows = []
    for r in results:
        rows.append([
            r["candidate_name"],
            r.get("candidate_dep_count", "—"),
            r.get("candidate_deps_mb", "—"),
        ])
    print(tabulate(rows, headers=headers, tablefmt="github"))


def report_contexts(results: list[dict]):
    print("\n" + "=" * 80)
    print("CONCURRENT CONTEXTS (500 docs each)")
    print("=" * 80)
    headers = [
        "Candidate",
        "1 ctx (MB)", "5 ctx (MB)", "10 ctx (MB)", "20 ctx (MB)",
        "Search 1ctx (ms)", "Search 20ctx (ms)",
    ]
    rows = []
    for r in results:
        rows.append([
            r["candidate_name"],
            r.get("rss_delta_mb_1ctx", "—"),
            r.get("rss_delta_mb_5ctx", "—"),
            r.get("rss_delta_mb_10ctx", "—"),
            r.get("rss_delta_mb_20ctx", "—"),
            r.get("search_avg_ms_1ctx", "—"),
            r.get("search_avg_ms_20ctx", "—"),
        ])
    print(tabulate(rows, headers=headers, tablefmt="github"))


REPORT_FNS = {
    "memory": report_memory,
    "scaling": report_scaling,
    "incremental": report_incremental,
    "recall": report_recall,
    "coldstart": report_coldstart,
    "deps": report_deps,
    "contexts": report_contexts,
}


def main():
    parser = argparse.ArgumentParser(description="Extended VectorDB benchmarks")
    parser.add_argument("--suite", nargs="*", choices=SUITES,
                        help="Suites to run (default: all)")
    parser.add_argument("--only", nargs="*",
                        help="Only run these candidates")
    args = parser.parse_args()

    suites = args.suite or SUITES
    candidates = args.only or CANDIDATES
    candidates = [c for c in candidates if c in CANDIDATES]

    if not candidates:
        print(f"No valid candidates. Available: {', '.join(CANDIDATES)}")
        sys.exit(1)

    print(f"Running {len(suites)} suite(s) for {len(candidates)} candidate(s)\n")

    all_results = {}

    for suite in suites:
        print(f"\n{'─' * 60}")
        print(f"Suite: {suite}")
        print(f"{'─' * 60}")

        suite_results = []
        for candidate in candidates:
            print(f"  [{candidate}]", end=" ", flush=True)
            # scaling and recall need longer timeout
            timeout = 600 if suite in ("scaling", "recall", "contexts") else 300
            result = run_suite(candidate, candidate, suite, timeout=timeout)
            if result:
                suite_results.append(result)
                print("OK")
            else:
                print("")  # newline after ERROR

        if suite_results:
            REPORT_FNS[suite](suite_results)
            all_results[suite] = suite_results

    # Save all results
    results_file = Path(__file__).parent / "benchmark_extended_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {results_file}")


if __name__ == "__main__":
    main()
