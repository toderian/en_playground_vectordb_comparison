#!/usr/bin/env python3
"""
Run integration tests for each candidate in its own isolated uv environment.

Usage:
    python run_tests.py                    # test all candidates
    python run_tests.py --only faiss usearch  # test specific candidates
    python run_tests.py -v                 # verbose pytest output
"""

from __future__ import annotations

import argparse
import subprocess
import sys

CANDIDATES = ["lancedb", "chromadb", "qdrant", "faiss", "milvus", "usearch"]


def run_tests_for(candidate: str, verbose: bool = False) -> tuple[bool, str]:
    """Run pytest for one candidate in its own uv --extra environment."""
    cmd = [
        "uv", "run", "--extra", candidate,
        "pytest", "tests/", "-x",
        f"-k=not DocSplitter",  # DocSplitter tests don't need adapters
    ]
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr
    return result.returncode == 0, output


def main():
    parser = argparse.ArgumentParser(description="Run integration tests per candidate")
    parser.add_argument("--only", nargs="*", help="Only test these candidates")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    candidates = args.only or CANDIDATES
    candidates = [c for c in candidates if c in CANDIDATES]

    if not candidates:
        print(f"No valid candidates. Available: {', '.join(CANDIDATES)}")
        sys.exit(1)

    # Run DocSplitter tests once (no adapter needed)
    print("  [doc_splitter]")
    cmd = ["uv", "run", "pytest", "tests/test_integration.py::TestDocSplitter", "-q"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode == 0:
        print("    PASSED")
    else:
        print(f"    FAILED\n{result.stdout}\n{result.stderr}")

    print(f"\nRunning integration tests for {len(candidates)} candidate(s)...\n")

    results = {}
    for candidate in candidates:
        print(f"  [{candidate}]")
        passed, output = run_tests_for(candidate, args.verbose)
        results[candidate] = passed

        if passed:
            # Extract summary line
            lines = output.strip().splitlines()
            summary = next((l for l in reversed(lines) if "passed" in l), "PASSED")
            print(f"    {summary.strip()}")
        else:
            print(f"    FAILED")
            # Show relevant failure output
            for line in output.strip().splitlines():
                if "FAILED" in line or "ERROR" in line or "AssertionError" in line:
                    print(f"    {line.strip()}")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for candidate, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {candidate:20s} {status}")

    failed = [c for c, p in results.items() if not p]
    if failed:
        print(f"\n{len(failed)} candidate(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} candidate(s) passed!")


if __name__ == "__main__":
    main()
