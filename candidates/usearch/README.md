# USearch (Unum Cloud)

| | |
|---|---|
| **Package** | [unum-cloud/USearch](https://github.com/unum-cloud/USearch) |
| **PyPI** | [`usearch`](https://pypi.org/project/usearch/) |
| **Latest version** | 2.23.0 |
| **Last release** | 2026-01-11 |
| **Python** | CPython 3.x (wheels: 3.9–3.13) |
| **Platforms** | Linux, macOS, Windows |
| **Deployment** | In-process only |
| **License** | Apache-2.0 |
| **Status** | Active — dev status: Production/Stable. C++ core, multi-language bindings. |

## Overview

USearch is a **single-file vector search engine** built as an improved
alternative to FAISS and hnswlib.  HNSW-based, with claims of significantly
better speed and lower memory usage than both.

## Why consider it

- **Faster than FAISS/hnswlib.** Benchmarks show improvements on both
  indexing and search.
- **mmap persistence.** `index.view("file")` memory-maps the index from
  disk — no need to load the full index into RAM.
- **Tiny footprint.** Single-file design, minimal dependencies.
- **Custom metrics.** Supports user-defined distance functions via Numba JIT.
- **Batch operations.** `add(keys, vectors, threads=N)` for parallel
  indexing.
- **Active development.** Frequent releases (v2.23+).

## Potential concerns

- Pure vector index — no built-in text/metadata storage (same trade-off as
  FAISS).  Adapter uses a JSON sidecar file.
- Newer and less widely deployed than FAISS.
- No server mode.

## API mapping

| edge_node operation | USearch equivalent |
|---|---|
| `HNSWVectorDB(workspace=...)` | `Index(ndim=..., metric='cos', dtype='f32')` |
| `db.index(inputs=...)` | `index.add(keys, vectors)` |
| `db.search(inputs=..., limit=k)` | `index.search(query, k)` → `.keys`, `.distances` |
| `db.num_docs()` | `len(index)` |

## Install

```bash
pip install usearch
```
