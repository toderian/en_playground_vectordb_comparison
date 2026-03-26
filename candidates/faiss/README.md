# FAISS

| | |
|---|---|
| **Package** | [facebookresearch/faiss](https://github.com/facebookresearch/faiss) |
| **Deployment** | In-process only |
| **License** | MIT |
| **Status** | Active (Meta / Facebook Research) |

## Overview

FAISS (Facebook AI Similarity Search) is a library for efficient similarity
search and clustering of dense vectors.  Written in C++ with Python bindings,
it is the de-facto standard for low-level vector search.

## Why consider it

- **Raw performance.** Highly optimised C++ with optional GPU support.
- **Mature and battle-tested.** Used in production at Meta scale.
- **Minimal dependencies.** `pip install faiss-cpu` — no server, no heavy
  framework.
- **Flexible index types.** Flat, IVF, HNSW, PQ, etc. — tune for your
  speed/accuracy/memory trade-off.

## Potential concerns

- **Low-level.** No built-in document storage — you must manage text/metadata
  separately (we use a JSON sidecar file in this adapter).
- **Manual persistence.** Must explicitly call `faiss.write_index()` /
  `faiss.read_index()`.
- **No server mode.** In-process only (there are third-party wrappers, but
  nothing official).
- **No filtering.** Metadata-based filtering requires a separate layer.

## API mapping

| edge_node operation | FAISS equivalent |
|---|---|
| `HNSWVectorDB(workspace=...)` | `faiss.IndexFlatIP(dim)` + manual file I/O |
| `db.index(inputs=...)` | `index.add(vectors)` + store metadata |
| `db.search(inputs=..., limit=k)` | `index.search(query, k)` → scores, indices |
| `db.num_docs()` | `index.ntotal` |

## Install

```bash
pip install faiss-cpu    # CPU-only
pip install faiss-gpu    # with CUDA support
```
