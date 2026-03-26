# Zvec (Alibaba)

| | |
|---|---|
| **Package** | [alibaba/zvec](https://github.com/alibaba/zvec) |
| **PyPI** | [`zvec`](https://pypi.org/project/zvec/) |
| **Latest version** | 0.2.1 |
| **Last release** | 2026-03-18 |
| **Python** | >=3.9 (classifiers: 3.10–3.14) |
| **Platforms** | Linux (x86_64, ARM64), macOS (ARM64) |
| **Deployment** | In-process only |
| **License** | Apache-2.0 |
| **Status** | Active, new — dev status: Alpha. C++ backend with Python bindings. |

## Overview

Zvec is an **embedded vector database** built on Alibaba's battle-tested
Proxima engine.  Marketed as "the SQLite of vector databases" — designed
specifically for on-device and edge computing use cases.

## Why consider it

- **Built for edge.** Explicitly designed for on-device RAG and resource-
  constrained environments.
- **High performance.** Claims >8,000 QPS on VectorDBBench Cohere 10M.
- **Auto-persistence.** Data persists to disk via `path=` parameter —
  no explicit save/load needed.
- **Crash recovery.** Built-in durability guarantees.
- **Hybrid search.** Supports dense + sparse vectors and filtered queries.
- **Low memory.** Configurable memory limits and mmap mode.

## Potential concerns

- Very new (Feb 2026) — small community (~750 GitHub stars).
- Python 3.10+ only (edge_node uses 3.10, so this should be fine).
- API surface is still evolving (v0.2.x).
- No built-in text/metadata storage in the current API — IDs are strings,
  text must be stored externally or via a future metadata feature.

## API mapping

| edge_node operation | Zvec equivalent |
|---|---|
| `HNSWVectorDB(workspace=...)` | `zvec.create_and_open(path=..., schema=...)` |
| `db.index(inputs=...)` | `collection.insert([zvec.Doc(...)])` |
| `db.search(inputs=..., limit=k)` | `collection.query(VectorQuery(...), topk=k)` |
| `db.num_docs()` | `collection.count()` |

## Install

```bash
pip install zvec
```
