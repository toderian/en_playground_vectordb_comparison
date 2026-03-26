# ChromaDB

| | |
|---|---|
| **Package** | [chroma-core/chroma](https://github.com/chroma-core/chroma) |
| **PyPI** | [`chromadb`](https://pypi.org/project/chromadb/) |
| **Latest version** | 1.5.5 |
| **Last release** | 2024-12-20 |
| **Python** | >=3.9 |
| **Deployment** | In-process (embedded) **and** sidecar (client-server) |
| **License** | Apache-2.0 |
| **Status** | Active, VC-backed |

## Overview

ChromaDB is an open-source embedding database designed to be simple to use.
It can run fully embedded (in-process) or as a standalone server accessed via
HTTP.

## Why consider it

- **Dual-mode.** Same Python API for embedded and client-server — easy to
  start embedded on edge and move to a server later.
- **Built-in persistence.** `PersistentClient` stores data to disk
  automatically.
- **Wide adoption.** Large community, extensive LangChain / LlamaIndex
  integrations.
- **Simple API.** Collections, add, query — minimal boilerplate.

## Potential concerns

- Embedded mode uses SQLite + hnswlib under the hood — can be memory-hungry
  on large collections.
- Server mode adds an HTTP hop (latency).
- ID-based deduplication: IDs must be strings, which differs from the
  integer `idx` used in edge_node.

## API mapping

| edge_node operation | ChromaDB equivalent |
|---|---|
| `HNSWVectorDB(workspace=...)` | `chromadb.PersistentClient(path=...)` |
| `db.index(inputs=...)` | `collection.add(ids, embeddings, documents)` |
| `db.search(inputs=..., limit=k)` | `collection.query(query_embeddings, n_results=k)` |
| `db.num_docs()` | `collection.count()` |

## Sidecar setup

```bash
docker compose up -d chromadb   # starts on port 8000
```

## Install

```bash
pip install chromadb
```
