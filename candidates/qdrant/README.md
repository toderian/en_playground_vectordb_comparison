# Qdrant

| | |
|---|---|
| **Package** | [qdrant/qdrant](https://github.com/qdrant/qdrant) |
| **PyPI** | [`qdrant-client`](https://pypi.org/project/qdrant-client/) |
| **Latest version** | 1.17.1 |
| **Last release** | 2026-03-13 |
| **Python** | >=3.10 (classifiers: 3.10–3.14) |
| **Deployment** | In-process (embedded) **and** sidecar (client-server) |
| **License** | Apache-2.0 |
| **Status** | Active, VC-backed (Qdrant Solutions GmbH) |

## Overview

Qdrant is a high-performance vector search engine written in **Rust**.  The
Python client supports both a local embedded mode (via `qdrant-client`
with on-disk storage) and full client-server via gRPC/REST.

## Why consider it

- **Rust core.** Excellent performance and memory safety.
- **Dual-mode.** `QdrantClient(path=...)` runs embedded; switch to
  `QdrantClient(host=...)` for server mode with zero API changes.
- **Rich filtering.** Payload-based filters during search (useful if we need
  metadata filtering later).
- **On-disk persistence.** Automatic, no manual save/load.
- **Actively maintained.** Frequent releases, strong community.

## Potential concerns

- Embedded mode ships a bundled Rust binary — adds ~50 MB to install.
- The embedded mode is labeled "local mode" and may not receive all
  server-mode features equally fast.
- Slightly more complex collection setup (must declare vector params).

## API mapping

| edge_node operation | Qdrant equivalent |
|---|---|
| `HNSWVectorDB(workspace=...)` | `QdrantClient(path=workspace)` + `create_collection(...)` |
| `db.index(inputs=...)` | `client.upsert(collection, points)` |
| `db.search(inputs=..., limit=k)` | `client.query_points(collection, query, limit=k)` |
| `db.num_docs()` | `client.get_collection(collection).points_count` |

## Sidecar setup

```bash
docker compose up -d qdrant   # REST on 6333, gRPC on 6334
```

## Install

```bash
pip install qdrant-client
```
