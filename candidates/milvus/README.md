# Milvus / Milvus Lite

| | |
|---|---|
| **Package** | [milvus-io/pymilvus](https://github.com/milvus-io/pymilvus) |
| **Deployment** | In-process (Milvus Lite) **and** sidecar (full Milvus) |
| **License** | Apache-2.0 |
| **Status** | Active, CNCF / LF AI & Data (Zilliz) |

## Overview

Milvus is a cloud-native vector database designed for scalable similarity
search.  **Milvus Lite** is an embedded version that runs in-process via
`pymilvus` — same API, no server needed.

## Why consider it

- **Unified API.** Code written for Milvus Lite works unchanged against a
  full Milvus cluster — easy path from edge to cloud.
- **Rich feature set.** Multiple index types, hybrid search (vector + scalar
  filtering), partitions, TTL.
- **Milvus Lite is truly embedded.** Just `pip install pymilvus` and point to
  a local file.
- **Strong ecosystem.** LangChain, LlamaIndex, Haystack integrations.

## Potential concerns

- Milvus Lite is relatively new — fewer battle-tested deployments at the edge.
- Full Milvus server has heavy dependencies (etcd, MinIO/S3) — sidecar setup
  is more complex than Qdrant or Chroma.
- `pymilvus` package is large (~100 MB installed with Lite dependencies).

## API mapping

| edge_node operation | Milvus equivalent |
|---|---|
| `HNSWVectorDB(workspace=...)` | `MilvusClient(uri="path/milvus.db")` + `create_collection(...)` |
| `db.index(inputs=...)` | `client.insert(collection, data)` |
| `db.search(inputs=..., limit=k)` | `client.search(collection, data, limit=k)` |
| `db.num_docs()` | `client.get_collection_stats(collection)["row_count"]` |

## Sidecar setup

```bash
docker compose up -d milvus   # REST on 19530
```

## Install

```bash
pip install pymilvus          # includes Milvus Lite
pip install "pymilvus[model]" # with model utilities
```
