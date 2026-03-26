# VectorDB Replacement — Comparison Playground

Edge Node currently uses [jina-ai/vectordb](https://github.com/jina-ai/vectordb)
for document embedding storage and semantic search.  The package is **unmaintained**
(last release 2023, repo archived).  This repo evaluates replacement candidates.

See [CURRENT_USAGE.md](CURRENT_USAGE.md) for a detailed breakdown of what
edge_node actually uses and the requirements any replacement must meet.

## Candidates

Each candidate lives in its own folder under `candidates/` with a dedicated
README, adapter implementation, and notes.

| Candidate | Deployment modes | Folder |
|---|---|---|
| **vectordb** (baseline) | in-process | [`candidates/vectordb_baseline/`](candidates/vectordb_baseline/) |
| **LanceDB** | in-process | [`candidates/lancedb/`](candidates/lancedb/) |
| **ChromaDB** | in-process, sidecar | [`candidates/chromadb/`](candidates/chromadb/) |
| **Qdrant** | in-process, sidecar | [`candidates/qdrant/`](candidates/qdrant/) |
| **FAISS** | in-process | [`candidates/faiss/`](candidates/faiss/) |
| **Milvus / Milvus Lite** | in-process, sidecar | [`candidates/milvus/`](candidates/milvus/) |

**In-process** = embedded, runs inside the edge node process, no extra
infrastructure.

**Sidecar** = runs as a separate container alongside the edge node, accessed
via HTTP/gRPC.

## Quick start

```bash
# 1. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies (installs all candidates)
pip install -r requirements.txt

# 3. Run benchmarks (in-process candidates only)
python benchmark.py

# 4. Run with sidecar candidates too (requires Docker)
docker compose up -d
python benchmark.py --sidecar

# 5. Clean up benchmark data
python benchmark.py --clean
docker compose down -v
```

## Benchmark options

```
python benchmark.py [OPTIONS]

  --num-docs N        Number of documents to index (default: 1000)
  --search-iters N    Number of search queries to run (default: 100)
  --sidecar           Also benchmark sidecar (Docker) candidates
  --only NAME [NAME]  Only run specific candidates (e.g., --only lancedb faiss)
  --clean             Remove benchmark workspace data and exit
```

## What the benchmark measures

| Metric | Description |
|---|---|
| Index time | Time to insert all documents (batch) |
| Docs/s | Indexing throughput |
| Search avg/p50/p99 | Search latency distribution |
| QPS | Search queries per second |
| Reopen time | Time to close and reopen the database |
| Persist OK | Whether documents survive a close/reopen cycle |
| Disk MB | On-disk storage size after indexing |

## Project structure

```
.
├── README.md               # This file
├── CURRENT_USAGE.md        # What edge_node uses today + replacement requirements
├── requirements.txt        # All dependencies
├── benchmark.py            # Benchmark harness
├── docker-compose.yml      # Sidecar containers (Chroma, Qdrant, Milvus)
└── candidates/
    ├── base.py             # Common interface (BaseVectorDB)
    ├── vectordb_baseline/  # Current implementation (reference)
    ├── lancedb/            # LanceDB candidate
    ├── chromadb/           # ChromaDB candidate
    ├── qdrant/             # Qdrant candidate
    ├── faiss/              # FAISS candidate
    └── milvus/             # Milvus / Milvus Lite candidate
```

## Adding a new candidate

1. Create a folder `candidates/<name>/`
2. Implement `BaseVectorDB` in `candidates/<name>/adapter.py`
3. Add `__init__.py` re-exporting the adapter class
4. Add a `README.md` with overview, pros/cons, and API mapping
5. Register it in `benchmark.py` (add a factory function and entry in the
   candidate lists)
