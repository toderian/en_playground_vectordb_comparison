# Candidates Summary

Side-by-side comparison of all vectordb replacement candidates evaluated in
this repo. For full details see each candidate's own
[README](vectordb_baseline/README.md).

## PyPI metadata

| Candidate | PyPI | Version | Last release | Python | Py 3.13 | License | Status |
|---|---|---|---|---|---|---|---|
| vectordb _(baseline)_ | [`vectordb`](https://pypi.org/project/vectordb/) | 0.0.21 | 2024-03-04 | 3.8–3.11 | :x: | Apache-2.0 | Unmaintained |
| LanceDB | [`lancedb`](https://pypi.org/project/lancedb/) | 0.30.1 | 2025-01-14 | 3.10–3.13 | :white_check_mark: | Apache-2.0 | Alpha |
| ChromaDB | [`chromadb`](https://pypi.org/project/chromadb/) | 1.5.5 | 2024-12-20 | >=3.9 | :white_check_mark: | Apache-2.0 | Active |
| Qdrant | [`qdrant-client`](https://pypi.org/project/qdrant-client/) | 1.17.1 | 2026-03-13 | 3.10–3.14 | :white_check_mark: | Apache-2.0 | Active |
| FAISS | [`faiss-cpu`](https://pypi.org/project/faiss-cpu/) | 1.13.2 | 2025-08-13 | 3.10–3.14 | :white_check_mark: | MIT + BSD-3 | Active |
| Milvus | [`pymilvus`](https://pypi.org/project/pymilvus/) | 2.6.10 | 2025-01-10 | 3.8–3.13 | :white_check_mark: | Apache-2.0 | Active |
| Zvec | [`zvec`](https://pypi.org/project/zvec/) | 0.2.1 | 2026-03-18 | 3.10–3.14 | :white_check_mark: | Apache-2.0 | Alpha |
| USearch | [`usearch`](https://pypi.org/project/usearch/) | 2.23.0 | 2026-01-11 | 3.9–3.13 | :white_check_mark: | Apache-2.0 | Production/Stable |

> The current baseline (`vectordb`) does not support Python 3.13 and has not
> been updated since March 2024. All replacement candidates support 3.13.

## Deployment modes

| Candidate | In-process | Sidecar (container) | Notes |
|---|---|---|---|
| vectordb _(baseline)_ | :white_check_mark: | :x: | hnswlib via docarray |
| LanceDB | :white_check_mark: | :x: | Embedded only, Lance columnar format |
| ChromaDB | :white_check_mark: | :white_check_mark: | `PersistentClient` / `HttpClient` — same API |
| Qdrant | :white_check_mark: | :white_check_mark: | `QdrantClient(path=)` / `QdrantClient(host=)` — same API |
| FAISS | :white_check_mark: | :x: | Pure library, no server mode |
| Milvus | :white_check_mark: | :white_check_mark: | Milvus Lite (file-based) / full Milvus server — same API |
| Zvec | :white_check_mark: | :x: | Embedded only, Proxima engine |
| USearch | :white_check_mark: | :x: | Pure library, no server mode |

## Feature comparison

| Feature | vectordb | LanceDB | ChromaDB | Qdrant | FAISS | Milvus | Zvec | USearch |
|---|---|---|---|---|---|---|---|---|
| **Text + vector storage** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: | :x: | :x: |
| **Auto-persistence** | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: | :white_check_mark: | :x: |
| **ANN index (HNSW etc.)** | :white_check_mark: | opt-in | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| **Metadata filtering** | :x: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: | :white_check_mark: | :x: |
| **Hybrid search** | :x: | :x: | :x: | :x: | :x: | :white_check_mark: | :white_check_mark: | :x: |
| **mmap / disk-backed** | :x: | :white_check_mark: | :x: | :white_check_mark: | :x: | :x: | :white_check_mark: | :white_check_mark: |
| **GPU support** | :x: | :x: | :x: | :x: | :white_check_mark: | :white_check_mark: | :x: | :x: |

- **Text + vector storage** — the database natively stores text/metadata alongside vectors (no sidecar JSON file needed).
- **Auto-persistence** — data survives process restart without an explicit `save()` call.
- **mmap / disk-backed** — can query data from disk without loading the full index into RAM.

## Pros & cons at a glance

| Candidate | Key strengths | Key concerns |
|---|---|---|
| **vectordb** _(baseline)_ | Current implementation, known behaviour | Unmaintained, no Py 3.13, heavy deps (jina) |
| **LanceDB** | Zero-config, disk-native, low memory, columnar | Young project (Alpha), brute-force by default |
| **ChromaDB** | Dual-mode, simple API, large community | Memory-hungry embedded mode (SQLite + hnswlib), string IDs only |
| **Qdrant** | Rust core, dual-mode, rich filtering, fast releases | ~50 MB install (bundled Rust binary), complex collection setup |
| **FAISS** | Raw performance, battle-tested at Meta scale, flexible indexes | Low-level (no text storage), manual persistence, no filtering |
| **Milvus** | Unified embedded-to-cloud API, rich features, strong ecosystem | Large install (~100 MB), heavy sidecar deps (etcd, MinIO) |
| **Zvec** | Edge-first design, high QPS, auto-persist, crash recovery | Very new (v0.2.x Alpha), small community, no text storage |
| **USearch** | Faster than FAISS, mmap from disk, tiny footprint, Production/Stable | No text storage, newer than FAISS, no server mode |

## Backing & community

| Candidate | Backed by | GitHub | Ecosystem integrations |
|---|---|---|---|
| vectordb | Jina AI | archived | docarray |
| LanceDB | LanceDB Inc. (VC-funded) | active | LangChain, LlamaIndex |
| ChromaDB | Chroma (VC-funded) | active | LangChain, LlamaIndex, many |
| Qdrant | Qdrant Solutions GmbH (VC-funded) | active | LangChain, LlamaIndex, Haystack |
| FAISS | Meta / Facebook Research | active | scikit-learn, many wrappers |
| Milvus | Zilliz / CNCF / LF AI & Data | active | LangChain, LlamaIndex, Haystack |
| Zvec | Alibaba | active (new) | minimal so far |
| USearch | Unum Cloud | active | standalone |

## Install

This project uses [uv](https://docs.astral.sh/uv/). Each candidate is an
optional dependency group — install only what you need:

```bash
uv run --extra all benchmark.py          # all candidates
uv run --extra lancedb benchmark.py      # just LanceDB
uv run --extra qdrant --extra faiss benchmark.py  # mix and match
```

Available extras: `baseline`, `lancedb`, `chromadb`, `qdrant`, `faiss`,
`milvus`, `zvec`, `usearch`, `all`.

<details>
<summary>Direct pip install (without uv)</summary>

```bash
pip install vectordb docarray[hnswlib]   # baseline
pip install lancedb                       # LanceDB
pip install chromadb                      # ChromaDB
pip install qdrant-client                 # Qdrant
pip install faiss-cpu                     # FAISS (CPU)
pip install pymilvus                      # Milvus Lite
pip install zvec                          # Zvec
pip install usearch                       # USearch
```
</details>
