# VectorDB Replacement — Final Verdict

> **Date:** 2026-04-06
> **Status:** Decision finalized
> **Full analysis:** [REPORT.md](REPORT.md)

## Decision

**FAISS** replaces `jina-ai/vectordb` as the Edge Node vector database.

## Installation

| Base Image | Package | Note |
|---|---|---|
| **CPU image** | `faiss-cpu>=1.7.4` | ~135 MB installed (from [faiss-wheels](https://github.com/faiss-wheels/faiss-wheels)) |
| **GPU image** | `faiss-cpu>=1.7.4` | CPU-only for now; GPU acceleration deferred (see [TODO_faiss_gpu.md](TODO_faiss_gpu.md)) |

> **Note on GPU package:** The old `faiss-gpu` PyPI package is abandoned (max 1.7.2, Python 3.10).
> The replacement is `faiss-gpu-cu12` (Meta, latest 1.14.1, Python 3.13 support).
> Deferred pending CUDA 12.8 compatibility validation.

### Transitive dependencies

| Package | Note |
|---|---|
| `numpy` | Already present in Edge Node |
| `packaging` | Already present in Edge Node |

**Net new dependency: one FAISS package per image.** No additional packages required.

### Runtime detection API

The adapter should auto-detect GPU availability and use the appropriate backend:

```python
import faiss

def _gpu_available() -> bool:
    """Check if FAISS GPU support is available."""
    return hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0

def create_index(embedding_size: int) -> faiss.Index:
    """Create a FAISS index, using GPU resources when available."""
    index = faiss.IndexFlatIP(embedding_size)
    if _gpu_available():
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
    return index
```

This keeps the adapter code identical across images — only the installed package differs. The same edge node codebase runs on both CPU and GPU nodes without branching.

## Why FAISS

| Criteria | Result |
|---|---|
| Weighted score | **4.70 / 5.00** (highest of all candidates) |
| Search recall | Perfect (100%) at all scales |
| Search latency | 0.5ms @ 1K docs, 19ms @ 50K docs |
| Index throughput | 8,500 docs/s |
| Memory footprint | 17 MB @ 1K docs |
| Dependency weight | 3 packages, 135 MB |
| Cold start | 60ms |
| Community | Meta-backed, 19M+ monthly PyPI downloads |
| Integration tests | 21/21 pass |

## Trade-offs (managed by adapter)

- **No built-in text storage** — adapter stores text in a sidecar JSON file (implemented and tested)
- **No auto-persistence** — adapter calls `save()` on `close()` (implemented and tested)
- **Brute-force flat index (IndexFlatIP)** — search scales linearly; acceptable for <50K docs, would need HNSW index for larger datasets

## Integration

Blast radius: **1 file** — `extensions/serving/base/base_doc_emb_serving.py` (~20 lines changed + drop-in adapter).

The adapter is ready at [`candidates/faiss/adapter.py`](candidates/faiss/adapter.py).

## Runner-up

**Zvec** (score 4.30) — strong alternative if FAISS ever becomes unsuitable. Perfect recall, 2 deps, designed for edge/on-device RAG. Currently alpha (v0.2.1).

## Not recommended

| Candidate | Reason |
|---|---|
| ChromaDB | 42% recall at 10K docs |
| USearch | 32% recall at 10K docs |
| Qdrant | Too slow for edge hardware |
| LanceDB | 60x slower search than FAISS, heavy deps |
