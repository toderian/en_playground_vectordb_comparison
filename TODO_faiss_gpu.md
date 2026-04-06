# TODO: Add faiss-gpu-cu12 support to GPU base image

> **Date:** 2026-04-06
> **Status:** Deferred

## Context

The GPU base image currently uses `faiss-cpu` (same as CPU image). GPU-accelerated FAISS is available via `faiss-gpu-cu12` (not the old abandoned `faiss-gpu` package).

## What needs to be done

1. **Validate `faiss-gpu-cu12` compatibility** with the GPU image's CUDA 12.8 stack (PyTorch cu128, cuDNN cu12, TensorRT cu12)
2. **Replace `faiss-cpu` with `faiss-gpu-cu12>=1.13.0`** in `base_edge_node_amd64_gpu/requirements.txt`
3. **Test** that the adapter's GPU auto-detection works (`faiss.get_num_gpus() > 0`, `index_cpu_to_gpu`, `index_gpu_to_cpu` for persistence)
4. **Benchmark** GPU vs CPU FAISS at edge_node's typical dataset sizes (<50K docs) to confirm the benefit is worth the extra dependency weight

## Package details

| Package | Latest | Python 3.13 | Note |
|---|---|---|---|
| `faiss-cpu` (faiss-wheels) | 1.13.2 | Yes | Currently used in both images |
| `faiss-gpu` (old) | 1.7.2 | No (max 3.10) | Abandoned, do NOT use |
| `faiss-gpu-cu12` (Meta) | 1.14.1.post1 | Yes (from 1.13.0) | Target package |

## Adapter readiness

The `FaissVectorDB` adapter in `edge_node/extensions/utils/faiss_vectordb.py` already has GPU support built in:
- `_gpu_available()` checks `faiss.get_num_gpus()`
- `_maybe_move_to_gpu()` uses `faiss.index_cpu_to_gpu()`
- `_save()` converts back to CPU via `faiss.index_gpu_to_cpu()` before writing

No adapter code changes needed — just swap the package in the GPU image requirements.
