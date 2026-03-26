# vectordb (Jina) — Baseline

| | |
|---|---|
| **Package** | [jina-ai/vectordb](https://github.com/jina-ai/vectordb) |
| **Version tested** | 0.0.21 |
| **Deployment** | In-process only |
| **License** | Apache-2.0 |
| **Status** | Unmaintained (last release 2023) |

## Why it is here

This is the **current implementation** used in Edge Node
(`extensions/serving/base/base_doc_emb_serving.py`). It serves as the
reference baseline so every replacement candidate is benchmarked
apples-to-apples against the status quo.

## How it works

- Wraps `docarray[hnswlib]` and runs an HNSW index in-process.
- Data model is defined as a `docarray.BaseDoc` subclass (`NaeuralDoc`).
- Persistence is file-based via the `workspace` parameter.

## Known issues

- **Unmaintained.** The GitHub repo is archived, no new releases.
- Pulls in `jina >= 3.20.0` as a transitive dependency (heavy).
- `docarray` itself underwent a major rewrite (v1 → v2) causing
  compatibility headaches.
- Fixed embedding size at class-definition time (not runtime-configurable
  without a workaround).

## Install

```bash
pip install vectordb docarray[hnswlib]
```
