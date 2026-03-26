# Current vectordb Usage in Edge Node

## Package

- **Library:** [jina-ai/vectordb](https://github.com/jina-ai/vectordb) (v0.0.21)
- **Status:** No longer maintained — last PyPI release 2023, repo archived
- **Transitive deps:** `jina >= 3.20.0`, `docarray[hnswlib] >= 0.34.0`

## Where it is used

Single file: `extensions/serving/base/base_doc_emb_serving.py`

The `BaseDocEmbServing` class provides document embedding + semantic search
across multiple named **contexts** (isolated workspaces).

## Data model

```python
from docarray import BaseDoc
from docarray.typing import NdArray

DOC_EMBEDDING_SIZE = 1024

class NaeuralDoc(BaseDoc):
    text: str = ''
    embedding: NdArray[DOC_EMBEDDING_SIZE]
    idx: int = -1
```

## API surface actually used

| Operation | Code | Notes |
|-----------|------|-------|
| **Create DB** | `HNSWVectorDB[NaeuralDoc](workspace=path)` | One instance per context |
| **Index** | `db.index(inputs=DocList[NaeuralDoc](docs))` | Batch insert |
| **Search** | `db.search(inputs=DocList[NaeuralDoc]([q]), limit=k)` | Returns `.matches` + `.scores` |
| **Count** | `db.num_docs()['num_docs']` | Total doc count |

No update or delete operations are used.

## Workflow

1. **Ingest:** Documents are split into overlapping segments (~1000 words, 50-word overlap)
2. **Embed:** Segments are embedded via a transformer model (e.g., MxBai-embed-large-v1, 1024-dim)
3. **Index:** Embedded segments are stored as `NaeuralDoc` records
4. **Query:** Query text → embed → `search(limit=k)` → results sorted by original `idx`

## Persistence

Each context persists to disk at: `{models_folder}/vectordb/{model_name}/{context}`

Contexts are backed up/restored via pickle so the system can reload after restart.

## Requirements for a replacement

A replacement must support:

1. **Batch insert** of (text, embedding_vector, integer_id) records
2. **k-NN search** by embedding vector, returning matched records + similarity scores
3. **Document count** query
4. **Disk persistence** (survive process restart)
5. **Multiple isolated collections/contexts** within the same process
6. **Embedding size of 1024** (but ideally configurable)
7. **Minimal external dependencies** — edge nodes run on constrained hardware

Nice-to-have:
- No separate server process (in-process / embedded mode)
- Python-first API
- Active maintenance and community
- Low memory footprint
- Optional client-server mode for scaling later
