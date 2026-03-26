# LanceDB

| | |
|---|---|
| **Package** | [lancedb/lancedb](https://github.com/lancedb/lancedb) |
| **PyPI** | [`lancedb`](https://pypi.org/project/lancedb/) |
| **Latest version** | 0.30.1 |
| **Last release** | 2025-01-14 |
| **Python** | >=3.10 (classifiers: 3.10–3.13) |
| **Deployment** | In-process (embedded) |
| **License** | Apache-2.0 |
| **Status** | Active, well-funded (LanceDB Inc.) — dev status: Alpha |

## Overview

LanceDB is a **serverless** vector database built on the
[Lance](https://github.com/lancedb/lance) columnar data format.  It runs
entirely in-process — no server, no containers, no configuration.

## Why consider it

- **Zero infrastructure.** Embedded by design, perfect for edge nodes.
- **Disk-native.** Data is persisted in Lance format automatically; survives
  restarts without explicit save/load.
- **Low memory footprint.** Reads data from disk on demand via memory-mapped I/O.
- **Columnar storage.** Efficient for filtered queries and metadata alongside
  vectors.
- **Active development.** Frequent releases, growing ecosystem.

## Potential concerns

- Relatively young project (first stable release 2023).
- Search is brute-force by default; ANN index must be explicitly created for
  large datasets (`create_index()`).
- No built-in client-server mode (embedded only).

## API mapping to current usage

| edge_node operation | LanceDB equivalent |
|---|---|
| `HNSWVectorDB(workspace=...)` | `lancedb.connect(path)` |
| `db.index(inputs=...)` | `table.add(rows)` |
| `db.search(inputs=..., limit=k)` | `table.search(vector).limit(k).to_list()` |
| `db.num_docs()` | `table.count_rows()` |

## Install

```bash
pip install lancedb
```
