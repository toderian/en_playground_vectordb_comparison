"""
Integration tests that reproduce the edge_node vectordb workflow.

Each test mirrors a real operation from BaseDocEmbServing:
  - batch indexing of embedded document segments
  - k-NN search returning matches + scores
  - document count queries
  - disk persistence across close/reopen cycles
  - multiple isolated contexts (collections)
  - incremental indexing (adding docs to an existing context)
  - search result ordering by document index

Run a single candidate in isolation:
    uv run --extra faiss pytest tests/ -v

Run all installed candidates:
    uv run --extra all pytest tests/ -v
"""

from __future__ import annotations

import numpy as np

from candidates.base import Document
from tests.conftest import (
    DOC_EMBEDDING_SIZE,
    make_embedding,
    make_similar_embedding,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_docs(n: int, start_idx: int = 0) -> list[Document]:
    """Create n documents with deterministic embeddings."""
    return [
        Document(
            text=f"Segment {start_idx + i}: sample text for testing.",
            embedding=make_embedding(seed=start_idx + i),
            idx=start_idx + i,
        )
        for i in range(n)
    ]


# ═════════════════════════════════════════════════════════════════════════════
# 1. BASIC OPERATIONS — mirrors edge_node's core API surface
# ═════════════════════════════════════════════════════════════════════════════


class TestBasicOperations:
    """Test the fundamental operations edge_node relies on."""

    def test_open_close(self, adapter_factory):
        """DB can be opened and closed without error."""
        db = adapter_factory()
        db.open()
        db.close()

    def test_index_and_count(self, adapter_factory):
        """Batch insert documents and verify count.

        Edge_node: ``self.__dbs[context].index(inputs=DocList[NaeuralDoc](lst_docs))``
        followed by ``self.__dbs[context].num_docs()['num_docs']``
        """
        db = adapter_factory()
        db.open()
        docs = _make_docs(50)
        db.index(docs)
        assert db.num_docs() == 50
        db.close()

    def test_index_batch_sizes(self, adapter_factory):
        """Edge_node indexes in variable batch sizes (configurable batch_size).

        Verify that multiple sequential batches accumulate correctly.
        """
        db = adapter_factory()
        db.open()

        batch1 = _make_docs(20, start_idx=0)
        batch2 = _make_docs(30, start_idx=20)
        batch3 = _make_docs(10, start_idx=50)

        db.index(batch1)
        assert db.num_docs() == 20

        db.index(batch2)
        assert db.num_docs() == 50

        db.index(batch3)
        assert db.num_docs() == 60

        db.close()

    def test_search_returns_results(self, adapter_factory):
        """Search returns matches with scores.

        Edge_node: ``search_results = self.__dbs[context].search(
            inputs=DocList[NaeuralDoc]([query_doc]), limit=k)[0]``
        """
        db = adapter_factory()
        db.open()

        docs = _make_docs(100)
        db.index(docs)

        query_emb = make_embedding(seed=0)  # same as doc 0
        results = db.search(query_emb, limit=10)

        assert len(results.matches) > 0
        assert len(results.matches) <= 10

        # Each result should have a score
        for match in results.matches:
            assert match.score is not None

        db.close()

    def test_search_limit(self, adapter_factory):
        """Search respects the limit parameter (edge_node default k=10)."""
        db = adapter_factory()
        db.open()

        docs = _make_docs(50)
        db.index(docs)

        for k in [1, 5, 10, 20]:
            results = db.search(make_embedding(seed=999), limit=k)
            assert len(results.matches) <= k

        db.close()

    def test_search_on_empty_db(self, adapter_factory):
        """Searching an empty context should return empty results, not crash."""
        db = adapter_factory()
        db.open()

        results = db.search(make_embedding(seed=0), limit=10)
        assert len(results.matches) == 0

        db.close()

    def test_num_docs_on_empty_db(self, adapter_factory):
        """Count on empty DB returns 0."""
        db = adapter_factory()
        db.open()
        assert db.num_docs() == 0
        db.close()


# ═════════════════════════════════════════════════════════════════════════════
# 2. PERSISTENCE — edge_node relies on data surviving restarts
# ═════════════════════════════════════════════════════════════════════════════


class TestPersistence:
    """Test that data survives close/reopen cycles.

    Edge_node: ``HNSWVectorDB[NaeuralDoc](workspace=path)`` reconstructs
    from disk on startup via ``__maybe_load_backup()``.
    """

    def test_docs_survive_reopen(self, adapter_factory, tmp_path):
        """Documents persist after close and reopen."""
        ws = tmp_path / "persist_test"

        # Write
        db = adapter_factory(workspace=ws)
        db.open()
        db.index(_make_docs(100))
        assert db.num_docs() == 100
        db.close()

        # Reopen
        db2 = adapter_factory(workspace=ws)
        db2.open()
        assert db2.num_docs() == 100
        db2.close()

    def test_search_after_reopen(self, adapter_factory, tmp_path):
        """Search still works after close/reopen."""
        ws = tmp_path / "search_persist"

        db = adapter_factory(workspace=ws)
        db.open()
        docs = _make_docs(50)
        db.index(docs)
        db.close()

        db2 = adapter_factory(workspace=ws)
        db2.open()
        results = db2.search(make_embedding(seed=0), limit=5)
        assert len(results.matches) > 0
        db2.close()

    def test_incremental_index_after_reopen(self, adapter_factory, tmp_path):
        """Can add more docs after a reopen (edge_node adds docs over time).

        Edge_node: ``curr_size = self.__dbs[context].num_docs()['num_docs']``
        then indexes new docs starting from curr_size.
        """
        ws = tmp_path / "incremental_persist"

        # First session: add 30 docs
        db = adapter_factory(workspace=ws)
        db.open()
        db.index(_make_docs(30, start_idx=0))
        assert db.num_docs() == 30
        db.close()

        # Second session: add 20 more
        db2 = adapter_factory(workspace=ws)
        db2.open()
        assert db2.num_docs() == 30
        db2.index(_make_docs(20, start_idx=30))
        assert db2.num_docs() == 50
        db2.close()

        # Third session: verify all 50 survived
        db3 = adapter_factory(workspace=ws)
        db3.open()
        assert db3.num_docs() == 50
        db3.close()


# ═════════════════════════════════════════════════════════════════════════════
# 3. MULTIPLE CONTEXTS — edge_node manages isolated workspaces
# ═════════════════════════════════════════════════════════════════════════════


class TestMultipleContexts:
    """Test context isolation.

    Edge_node: ``self.__dbs = {}`` with one HNSWVectorDB per context,
    each using a separate workspace path.
    """

    def test_contexts_are_isolated(self, adapter_factory, tmp_path):
        """Documents in one context are invisible to another."""
        ws_a = tmp_path / "context_a"
        ws_b = tmp_path / "context_b"

        db_a = adapter_factory(workspace=ws_a)
        db_b = adapter_factory(workspace=ws_b)

        db_a.open()
        db_b.open()

        db_a.index(_make_docs(40, start_idx=0))
        db_b.index(_make_docs(20, start_idx=100))

        assert db_a.num_docs() == 40
        assert db_b.num_docs() == 20

        db_a.close()
        db_b.close()

    def test_contexts_persist_independently(self, adapter_factory, tmp_path):
        """Each context persists separately on disk."""
        ws_a = tmp_path / "ctx_persist_a"
        ws_b = tmp_path / "ctx_persist_b"

        # Write to both
        db_a = adapter_factory(workspace=ws_a)
        db_a.open()
        db_a.index(_make_docs(25, start_idx=0))
        db_a.close()

        db_b = adapter_factory(workspace=ws_b)
        db_b.open()
        db_b.index(_make_docs(15, start_idx=100))
        db_b.close()

        # Reopen and verify
        db_a2 = adapter_factory(workspace=ws_a)
        db_a2.open()
        assert db_a2.num_docs() == 25
        db_a2.close()

        db_b2 = adapter_factory(workspace=ws_b)
        db_b2.open()
        assert db_b2.num_docs() == 15
        db_b2.close()

    def test_search_does_not_cross_contexts(self, adapter_factory, tmp_path):
        """A query in context A must not return documents from context B."""
        ws_a = tmp_path / "ctx_cross_a"
        ws_b = tmp_path / "ctx_cross_b"

        # Use a known embedding for context A
        emb_a = make_embedding(seed=42)
        doc_a = Document(text="only in context A", embedding=emb_a, idx=0)

        db_a = adapter_factory(workspace=ws_a)
        db_a.open()
        db_a.index([doc_a])

        # Context B has different documents
        db_b = adapter_factory(workspace=ws_b)
        db_b.open()
        db_b.index(_make_docs(10, start_idx=200))

        # Search B with A's embedding — should not find A's doc text
        results = db_b.search(emb_a, limit=10)
        for m in results.matches:
            assert m.document.text != "only in context A"

        db_a.close()
        db_b.close()


# ═════════════════════════════════════════════════════════════════════════════
# 4. SEARCH QUALITY — edge_node expects nearest neighbours to be meaningful
# ═════════════════════════════════════════════════════════════════════════════


class TestSearchQuality:
    """Verify that search returns semantically nearest documents."""

    def test_exact_match_is_top_result(self, adapter_factory):
        """Querying with a document's own embedding should return it first."""
        db = adapter_factory()
        db.open()

        docs = _make_docs(100)
        db.index(docs)

        # Query with doc 42's embedding
        target_emb = make_embedding(seed=42)
        results = db.search(target_emb, limit=5)

        assert len(results.matches) > 0
        top = results.matches[0]
        assert top.document.idx == 42
        assert top.document.text == "Segment 42: sample text for testing."

    def test_similar_embedding_ranks_high(self, adapter_factory):
        """A query close to a document embedding should rank it in top results."""
        db = adapter_factory()
        db.open()

        docs = _make_docs(200)
        db.index(docs)

        # Create query similar to doc 77
        base_emb = make_embedding(seed=77)
        query = make_similar_embedding(base_emb, noise=0.05, seed=999)

        results = db.search(query, limit=10)
        top_idxs = [m.document.idx for m in results.matches]
        assert 77 in top_idxs, f"Expected doc 77 in top 10, got {top_idxs}"

    def test_results_sorted_by_relevance(self, adapter_factory):
        """Scores should be monotonically non-increasing (best first).

        Note: some DBs use distance (lower=better), others use similarity
        (higher=better). We check that scores are consistently ordered.
        """
        db = adapter_factory()
        db.open()

        docs = _make_docs(100)
        db.index(docs)

        results = db.search(make_embedding(seed=0), limit=20)
        scores = [m.score for m in results.matches]

        if len(scores) >= 2:
            # Check if sorted ascending (distance) or descending (similarity)
            is_ascending = all(a <= b + 1e-6 for a, b in zip(scores, scores[1:]))
            is_descending = all(a >= b - 1e-6 for a, b in zip(scores, scores[1:]))
            assert is_ascending or is_descending, (
                f"Scores not consistently ordered: {scores[:5]}..."
            )


# ═════════════════════════════════════════════════════════════════════════════
# 5. EDGE NODE WORKFLOW — end-to-end simulation
# ═════════════════════════════════════════════════════════════════════════════


class TestEdgeNodeWorkflow:
    """Full workflow simulation matching BaseDocEmbServing._predict()."""

    def test_add_doc_then_query(self, adapter_factory, doc_splitter):
        """Simulate the ADD_DOC → QUERY cycle from edge_node.

        Steps:
        1. Split documents into segments (DocSplitter)
        2. Embed segments (simulated with deterministic embeddings)
        3. Index segments with sequential idx
        4. Query and sort results by idx (edge_node sorts by idx)
        """
        db = adapter_factory()
        db.open()

        # Simulate ingesting two documents
        raw_docs = [
            "This is the first document about machine learning and neural networks. " * 20,
            "This is the second document about database systems and indexing. " * 20,
        ]

        # 1. Split
        segments = doc_splitter.split_documents(raw_docs)
        assert len(segments) >= 2

        # 2. Embed (simulated) and create docs
        docs = [
            Document(
                text=seg,
                embedding=make_embedding(seed=i),
                idx=i,
            )
            for i, seg in enumerate(segments)
        ]

        # 3. Index
        db.index(docs)
        assert db.num_docs() == len(segments)

        # 4. Query — use embedding similar to first segment
        query_emb = make_similar_embedding(
            make_embedding(seed=0), noise=0.1, seed=42
        )
        results = db.search(query_emb, limit=5)
        assert len(results.matches) > 0

        # 5. Sort by idx (edge_node does this)
        sorted_matches = sorted(results.matches, key=lambda m: m.document.idx)
        idxs = [m.document.idx for m in sorted_matches]
        assert idxs == sorted(idxs)

        db.close()

    def test_incremental_add_docs_workflow(self, adapter_factory, tmp_path):
        """Simulate edge_node adding docs over time across sessions.

        Edge_node pattern:
            curr_size = self.__dbs[context].num_docs()['num_docs']
            lst_docs = [NaeuralDoc(..., idx=curr_size + i) for i, ...]
            self.__dbs[context].index(inputs=DocList[NaeuralDoc](lst_docs))
        """
        ws = tmp_path / "incremental_workflow"

        # Session 1: ingest first batch
        db = adapter_factory(workspace=ws)
        db.open()
        curr_size = db.num_docs()
        assert curr_size == 0

        batch1 = _make_docs(20, start_idx=curr_size)
        db.index(batch1)
        db.close()

        # Session 2: ingest second batch
        db = adapter_factory(workspace=ws)
        db.open()
        curr_size = db.num_docs()
        assert curr_size == 20

        batch2 = _make_docs(15, start_idx=curr_size)
        db.index(batch2)

        assert db.num_docs() == 35

        # Search should span both batches
        results = db.search(make_embedding(seed=0), limit=10)
        assert len(results.matches) > 0

        # First batch doc should still be findable
        results = db.search(make_embedding(seed=0), limit=1)
        assert results.matches[0].document.idx == 0

        db.close()

    def test_multi_context_workflow(self, adapter_factory, tmp_path):
        """Simulate managing multiple contexts like edge_node does.

        Edge_node: ``self.__dbs = {}`` — dict of context → vectordb
        """
        contexts = {}

        # Create 3 contexts (like edge_node's context management)
        for ctx_name in ["default", "context_project1", "context_project2"]:
            ws = tmp_path / "vectordb" / "model_name" / ctx_name
            db = adapter_factory(workspace=ws)
            db.open()
            contexts[ctx_name] = db

        # Add different amounts of docs to each
        contexts["default"].index(_make_docs(10, start_idx=0))
        contexts["context_project1"].index(_make_docs(25, start_idx=100))
        contexts["context_project2"].index(_make_docs(5, start_idx=200))

        # Verify counts
        assert contexts["default"].num_docs() == 10
        assert contexts["context_project1"].num_docs() == 25
        assert contexts["context_project2"].num_docs() == 5

        # Search in specific context
        results = contexts["context_project1"].search(
            make_embedding(seed=100), limit=5
        )
        assert len(results.matches) > 0
        assert results.matches[0].document.idx == 100

        # Close all (simulate shutdown)
        for db in contexts.values():
            db.close()

        # Reopen all (simulate __maybe_load_backup)
        for ctx_name in ["default", "context_project1", "context_project2"]:
            ws = tmp_path / "vectordb" / "model_name" / ctx_name
            db = adapter_factory(workspace=ws)
            db.open()
            contexts[ctx_name] = db

        assert contexts["default"].num_docs() == 10
        assert contexts["context_project1"].num_docs() == 25
        assert contexts["context_project2"].num_docs() == 5

        for db in contexts.values():
            db.close()

    def test_1024_dim_embeddings(self, adapter_factory):
        """Verify that 1024-dimensional embeddings work correctly.

        Edge_node uses DOC_EMBEDDING_SIZE = 1024 exclusively.
        """
        db = adapter_factory()
        db.open()

        emb = make_embedding(seed=0, dim=DOC_EMBEDDING_SIZE)
        assert len(emb) == 1024

        doc = Document(text="1024-dim test", embedding=emb, idx=0)
        db.index([doc])
        assert db.num_docs() == 1

        results = db.search(emb, limit=1)
        assert len(results.matches) == 1

        db.close()

    def test_large_batch_index(self, adapter_factory):
        """Index a larger batch simulating a real document ingestion.

        Edge_node processes documents that can produce hundreds of segments.
        """
        db = adapter_factory()
        db.open()

        docs = _make_docs(500)
        # Index in batches of 200 (like benchmark.py)
        for start in range(0, len(docs), 200):
            db.index(docs[start : start + 200])

        assert db.num_docs() == 500

        # Search should still work fast
        results = db.search(make_embedding(seed=250), limit=10)
        assert len(results.matches) > 0
        assert results.matches[0].document.idx == 250

        db.close()


# ═════════════════════════════════════════════════════════════════════════════
# 6. DOC SPLITTER — verify the ported edge_node splitter
# ═════════════════════════════════════════════════════════════════════════════


class TestDocSplitter:
    """Test the ported DocSplitter to ensure test data is realistic."""

    def test_short_doc_no_split(self, doc_splitter):
        """A short document should remain a single segment."""
        text = "Hello world, this is a short document."
        segments = doc_splitter.split_document(text)
        assert len(segments) == 1

    def test_long_doc_splits(self, doc_splitter):
        """A long document should be split into multiple overlapping segments."""
        # Create a document with ~2500 words (use plain words to match the regex)
        words = [f"document" for _ in range(2500)]
        text = " ".join(words)
        segments = doc_splitter.split_document(text)
        assert len(segments) >= 3

    def test_overlap_preserves_context(self, doc_splitter):
        """Adjacent segments should share overlapping words."""
        text = " ".join(f"word{i}" for i in range(2000))
        segments = doc_splitter.split_document(text)
        if len(segments) >= 2:
            words_seg0 = set(segments[0].split())
            words_seg1 = set(segments[1].split())
            overlap = words_seg0 & words_seg1
            assert len(overlap) > 0, "Segments should have overlapping words"

    def test_split_multiple_documents(self, doc_splitter):
        """Splitting multiple documents concatenates their segments."""
        docs = [
            "Short doc one.",
            "Short doc two.",
        ]
        segments = doc_splitter.split_documents(docs)
        assert len(segments) == 2
