"""Tests for document ingestion with contextualized chunk embeddings
(voyage-context-4) and the Documents admin surface."""
from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from config import Config
from documents import service as docs
from documents.service import (
    ContextualEmbedder,
    DocumentIngestError,
    cosine,
    extract_text,
    ingest_document,
    delete_document,
    list_documents,
    search_documents,
    segment_text,
)

# Contextualized chunk embedding model used throughout the pipeline.
MODEL = "voyage-context-4"


def _run(coro):
    return asyncio.run(coro)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _make_pdf(pages_text: list[str]) -> bytes:
    """Synthesize a tiny text-bearing PDF with pypdf (real /Resources font)."""
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, StreamObject

    writer = PdfWriter()
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    resources = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    for text in pages_text:
        stream = StreamObject()
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode())
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Contents")] = stream
        page[NameObject("/Resources")] = resources
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class FakeCursor(list):
    def sort(self, key=None, direction=1):
        return self


class FakeCollection:
    """Miniature pymongo collection for unit tests (mirrors test_admin)."""

    def __init__(self, docs=None):
        self.docs = list(docs or [])
        self.calls = []

    def find(self, query):
        self.calls.append(("find", query))

        def match(doc):
            for field, value in query.items():
                if field == "_id" and doc.get("_id") != value:
                    return False
                if field == "doc_id" and doc.get("doc_id") != value:
                    return False
            return True

        return FakeCursor([d for d in self.docs if match(d)])

    def find_one(self, query):
        rows = self.find(query)
        return rows[0] if rows else None

    def insert_one(self, doc):
        self.calls.append(("insert_one", doc))
        self.docs.append(doc)
        return doc

    def update_one(self, query, update, upsert=False):
        self.calls.append(("update_one", query, update))
        rows = self.find(query)
        if rows:
            doc = rows[0]
            for field, value in update.get("$set", {}).items():
                doc[field] = value
            return 1
        if upsert:
            doc = dict(query)
            for field, value in update.get("$set", {}).items():
                doc[field] = value
            self.docs.append(doc)
        return 0

    def delete_one(self, query):
        rows = self.find(query)
        if not rows:
            return 0
        self.docs.remove(rows[0])
        return 1

    def delete_many(self, query):
        rows = self.find(query)
        for doc in rows:
            self.docs.remove(doc)
        return len(rows)


class FakeEmbedder:
    """Deterministic embedder: one chunk per segment, query vector [1, 0, 0]."""

    def __init__(self, total_tokens=11, fail=False):
        self.total_tokens = total_tokens
        self.fail = fail
        self.last_segments = None
        self.calls = []  # every embed_documents batch received

    async def embed_documents(self, segments):
        if self.fail:
            raise DocumentIngestError("voyage contextualized API unavailable (503)")
        self.last_segments = segments
        self.calls.append(list(segments))
        chunks = []
        for index, segment in enumerate(segments):
            chunks.append(
                {"text": segment, "embedding": [0.2, 0.3, float(index + 1)]}
            )
        return chunks, self.total_tokens

    async def embed_query(self, query: str):
        return [1.0, 0.0, 0.0]


def _books_txt() -> str:
    return "WiredTiger keeps data in an in-memory cache.\n\n" \
        "Fragmentation limits write throughput.\n\n" \
        "In-place updates double the working set."


class TestSegmentText:
    def test_small_text_is_one_segment(self):
        assert segment_text("short doc") == ["short doc"]

    def test_segments_bounded_and_nonempty(self):
        paragraphs = ["\u00b7" * 5000] * 50  # 50 * ~5K chars
        text = "\n\n".join(paragraphs)
        parts = segment_text(text, max_chars=30000)
        assert len(parts) >= 2
        for part in parts:
            assert len(part) > 0
            assert len(part) <= 30000
        # every paragraph survives intact (boundary separators may be dropped)
        assert "\n\n".join(parts).replace("\n\n", "") == text.replace("\n\n", "")

    def test_boundary_prefers_paragraph_break(self):
        para_a = "A" * 60000
        para_b = "B" * 60000
        parts = segment_text(f"{para_a}\n\n{para_b}", max_chars=100000)
        assert parts == [para_a, para_b]

    def test_oversized_paragraph_is_hard_split(self):
        parts = segment_text("Z" * 5000, max_chars=100)
        assert all(len(p) <= 100 for p in parts)
        assert "".join(parts) == "Z" * 5000


class TestPdfExtraction:
    def test_extracts_text_and_page_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book.pdf"
            path.write_bytes(_make_pdf(["WiredTiger in-memory cache", "Fragmentation"]))
            text, pages = extract_text(path)
        assert "WiredTiger" in text and "Fragmentation" in text
        assert pages == 2

    def test_text_files_extract_with_zero_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.md"
            path.write_text("# Title\n\nBody text")
            text, pages = extract_text(path)
        assert "# Title" in text and pages == 0

    def test_unsupported_extension_raises(self):
        try:
            extract_text(Path("archive.zip"))
        except DocumentIngestError as exc:
            assert ".zip" in str(exc)
        else:
            raise AssertionError("expected DocumentIngestError")


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload) if not isinstance(payload, str) else payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                f"upstream {self.status_code}",
                request=mock.Mock(),
                response=self,
            )

    def json(self):
        return self._payload


def _patch_httpx(payload, status=200):
    """Return a context manager whose posted (url, json) can be inspected."""

    class _Client:
        def __init__(self, result):
            self.result = result
            self.posted = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json=None, headers=None):
            self.posted.append((url, json, headers))
            return self.result

    sentinel = _Client(_FakeResponse(payload, status=status))

    def factory(*args, **kwargs):
        return sentinel

    patcher = mock.patch("documents.service.httpx.AsyncClient", side_effect=factory)
    return patcher, sentinel


def _doc_payload(chunks, total_tokens=42):
    return {
        "data": [{"data": chunks, "index": 0}],
        "model": MODEL,
        "usage": {"total_tokens": total_tokens},
    }


class TestContextualEmbedder:
    def test_embed_documents_posts_document_payload_and_parses_chunks(self):
        payload = _doc_payload(
            [{"embedding": [0.1, 0.2], "text": "chunk a", "index": 0}], total_tokens=7
        )
        patcher, client = _patch_httpx(payload)
        with patcher:
            chunks, total = _run(
                ContextualEmbedder(api_key="k").embed_documents(["seg one", "seg two"])
            )
        url, body, headers = client.posted[0]
        assert body == {
            "model": MODEL,
            "inputs": ["seg one", "seg two"],
            "input_type": "document",
            "enable_auto_chunking": True,
            "chunk_size": 512,
        }
        assert headers["Authorization"] == "Bearer k"
        assert chunks == [{"text": "chunk a", "embedding": [0.1, 0.2]}]
        assert total == 7

    def test_embed_query_posts_query_input_type(self):
        patcher, client = _patch_httpx(
            _doc_payload([{"embedding": [0.9, 0.1], "text": "", "index": 0}])
        )
        with patcher:
            vector = _run(ContextualEmbedder().embed_query("cache sizing"))
        url, body, headers = client.posted[0]
        assert body["input_type"] == "query"
        assert body["inputs"] == ["cache sizing"]
        assert vector == [0.9, 0.1]

    def test_upstream_error_surfaces_as_document_ingest_error(self):
        patcher, _ = _patch_httpx({}, status=429)
        with patcher:
            try:
                _run(ContextualEmbedder().embed_documents(["x"]))
            except DocumentIngestError as exc:
                assert "429" in str(exc)
            else:
                raise AssertionError("expected DocumentIngestError")


class TestIngestDocument:
    def _collections(self):
        return FakeCollection(), FakeCollection()

    def _txt(self, tmp: str, content: str) -> str:
        path = Path(tmp) / "notes.txt"
        path.write_text(content)
        return str(path)

    def test_ingest_marks_ready_and_stores_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs_coll, chunks_coll = self._collections()
            source = self._txt(tmp, "WiredTiger cache\n\nFragmentation\n\nIn-place updates")
            result = _run(
                ingest_document(
                    "doc1", source, "notes.txt", docs_coll, chunks_coll,
                    embedder=FakeEmbedder(), title="MongoDB internals", max_chars=20,
                )
            )
        assert result["status"] == "ready"
        assert result["filename"] == "notes.txt"
        assert result["title"] == "MongoDB internals"
        assert result["pages"] == 0
        assert result["chunk_count"] == 3
        assert result["tokens_total"] == 11
        assert result["model"] == MODEL
        assert result["error"] is None
        chunks = sorted(chunks_coll.docs, key=lambda c: c["chunk_index"])
        assert [c["doc_id"] for c in chunks] == ["doc1"] * 3
        assert [c["text"] for c in chunks] == [
            "WiredTiger cache", "Fragmentation", "In-place updates",
        ]
        assert chunks[0]["embedding"] == [0.2, 0.3, 1.0]
        assert chunks[2]["embedding"] == [0.2, 0.3, 3.0]

    def test_ingest_failure_marks_failed_with_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs_coll, chunks_coll = self._collections()
            source = self._txt(tmp, "body")
            result = _run(
                ingest_document(
                    "doc1", source, "notes.txt", docs_coll, chunks_coll,
                    embedder=FakeEmbedder(fail=True),
                )
            )
        assert result["status"] == "failed"
        assert "503" in result["error"]
        assert result["chunk_count"] == 0
        assert chunks_coll.docs == []

    def test_reingest_failed_document_replaces_old_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs_coll, chunks_coll = self._collections()
            source = self._txt(tmp, "WiredTiger cache\n\nFragmentation\n\nIn-place updates")
            _run(
                ingest_document(
                    "doc1", source, "notes.txt", docs_coll, chunks_coll,
                    embedder=FakeEmbedder(fail=True),
                )
            )
            result = _run(
                ingest_document(
                    "doc1", source, "notes.txt", docs_coll, chunks_coll,
                    embedder=FakeEmbedder(), max_chars=20,
                )
            )
        assert result["status"] == "ready"
        assert result["error"] is None
        assert len(chunks_coll.docs) == 3  # replaced, not duplicated

    def test_ingest_batches_segments_under_token_budget(self):
        # The contextualized API caps the whole REQUEST at 120K tokens, so
        # segments must be embedded in char-budgeted batches (never one giant
        # request). Chunk order must still follow document order.
        with tempfile.TemporaryDirectory() as tmp:
            docs_coll, chunks_coll = self._collections()
            source = self._txt(tmp, "AAAAAAAA\n\nBBBBBBBB\n\nCCCCCCCC")
            embedder = FakeEmbedder()
            with mock.patch.object(docs, "MAX_BATCH_CHARS", 18):
                result = _run(
                    ingest_document(
                        "doc1", source, "notes.txt", docs_coll, chunks_coll,
                        embedder=embedder, max_chars=2,
                    )
                )
        assert result["status"] == "ready"
        assert embedder.calls == [["AAAAAAAA", "BBBBBBBB"], ["CCCCCCCC"]]
        chunks = sorted(chunks_coll.docs, key=lambda c: c["chunk_index"])
        assert [c["text"] for c in chunks] == ["AAAAAAAA", "BBBBBBBB", "CCCCCCCC"]
        assert result["tokens_total"] == 11 * 2  # summed across batches


class TestDocumentSearch:
    def _seed(self):
        docs_coll = FakeCollection(
            [
                {"_id": "book", "doc_id": "book", "title": "High Perf Mongo", "status": "ready"},
                {"_id": "other", "doc_id": "other", "title": "Other notes", "status": "ready"},
            ]
        )
        chunks_coll = FakeCollection(
            [
                {"doc_id": "book", "chunk_index": 0, "text": "in-memory cache", "embedding": [1.0, 0.0, 0.0]},
                {"doc_id": "book", "chunk_index": 1, "text": "WiredTiger", "embedding": [0.5, 0.5, 0.0]},
                {"doc_id": "other", "chunk_index": 0, "text": "fragmentation", "embedding": [0.3, 0.0, 0.95]},
            ]
        )
        return docs_coll, chunks_coll

    def test_returns_top_k_ordered_by_cosine(self):
        docs_coll, chunks_coll = self._seed()
        results = _run(
            search_documents(
                "cache", docs_coll, chunks_coll, embedder=FakeEmbedder(), k=2
            )
        )
        assert [r["chunk_index"] for r in results] == [0, 1]
        assert all(r["document_id"] == "book" for r in results)
        assert results[0]["document_title"] == "High Perf Mongo"
        assert results[0]["score"] > results[1]["score"] > 0
        assert results[0]["text"].startswith("in-memory")

    def test_doc_filter_restricts_to_one_document(self):
        docs_coll, chunks_coll = self._seed()
        results = _run(
            search_documents(
                "cache", docs_coll, chunks_coll, embedder=FakeEmbedder(), doc_id="other"
            )
        )
        assert [r["document_id"] for r in results] == ["other"]

    def test_empty_store_returns_empty(self):
        results = _run(
            search_documents("x", FakeCollection(), FakeCollection(), embedder=FakeEmbedder())
        )
        assert results == []


class TestDeleteDocument:
    def test_delete_removes_doc_chunks_and_file(self):
        docs_coll = FakeCollection(
            [{"_id": "doc1", "doc_id": "doc1", "path": "/tmp/nope.txt"}]
        )
        chunks_coll = FakeCollection(
            [{"doc_id": "doc1", "chunk_index": 0, "text": "x", "embedding": [1.0]}]
        )
        with tempfile.TemporaryDirectory() as tmp:
            stored = Path(tmp) / "doc1.txt"
            stored.write_text("data")
            docs_coll.docs[0]["path"] = str(stored)
            assert delete_document("doc1", docs_coll, chunks_coll) is True
        assert docs_coll.docs == []
        assert chunks_coll.docs == []
        assert not stored.exists()
        assert delete_document("doc1", docs_coll, chunks_coll) is False


class TestListDocuments:
    def test_newest_first_and_public_shape(self):
        coll = FakeCollection(
            [
                {"_id": "old", "doc_id": "old", "filename": "a.txt", "status": "ready",
                 "chunk_count": 2, "created_at": "2026-01-01T00:00:00+00:00",
                 "title": "", "pages": 0, "tokens_total": 5, "model": MODEL,
                 "error": None, "path": "/tmp/a"},
                {"_id": "new", "doc_id": "new", "filename": "b.pdf", "status": "ingesting",
                 "chunk_count": 0, "created_at": "2026-02-01T00:00:00+00:00",
                 "title": "B", "pages": 9, "tokens_total": 0, "model": MODEL,
                 "error": None, "path": "/tmp/b"},
            ]
        )
        rows = list_documents(coll)
        assert [r["doc_id"] for r in rows] == ["new", "old"]
        row = rows[0]
        assert row["status"] == "ingesting"
        assert row["title"] == "B"
        assert "path" not in row  # never leak internal storage paths


class TestSearchSync:
    """The agent tool path is synchronous (execute_tool_call is sync)."""

    def test_sync_search_matches_async_results(self):
        # same seed as TestDocumentSearch, inline (no cross-test coupling)
        docs_coll = FakeCollection(
            [{"_id": "book", "doc_id": "book", "title": "High Perf Mongo", "status": "ready"}]
        )
        chunks_coll = FakeCollection(
            [
                {"doc_id": "book", "chunk_index": 0, "text": "in-memory cache", "embedding": [1.0, 0.0, 0.0]},
                {"doc_id": "book", "chunk_index": 1, "text": "WiredTiger", "embedding": [0.0, 1.0, 0.0]},
            ]
        )
        with mock.patch.object(docs, "ContextualEmbedder") as embedder_cls:
            embedder_cls.return_value = mock.Mock(
                **{"embed_query_sync.return_value": [1.0, 0.0, 0.0]}
            )
            results = docs.search_documents_sync("cache", docs_coll, chunks_coll, k=1)
        assert [r["chunk_index"] for r in results] == [0]
        assert results[0]["score"] > 0.9


class TestDocumentsSearchTool:
    """documents_search in the agent toolset (admin/executor.py)."""

    def _schema(self):
        from admin.executor import DEFAULT_AGENT_TOOLS

        for tool in DEFAULT_AGENT_TOOLS:
            fn = tool.get("function") or {}
            if fn.get("name") == "documents_search":
                return tool
        raise AssertionError("documents_search missing from DEFAULT_AGENT_TOOLS")

    def test_tool_is_in_default_agent_toolset(self):
        schema = self._schema()
        params = schema["function"]["parameters"]
        assert "query" in params["required"]
        assert "query" in params["properties"]
        assert "k" in params["properties"]
        assert "document" in params["properties"]

    def test_execute_returns_top_hits_with_scores(self):
        from admin.executor import execute_tool_call
        from pathlib import Path as P

        hits = [
            {"document_id": "book", "document_title": "High Perf Mongo",
             "chunk_index": 3, "score": 0.87, "text": "WiredTiger evicts cold pages"},
        ]
        with mock.patch("admin.executor._memories_for_query", return_value=[]):
            with mock.patch(
                "admin.executor.documents_service.search_documents_sync", return_value=hits
            ) as fake:
                result = execute_tool_call(
                    {"name": "documents_search", "arguments": '{"query": "cache eviction", "k": 3}'},
                    P("/tmp"),
                )
        assert result["name"] == "documents_search"
        assert "0.87" in result["output"]
        assert "High Perf Mongo" in result["output"]
        assert "WiredTiger evicts cold pages" in result["output"]
        assert fake.call_args.kwargs["k"] == 3

    def test_execute_surfaces_empty_library(self):
        from admin.executor import execute_tool_call
        from pathlib import Path as P

        with mock.patch("admin.executor._memories_for_query", return_value=[]):
            with mock.patch("admin.executor.documents_service.search_documents_sync", return_value=[]):
                result = execute_tool_call(
                    {"name": "documents_search", "arguments": '{"query": "anything"}'},
                    P("/tmp"),
                )
        assert result["output"].startswith("[error]")
        assert "no matches" in result["output"]

    def test_execute_surfaces_failures(self):
        from admin.executor import execute_tool_call
        from pathlib import Path as P

        with mock.patch("admin.executor._memories_for_query", return_value=[]):
            with mock.patch(
                "admin.executor.documents_service.search_documents_sync",
                side_effect=RuntimeError("index down"),
            ):
                result = execute_tool_call(
                    {"name": "documents_search", "arguments": '{"query": "x"}'},
                    P("/tmp"),
                )
        assert result["output"].startswith("[error]")
        assert "index down" in result["output"]

    # --- memory-first: Pensive consults its long-term memory BEFORE the corpus ---

    def _call(self, query='{"query": "eviction"}', workspace="/tmp/ag_123"):
        from admin.executor import execute_tool_call
        from pathlib import Path as P

        return execute_tool_call(
            {"name": "documents_search", "arguments": query}, P(workspace)
        )

    def _memory_hit(self, score=0.91, text="WiredTiger eviction begins when the dirty-page target is reached",
                    source="semantic", timestamp="2026-11-01T00:00:00+00:00"):
        return {
            "text": text,
            "source": source,
            "similarity_score": score,
            "timestamp": timestamp,
        }

    def test_confident_memory_hit_skips_the_corpus(self):
        with mock.patch("admin.executor._memories_for_query") as memory:
            memory.return_value = [self._memory_hit()]
            with mock.patch("admin.executor.documents_service.search_documents_sync") as corpus:
                result = self._call()
        assert corpus.call_count == 0  # memory answered; no corpus query
        assert result["output"].startswith("[ok]")
        assert "long-term memory" in result["output"]
        assert "no corpus query needed" in result["output"]
        assert "WiredTiger eviction begins" in result["output"]
        assert "semantic" in result["output"]

    def test_weak_memory_match_falls_through_to_corpus(self):
        with mock.patch("admin.executor._memories_for_query") as memory:
            memory.return_value = [self._memory_hit(score=0.55)]
            with mock.patch(
                "admin.executor.documents_service.search_documents_sync"
            ) as corpus:
                corpus.return_value = [
                    {"document_id": "book", "document_title": "Book", "chunk_index": 3,
                     "score": 0.5, "text": "cache eviction"}
                ]
                result = self._call()
        corpus.assert_called_once()
        assert "(chunk 3)" in result["output"]

    def test_empty_memory_falls_through_to_corpus(self):
        with mock.patch("admin.executor._memories_for_query", return_value=[]):
            with mock.patch(
                "admin.executor.documents_service.search_documents_sync"
            ) as corpus:
                corpus.return_value = [
                    {"document_id": "book", "document_title": "Book", "chunk_index": 3,
                     "score": 0.5, "text": "cache eviction"}
                ]
                result = self._call()
        corpus.assert_called_once()

    def test_memory_search_failure_degrades_to_corpus(self):
        # the REAL _memories_for_query catches store failures internally, so
        # a broken store must never block the corpus path
        with mock.patch("utils.llm.get_embeddings", side_effect=RuntimeError("store down")):
            with mock.patch(
                "admin.executor.documents_service.search_documents_sync"
            ) as corpus:
                corpus.return_value = [
                    {"document_id": "book", "document_title": "Book", "chunk_index": 3,
                     "score": 0.5, "text": "cache eviction"}
                ]
                result = self._call()
        corpus.assert_called_once()

    def test_memory_lookup_uses_agent_scoped_user_id(self):
        # user_id threads from the run (agent:<id>) into the memory search;
        # derived from the workspace directory name when not provided.
        from admin.executor import execute_tool_call
        from pathlib import Path as P

        with mock.patch("admin.executor._memories_for_query") as memory:
            memory.return_value = [self._memory_hit()]
            with mock.patch("admin.executor.documents_service.search_documents_sync") as corpus:
                execute_tool_call(
                    {"name": "documents_search", "arguments": '{"query": "x"}'},
                    P("/tmp/sub-abc123"),
                    user_id="agent:sub-abc123",
                )
        memory.assert_called_once_with("x", "agent:sub-abc123")


class TestMemoriesForQuery:
    """The memory sweep: own + default + other agents' namespaces."""

    def _sweep(self, docs, embedder_vector=(1.0, 0.0)):
        from admin.executor import _memories_for_query

        coll = FakeCollection(docs)
        client = mock.Mock()
        db = mock.Mock()
        db.__getitem__ = mock.Mock(side_effect=lambda name: coll if name == "agent_memories" else mock.Mock())
        client.__getitem__ = mock.Mock(return_value=db)
        embeddings = mock.Mock()
        embeddings.embed_query = mock.Mock(return_value=list(embedder_vector))
        with mock.patch("persistence.mongo.get_client", return_value=client):
            with mock.patch("utils.llm.get_embeddings", return_value=embeddings):
                return _memories_for_query("cache eviction", "agent:aaa")

    def _doc(self, namespace, content, embedding, **value_extra):
        return {
            "namespace": namespace,
            "embedding": embedding,
            "value": {"content": content, "type": "semantic", **value_extra},
        }

    def test_covers_own_default_and_other_agents_but_not_foreign_users(self):
        hits = self._sweep(
            [
                self._doc(["users", "agent:aaa", "memories", "semantic"], "own fact", [1.0, 0.0]),
                self._doc(["users", "default", "memories", "semantic"], "user fact", [0.8, 0.6]),
                self._doc(["users", "agent:bbb", "memories", "semantic"], "other agent fact", [0.6, 0.8]),
                self._doc(["users", "stranger", "memories", "semantic"], "foreign fact", [0.9, 0.1]),
            ]
        )
        texts = [h["text"] for h in hits]
        assert texts == ["own fact", "user fact", "other agent fact"]  # score-desc
        assert "foreign fact" not in texts
        assert hits[0]["similarity_score"] == 1.0
        assert hits[0]["source"] == "semantic"

    def test_skips_superseded_and_empty_memories(self):
        hits = self._sweep(
            [
                self._doc(["users", "default", "memories", "semantic"], "good fact", [1.0, 0.0]),
                self._doc(
                    ["users", "default", "memories", "semantic"], "stale fact", [1.0, 0.0],
                    superseded_by_snapshot=True,
                ),
                self._doc(["users", "default", "memories", "semantic"], "", [1.0, 0.0]),
            ]
        )
        assert [h["text"] for h in hits] == ["good fact"]

    def test_unreachable_store_returns_empty(self):
        from admin.executor import _memories_for_query

        with mock.patch("persistence.mongo.get_client", side_effect=RuntimeError("db down")):
            assert _memories_for_query("q", "agent:aaa") == []


class TestCitationFooter:
    """One-line Sources footer for a grounded answer (never inline).

    The line is parsed deterministically from the documents_search tool
    digests stored in the run's step transcripts and lists the ingested
    resource(s) — proof of grounding, nothing more.
    """

    DIGEST = "\n".join([
        "[ok] 5 matching chunk(s) from the ingested corpus:",
        "  [0.52] High Performance MongoDB — Asya Kamsky (chunk 202): Storage Engines154 Changing how eviction works While Atlas",
        "  [0.48] High Performance MongoDB — Asya Kamsky (chunk 386): Chapter 14 321 Addressing disk I/O bottlenecks",
    ])

    @staticmethod
    def _steps_with(outputs):
        return [
            {"response": "a", "tool_calls": [{"name": "documents_search", "output": out}]}
            for out in outputs
        ]

    def test_parses_digest_into_one_line(self):
        footer = docs.citation_footer(self._steps_with([self.DIGEST]))
        assert footer == "Sources: High Performance MongoDB — Asya Kamsky"
        assert "chunk" not in footer
        assert "[0.52]" not in footer
        assert "“" not in footer

    def test_dedupes_repeated_chunks_keeping_best_score(self):
        first = "  [0.40] Book (chunk 7): first snippet"
        second = "  [0.55] Book (chunk 7): second deeper snippet"
        footer = docs.citation_footer(self._steps_with([first, second]))
        assert footer == "Sources: Book"

    def test_multiple_resources_listed_in_one_line(self):
        a = "  [0.40] Book A (chunk 1): snippet"
        b = "  [0.55] Book B (chunk 2): snippet"
        footer = docs.citation_footer(self._steps_with([a, b]))
        assert footer == "Sources: Book B, Book A"  # best score first

    def test_none_without_grounded_calls(self):
        assert docs.citation_footer([]) is None
        assert docs.citation_footer(
            [{"tool_calls": [{"name": "web_search", "output": "[ok] web hit"}]}]
        ) is None
        assert docs.citation_footer(
            self._steps_with(["[error] documents_search found no matches — try different wording"])
        ) is None

    def test_resources_capped_keeping_line_short(self):
        digest = "\n".join(
            [f"  [{0.9 - i / 100:.2f}] Title {i} (chunk {i}): snippet" for i in range(12)]
        )
        footer = docs.citation_footer(self._steps_with([digest]))
        assert "…" in footer
        assert len(footer.split(", ")) <= 9  # "Sources:" + 8 titles + truncation
        assert "Title 0" in footer and "Title 11" not in footer


class TestDocumentsRoutes(unittest.TestCase):
    """Thin route layer: /api/admin/documents/* (see api/admin_routes.py)."""

    def setUp(self):
        from api import admin_routes

        self.routes = admin_routes

    def test_list_documents_route(self):
        with mock.patch.object(self.routes.docs, "list_documents") as fake:
            fake.return_value = [{"doc_id": "d1", "filename": "x.pdf"}]
            body = _run(self.routes.list_documents())
        assert body == {"documents": [{"doc_id": "d1", "filename": "x.pdf"}]}

    def test_upload_route_saves_file_and_schedules_ingest(self):
        from starlette.datastructures import UploadFile

        with tempfile.TemporaryDirectory() as tmp:
            uploaded = UploadFile(file=io.BytesIO(b"hello world"), filename="hello.txt")
            with mock.patch.object(Config, "DOCUMENTS_DIR", tmp):
                with mock.patch.object(
                    self.routes.docs, "docs_collection"
                ) as get_docs, mock.patch.object(
                    self.routes.docs, "chunks_collection"
                ) as get_chunks:
                    docs_coll = FakeCollection()
                    chunks_coll = FakeCollection()
                    get_docs.return_value = docs_coll
                    get_chunks.return_value = chunks_coll
                    body = _run(
                        self.routes.upload_document(file=uploaded, title="Hello")
                    )
            assert body["document"]["filename"] == "hello.txt"
            assert body["document"]["title"] == "Hello"
            assert body["document"]["status"] == "ingesting"
            assert body["document"]["size_bytes"] == 11
            doc = docs_coll.docs[0]
            saved = Path(doc["path"])
            assert saved.read_text() == "hello world"
            assert doc["status"] == "ingesting"

    def test_reembed_route_resets_status_and_schedules(self):
        from starlette.background import BackgroundTasks

        failed = {
            "_id": "doc1", "doc_id": "doc1", "filename": "x.pdf",
            "path": "/tmp/x.pdf", "title": "X", "status": "failed",
            "error": "boom", "chunk_count": 0,
        }
        ingesting = dict(failed, status="ingesting", error=None)
        with mock.patch.object(self.routes.docs, "get_document") as get_doc:
            get_doc.side_effect = [failed, ingesting]
            with mock.patch.object(self.routes.docs, "set_status") as set_status:
                body = _run(
                    self.routes.reembed_document("doc1", background_tasks=BackgroundTasks())
                )
        assert body["document"]["status"] == "ingesting"
        assert body["document"]["error"] is None
        set_status.assert_called_once_with("doc1", mock.ANY, "ingesting", error=None)

    def test_upload_rejects_unsupported_extension(self):
        from starlette.datastructures import UploadFile

        try:
            with mock.patch.object(Config, "DOCUMENTS_DIR", "/tmp/nope"):
                _run(
                    self.routes.upload_document(
                        file=UploadFile(file=io.BytesIO(b"x"), filename="z.docx"),
                        title=None,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            assert getattr(exc, "status_code", None) == 400
        else:
            raise AssertionError("expected 400 for unsupported extension")

    def test_search_route_passes_through(self):
        with mock.patch.object(self.routes.docs, "search_documents") as fake:
            fake.return_value = [{"document_id": "d", "score": 0.9, "text": "hit"}]
            body = _run(
                self.routes.search_documents(
                    self.routes.DocumentSearchRequest(query="q", k=3)
                )
            )
        assert body["results"][0]["text"] == "hit"
        fake.assert_called_once()
        args = fake.call_args
        assert args.kwargs["k"] == 3 and args.kwargs["query"] == "q"

    def test_delete_route_not_found(self):
        with mock.patch.object(self.routes.docs, "delete_document") as fake:
            fake.return_value = False
            try:
                _run(self.routes.delete_document("nope"))
            except Exception as exc:  # noqa: BLE001
                assert getattr(exc, "status_code", None) == 404
            else:
                raise AssertionError("expected 404")

    def test_delete_route_ok(self):
        with mock.patch.object(self.routes.docs, "delete_document") as fake:
            fake.return_value = True
            assert _run(self.routes.delete_document("doc1")) == {"deleted": True}

    def test_chunks_route_returns_text_preview(self):
        with mock.patch.object(self.routes.docs, "get_chunks") as fake:
            fake.return_value = [{"chunk_index": 0, "text": "cache"}]
            body = _run(self.routes.list_document_chunks("doc1", limit=5))
        assert body["doc_id"] == "doc1"
        assert body["total"] == 1
        assert body["chunks"][0]["text"] == "cache"

class PensiveApiToolTest(unittest.TestCase):
    """Agents must reach Pensive's own API natively — never by curling it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _execute(self, args: dict):
        from admin import executor

        return executor.execute_tool_call(
            {"name": "pensive_api", "arguments": json.dumps(args)},
            Path(self.tmp.name),
        )

    def test_tool_schema_registers_native_operations(self):
        from admin import executor

        tool = next(
            t for t in executor.DEFAULT_AGENT_TOOLS
            if (t.get("function") or {}).get("name") == "pensive_api"
        )
        fn = tool["function"]
        operations = fn["parameters"]["properties"]["operation"]["enum"]
        self.assertEqual(operations, ["list_documents", "search_documents"])
        description = fn["description"].lower()
        self.assertIn("never", description)
        self.assertIn("curl", description)

    def test_list_documents_returns_inventory(self):
        from admin import executor

        with mock.patch(
            "admin.executor.documents_service.docs_collection", return_value=object()
        ), mock.patch(
            "admin.executor.documents_service.list_documents",
            return_value=[
                {"id": "doc_1", "name": "hpmongo.pdf", "status": "ready", "created_at": "t1"},
                {"id": "doc_2", "name": "notes.txt", "status": "ready", "created_at": "t2"},
            ],
        ):
            result = self._execute({"operation": "list_documents"})
        self.assertEqual(result["name"], "pensive_api")
        self.assertIn("2 document(s)", result["output"])
        self.assertIn("doc_1", result["output"])
        self.assertIn("hpmongo.pdf", result["output"])

    def test_search_documents_delegates_to_corpus_search(self):
        from admin import executor

        with mock.patch(
            "admin.executor._handle_documents_search", return_value="[ok] chunk results"
        ) as handler:
            result = self._execute({"operation": "search_documents", "query": "wiredtiger"})
        handler.assert_called_once()
        self.assertIn("[ok]", result["output"])

    def test_unknown_operation_returns_guidance(self):
        result = self._execute({"operation": "drop_everything"})
        self.assertIn("[error]", result["output"])
        self.assertIn("list_documents", result["output"])
