"""Document management: queryable embeddings from uploaded documents.

Ingestion pipeline
------------------
1. Extract text from an uploaded file (PDF via pypdf, or plain txt/md).
2. Split it into bounded *segments* (the contextualized embedding API bills a
   120K-token budget per request, so segments stay well under that).
3. Call the Voyage contextualized embeddings endpoint (``voyage-context-4``)
   with ``input_type=document`` and server-side auto-chunking (512 tokens):
   the backend splits each segment into chunks and embeds every chunk in the
   context of its full segment, so vectors capture document context without
   any manual metadata augmentation.
4. Store chunk text + vector pairs in ``document_chunks`` and status bookkeeping
   in ``documents`` (ingesting → ready | failed).

Query side
----------
The same endpoint encodes the query with ``input_type=query``; the top-k
matching chunks are found with cosine similarity over the stored vectors.
At lab scale (a handful of documents, thousands of chunks) a pure-Python scan
is fast enough and needs no Atlas Search index; swap in ``$vectorSearch`` when
the corpus grows.
"""
from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from config import Config
from persistence.mongo import get_client

logger = logging.getLogger(__name__)

# Supported upload types: PDF + anything trivially chunkable as text.
SUPPORTED_SUFFIXES = frozenset({".pdf", ".txt", ".md"})

# Segments this large are comfortably inside the API's 120K-token per-request
# budget even after automatic chunking (chunks carry overhead tokens).
MAX_SEGMENT_CHARS = 90000

# Total characters per embedding REQUEST. The contextualized API caps the
# whole batch (all inputs in one call) at 120K tokens, so segments are sent
# in char-budgeted batches; each input is still auto-chunked and embedded in
# the context of its own segment. 300K chars ≈ 75K tokens — headroom for
# token-dense text (code listings) without wasting the window.
MAX_BATCH_CHARS = 300000

# Server-side chunking size in tokens (contextualized embedding API).
CHUNK_SIZE = 512


class DocumentIngestError(RuntimeError):
    """Raised when a document cannot be extracted or embedded."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- collections -------------------------------------------------------------


def docs_collection():
    return get_client()[Config.MONGODB_DB]["documents"]


def chunks_collection():
    return get_client()[Config.MONGODB_DB]["document_chunks"]


# --- extraction and segmentation ---------------------------------------------


def extract_text(source_path: str | Path) -> tuple[str, int]:
    """Return (text, page_count) for a PDF / txt / md file.

    Page count is 0 for plain-text files. Raises DocumentIngestError for
    unsupported types or unreadable content.
    """
    path = Path(source_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - guarded by requirements
            raise DocumentIngestError("pypdf is not installed") from exc
        try:
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:  # noqa: BLE001 - broken/encrypted PDFs vary wildly
            raise DocumentIngestError(f"could not read PDF: {type(exc).__name__}: {exc}") from exc
        return "\n\n".join(pages), len(pages)
    if suffix in {".txt", ".md"}:
        try:
            return path.read_text(encoding="utf-8", errors="replace"), 0
        except OSError as exc:
            raise DocumentIngestError(f"could not read {suffix} file: {exc}") from exc
    raise DocumentIngestError(
        f"unsupported file type {suffix or '(none)'} — supported: "
        f"{', '.join(sorted(SUPPORTED_SUFFIXES))}"
    )


def segment_text(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> list[str]:
    """Split text into bounded segments, preferring paragraph boundaries.

    Paragraphs are kept whole when they fit; an oversized paragraph is
    hard-split so every segment respects the token budget. Separators between
    segments may be dropped at cut boundaries (each chunk is self-contained,
    so a lost paragraph break is harmless for embedding quality).
    """
    max_chars = max(16, int(max_chars))
    segments: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            segments.append("\n\n".join(current))
        current = []
        current_len = 0

    for paragraph in text.split("\n\n"):
        if not paragraph:
            continue
        if len(paragraph) > max_chars:
            # Oversized paragraph: hard-split at the cap (contiguous cut, so
            # the full text is still recoverable).
            flush()
            while len(paragraph) > max_chars:
                segments.append(paragraph[:max_chars])
                paragraph = paragraph[max_chars:]
            current.append(paragraph)
            current_len = len(paragraph)
            continue
        if current and current_len + 2 + len(paragraph) > max_chars:
            flush()
        if current:
            current_len += 2
        current.append(paragraph)
        current_len += len(paragraph)
    flush()
    return segments


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# --- contextualized embedding client -----------------------------------------


class ContextualEmbedder:
    """Client for the Voyage contextualized embeddings endpoint.

    ``embed_documents`` sends segments with ``input_type=document`` and
    server-side auto-chunking, so each returned chunk carries an embedding
    computed in the context of its whole segment. ``embed_query`` uses
    ``input_type=query``; both encodings are compatible for retrieval.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0,
    ) -> None:
        self._url = url or Config.CONTEXT_EMBEDDING_URI
        self._api_key = api_key if api_key is not None else Config.LLM_EMBEDDING_KEY
        self._model = model or Config.CONTEXT_EMBEDDING_MODEL
        self._timeout = timeout

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    @staticmethod
    def _parse(data: dict) -> tuple[list[dict], int]:
        """Parse the contextualized response into (chunks, total_tokens)."""
        chunks: list[dict] = []
        for doc in data.get("data") or []:
            for item in doc.get("data") or []:
                chunks.append(
                    {
                        "text": str(item.get("text") or ""),
                        "embedding": list(item.get("embedding") or []),
                    }
                )
        usage = data.get("usage") or {}
        return chunks, int(usage.get("total_tokens") or 0)

    def _raise(self, response: httpx.Response) -> None:
        raise DocumentIngestError(
            f"contextualized embedding API {response.status_code}: {response.text[:300]}"
        )

    def _payload(self, inputs: list[str], input_type: str) -> dict:
        payload: dict = {
            "model": self._model,
            "inputs": inputs,
            "input_type": input_type,
        }
        if input_type == "document":
            payload["enable_auto_chunking"] = True
            payload["chunk_size"] = CHUNK_SIZE
        return payload

    async def embed_documents(self, segments: list[str]) -> tuple[list[dict], int]:
        """Embed document segments; returns (chunks, total_tokens)."""
        return await self.embed_documents_core(list(segments), "document")

    async def embed_query(self, query: str) -> list[float]:
        chunks, _ = await self.embed_documents_core([query], "query")
        return chunks[0]["embedding"] if chunks else []

    async def embed_documents_core(self, inputs: list[str], input_type: str) -> tuple[list[dict], int]:
        """Shared async POST used by documents and queries."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                self._url, json=self._payload(inputs, input_type), headers=self._headers()
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                self._raise(response)
        return self._parse(response.json())

    def embed_documents_sync(self, segments: list[str]) -> tuple[list[dict], int]:
        """Synchronous document embedding (agent tool path)."""
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                self._url, json=self._payload(list(segments), "document"), headers=self._headers()
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                self._raise(response)
        return self._parse(response.json())

    def embed_query_sync(self, query: str) -> list[float]:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                self._url, json=self._payload([query], "query"), headers=self._headers()
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                self._raise(response)
        chunks, _ = self._parse(response.json())
        return chunks[0]["embedding"] if chunks else []


# --- document lifecycle -------------------------------------------------------


def create_document(
    doc_id: str,
    filename: str,
    stored_path: str,
    size_bytes: int,
    title: Optional[str] = None,
) -> dict:
    """Document record in 'ingesting' state (uploaded, not yet embedded)."""
    now = _now()
    return {
        "_id": doc_id,
        "doc_id": doc_id,
        "filename": filename,
        "title": (title or "").strip() or Path(filename).stem,
        "path": stored_path,
        "suffix": Path(filename).suffix.lower(),
        "size_bytes": int(size_bytes),
        "pages": 0,
        "status": "ingesting",
        "error": None,
        "chunk_count": 0,
        "tokens_total": 0,
        "model": Config.CONTEXT_EMBEDDING_MODEL,
        "created_at": now,
        "updated_at": now,
    }


def set_status(doc_id: str, docs_coll, status: str, **fields) -> None:
    set_fields = {"status": status, "updated_at": _now(), **fields}
    logger.info("document %s → %s", doc_id, status)
    docs_coll.update_one({"_id": doc_id}, {"$set": set_fields})


def get_document(doc_id: str, docs_coll) -> Optional[dict]:
    return docs_coll.find_one({"_id": doc_id})


def public_doc(doc: dict) -> dict:
    """Public document shape: never leak internal storage paths."""
    return {key: value for key, value in dict(doc).items() if key not in {"_id", "path"}}


def list_documents(docs_coll) -> list[dict]:
    rows = list(docs_coll.find({}))
    rows.sort(key=lambda d: str(d.get("created_at") or ""), reverse=True)
    return [public_doc(doc) for doc in rows]


def delete_document(doc_id: str, docs_coll, chunks_coll) -> bool:
    doc = get_document(doc_id, docs_coll)
    if doc is None:
        return False
    stored = doc.get("path")
    if stored:
        try:
            Path(stored).unlink(missing_ok=True)
        except OSError as exc:  # noqa: BLE001
            logger.warning("document file cleanup failed for %s: %s", doc_id, exc)
    chunks_coll.delete_many({"doc_id": doc_id})
    docs_coll.delete_one({"_id": doc_id})
    return True


def get_chunks(doc_id: str, chunks_coll, limit: int = 25) -> list[dict]:
    """Chunk text preview (no embeddings) sorted by chunk order."""
    limit = max(1, min(int(limit), 100))
    rows = sorted(
        chunks_coll.find({"doc_id": doc_id}), key=lambda c: int(c.get("chunk_index") or 0)
    )
    return [{"chunk_index": int(c.get("chunk_index") or 0), "text": str(c.get("text") or "")} for c in rows[:limit]]


async def ingest_document(
    doc_id: str,
    source_path: str | Path,
    filename: str,
    docs_coll=None,
    chunks_coll=None,
    embedder: Optional[ContextualEmbedder] = None,
    title: Optional[str] = None,
    max_chars: int = MAX_SEGMENT_CHARS,
) -> dict:
    """Extract, embed (voyage-context-4, contextualized) and index a document.

    Runs in the background after upload / re-embed. Never raises: failures
    are recorded on the document record so the UI can offer a retry.
    """
    # NOTE: pymongo Collection objects raise on truthiness tests, so the
    # "or" fallback pattern is deliberately avoided here.
    docs_coll = docs_collection() if docs_coll is None else docs_coll
    chunks_coll = chunks_collection() if chunks_coll is None else chunks_coll
    embedder = ContextualEmbedder() if embedder is None else embedder
    try:
        if get_document(doc_id, docs_coll) is None:
            # Belt-and-braces: the caller (upload/reembed) always inserts the
            # record first, but a direct ingest should still bookend itself.
            docs_coll.insert_one(
                create_document(doc_id, filename, str(source_path), 0, title=title)
            )
        logger.info("ingesting %s (%s)", filename, source_path)
        text, pages = extract_text(source_path)
        if not text.strip():
            raise DocumentIngestError("no text content found in the file")
        segments = segment_text(text, max_chars=max_chars)
        logger.info("document %s: %d segments, %d pages", doc_id, len(segments), pages)
        # Batch segments into requests that stay inside the API's per-request
        # token budget; chunk order follows document order.
        batches: list[list[str]] = []
        current: list[str] = []
        current_chars = 0
        for segment in segments:
            if current and current_chars + len(segment) > MAX_BATCH_CHARS:
                batches.append(current)
                current = []
                current_chars = 0
            current.append(segment)
            current_chars += len(segment)
        if current:
            batches.append(current)
        chunks: list[dict] = []
        total_tokens = 0
        for batch in batches:
            batch_chunks, batch_tokens = await embedder.embed_documents(batch)
            chunks.extend(batch_chunks)
            total_tokens += int(batch_tokens)
        if not chunks:
            raise DocumentIngestError("embedding returned no chunks")
        chunks_coll.delete_many({"doc_id": doc_id})
        for index, chunk in enumerate(chunks):
            chunks_coll.insert_one(
                {
                    "doc_id": doc_id,
                    "chunk_index": index,
                    "text": str(chunk.get("text") or "")[:4000],
                    "embedding": list(chunk.get("embedding") or []),
                }
            )
        set_status(
            doc_id,
            docs_coll,
            "ready",
            chunk_count=len(chunks),
            pages=pages,
            tokens_total=int(total_tokens),
            error=None,
        )
    except DocumentIngestError as exc:
        logger.warning("document %s failed: %s", doc_id, exc)
        set_status(doc_id, docs_coll, "failed", error=str(exc))
    doc = get_document(doc_id, docs_coll) or {}
    return public_doc(doc)


# --- retrieval ---------------------------------------------------------------

# One line of the documents_search digest, e.g.
#   [0.52] High Performance MongoDB — Asya Kamsky (chunk 202): text…
_CITATION_LINE = re.compile(
    r"^\s*\[([0-9.]+)\]\s+(.+?)\s+\(chunk\s+(\d+)\):\s*(.*)$"
)
_CITE_LIMIT = 8


def citation_footer(steps: list[dict], limit: int = _CITE_LIMIT) -> Optional[str]:
    """One-line Sources suffix for an answer grounded in ``documents_search``.

    Parses the chunk digests already recorded in the run's step transcripts
    and returns a single line naming the ingested resource(s) referenced —
    e.g. ``Sources: High Performance MongoDB — Asya Kamsky`` — deduped by
    title, best chunk score first, capped so the line stays short. Returns
    ``None`` when the run never retrieved chunks, so ungrounded answers are
    never decorated with fabricated sources.
    """
    best: dict[tuple[str, int], float] = {}
    for step in steps or []:
        for call in step.get("tool_calls") or []:
            if call.get("name") != "documents_search":
                continue
            for line in str(call.get("output") or "").splitlines():
                match = _CITATION_LINE.match(line.strip())
                if not match:
                    continue
                score = float(match.group(1))
                title = match.group(2).strip()
                chunk = int(match.group(3))
                key = (title, chunk)
                if key not in best or score > best[key]:
                    best[key] = score
    if not best:
        return None
    ordered = sorted(best.items(), key=lambda pair: pair[1], reverse=True)
    titles: list[str] = []
    for (title, _chunk), _score in ordered:
        if title not in titles:
            titles.append(title)
        if len(titles) >= max(1, int(limit)):
            break
    line = "Sources: " + ", ".join(titles)
    if len(titles) < len({title for title, _ in best}):
        line += " …"
    return line


def _rank(
    vector: list[float],
    docs_coll,
    chunks_coll,
    k: int,
    doc_id: Optional[str] = None,
) -> list[dict]:
    query = {"doc_id": doc_id} if doc_id else {}
    rows = list(chunks_coll.find(query))
    scored = []
    for chunk in rows:
        score = cosine(vector, list(chunk.get("embedding") or []))
        if score <= 0.0:
            continue
        scored.append((score, chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    results = []
    titles: dict[str, str] = {}
    for score, chunk in scored[:k]:
        owner = str(chunk.get("doc_id") or "")
        if owner not in titles:
            owner_doc = docs_coll.find_one({"_id": owner})
            titles[owner] = (owner_doc or {}).get("title") or (owner_doc or {}).get("filename") or owner
        results.append(
            {
                "document_id": owner,
                "document_title": titles[owner],
                "chunk_index": int(chunk.get("chunk_index") or 0),
                "score": round(score, 4),
                "text": str(chunk.get("text") or ""),
            }
        )
    return results


async def search_documents(
    query: str,
    docs_coll=None,
    chunks_coll=None,
    embedder: Optional[ContextualEmbedder] = None,
    k: int = 5,
    doc_id: Optional[str] = None,
) -> list[dict]:
    """Top-k chunks for a query (async path; UI + re-ranking)."""
    embedder = ContextualEmbedder() if embedder is None else embedder
    vector = await embedder.embed_query(query)
    return _rank(
        vector,
        docs_collection() if docs_coll is None else docs_coll,
        chunks_collection() if chunks_coll is None else chunks_coll,
        k=max(1, min(int(k), 20)),
        doc_id=doc_id,
    )


def search_documents_sync(
    query: str,
    docs_coll=None,
    chunks_coll=None,
    k: int = 5,
    doc_id: Optional[str] = None,
) -> list[dict]:
    """Top-k chunks for a query (sync path; the documents_search agent tool)."""
    embedder = ContextualEmbedder()
    vector = embedder.embed_query_sync(query)
    return _rank(
        vector,
        docs_collection() if docs_coll is None else docs_coll,
        chunks_collection() if chunks_coll is None else chunks_coll,
        k=max(1, min(int(k), 20)),
        doc_id=doc_id,
    )