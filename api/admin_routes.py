"""Pensive admin REST API: memory curation, system prompt, and test agents."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from admin import service as admin
from config import Config
from documents import service as docs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])

# Uploaded originals cap — big enough for books/whitepapers, small enough to
# keep the documents volume and extraction times sane.
MAX_UPLOAD_BYTES = 128 * 1024 * 1024


def _bad_request(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


class MemoryUpdateRequest(BaseModel):
    content: str


class MemoryMergeRequest(BaseModel):
    user_id: str = "default"
    mem_type: Optional[str] = None


class SystemPromptRequest(BaseModel):
    text: str = ""


class AgentRequest(BaseModel):
    name: str
    system_prompt: str = ""
    tools: Optional[list[dict]] = None


class AgentUpdateRequest(BaseModel):
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    tools: Optional[list[dict]] = None


class AgentRunRequest(BaseModel):
    task: str


class DocumentSearchRequest(BaseModel):
    query: str
    k: int = 5
    doc_id: Optional[str] = None


@router.get("/memory-types")
async def get_memory_types() -> dict:
    return {"types": admin.memory_types()}


@router.get("/memories")
async def list_memories(user_id: str = "default", mem_type: Optional[str] = None) -> dict:
    try:
        rows = admin.list_memories(user_id, mem_type)
    except ValueError as exc:
        raise _bad_request(exc)
    counts = admin.memory_counts(user_id)
    return {"user_id": user_id, "memories": rows, "counts": counts}


@router.put("/memories/{mem_type}/{key}")
async def update_memory(
    mem_type: str, key: str, body: MemoryUpdateRequest, user_id: str = "default"
) -> dict:
    try:
        record = admin.update_memory(user_id, mem_type, key, body.content)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise _bad_request(exc)
    return {"memory": record}


@router.post("/memories/merge-duplicates")
async def merge_duplicate_memories(body: MemoryMergeRequest) -> dict:
    """Group semantically similar memories into one complete curated memory
    each (newest record wins as anchor, contents merged, copies deleted)."""
    try:
        return await admin.merge_duplicate_memories(body.user_id, body.mem_type)
    except ValueError as exc:
        raise _bad_request(exc)


@router.delete("/memories/{mem_type}/{key}")
async def delete_memory(mem_type: str, key: str, user_id: str = "default") -> dict:
    try:
        deleted = admin.delete_memory(user_id, mem_type, key)
    except ValueError as exc:
        raise _bad_request(exc)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"memory not found: {mem_type}/{key}")
    return {"deleted": True}


@router.get("/system-prompt")
async def get_system_prompt() -> dict:
    return {"text": admin.get_system_prompt()}


@router.put("/system-prompt")
async def set_system_prompt(body: SystemPromptRequest) -> dict:
    text = admin.set_system_prompt(body.text)
    return {"text": text}


@router.get("/agents")
async def list_agents() -> dict:
    return {"agents": admin.list_agents()}


@router.post("/agents")
async def create_agent(body: AgentRequest) -> dict:
    try:
        agent = admin.create_agent(body.name, body.system_prompt, body.tools or [])
    except ValueError as exc:
        raise _bad_request(exc)
    return {"agent": agent}


@router.put("/agents/{agent_id}")
async def update_agent(agent_id: str, body: AgentUpdateRequest) -> dict:
    try:
        agent = admin.update_agent(
            agent_id,
            name=body.name,
            system_prompt=body.system_prompt,
            tools=body.tools,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise _bad_request(exc)
    return {"agent": agent}


@router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str) -> dict:
    if not admin.delete_agent(agent_id):
        raise HTTPException(status_code=404, detail=f"agent not found: {agent_id}")
    return {"deleted": True}


@router.post("/agents/{agent_id}/run")
async def run_agent(agent_id: str, body: AgentRunRequest, request: Request) -> dict:
    agent = admin.get_agent(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"agent not found: {agent_id}")
    graph: Any = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="pensive graph not ready")
    try:
        result = await admin.run_agent(agent, body.task, graph)
    except ValueError as exc:
        raise _bad_request(exc)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    try:
        record = admin.record_run(agent_id, body.task, result, executor="test")
        result = dict(result, run_id=record["run_id"])
    except Exception:  # noqa: BLE001 — history is best-effort, never fails the run
        logger.exception("Could not record agent run history")
    return result


@router.get("/agents/{agent_id}/runs")
async def list_agent_runs(agent_id: str) -> dict:
    return {"agent_id": agent_id, "runs": admin.list_runs(agent_id)}


@router.get("/runs")
async def list_recent_runs(
    limit: int = 25,
    sort_by: str = "created_at",
    order: str = "desc",
    executor: Optional[str] = None,
) -> dict:
    """Global workflow feed: latest runs across lab agents AND spawned
    subagents (``sub-*``). Sortable by ``created_at`` or efficiency metrics
    (``tokens_total``, ``duration_ms``, …) to hunt for low-token,
    high-performing agents worth promoting into reusable tools; filter to
    ``executor=pensive|test`` for who ran the workflow."""
    limit = max(1, min(int(limit), 100))
    try:
        runs = admin.list_recent_runs(
            limit=limit, sort_by=sort_by, order=order, executor=executor
        )
    except ValueError as exc:
        raise _bad_request(exc)
    return {"runs": runs}


@router.get("/agents/{agent_id}/workspace")
async def agent_workspace(agent_id: str) -> dict:
    return {"agent_id": agent_id, "files": admin.list_workspace_files(agent_id)}


@router.get("/agents/{agent_id}/workspace/file")
async def agent_workspace_file(agent_id: str, path: str) -> dict:
    try:
        content = admin.read_workspace_file(agent_id, path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise _bad_request(exc)
    return {"path": path, "content": content}


# --- documents: ingestion + library ------------------------------------------
#
# Uploads are embedded in the background with contextualized chunk embeddings
# (voyage-context-4) and indexed in document_chunks. The documents_search tool
# then lets agents query the corpus on demand — nothing is injected into
# prompts automatically.


@router.get("/documents")
async def list_documents() -> dict:
    return {"documents": docs.list_documents(docs.docs_collection())}


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
) -> dict:
    """Accept a PDF/txt/md upload; the actual embedding runs in the background.

    The document is recorded as ``ingesting`` immediately so the UI can show
    progress; a BackgroundTask drives it to ``ready`` or ``failed``.
    """
    if background_tasks is None:
        background_tasks = BackgroundTasks()
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in docs.SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unsupported file type {suffix or '(none)'} — supported: "
                f"{', '.join(sorted(docs.SUPPORTED_SUFFIXES))}"
            ),
        )
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file too large ({len(content)} bytes, max {MAX_UPLOAD_BYTES})",
        )
    doc_id = uuid.uuid4().hex[:12]
    stored_root = Path(Config.DOCUMENTS_DIR).expanduser()
    stored_root.mkdir(parents=True, exist_ok=True)
    stored_path = stored_root / f"{doc_id}{suffix}"
    stored_path.write_bytes(content)
    docs_coll = docs.docs_collection()
    chunks_coll = docs.chunks_collection()
    record = docs.create_document(doc_id, filename, str(stored_path), len(content), title=title)
    docs_coll.insert_one(record)
    background_tasks.add_task(
        docs.ingest_document, doc_id, str(stored_path), filename, docs_coll, chunks_coll, title=title
    )
    return {"document": docs.public_doc(record)}


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    if not docs.delete_document(doc_id, docs.docs_collection(), docs.chunks_collection()):
        raise HTTPException(status_code=404, detail=f"document not found: {doc_id}")
    return {"deleted": True}


@router.post("/documents/{doc_id}/reembed")
async def reembed_document(doc_id: str, background_tasks: BackgroundTasks = None) -> dict:
    """Retry a failed ingest (or re-run a successful one) in the background."""
    if background_tasks is None:
        background_tasks = BackgroundTasks()
    docs_coll = docs.docs_collection()
    record = docs.get_document(doc_id, docs_coll)
    if record is None:
        raise HTTPException(status_code=404, detail=f"document not found: {doc_id}")
    docs.set_status(doc_id, docs_coll, "ingesting", error=None)
    chunks_coll = docs.chunks_collection()
    background_tasks.add_task(
        docs.ingest_document,
        doc_id,
        record["path"],
        record["filename"],
        docs_coll,
        chunks_coll,
        title=record.get("title"),
    )
    return {"document": docs.public_doc(docs.get_document(doc_id, docs_coll))}


@router.get("/documents/{doc_id}/chunks")
async def list_document_chunks(doc_id: str, limit: int = 25) -> dict:
    """Chunk text preview (no embeddings) for the library UI."""
    chunks = docs.get_chunks(doc_id, docs.chunks_collection(), limit=limit)
    return {"doc_id": doc_id, "total": len(chunks), "chunks": chunks}


@router.post("/documents/search")
async def search_documents(body: DocumentSearchRequest) -> dict:
    """Semantic search over ingested chunks (contextualized embeddings)."""
    results = await docs.search_documents(
        query=body.query,
        docs_coll=docs.docs_collection(),
        chunks_coll=docs.chunks_collection(),
        k=body.k,
        doc_id=body.doc_id,
    )
    return {"query": body.query, "results": results}