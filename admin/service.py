"""Pensive admin service: memory curation, system prompt, and isolated test agents.

All curation writes go through the LangGraph MongoDBStore so vector embeddings
are (re)computed automatically; listing reads the underlying Mongo collection
(the Store API has no list-items call).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Collection, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from admin.executor import DEFAULT_AGENT_TOOLS, execute_tool_call, safe_workspace_path, workspace_root
from documents import service as documents_service
from glean.client import get_glean_client
from utils.llm import get_responses_client
from config import Config
from memory.memories import (
    MEMORY_MERGE_OVERLAP,
    MEMORY_MERGE_SCORE,
    MEMORY_TYPES,
    _token_containment,
    consolidate_memories,
    lexical_similarity,
    merge_texts,
    namespace,
)
from persistence.mongo import get_client, get_store

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_ID = "system_prompt"
DEFAULT_USER_ID = "default"

# Display metadata for the four long-term memory types.
MEMORY_TYPE_META: dict[str, dict] = {
    "semantic": {
        "description": "User facts and preferences",
    },
    "episodic": {
        "description": "Session events, decisions, and outcomes",
    },
    "taxonomic": {
        "description": "Domain-specific vocabulary and reusable concepts (e.g. corporate policies)",
    },
    "procedural": {
        "description": "Reusable workflows and agent runbooks — skills",
    },
}


def memory_types() -> list[dict]:
    """Return the four memory types with display descriptions."""
    return [
        {"type": mem_type, **MEMORY_TYPE_META[mem_type]}
        for mem_type in MEMORY_TYPES
    ]


# Workflow-history retention: at most this many run documents, and none older
# than this many days — whichever limit is hit first.
RUN_HISTORY_MAX_DOCS = 1500
RUN_HISTORY_MAX_AGE_DAYS = 30


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memories_collection(collection: Collection | None = None) -> Collection:
    if collection is not None:
        return collection
    return get_client()[Config.MONGODB_DB]["agent_memories"]


def _settings_collection(collection: Collection | None = None) -> Collection:
    if collection is not None:
        return collection
    return get_client()[Config.MONGODB_DB]["pensive_admin_settings"]


def _agents_collection(collection: Collection | None = None) -> Collection:
    if collection is not None:
        return collection
    return get_client()[Config.MONGODB_DB]["pensive_agents"]


def _runs_collection(collection: Collection | None = None) -> Collection:
    if collection is not None:
        return collection
    return get_client()[Config.MONGODB_DB]["pensive_agent_runs"]


def _resolve_store(store) -> Any:
    return store if store is not None else get_store()


# --- system prompt -----------------------------------------------------------

def get_system_prompt(collection: Collection | None = None) -> str:
    """Return the runtime system prompt (MongoDB override or env default)."""
    doc = _settings_collection(collection).find_one({"_id": SYSTEM_PROMPT_ID})
    if doc and doc.get("text"):
        return str(doc["text"])
    return Config.SYSTEM_PROMPT


def set_system_prompt(text: str, collection: Collection | None = None) -> str:
    """Persist the runtime system prompt. The empty string restores the default."""
    _settings_collection(collection).update_one(
        {"_id": SYSTEM_PROMPT_ID},
        {"$set": {"text": text, "updated_at": _now()}},
        upsert=True,
    )
    return text


# --- memory curation ---------------------------------------------------------

def list_memories(
    user_id: str = DEFAULT_USER_ID,
    mem_type: Optional[str] = None,
    *,
    collection: Collection | None = None,
    store=None,
) -> list[dict]:
    """List curatable memories for a user, optionally filtered by type."""
    del store  # store kept in the signature for dependency symmetry
    query: dict = {"value.user_id": user_id}
    if mem_type:
        query["value.type"] = mem_type
    rows: list[dict] = []
    for doc in _memories_collection(collection).find(query).sort("updated_at", -1):
        value = doc.get("value")
        if not isinstance(value, dict):
            continue
        rows.append(
            {
                "key": doc.get("key") or "",
                "type": value.get("type") or mem_type or "",
                "label": value.get("label") or "",
                "content": value.get("content") or "",
                "recorded_at": value.get("recorded_at"),
                "updated_at": doc.get("updated_at") or value.get("updated_at"),
            }
        )
    return rows


def memory_counts(
    user_id: str = DEFAULT_USER_ID,
    *,
    collection: Collection | None = None,
    store=None,
) -> dict[str, int]:
    """Per-type memory counts for a user, always computed UNFILTERED so panel
    badges stay correct regardless of any active type filter in the UI."""
    del store
    counts: dict[str, int] = {mem_type: 0 for mem_type in MEMORY_TYPES}
    for doc in _memories_collection(collection).find({"value.user_id": user_id}):
        value = doc.get("value")
        if not isinstance(value, dict):
            continue  # legacy string-encoded records are not curatable
        mem_type = value.get("type")
        if mem_type in counts:
            counts[mem_type] += 1
    return counts


def _record_from_value(key: str, value: dict) -> dict:
    return {
        "key": key,
        "type": value.get("type") or "",
        "label": value.get("label") or "",
        "content": value.get("content") or "",
        "recorded_at": value.get("recorded_at"),
        "updated_at": value.get("updated_at"),
    }


def update_memory(
    user_id: str,
    mem_type: str,
    key: str,
    content: str,
    *,
    collection: Collection | None = None,
    store=None,
) -> dict:
    """Correct a memory in place. ``store.put`` on the same key re-embeds it."""
    if mem_type not in MEMORY_TYPES:
        raise ValueError(f"unknown memory type: {mem_type}")
    if not content.strip():
        raise ValueError("memory content cannot be empty")
    coll = _memories_collection(collection)
    existing = coll.find_one(
        {"key": key, "value.user_id": user_id, "value.type": mem_type}
    )
    if existing is None:
        raise KeyError(f"memory not found: {mem_type}/{key}")
    value = dict(existing.get("value") or {})
    value["content"] = content
    value["updated_at"] = _now()
    _resolve_store(store).put(namespace(user_id, mem_type), key, value)
    return _record_from_value(key, value)


def delete_memory(
    user_id: str,
    mem_type: str,
    key: str,
    *,
    collection: Collection | None = None,
    store=None,
) -> bool:
    """Prune a memory. Returns True when a record was removed."""
    if mem_type not in MEMORY_TYPES:
        raise ValueError(f"unknown memory type: {mem_type}")
    coll = _memories_collection(collection)
    exists = coll.find_one(
        {"key": key, "value.user_id": user_id, "value.type": mem_type}
    ) is not None
    if exists:
        _resolve_store(store).delete(namespace(user_id, mem_type), key)
    return exists


def _similar_memory_peers(
    store: Any,
    user_id: str,
    mem_type: str,
    anchor_key: str,
    anchor_content: str,
    rows: list[dict],
    grouped: set,
) -> list[dict]:
    """Near-duplicate rows of the same memory (vector sim with lexical
    fallback), excluding the anchor and already-grouped keys."""
    try:
        hits = store.search(namespace(user_id, mem_type), query=anchor_content, limit=8) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Memory merge search failed (%s: %s)", type(exc).__name__, exc
        )
        hits = []
    peers: list[dict] = []
    by_key = {r["key"]: r for r in rows}
    for item in hits:
        if isinstance(item, dict):
            value = item.get("value") or {}
            hit_key = item.get("key") or value.get("key")
            score = item.get("score")
        else:
            value = getattr(item, "value", None) or {}
            hit_key = getattr(item, "key", None) or value.get("key")
            score = getattr(item, "score", None)
        if not hit_key or hit_key == anchor_key or hit_key in grouped:
            continue
        if str(value.get("user_id") or "") != user_id:
            continue
        row = by_key.get(hit_key)
        if row is None:
            continue
        if score is not None:
            if float(score) < MEMORY_MERGE_SCORE:
                continue
        elif lexical_similarity(anchor_content, str(row.get("content") or "")) < MEMORY_MERGE_OVERLAP:
            continue
        peers.append(row)
    return peers


async def merge_duplicate_memories(
    user_id: str = DEFAULT_USER_ID,
    mem_type: Optional[str] = None,
    *,
    collection: Collection | None = None,
    store=None,
) -> dict:
    """Query all semantically similar memories together and curate each group
    into ONE complete memory: the most recently updated record is the anchor
    (its key, label, and newest position), near-identical lines collapse
    heuristically (longer version wins), an LLM consolidation pass folds the
    group's fragments into one flowing curated memory (best-effort; the
    heuristic result stands if the model fails), and the duplicate copies are
    deleted. Returns scan/merge/removal counts. Similarity detection uses
    vector search with a lexical fallback when embeddings are unavailable."""
    store_resolved = _resolve_store(store)
    types = [mem_type] if mem_type else list(MEMORY_TYPES)
    client = get_responses_client()
    scanned = 0
    merged = 0
    removed = 0
    polished = 0
    for mem_type_current in types:
        rows = list_memories(user_id, mem_type_current, collection=collection)
        grouped: set = set()
        for row in rows:  # newest-first: the anchor is the most recent record
            scanned += 1
            key = row["key"]
            if key in grouped:
                continue
            grouped.add(key)
            content = row["content"] or ""
            # Self-heal: a record that is itself a literal concatenation of
            # near-identical fragments collapses to its fuller version; rows
            # with suspiciously overlapping lines (paraphrase-level dupes)
            # get the LLM consolidation pass so they read as ONE memory.
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            if len(lines) > 1:
                collapsed = lines[0]
                for line in lines[1:]:
                    collapsed = merge_texts(collapsed, line)
                suspicious = any(
                    _token_containment(a, b) >= 0.5
                    for i, a in enumerate(lines)
                    for b in lines[i + 1 :]
                )
                polished_text = ""
                if suspicious or collapsed != content.strip():
                    polished_text = await consolidate_memories(client, lines) or collapsed
                if polished_text and polished_text != content.strip():
                    content = "\n".join(l.strip() for l in polished_text.splitlines() if l.strip())
                    polished += 1
                    store_resolved.put(
                        namespace(user_id, mem_type_current),
                        key,
                        {
                            "content": content,
                            "user_id": user_id,
                            "type": mem_type_current,
                            "label": row.get("label") or mem_type_current,
                            "recorded_at": row.get("recorded_at"),
                            "updated_at": _now(),
                        },
                    )
            peers = _similar_memory_peers(
                store_resolved, user_id, mem_type_current, key, content, rows, grouped
            )
            if not peers:
                continue
            merged += 1
            fragments = [content] + [p["content"] for p in peers]
            recorded_at = row.get("recorded_at")
            for peer in peers:
                recorded_at = recorded_at or peer.get("recorded_at")
                grouped.add(peer["key"])
                removed += 1
            collapsed = fragments[0]
            for fragment in fragments[1:]:
                collapsed = merge_texts(collapsed, fragment)
            # LLM polish: fold the collapsed fragments into one curated memory
            polished_text = await consolidate_memories(client, fragments)
            content = polished_text or collapsed
            content = "\n".join(l.strip() for l in content.splitlines() if l.strip())
            store_resolved.put(
                namespace(user_id, mem_type_current),
                key,
                {
                    "content": content,
                    "user_id": user_id,
                    "type": mem_type_current,
                    "label": row.get("label") or mem_type_current,
                    "recorded_at": recorded_at,
                    "updated_at": _now(),
                },
            )
            for peer in peers:
                store_resolved.delete(namespace(user_id, mem_type_current), peer["key"])
    return {"scanned": scanned, "merged": merged, "removed": removed, "polished": polished}


# --- test agents -------------------------------------------------------------

def _public_agent(doc: dict) -> dict:
    return {
        "id": doc.get("_id") or "",
        "name": doc.get("name") or "",
        "system_prompt": doc.get("system_prompt") or "",
        "tools": doc.get("tools") or [],
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


def list_agents(collection: Collection | None = None) -> list[dict]:
    coll = _agents_collection(collection)
    return [_public_agent(doc) for doc in coll.find({}).sort("name", 1)]


def get_agent(agent_id: str, collection: Collection | None = None) -> Optional[dict]:
    doc = _agents_collection(collection).find_one({"_id": agent_id})
    return _public_agent(doc) if doc else None


_AUTO_NAME_ADJECTIVES = (
    "amber", "azure", "coral", "emerald", "jade", "lunar", "rune", "solar", "velvet", "whisper",
)
_AUTO_NAME_NOUNS = (
    "scribe", "forger", "quill", "vault", "diary", "weaver", "keeper", "cipher", "folio", "oracle",
)


def autogen_agent_name() -> str:
    """Generate a friendly agent name when the user leaves it blank."""
    adjective = random.choice(_AUTO_NAME_ADJECTIVES)
    noun = random.choice(_AUTO_NAME_NOUNS)
    return f"{adjective}-{noun}-{uuid.uuid4().hex[:4]}"


def create_agent(
    name: str,
    system_prompt: str,
    tools: list[dict],
    *,
    collection: Collection | None = None,
) -> dict:
    name = str(name or "").strip() or autogen_agent_name()
    agent_id = uuid.uuid4().hex[:12]
    now = _now()
    doc = {
        "_id": agent_id,
        "name": name,
        "system_prompt": system_prompt,
        "tools": tools or DEFAULT_AGENT_TOOLS,
        "created_at": now,
        "updated_at": now,
    }
    _agents_collection(collection).insert_one(doc)
    return _public_agent(doc)


def update_agent(
    agent_id: str,
    name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    tools: Optional[list[dict]] = None,
    *,
    collection: Collection | None = None,
) -> dict:
    coll = _agents_collection(collection)
    doc = coll.find_one({"_id": agent_id})
    if doc is None:
        raise KeyError(f"agent not found: {agent_id}")
    updates: dict = {}
    if name is not None:
        if not str(name).strip():
            raise ValueError("agent name cannot be empty")
        updates["name"] = str(name).strip()
    if system_prompt is not None:
        updates["system_prompt"] = system_prompt
    if tools is not None:
        updates["tools"] = tools
    updates["updated_at"] = _now()
    coll.update_one({"_id": agent_id}, {"$set": updates})
    merged = dict(doc)
    merged.update(updates)
    return _public_agent(merged)


def delete_agent(agent_id: str, *, collection: Collection | None = None) -> bool:
    result = _agents_collection(collection).delete_one({"_id": agent_id})
    count = getattr(result, "deleted_count", result)
    return int(count or 0) > 0


# --- run history -------------------------------------------------------------

def _public_run(doc: dict) -> dict:
    return {
        "run_id": doc.get("_id") or "",
        "agent_id": doc.get("agent_id") or "",
        "task": doc.get("task") or "",
        "answer": doc.get("answer") or "",
        "decisions": doc.get("decisions") or [],
        "tool_calls": doc.get("tool_calls") or [],
        "steps": doc.get("steps") or [],
        "workspace": doc.get("workspace") or "",
        "status": doc.get("status") or "",
        "created_at": doc.get("created_at"),
        # Who executed the workflow: "pensive" (subagent spawned from a main
        # chat) or "test" (console lab-agent run). Legacy runs fall back to
        # the sub- agent-id prefix convention.
        "executor": doc.get("executor") or (
            "pensive" if str(doc.get("agent_id") or "").startswith("sub-") else "test"
        ),
        # Efficiency metrics (None on legacy runs so the UI can hide them).
        "tokens_in": doc.get("tokens_in"),
        "tokens_out": doc.get("tokens_out"),
        "tokens_total": (
            int(doc.get("tokens_in") or 0) + int(doc.get("tokens_out") or 0)
        )
        if doc.get("tokens_in") is not None or doc.get("tokens_out") is not None
        else 0,
        "duration_ms": doc.get("duration_ms"),
        # Legacy runs (recorded before the metrics existed) still carry their
        # step transcripts — derive the call/step totals from them so the
        # top-level view aggregates everything.
        "tool_call_count": (
            doc.get("tool_call_count")
            if doc.get("tool_call_count") is not None
            else (
                sum(len(s.get("tool_calls") or []) for s in (doc.get("steps") or []))
                or (len(doc.get("tool_calls") or []) or None)
            )
        ),
        "step_count": (
            doc.get("step_count")
            if doc.get("step_count") is not None
            else (len(doc.get("steps") or []) or None)
        ),
        # Definition of the agent that ran (name/system prompt/tools) when it
        # was an ephemeral subagent — lets the console promote it to a
        # reusable specialist agent.
        "agent": doc.get("agent"),
    }


def record_run(
    agent_id: str,
    task: str,
    result: dict,
    *,
    agent_def: Optional[dict] = None,
    executor: Optional[str] = None,
    collection: Collection | None = None,
) -> dict:
    """Persist a completed agent run so past work can be reviewed and re-run.

    The result is stored with bounded sizes (executed tool-call outputs are
    already truncated by the executor) so history stays light. ``agent_def``
    captures the ephemeral agent's definition (name / system prompt / tools)
    so a useful subagent can later be promoted to a reusable lab agent.
    ``executor`` tags who ran it: "pensive" (spawned from a chat) or "test"
    (console lab-agent run).
    """
    doc = {
        "_id": uuid.uuid4().hex[:12],
        "agent_id": agent_id,
        "task": str(task or "")[:2000],
        "answer": str(result.get("answer") or "")[:20000],
        "decisions": (result.get("decisions") or [])[:50],
        "tool_calls": (result.get("tool_calls") or [])[:50],
        "steps": (result.get("steps") or [])[:40],
        "workspace": str(result.get("workspace") or ""),
        "status": "completed",
        "created_at": _now(),
        "executor": str(executor or "")[:16] or None,
        "tokens_in": _positive_int(result.get("tokens_in")),
        "tokens_out": _positive_int(result.get("tokens_out")),
        "duration_ms": _positive_int(result.get("duration_ms")),
        "tool_call_count": _positive_int(result.get("tool_call_count")),
        "step_count": _positive_int(result.get("step_count")),
    }
    if agent_def and isinstance(agent_def, dict):
        tools = [t for t in agent_def.get("tools") or [] if isinstance(t, dict)][:12]
        doc["agent"] = {
            "name": str(agent_def.get("name") or "")[:200],
            "system_prompt": str(agent_def.get("system_prompt") or "")[:4000],
            "tools": tools,
        }
    _runs_collection(collection).insert_one(doc)
    try:
        prune_runs(collection=collection)
    except Exception:  # noqa: BLE001 — retention is best-effort
        logger.exception("could not prune run history")
    return _public_run(doc)


def prune_runs(*, collection: Collection | None = None) -> dict:
    """Enforce workflow-history retention: at most ``RUN_HISTORY_MAX_DOCS``
    documents and none older than ``RUN_HISTORY_MAX_AGE_DAYS`` days —
    whichever limit is hit first. Runs are stored with ISO-8601 UTC
    ``created_at`` strings, which sort/compare lexicographically, so the
    app-side scan is safe and exact."""
    coll = _runs_collection(collection)
    docs = list(coll.find({}))
    if not docs:
        return {"removed": 0}
    docs.sort(key=lambda d: d.get("created_at") or "", reverse=True)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=RUN_HISTORY_MAX_AGE_DAYS)).isoformat()
    doomed = [d["_id"] for d in docs[RUN_HISTORY_MAX_DOCS:]]
    doomed += [
        d["_id"]
        for d in docs[:RUN_HISTORY_MAX_DOCS]
        if d.get("created_at") and d["created_at"] < cutoff
    ]
    for run_id in doomed:
        coll.delete_one({"_id": run_id})
    return {"removed": len(doomed)}


def list_runs(
    agent_id: str,
    limit: int = 20,
    *,
    collection: Collection | None = None,
) -> list[dict]:
    """Agent runs, newest first."""
    docs = _runs_collection(collection).find({"agent_id": agent_id})
    docs = sorted(docs, key=lambda d: d.get("created_at") or "", reverse=True)
    return [_public_run(d) for d in docs[:limit]]


def _positive_int(value) -> int:
    """Coerce a metric to a non-negative int (0 when missing/invalid)."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def list_recent_runs(
    limit: int = 25,
    *,
    sort_by: str = "created_at",
    order: str = "desc",
    executor: Optional[str] = None,
    collection: Collection | None = None,
) -> list[dict]:
    """Most recent runs across ALL agents (lab agents and ephemeral spawned
    ``sub-*`` subagents) — the console's global workflow feed.

    ``sort_by`` is a whitelisted field (``created_at`` or an efficiency
    metric: ``tokens_in``, ``tokens_out``, ``tokens_total``, ``duration_ms``,
    ``tool_call_count``, ``step_count``); ``order`` is ``asc`` or ``desc``
    (default: newest first). ``executor`` narrows to ``pensive`` (subagents
    spawned from a chat) or ``test`` (console lab-agent runs); legacy runs
    are matched on their derived tag.
    """
    if sort_by not in RUN_SORT_FIELDS:
        raise ValueError(f"unknown sort field: {sort_by}")
    if executor is not None and executor not in ("pensive", "test"):
        raise ValueError(f"unknown executor: {executor}")
    descending = order != "asc"
    runs = [
        _public_run(d)
        for d in _runs_collection(collection).find({})
    ]
    if executor is not None:
        runs = [r for r in runs if r["executor"] == executor]
    runs.sort(
        key=lambda run: run.get(sort_by) if sort_by == "created_at" else int(run.get(sort_by) or 0),
        reverse=descending,
    )
    return runs[:limit]


RUN_SORT_FIELDS = frozenset(
    {
        "created_at",
        "tokens_in",
        "tokens_out",
        "tokens_total",
        "duration_ms",
        "tool_call_count",
        "step_count",
    }
)


# --- workspace browsing ------------------------------------------------------

_SKIP_WORKSPACE_DIRS = {"__pycache__", ".pytest_cache", ".git", ".venv", "node_modules"}


def list_workspace_files(
    agent_id: str,
    *,
    base: Path | None = None,
    limit: int = 200,
) -> list[dict]:
    """Flat listing of the agent's workspace: relative paths + byte sizes."""
    workspace = workspace_root(agent_id, base=base)
    if not workspace.exists():
        return []
    files: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(workspace):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_WORKSPACE_DIRS)
        rel_dir = Path(dirpath).relative_to(workspace)
        for filename in sorted(filenames):
            if filename.endswith(".pyc"):
                continue
            full = Path(dirpath) / filename
            try:
                size = full.stat().st_size
            except OSError:
                size = 0
            rel = filename if str(rel_dir) == "." else (rel_dir / filename).as_posix()
            files.append({"path": rel, "size": size})
            if len(files) >= limit:
                return files
    return files


def read_workspace_file(
    agent_id: str,
    rel_path: str,
    *,
    base: Path | None = None,
) -> str:
    """Read a file from the agent workspace (path escapes are rejected)."""
    workspace = workspace_root(agent_id, base=base)
    path = safe_workspace_path(workspace, rel_path)
    if not path.is_file():
        raise FileNotFoundError(f"workspace file not found: {rel_path}")
    return path.read_text(encoding="utf-8", errors="replace")


def _to_langchain_tool_call(call: dict) -> dict:
    """Convert a model tool call (id/name/arguments) to LangChain form."""
    try:
        args = json.loads(call.get("arguments") or "{}")
    except (ValueError, TypeError):
        args = {}
    return {
        "id": call.get("id") or "",
        "name": call.get("name") or "",
        "args": args if isinstance(args, dict) else {},
    }


_FAILURE_PREFIX = "Sorry, I couldn't reach the language model"


def _is_failure_answer(answer: str) -> bool:
    return answer.startswith(_FAILURE_PREFIX)


def _all_unparseable_arguments(tool_calls: list[dict]) -> bool:
    """True when EVERY tool call in a step carries invalid JSON arguments.

    A single broken call is executed normally (the executor reports the parse
    error to the model, which can fix that one payload); when the whole step
    comes back broken, re-sample the step instead of burning it.
    """
    for call in tool_calls:
        try:
            json.loads(str(call.get("arguments") or "{}"))
        except (ValueError, TypeError):
            continue
        return False
    return True


# Token economy for agent tool loops: the model only ever sees a windowed
# slice of the transcript (task frame + most recent activity), never the full
# unbounded history — long loops were retransmitting everything on every step
# (323k input tokens for one research run). The run record keeps the FULL
# transcript; only the model payload is windowed.
_TRANSCRIPT_WINDOW = 8
# Model-visible cap for a single tool result replayed into the transcript.
_TRANSCRIPT_RESULT_CAP = 8000


def _agent_model_messages(transcript: list) -> list:
    """Task frame + the most recent activity; middle history is dropped.

    The first message is always the task. Keeping it plus the last N-1
    messages preserves grounding (what to do) and immediate context (what
    just happened) while cutting retransmission cost by the loop length.
    """
    if len(transcript) <= _TRANSCRIPT_WINDOW:
        return list(transcript)
    return [transcript[0]] + list(transcript[-(_TRANSCRIPT_WINDOW - 1):])


def _cap_tool_result(text: str) -> str:
    """Bound a tool result replayed into the model's transcript."""
    return str(text or "")[:_TRANSCRIPT_RESULT_CAP]


# Once a run has been going for a while, the model must stop researching and
# write its best answer with what it has — convergence over completeness.
_CONVERGE_NUDGE_AFTER = 6
_CONVERGE_NUDGE = (
    "You have used many steps. STOP researching now and write your final "
    "answer using the information you already gathered — do not fetch more."
)


def _nudge_toward_convergence(transcript: list, windowed: list) -> list:
    """Prepend a convergence nudge once the run is past the threshold."""
    if len(transcript) < _CONVERGE_NUDGE_AFTER:
        return windowed
    return [SystemMessage(content=_CONVERGE_NUDGE)] + list(windowed)


async def _invoke_agent_step(
    graph: Any,
    agent: dict,
    user_id: str,
    thread_id: str,
    transcript: list,
    task: str,
    tools: list | None = None,
) -> tuple[str, list[dict], list[dict], int, int, list[dict]]:
    """Run the graph once for the agent; returns
    (answer, tool_calls, decisions, tokens_in, tokens_out, server_tool_log)."""
    input_state: dict = {
        "messages": [HumanMessage(content=task)],
        "prompt_messages": _nudge_toward_convergence(
            transcript, _agent_model_messages(transcript)
        ),
        "decisions": [],
        "user_id": user_id,
        "intent": "",
        "retrieved_memories": [],
        "response": "",
        "memory_needed": False,
        "memory_query": "",
        "memory_candidates": [],
        "memory_context": [],
        "tools": tools if tools is not None else (agent.get("tools") or DEFAULT_AGENT_TOOLS),
        "tool_choice": None,
        "tool_calls": [],
        "system_prompt": agent.get("system_prompt") or "",
        # Subagents never see the spawn/spawn tool themselves — no nesting.
        "offer_server_tools": False,
    }
    config: dict = {
        "configurable": {"thread_id": thread_id, "user_id": user_id},
        "tags": ["pensive", "agent-test"],
        "run_name": f"agent:{agent.get('id', '')}:run",
    }

    final_state: dict = {}
    decisions: list[dict] = []
    try:
        async for mode, chunk in graph.astream(
            input_state,
            config=config,
            stream_mode=["custom", "values"],
        ):
            if mode == "custom" and isinstance(chunk, dict) and chunk.get("type") == "decision":
                decisions.append(chunk)
            elif mode == "values" and isinstance(chunk, dict):
                final_state = chunk
    except Exception as exc:
        logger.exception("Agent step failed")
        raise RuntimeError(f"agent run failed: {type(exc).__name__}: {exc}") from exc

    return (
        final_state.get("response") or "",
        final_state.get("tool_calls") or [],
        decisions,
        int(final_state.get("tokens_in") or 0),
        int(final_state.get("tokens_out") or 0),
        final_state.get("server_tool_log") or [],
    )


async def run_agent(
    agent: dict,
    task: str,
    graph: Any,
    *,
    max_steps: int = 10,
    workspace_base: Path | None = None,
) -> dict:
    """Execute a task with the agent's system prompt and tools in complete
    isolation: memory is scoped to ``agent:{id}`` and file/command tools run
    inside the agent's own workspace directory — nothing leaks into (or out
    of) the main pensive context.

    Tool calls are EXECUTED and their results fed back to the model until it
    produces a final answer (or ``max_steps`` is reached).
    """
    task = (task or "").strip()
    if not task:
        raise ValueError("task cannot be empty")
    agent_id = agent.get("id") or ""
    user_id = f"agent:{agent_id}"
    digest = hashlib.sha256(task.encode()).hexdigest()[:12]
    thread_id = f"{user_id}:{digest}"
    workspace = workspace_root(agent_id, base=workspace_base)
    workspace.mkdir(parents=True, exist_ok=True)

    # Every agent run (lab agents AND spawned subagents) carries the discovered
    # Glean tools, so "search exclusively via Glean" tasks never fall back to
    # the public web tools. Schemas are refreshed (~once a minute) and deduped
    # against the agent's own tool list.
    agent_tools = list(agent.get("tools") or DEFAULT_AGENT_TOOLS)
    try:
        await get_glean_client().refresh_tools_if_stale()
    except Exception:  # noqa: BLE001 — best-effort; no credentials is fine
        logger.debug("Glean snapshot refresh skipped", exc_info=True)
    existing_names = {(t.get("function") or {}).get("name") for t in agent_tools}
    for schema in get_glean_client().tool_schemas():
        name = (schema.get("function") or {}).get("name")
        if name and name not in existing_names:
            agent_tools.append(schema)
            existing_names.add(name)

    transcript: list = [HumanMessage(content=task)]
    decisions_all: list[dict] = []
    executed_calls: list[dict] = []
    steps: list[dict] = []
    answer = ""
    tool_calls: list[dict] = []
    tokens_in_total = 0
    tokens_out_total = 0
    started = time.monotonic()

    for _ in range(max_steps + 1):
        # Transient upstream failures surface as a "Sorry..." answer from the
        # generate node, and truncated tool-call arguments arrive as invalid
        # JSON. Retry a few times before giving up; retries re-sample the same
        # step WITHOUT touching the transcript, so they burn time, not steps,
        # and never pollute the conversation the model sees.
        retries = 0
        while True:
            answer, tool_calls, decisions, step_tokens_in, step_tokens_out, server_log = (
                await _invoke_agent_step(
                    graph, agent, user_id, thread_id, transcript, task, tools=agent_tools
                )
            )
            tokens_in_total += step_tokens_in
            tokens_out_total += step_tokens_out
            decisions_all.extend(decisions)
            needs_retry = (tool_calls and _all_unparseable_arguments(tool_calls)) or (
                not tool_calls and _is_failure_answer(answer)
            )
            if not needs_retry or retries >= 2:
                break
            retries += 1
            await asyncio.sleep(0.5)
        if not tool_calls and not server_log:
            break

        # Server-loop executions (glean/spawn/project) happened inside the
        # graph; fold them into the step transcript and totals so the history
        # aggregates EVERY tool call, not just the relayed executor ones.
        for call in server_log:
            executed_calls.append(
                {
                    "name": call.get("name") or "",
                    "arguments": str(call.get("arguments") or "")[:300],
                    "output": str(call.get("output") or "")[:500],
                }
            )
        step_calls: list[dict] = [
            {
                "name": call.get("name") or "",
                # Full payloads (bounded) so the UI can replay exactly what
                # the agent did at each step.
                "arguments": str(call.get("arguments") or "")[:12000],
                "output": str(call.get("output") or "")[:8000],
            }
            for call in server_log
        ]
        if tool_calls:
            transcript.append(
                AIMessage(
                    content=answer,
                    tool_calls=[_to_langchain_tool_call(call) for call in tool_calls],
                )
            )
            for call in tool_calls:
                executed = execute_tool_call(call, workspace, user_id=user_id)
                executed_calls.append(
                    {
                        "name": executed["name"],
                        "arguments": json.dumps(call.get("arguments") or "")[:300],
                        "output": executed["output"][:500],
                    }
                )
                step_calls.append(
                    {
                        "name": executed["name"],
                        "arguments": str(call.get("arguments") or "")[:12000],
                        "output": executed["output"][:8000],
                    }
                )
                transcript.append(
                    ToolMessage(
                        content=_cap_tool_result(executed["output"]),
                        tool_call_id=call.get("id") or "",
                    )
                )
        steps.append(
            {
                "response": answer[:2000],
                "decisions": decisions[:10],
                "tool_calls": step_calls,
            }
        )
        if not tool_calls:
            break

    # Grounding receipts: when the run retrieved corpus chunks, the answer
    # ends with a deterministic Sources block (parsed from the stored tool
    # digests) — never inline, so the answer text stays clean to copy-paste.
    footer = documents_service.citation_footer(steps)
    if footer and answer and "Sources:" not in answer:
        answer = f"{answer}\n\n{footer}"

    return {
        "answer": answer,
        "decisions": decisions_all,
        "tool_calls": executed_calls,
        "steps": steps,
        "workspace": str(workspace),
        # Efficiency metrics: tokens consumed (summed across every LLM
        # round-trip, retries included), wall-clock duration, and activity
        # counts — so the workflows list can rank agents by token economy.
        "tokens_in": tokens_in_total,
        "tokens_out": tokens_out_total,
        "token_total": tokens_in_total + tokens_out_total,
        "duration_ms": int((time.monotonic() - started) * 1000),
        "tool_call_count": len(executed_calls),
        "step_count": len(steps),
    }