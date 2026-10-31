"""Pensive admin service: memory curation, system prompt, and isolated test agents.

All curation writes go through the LangGraph MongoDBStore so vector embeddings
are (re)computed automatically; listing reads the underlying Mongo collection
(the Store API has no list-items call).
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Collection, Optional

from langchain_core.messages import HumanMessage

from config import Config
from memory.memories import MEMORY_TYPES, namespace
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


def create_agent(
    name: str,
    system_prompt: str,
    tools: list[dict],
    *,
    collection: Collection | None = None,
) -> dict:
    if not name.strip():
        raise ValueError("agent name cannot be empty")
    agent_id = uuid.uuid4().hex[:12]
    now = _now()
    doc = {
        "_id": agent_id,
        "name": name.strip(),
        "system_prompt": system_prompt,
        "tools": tools or [],
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


async def run_agent(agent: dict, task: str, graph: Any) -> dict:
    """Execute a task with the agent's system prompt and tools in complete
    isolation: memory is scoped to ``agent:{id}`` so nothing leaks into (or
    out of) the main pensive context."""
    task = (task or "").strip()
    if not task:
        raise ValueError("task cannot be empty")
    agent_id = agent.get("id") or ""
    user_id = f"agent:{agent_id}"
    digest = hashlib.sha256(task.encode()).hexdigest()[:12]
    task_message = HumanMessage(content=task)
    input_state: dict = {
        "messages": [task_message],
        "prompt_messages": [task_message],
        "decisions": [],
        "user_id": user_id,
        "intent": "",
        "retrieved_memories": [],
        "response": "",
        "memory_needed": False,
        "memory_query": "",
        "memory_candidates": [],
        "memory_context": [],
        "tools": agent.get("tools") or [],
        "tool_choice": None,
        "tool_calls": [],
        "system_prompt": agent.get("system_prompt") or "",
    }
    config: dict = {
        "configurable": {"thread_id": f"{user_id}:{digest}", "user_id": user_id},
        "tags": ["pensive", "agent-test"],
        "run_name": f"agent:{agent_id}:run",
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
        logger.exception("Agent run failed")
        raise RuntimeError(f"agent run failed: {type(exc).__name__}: {exc}") from exc

    return {
        "answer": final_state.get("response") or "",
        "decisions": decisions,
        "tool_calls": final_state.get("tool_calls") or [],
    }