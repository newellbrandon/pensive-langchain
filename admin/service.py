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
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from admin.executor import DEFAULT_AGENT_TOOLS, execute_tool_call, safe_workspace_path, workspace_root
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
    }


def record_run(
    agent_id: str,
    task: str,
    result: dict,
    *,
    collection: Collection | None = None,
) -> dict:
    """Persist a completed agent run so past work can be reviewed and re-run.

    The result is stored with bounded sizes (executed tool-call outputs are
    already truncated by the executor) so history stays light.
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
    }
    _runs_collection(collection).insert_one(doc)
    return _public_run(doc)


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


async def _invoke_agent_step(
    graph: Any,
    agent: dict,
    user_id: str,
    thread_id: str,
    transcript: list,
    task: str,
) -> tuple[str, list[dict], list[dict]]:
    """Run the graph once for the agent; returns (answer, tool_calls, decisions)."""
    input_state: dict = {
        "messages": [HumanMessage(content=task)],
        "prompt_messages": list(transcript),
        "decisions": [],
        "user_id": user_id,
        "intent": "",
        "retrieved_memories": [],
        "response": "",
        "memory_needed": False,
        "memory_query": "",
        "memory_candidates": [],
        "memory_context": [],
        "tools": agent.get("tools") or DEFAULT_AGENT_TOOLS,
        "tool_choice": None,
        "tool_calls": [],
        "system_prompt": agent.get("system_prompt") or "",
        # Subagents never see the spawn tool themselves — no nesting.
        "offer_spawn_tool": False,
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
    )


async def run_agent(
    agent: dict,
    task: str,
    graph: Any,
    *,
    max_steps: int = 24,
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

    transcript: list = [HumanMessage(content=task)]
    decisions_all: list[dict] = []
    executed_calls: list[dict] = []
    steps: list[dict] = []
    answer = ""
    tool_calls: list[dict] = []

    for _ in range(max_steps + 1):
        # Transient upstream failures surface as a "Sorry..." answer from the
        # generate node, and truncated tool-call arguments arrive as invalid
        # JSON. Retry a few times before giving up; retries re-sample the same
        # step WITHOUT touching the transcript, so they burn time, not steps,
        # and never pollute the conversation the model sees.
        retries = 0
        while True:
            answer, tool_calls, decisions = await _invoke_agent_step(
                graph, agent, user_id, thread_id, transcript, task
            )
            decisions_all.extend(decisions)
            needs_retry = (tool_calls and _all_unparseable_arguments(tool_calls)) or (
                not tool_calls and _is_failure_answer(answer)
            )
            if not needs_retry or retries >= 2:
                break
            retries += 1
            await asyncio.sleep(0.5)
        if not tool_calls:
            break

        transcript.append(
            AIMessage(
                content=answer,
                tool_calls=[_to_langchain_tool_call(call) for call in tool_calls],
            )
        )
        step_calls: list[dict] = []
        for call in tool_calls:
            executed = execute_tool_call(call, workspace)
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
                    # Full payloads (bounded) so the UI can replay exactly what
                    # the agent did at each step.
                    "arguments": str(call.get("arguments") or "")[:12000],
                    "output": executed["output"][:8000],
                }
            )
            transcript.append(
                ToolMessage(content=executed["output"], tool_call_id=call.get("id") or "")
            )
        steps.append(
            {
                "response": answer[:2000],
                "decisions": decisions[:10],
                "tool_calls": step_calls,
            }
        )

    return {
        "answer": answer,
        "decisions": decisions_all,
        "tool_calls": executed_calls,
        "steps": steps,
        "workspace": str(workspace),
    }