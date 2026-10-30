"""Four-type long-term memory (semantic / episodic / taxonomic / procedural).

Ported from leafy10's ``memory_pipeline.py`` and adapted to Pensive:

- Writes go to per-type namespaces on the LangGraph MongoDBStore:
  ``("users", <user_id>, "memories", <type>)``. Keys are derived from
  ``<type>:<label>`` so re-saving the same label supersedes the prior value
  (last-write-wins), matching leafy10's newest-fact-wins semantics.
- Retrieval vector-searches whichever source types are requested, drops
  superseded/empty facts, then reranks (Voyage, with a vector-similarity
  fallback) and orders newest-first for the prompt context block.
- Intent planning and typed extraction both use Pensive's OpenAI Responses
  client (structured output is emulated by asking for JSON).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Sequence

import httpx
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from utils.llm import get_responses_client

logger = logging.getLogger(__name__)

MEMORY_TYPES = ("semantic", "episodic", "taxonomic", "procedural")
ALL_TYPES = list(MEMORY_TYPES)
SEARCH_TOP_K = 5
RERANK_TOP_K = 5
RECENT_TURNS = 6

DEFAULT_VOYAGE_RERANK_URL = "https://ai.mongodb.com/v1/rerank"
VOYAGE_RERANK_MODEL = "rerank-2.5-lite"

INTENT_PROMPT = (
    "You plan long-term memory lookups for an assistant. Read the latest user "
    "message in the context of the recent conversation and decide whether remembered "
    "facts about this user (name, interests, goals, location, past conversations, "
    "preferences, skills, how they like things done) could improve the reply.\n"
    "Lean toward needs_memory=true for anything that can be personalized. Set it to "
    "false only for mechanical requests such as today's date or a factual web lookup.\n"
    "Write `query` as a standalone semantic search query describing the user facts "
    "that would help, not a restatement of the request.\n"
    'Return JSON only: {"intent": str, "needs_memory": bool, "query": str, "rationale": str}.'
)

EXTRACT_PROMPT = (
    "Extract durable long-term memories from the user's message and classify each into "
    "one of these types:\n"
    "- semantic: stable facts and preferences about the user\n"
    "- episodic: a specific past event or interaction, no longer true-as-fact but recallable\n"
    "- taxonomic: categories or domain knowledge the user relies on\n"
    "- procedural: how the user does things (steps, preferences for output)\n"
    "Save only explicit, lasting information. Do not save questions or transient chit-chat. "
    'Return JSON only: {"memories": [{"type": "semantic|episodic|taxonomic|procedural", '
    '"text": str, "label": str}]}. An empty list means nothing durable was shared.'
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


class MemoryPlan(BaseModel):
    intent: str = Field(description="Short description of what the user wants.")
    needs_memory: bool = Field(description="Whether remembered user facts could help.")
    query: str = Field(description="Standalone semantic search query for long-term memory.")
    rationale: str = Field(description="Why memory is or is not needed.")


def latest_user_text(messages: Sequence[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _extract_json(text: str, pattern) -> Any | None:
    if not text:
        return None
    match = pattern.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except (ValueError, TypeError) as exc:
        logger.warning("Could not parse JSON from LLM reply: %s", exc)
        return None


# --- intent planning ---------------------------------------------------------

async def plan_memory_query(client: Any, messages: Sequence[AnyMessage]) -> MemoryPlan:
    """Ask the LLM for intent and a semantic search query (JSON response)."""
    recent = [m for m in messages if not isinstance(m, SystemMessage)][-RECENT_TURNS:]
    try:
        raw = await client.agenerate([SystemMessage(content=INTENT_PROMPT), *recent])
        data = _extract_json(str(raw), _JSON_OBJECT)
        if isinstance(data, dict):
            return MemoryPlan(
                intent=str(data.get("intent") or ""),
                needs_memory=bool(data.get("needs_memory")),
                query=str(data.get("query") or ""),
                rationale=str(data.get("rationale") or ""),
            )
    except Exception:
        logger.exception("Memory intent analysis failed; searching with the raw user message")
    text = latest_user_text(messages)
    return MemoryPlan(
        intent=text,
        needs_memory=bool(text),
        query=text,
        rationale="Intent analysis unavailable",
    )


# --- storage -------------------------------------------------------------

def namespace(user_id: str, mem_type: str) -> tuple[str, ...]:
    return ("users", user_id, "memories", mem_type)


def _recorded_at(value: dict) -> str | None:
    ts = value.get("recorded_at") or value.get("updated_at")
    return str(ts) if ts else None


def save_memories(store: Any, user_id: str, memories: list[dict]) -> int:
    """Persist typed memories; same (type,label) supersedes (overwrites) prior value."""
    if store is None:
        return 0
    saved = 0
    for item in memories:
        mem_type = str(item.get("type") or "").strip().lower()
        if mem_type not in MEMORY_TYPES:
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        label = str(item.get("label") or f"{mem_type}_{hashlib.sha256(text.encode()).hexdigest()[:8]}").strip()
        key = hashlib.sha256(f"{mem_type}:{label}".encode()).hexdigest()[:16]
        try:
            store.put(
                namespace(user_id, mem_type),
                key,
                {
                    "content": text,
                    "user_id": user_id,
                    "type": mem_type,
                    "label": label,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            saved += 1
        except Exception as exc:
            logger.warning("Memory save failed (%s: %s)", type(exc).__name__, exc)
    return saved


async def extract_memories(client: Any, query: str, user_id: str) -> list[dict]:
    """Ask the LLM to extract and classify durable memories from the user message."""
    try:
        raw = await client.agenerate(
            [SystemMessage(content=EXTRACT_PROMPT), HumanMessage(content=query)]
        )
        data = _extract_json(str(raw), _JSON_OBJECT)
        memories = data.get("memories") if isinstance(data, dict) else None
        if not isinstance(memories, list):
            return []
        cleaned = []
        for item in memories:
            if not isinstance(item, dict):
                continue
            mem_type = str(item.get("type") or "").strip().lower()
            if mem_type in MEMORY_TYPES and str(item.get("text") or "").strip():
                cleaned.append(
                    {"type": mem_type, "text": str(item["text"]).strip(), "label": str(item.get("label") or "")}
                )
        return cleaned
    except Exception:
        logger.exception("Memory extraction failed")
        return []


async def extract_and_save(client: Any, store: Any, query: str, user_id: str) -> int:
    """Extract typed memories from the latest user message and persist them."""
    memories = await extract_memories(client, query, user_id)
    return save_memories(store, user_id, memories)


# --- retrieval -----------------------------------------------------------

def _search_source(store: Any, user_id: str, mem_type: str, query: str) -> list[dict]:
    if store is None or not query:
        return []
    try:
        results = store.search(namespace(user_id, mem_type), query=query, limit=SEARCH_TOP_K)
    except Exception as exc:
        logger.warning(
            "Memory search failed for %s (%s: %s)", mem_type, type(exc).__name__, exc
        )
        return []
    hits = []
    for item in results or []:
        value = getattr(item, "value", None) or {}
        if value.get("superseded_by_version_id") or value.get("superseded_by_snapshot"):
            continue
        text = str(value.get("content") or "")
        if not text:
            continue
        hits.append(
            {
                "text": text,
                "source": value.get("type") or mem_type,
                "similarity_score": getattr(item, "score", None),
                "timestamp": _recorded_at(value),
            }
        )
    return hits


def retrieve(
    store: Any,
    query: str,
    user_id: str,
    sources: Sequence[str] = ALL_TYPES,
) -> list[dict]:
    """Vector-search the requested memory types; return [] when memory is unavailable."""
    if store is None or not query or not user_id:
        return []
    candidates: list[dict] = []
    for mem_type in sources:
        if mem_type in MEMORY_TYPES:
            candidates.extend(_search_source(store, user_id, mem_type, query))
    return candidates


def _by_similarity(candidates: list[dict], top_k: int) -> list[dict]:
    ordered = sorted(
        candidates,
        key=lambda c: c["similarity_score"] if c["similarity_score"] is not None else float("-inf"),
        reverse=True,
    )
    return [
        {**c, "relevance_score": None, "rank": rank}
        for rank, c in enumerate(ordered[:top_k], start=1)
    ]


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = RERANK_TOP_K,
    *,
    transport: httpx.BaseTransport | None = None,
) -> list[dict]:
    """Rerank candidates with Voyage; fall back to vector similarity on failure/missing key."""
    if not candidates:
        return []
    api_key = os.environ.get("VOYAGE_API_KEY", "")
    if not api_key:
        logger.warning("VOYAGE_API_KEY is missing; ranking memory by vector similarity")
        return _by_similarity(candidates, top_k)
    try:
        with httpx.Client(timeout=10.0, transport=transport) as http:
            response = http.post(
                os.environ.get("VOYAGE_RERANK_URL") or DEFAULT_VOYAGE_RERANK_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "query": query,
                    "documents": [c["text"] for c in candidates],
                    "model": VOYAGE_RERANK_MODEL,
                    "top_k": top_k,
                },
            )
            response.raise_for_status()
            results = response.json()["data"]
        return [
            {
                **candidates[item["index"]],
                "relevance_score": float(item["relevance_score"]),
                "rank": rank,
            }
            for rank, item in enumerate(results[:top_k], start=1)
        ]
    except Exception as exc:
        logger.warning(
            "Voyage rerank failed (%s: %s); ranking memory by vector similarity",
            type(exc).__name__,
            exc,
        )
        return _by_similarity(candidates, top_k)


# --- context -------------------------------------------------------------

def chronological(hits: list[dict]) -> list[dict]:
    """Newest first; memories without a known time go last."""
    return sorted(hits, key=lambda h: h.get("timestamp") or "", reverse=True)


def _recorded(hit: dict) -> str:
    ts = hit.get("timestamp")
    if not ts:
        return "date unknown"
    try:
        return datetime.fromisoformat(str(ts)).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return "date unknown"


def format_context(hits: list[dict], now: datetime | None = None) -> str:
    """Build the dated prompt block of relevant memory facts, newest first."""
    if not hits:
        return ""
    now = now or datetime.now(timezone.utc)
    lines = [
        f"Current time: {now.astimezone(timezone.utc):%Y-%m-%d %H:%M UTC}.",
        "Relevant long-term memory about this user, newest first "
        "(retrieved automatically; the user did not ask for it). "
        "When facts conflict, the newest recorded one is the current truth:",
    ]
    lines += [
        f"- [{_recorded(hit)}, {hit['source']}] {hit['text']}"
        for hit in chronological(hits)
    ]
    return "\n".join(lines)


def build_prompt_messages(
    system_prompt: str,
    messages: Sequence[AnyMessage],
    memory_context: list[dict],
) -> list[AnyMessage]:
    """Assemble the per-call prompt; memory context is never written back to state."""
    prompt: list[AnyMessage] = [SystemMessage(content=system_prompt)]
    memory_block = format_context(memory_context)
    if memory_block:
        prompt.append(SystemMessage(content=memory_block))
    for message in messages:
        if isinstance(message, SystemMessage):
            continue
        prompt.append(message)
    return prompt


def default_client():
    """Return the configured OpenAI Responses client used for memory planning/extraction."""
    return get_responses_client()