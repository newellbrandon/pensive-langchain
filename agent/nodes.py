"""LangGraph agent nodes with four-type long-term memory and streamed decisions."""
from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from agent.state import AgentState
from memory import memories
from utils.llm import get_responses_client

logger = logging.getLogger(__name__)


def _emit_decision(step: str, decision: str, reason: str) -> None:
    writer = get_stream_writer()
    writer({"type": "decision", "step": step, "decision": decision, "reason": reason})


def _latest_user_message(state: AgentState) -> str:
    return memories.latest_user_text(state["messages"])


async def analyze_intent(state: AgentState) -> dict:
    """Decide whether long-term memory is needed and build a search query."""
    client = get_responses_client()
    plan = await memories.plan_memory_query(client, state["messages"])
    intent = "memory" if plan.needs_memory else "general"
    _emit_decision("classify_intent", intent, plan.rationale)
    decision = {
        "step": "classify_intent",
        "decision": intent,
        "reason": plan.rationale,
    }
    return {
        "intent": intent,
        "memory_needed": plan.needs_memory,
        "memory_query": plan.query,
        "decisions": [decision],
    }


def route_after_intent(state: AgentState) -> Literal["search_long_term_memory", "generate_response"]:
    """Route to memory search when the intent analysis decided it is needed."""
    if state.get("memory_needed") and state.get("user_id"):
        return "search_long_term_memory"
    return "generate_response"


async def search_long_term_memory(state: AgentState, runtime: Runtime) -> dict:
    """Search all four memory types for the current user and rerank the hits."""
    user_id = state.get("user_id")
    query = state.get("memory_query") or ""
    store: BaseStore | None = runtime.store

    candidates = memories.retrieve(store, query, user_id, sources=memories.ALL_TYPES)
    hits = memories.rerank(query, candidates) if query else []

    counts = {mem_type: 0 for mem_type in memories.MEMORY_TYPES}
    for hit in hits:
        counts[hit.get("source", "unknown")] = counts.get(hit.get("source", "unknown"), 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
    reason = (
        f"Found {len(hits)} relevant memories ({summary})."
        if summary
        else "No relevant memories found in any type."
    )

    _emit_decision("search_long_term_memory", f"{len(hits)} results", reason)
    decision = {
        "step": "search_long_term_memory",
        "decision": str(len(hits)),
        "reason": reason,
    }
    return {
        "memory_candidates": candidates,
        "memory_context": hits,
        "retrieved_memories": hits,
        "decisions": [decision],
    }


async def generate_response(state: AgentState) -> dict:
    """Generate an assistant response using checkpoint history and retrieved memories."""
    query = _latest_user_message(state)
    memory_context = state.get("memory_context") or []

    system_prompt = (
        "You are Pensive, a helpful assistant with access to long-term memory that spans "
        "semantic facts, past episodes, domain knowledge, and how the user likes things "
        "done. Use remembered details to personalize your reply. Be concise. Never claim "
        "to remember anything not in the supplied memory."
    )
    client = get_responses_client()
    try:
        prompt_messages = memories.build_prompt_messages(system_prompt, [HumanMessage(content=query)], memory_context)
        content = await client.agenerate(prompt_messages)
        reason = f"Generated answer ({len(content)} chars)."
    except Exception as exc:
        logger.error("LLM generation failed: %s", type(exc).__name__)
        content = (
            "Sorry, I couldn't reach the language model. "
            f"Error: {type(exc).__name__}"
        )
        reason = f"Generation failed: {type(exc).__name__}"

    _emit_decision("generate_response", "respond", reason)
    decision = {
        "step": "generate_response",
        "decision": "respond",
        "reason": reason,
    }
    return {
        "response": content,
        "messages": [AIMessage(content=content)],
        "decisions": [decision],
    }


async def save_memory(state: AgentState, runtime: Runtime) -> dict:
    """Extract durable facts from the user message and persist them by type."""
    query = _latest_user_message(state)
    user_id = state.get("user_id")
    store: BaseStore | None = runtime.store

    if store is None or not user_id:
        _emit_decision("save_memory", "skipped", "No store available.")
        return {
            "decisions": [
                {"step": "save_memory", "decision": "skipped", "reason": "No store available."}
            ]
        }

    client = get_responses_client()
    saved = await memories.extract_and_save(client, store, query, user_id)

    if saved:
        reason = f"Saved {saved} memory item(s)."
        decision_label = "saved"
    else:
        reason = "No durable fact to save."
        decision_label = "skipped"

    _emit_decision("save_memory", decision_label, reason)
    return {
        "decisions": [
            {"step": "save_memory", "decision": decision_label, "reason": reason}
        ]
    }