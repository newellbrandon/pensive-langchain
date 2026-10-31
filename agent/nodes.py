"""LangGraph agent nodes with four-type long-term memory and streamed decisions."""
from __future__ import annotations

import json
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
    context = state.get("prompt_messages") or state["messages"]
    client = get_responses_client()
    plan = await memories.plan_memory_query(client, context)
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
    """Generate an assistant response using the full incoming context + retrieved memory.

    Tool definitions from the harness are forwarded to the model so it can
    request read/write/execute operations; any tool calls it emits are relayed
    back to the client in the response.
    """
    query = _latest_user_message(state)
    memory_context = state.get("memory_context") or []

    # Honoring the full request transcript (harness context) is what lets this
    # act as an OpenCode coding model: the client's system prompt, tool results,
    # and conversation history all flow to the model. Pensive adds persistent
    # memory on top and does not override the caller's instructions.
    transcript = state.get("prompt_messages") or state["messages"]
    memory_block = memories.format_context(memory_context)
    prompt_messages: list = []
    system_prompt = state.get("system_prompt") or ""
    if system_prompt:
        prompt_messages.append(SystemMessage(content=system_prompt))
    if memory_block:
        prompt_messages.append(SystemMessage(content=memory_block))
    prompt_messages.extend(transcript)

    client = get_responses_client()
    try:
        result = await client.acomplete(
            prompt_messages,
            tools=state.get("tools") or None,
            tool_choice=state.get("tool_choice"),
        )
        content = result.text
        tool_calls = result.tool_calls
        if tool_calls:
            reason = f"Requested {len(tool_calls)} tool call(s)."
        else:
            reason = f"Generated answer ({len(content)} chars)."
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        logger.error("LLM generation failed: %s", detail)
        content = (
            "Sorry, I couldn't reach the language model. "
            f"Error: {detail[:300]}"
        )
        tool_calls = []
        reason = f"Generation failed: {detail[:200]}"

    _emit_decision("generate_response", "tool_calls" if tool_calls else "respond", reason)
    decision = {
        "step": "generate_response",
        "decision": "tool_calls" if tool_calls else "respond",
        "reason": reason,
    }
    assistant_message = AIMessage(content=content)
    if tool_calls:
        assistant_message = AIMessage(
            content=content,
            tool_calls=[
                {
                    "id": call.get("id", ""),
                    "name": call.get("name", ""),
                    "args": _parse_arguments(call.get("arguments", "")),
                }
                for call in tool_calls
            ],
        )
    return {
        "response": content,
        "tool_calls": tool_calls,
        "messages": [assistant_message],
        "decisions": [decision],
    }


def _parse_arguments(arguments: str) -> dict:
    try:
        parsed = json.loads(arguments or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


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