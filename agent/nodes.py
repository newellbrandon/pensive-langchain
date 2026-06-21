"""LangGraph agent nodes with streamed workflow decisions."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from agent.state import AgentState
from utils.llm import get_chat_model

logger = logging.getLogger(__name__)

MEMORY_KEYWORDS = re.compile(
    r"\b(remember|recall|my|mine|preference|favorite|favourite|last time|you know|about me)\b",
    re.IGNORECASE,
)


def _emit_decision(step: str, decision: str, reason: str) -> None:
    writer = get_stream_writer()
    writer({"type": "decision", "step": step, "decision": decision, "reason": reason})


def _latest_user_message(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                return " ".join(parts)
    return ""


def classify_intent(state: AgentState) -> dict:
    """Decide whether long-term memory search is needed."""
    query = _latest_user_message(state)
    needs_memory = bool(MEMORY_KEYWORDS.search(query))
    intent = "memory" if needs_memory else "general"
    reason = (
        "Query references personal context or recall."
        if needs_memory
        else "General question; skip memory search."
    )
    _emit_decision("classify_intent", intent, reason)
    decision = {
        "step": "classify_intent",
        "decision": intent,
        "reason": reason,
    }
    return {"intent": intent, "decisions": [decision]}


def search_long_term_memory(state: AgentState, runtime: Runtime) -> dict:
    """Search MongoDB store for relevant long-term memories."""
    query = _latest_user_message(state)
    user_id = state.get("user_id")
    if not user_id:
        raise RuntimeError("user_id missing from agent state")
    namespace = ("users", user_id, "memories")

    store: BaseStore | None = runtime.store
    memories: list[dict] = []
    reason = "No store available."

    if store is not None:
        try:
            results = store.search(namespace, query=query, limit=3)
            memories = [
                {
                    "key": item.key,
                    "content": item.value.get("content", ""),
                    "score": getattr(item, "score", None),
                }
                for item in results
            ]
            reason = f"Found {len(memories)} relevant memories."
        except Exception as exc:
            logger.warning("Memory search failed: %s", type(exc).__name__)
            reason = f"Memory search failed: {type(exc).__name__}"

    _emit_decision("search_long_term_memory", f"{len(memories)} results", reason)
    decision = {
        "step": "search_long_term_memory",
        "decision": str(len(memories)),
        "reason": reason,
    }
    return {"retrieved_memories": memories, "decisions": [decision]}


async def generate_response(state: AgentState) -> dict:
    """Generate an assistant response using checkpoint history and retrieved memories."""
    query = _latest_user_message(state)
    memories = state.get("retrieved_memories") or []
    memory_lines = [
        f"- {item.get('content', '')}" for item in memories if item.get("content")
    ]
    memory_block = "\n".join(memory_lines) if memory_lines else "No relevant long-term memories."

    system_prompt = (
        "You are Pensive, a helpful assistant with access to long-term memory.\n"
        "Use retrieved memories when relevant. Be concise.\n\n"
        f"Retrieved memories:\n{memory_block}"
    )

    _emit_decision(
        "generate_response",
        "respond",
        f"Generating answer using {len(memories)} memories.",
    )

    model = get_chat_model()
    try:
        response = await model.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query),
            ]
        )
        content = response.content
        if not isinstance(content, str):
            content = str(content)
        reason = f"Generated answer ({len(content)} chars)."
    except Exception as exc:
        logger.error("LLM generation failed: %s", type(exc).__name__)
        content = (
            "Sorry, I couldn't reach the language model. "
            f"Error: {type(exc).__name__}"
        )
        reason = f"Generation failed: {type(exc).__name__}"

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
    """Extract and persist a durable fact when the user shares one."""
    query = _latest_user_message(state)
    user_id = state.get("user_id")
    if not user_id:
        raise RuntimeError("user_id missing from agent state")
    store: BaseStore | None = runtime.store

    if store is None:
        _emit_decision("save_memory", "skipped", "No store available.")
        return {
            "decisions": [
                {
                    "step": "save_memory",
                    "decision": "skipped",
                    "reason": "No store available.",
                }
            ]
        }

    model = get_chat_model()
    save = False
    content = ""
    try:
        extraction = await model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "Extract ONE durable user fact from the message, if any. "
                        'Return JSON only: {"save": true/false, "content": "fact text"}. '
                        "Save only explicit preferences, names, or lasting facts. "
                        "Do not save questions or transient chit-chat."
                    )
                ),
                HumanMessage(content=query),
            ]
        )
        raw = extraction.content if isinstance(extraction.content, str) else str(extraction.content)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        payload = json.loads(match.group(0) if match else raw)
        save = bool(payload.get("save"))
        content = str(payload.get("content", "")).strip()
    except Exception as exc:
        logger.warning("Memory extraction failed: %s", type(exc).__name__)
        save = False

    if save and content:
        key = hashlib.sha256(content.encode()).hexdigest()[:16]
        namespace = ("users", user_id, "memories")
        try:
            store.put(
                namespace,
                key,
                {"content": content, "user_id": user_id},
            )
            reason = f"Saved memory: {content[:80]}"
            decision_label = "saved"
        except Exception as exc:
            logger.warning("Memory save failed: %s", type(exc).__name__)
            reason = f"Save failed: {type(exc).__name__}"
            decision_label = "skipped"
    else:
        reason = "No durable fact to save."
        decision_label = "skipped"

    _emit_decision("save_memory", decision_label, reason)
    return {
        "decisions": [
            {
                "step": "save_memory",
                "decision": decision_label,
                "reason": reason,
            }
        ]
    }


def route_after_classify(state: AgentState) -> Literal["search_long_term_memory", "generate_response"]:
    """Route to memory search or direct generation."""
    if state.get("intent") == "memory":
        return "search_long_term_memory"
    return "generate_response"
