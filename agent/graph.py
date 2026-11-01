"""LangGraph workflow definition and compilation."""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    analyze_intent,
    generate_response,
    route_after_intent,
    save_memory,
    search_long_term_memory,
)
from agent.state import AgentState
from persistence.mongo import get_checkpointer, get_store

_active_graph: Any = None


def set_active_graph(graph: Any) -> None:
    """Register the compiled graph for subagent execution (startup wiring)."""
    global _active_graph
    _active_graph = graph


def active_graph() -> Any:
    """The compiled graph subagents run on, or None before wiring."""
    return _active_graph


def build_graph():
    """Build and compile the demo agent graph with MongoDB persistence."""
    builder = StateGraph(AgentState)
    builder.add_node("analyze_intent", analyze_intent)
    builder.add_node("search_long_term_memory", search_long_term_memory)
    builder.add_node("generate_response", generate_response)
    builder.add_node("save_memory", save_memory)

    builder.add_edge(START, "analyze_intent")
    builder.add_conditional_edges(
        "analyze_intent",
        route_after_intent,
        ["search_long_term_memory", "generate_response"],
    )
    builder.add_edge("search_long_term_memory", "generate_response")
    builder.add_edge("generate_response", "save_memory")
    builder.add_edge("save_memory", END)

    return builder.compile(checkpointer=get_checkpointer(), store=get_store())