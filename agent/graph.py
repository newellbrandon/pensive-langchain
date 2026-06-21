"""LangGraph workflow definition and compilation."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    classify_intent,
    generate_response,
    route_after_classify,
    save_memory,
    search_long_term_memory,
)
from agent.state import AgentState
from persistence.mongo import get_checkpointer, get_store


def build_graph():
    """Build and compile the demo agent graph with MongoDB persistence."""
    builder = StateGraph(AgentState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("search_long_term_memory", search_long_term_memory)
    builder.add_node("generate_response", generate_response)
    builder.add_node("save_memory", save_memory)

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        ["search_long_term_memory", "generate_response"],
    )
    builder.add_edge("search_long_term_memory", "generate_response")
    builder.add_edge("generate_response", "save_memory")
    builder.add_edge("save_memory", END)

    return builder.compile(checkpointer=get_checkpointer(), store=get_store())
