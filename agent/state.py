"""LangGraph agent state schema."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state for the demo agent workflow."""

    messages: Annotated[list[AnyMessage], add_messages]
    decisions: Annotated[list[dict], operator.add]
    user_id: str
    intent: str
    retrieved_memories: list[dict]
    response: str
    # Editable assistant persona (env default; admin UI override). Empty means
    # honor the client's own instructions without adding a persona.
    system_prompt: str
    # Harness tool-calling: definitions forwarded from the client and any
    # tool calls the model emitted for this turn (relayed back to the client).
    tools: list[dict]
    tool_choice: object | None
    tool_calls: list[dict]
    # Full incoming request transcript for this turn (harness context). Kept
    # separate from `messages` so it is not re-accumulated by the checkpointer.
    prompt_messages: list[AnyMessage]
    # Four-type long-term memory (semantic/episodic/taxonomic/procedural).
    memory_needed: bool
    memory_query: str
    memory_candidates: list[dict]
    memory_context: list[dict]
