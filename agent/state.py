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
