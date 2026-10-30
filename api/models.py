"""Pydantic models for OpenAI-compatible API requests and responses."""
from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    # content may be null on assistant tool-call turns (OpenAI convention).
    content: Optional[Union[str, list[Any]]] = None
    # OpenAI tool-calling fields preserved for harness integration.
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict]] = None


class ChatCompletionRequest(BaseModel):
    model: str = "pensive"
    messages: list[ChatMessage]
    stream: bool = False
    tools: Optional[list[dict]] = None
    tool_choice: Optional[Any] = None
    user: Optional[Union[str, dict[str, Any]]] = None
    metadata: Optional[dict[str, Any]] = None
    # OpenWebUI session fields (forwarded on external OpenAI connections)
    chat_id: Optional[str] = None
    session_id: Optional[str] = None
    id: Optional[str] = Field(default=None, description="OpenWebUI assistant message id")


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "pensive"


class ModelsListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]
