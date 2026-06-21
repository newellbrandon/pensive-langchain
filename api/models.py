"""Pydantic models for OpenAI-compatible API requests and responses."""
from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, list[Any]]


class ChatCompletionRequest(BaseModel):
    model: str = "pensive"
    messages: list[ChatMessage]
    stream: bool = False
    user: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "pensive"


class ModelsListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]
