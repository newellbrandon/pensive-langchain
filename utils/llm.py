"""LLM and embedding helpers."""
from __future__ import annotations

from functools import cached_property
from typing import Any, List

import httpx
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import _USER_AGENT
from langchain_core.embeddings import Embeddings
from langsmith import traceable

from config import Config

GROVE_ANTHROPIC_VERSION = "2023-06-01"


class GroveChatAnthropic(ChatAnthropic):
    """ChatAnthropic configured for the Grove Anthropic gateway."""

    @cached_property
    def _client_params(self) -> dict[str, Any]:
        default_headers = {
            "User-Agent": _USER_AGENT,
            "api-key": Config.LLM_KEY,
            "anthropic-version": GROVE_ANTHROPIC_VERSION,
        }
        if self.default_headers:
            default_headers.update(self.default_headers)

        client_params: dict[str, Any] = {
            # Grove authenticates via the api-key header, not x-api-key.
            "api_key": "not-used-by-grove",
            "base_url": self.anthropic_api_url,
            "max_retries": self.max_retries,
            "default_headers": default_headers,
        }
        if self.default_request_timeout is None or self.default_request_timeout > 0:
            client_params["timeout"] = self.default_request_timeout
        return client_params


def get_chat_model() -> GroveChatAnthropic:
    """Return a Grove-configured Anthropic chat model."""
    return GroveChatAnthropic(
        model=Config.LLM_MODEL,
        anthropic_api_url=Config.anthropic_base_url(),
        max_tokens=1024,
    )


class VoyageEmbeddings(Embeddings):
    """Voyage embeddings via MongoDB AI embeddings endpoint."""

    def __init__(self) -> None:
        self._client = httpx.Client(timeout=60.0)

    @traceable(run_type="embedding", name="voyage_embed")
    def _embed(self, texts: List[str]) -> List[List[float]]:
        response = self._client.post(
            Config.LLM_EMBEDDING_URI,
            json={
                "model": Config.LLM_EMBEDDING_MODEL,
                "input": texts,
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {Config.LLM_EMBEDDING_KEY}",
            },
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        embeddings: List[List[float]] = []
        for item in sorted(data, key=lambda row: row.get("index", 0)):
            vector = item.get("embedding")
            if vector:
                embeddings.append(vector)
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        vectors = self._embed([text])
        return vectors[0] if vectors else []


def get_embeddings() -> VoyageEmbeddings:
    """Return configured Voyage embeddings client."""
    return VoyageEmbeddings()
