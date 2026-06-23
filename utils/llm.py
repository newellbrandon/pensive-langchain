"""LLM and embedding helpers."""
from __future__ import annotations

from typing import List

import httpx
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langsmith import traceable

from config import Config


def get_chat_model() -> ChatOpenAI:
    """Return a Grove-configured OpenAI-compatible chat model."""
    return ChatOpenAI(
        model=Config.LLM_MODEL,
        openai_api_base=Config.openai_base_url(),
        # Grove authenticates via the api-key header, not Authorization Bearer.
        openai_api_key="not-used-by-grove",
        default_headers={"api-key": Config.LLM_KEY},
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
