"""LLM and embedding helpers."""
from __future__ import annotations

from typing import List

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from langsmith import traceable

from config import Config


class ResponsesClient:
    """Minimal async client for the OpenAI Responses API.

    The AI gateway exposes a reasoning model (e.g. ``gpt-5.6-luna``) via the
    Responses API. ``ChatOpenAI`` cannot be used here because it only speaks
    Chat Completions and posts to the wrong pathway for a ``/responses`` URI.

    The final assistant answer lives in ``output[].content[].text`` on message
    items (``phase == "final_answer"``); reasoning/"thinking" items are
    intentionally ignored so only the real answer reaches the caller.
    """

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        model: str,
        max_output_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout

    def _message_text(self, message: BaseMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return " ".join(parts)
        return str(content)

    async def agenerate(self, messages: List[BaseMessage]) -> str:
        """POST messages to the Responses API and return the final answer text."""
        instructions: List[str] = []
        input_items: list[dict] = []

        for message in messages:
            text = self._message_text(message)
            role = getattr(message, "type", "")
            if role == "system":
                if text:
                    instructions.append(text)
            elif role in ("human", "user"):
                if text:
                    input_items.append(
                        {"role": "user", "content": [{"type": "input_text", "text": text}]}
                    )
            elif role in ("ai", "assistant"):
                # Skip empty assistant turns (e.g. tool-call-only messages); the
                # tool results that follow carry the observable content.
                if text:
                    input_items.append(
                        {"role": "assistant", "content": [{"type": "output_text", "text": text}]}
                    )
            elif text:
                # Preserve anything else (tool results, developer notes) so full
                # request context is never dropped by the harness.
                input_items.append(
                    {"role": "user", "content": [{"type": "input_text", "text": f"[{role}] {text}"}]}
                )

        payload: dict = {
            "model": self._model,
            "input": input_items,
            "max_output_tokens": self._max_output_tokens,
        }
        if instructions:
            payload["instructions"] = "\n".join(instructions)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(self._url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        return self._extract_answer(data)

    @staticmethod
    def _extract_answer(data: dict) -> str:
        """Return the concatenated final-answer text, ignoring reasoning items."""
        parts: List[str] = []
        for item in data.get("output") or []:
            if item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    parts.append(str(part.get("text", "")))
        return "".join(parts)


def get_responses_client() -> ResponsesClient:
    """Return an async client for the configured AI gateway Responses API."""
    return ResponsesClient(
        url=Config.responses_url(),
        api_key=Config.LLM_KEY,
        model=Config.LLM_MODEL,
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
