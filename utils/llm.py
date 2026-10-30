"""LLM and embedding helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from langsmith import traceable

from config import Config


@dataclass
class GenerationResult:
    """Result of a Responses API completion: answer text + tool calls."""

    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)


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

    @staticmethod
    def _normalize_tools(tools: Optional[list]) -> list[dict]:
        """Accept OpenAI Chat (``function`` wrapper) or Responses tool format."""
        normalized: list[dict] = []
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                fn = tool["function"]
                entry: dict = {
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", "") or "",
                    "parameters": fn.get("parameters") or {},
                }
                strict = fn.get("strict")
                if strict is not None:
                    entry["strict"] = strict
                normalized.append(entry)
            elif tool.get("type") == "function" and tool.get("name"):
                normalized.append(dict(tool))
        return normalized

    def _build_input_items(self, messages: List[BaseMessage]) -> tuple[list[dict], list[str]]:
        """Convert LangChain messages into Responses API input items.

        Preserves the harness contract exactly: assistant tool calls replay as
        ``function_call`` content blocks and tool results as
        ``function_call_output`` items (with their ``call_id`` pairings), so a
        multi-turn tool conversation remains coherent for the upstream model.
        """
        input_items: list[dict] = []
        instructions: list[str] = []

        for message in messages:
            text = self._message_text(message)
            role = getattr(message, "type", "")

            if role == "system":
                if text:
                    instructions.append(text)
                continue

            if role in ("human", "user"):
                if text:
                    input_items.append(
                        {"role": "user", "content": [{"type": "input_text", "text": text}]}
                    )
                continue

            if role == "tool":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": getattr(message, "tool_call_id", None) or "",
                        "output": text,
                    }
                )
                continue

            if role in ("ai", "assistant"):
                if text:
                    input_items.append(
                        {"role": "assistant", "content": [{"type": "output_text", "text": text}]}
                    )
                # Tool calls replay as TOP-LEVEL function_call input items:
                # the AI gateway rejects function_call content blocks inside
                # assistant messages and validates that every output is paired
                # with a matching call.
                for call in getattr(message, "tool_calls", None) or []:
                    arguments = call.get("args")
                    if isinstance(arguments, dict):
                        arguments = json.dumps(arguments)
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": call.get("id") or "",
                            "name": call.get("name") or "",
                            "arguments": arguments or "{}",
                            "status": "completed",
                        }
                    )
                continue

            if text:
                # Preserve anything else (developer notes) so full request
                # context is never dropped by the harness.
                input_items.append(
                    {"role": "user", "content": [{"type": "input_text", "text": f"[{role}] {text}"}]}
                )

        return input_items, instructions

    async def acomplete(
        self,
        messages: List[BaseMessage],
        tools: Optional[list] = None,
        tool_choice: Optional[dict] = None,
    ) -> GenerationResult:
        """POST messages (+ optional tools) to the Responses API.

        Returns the final answer text and any ``function_call`` output items.
        """
        input_items, instructions = self._build_input_items(messages)

        payload: dict = {
            "model": self._model,
            "input": input_items,
            "max_output_tokens": self._max_output_tokens,
        }
        if instructions:
            payload["instructions"] = "\n".join(instructions)
        normalized_tools = self._normalize_tools(tools)
        if normalized_tools:
            payload["tools"] = normalized_tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(self._url, json=payload, headers=headers)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                # Surface the upstream error body (the gateway's message is the
                # only way to diagnose rejected payloads).
                raise httpx.HTTPStatusError(
                    f"Upstream Responses API {response.status_code}: {response.text[:500]}",
                    request=response.request,
                    response=response,
                ) from None
            data = response.json()

        return GenerationResult(
            text=self._extract_answer(data),
            tool_calls=self._extract_tool_calls(data),
        )

    async def agenerate(self, messages: List[BaseMessage]) -> str:
        """POST messages to the Responses API and return the final answer text."""
        result = await self.acomplete(messages)
        return result.text

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

    @staticmethod
    def _extract_tool_calls(data: dict) -> list[dict]:
        """Return ``function_call`` output items as {id, name, arguments}."""
        calls: list[dict] = []
        for item in data.get("output") or []:
            if item.get("type") != "function_call":
                continue
            calls.append(
                {
                    "id": item.get("call_id") or item.get("id") or "",
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments") or "",
                }
            )
        return calls


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
