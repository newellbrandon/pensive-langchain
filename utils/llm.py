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
    """Result of a Responses API completion: answer text + tool calls.

    ``tokens_*`` carry the upstream usage report (0 when the gateway omits it)
    so runs can be ranked by token efficiency.
    """

    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0


# Output budget for upstream completions. Generous on purpose: reasoning
# models spend tokens on chain-of-thought BEFORE emitting function_call
# items, and a truncated (or never-emitted) function_call arrives as cut-off
# JSON arguments — 4096 was too tight for deep planning + large single-call
# file writes. A ceiling, not a target: typical turns stay far below it.
DEFAULT_MAX_OUTPUT_TOKENS = 16384


class ResponsesClient:
    """Minimal async client for the OpenAI Responses API.

    The AI gateway exposes a reasoning model (e.g. ``gpt-5.6-luna``) via the
    Responses API. ``ChatOpenAI`` cannot be used here because it only speaks
    Chat Completions and posts to the wrong pathway for a ``/responses`` URI.

    The final assistant answer lives in ``output[].content[].text`` on message
    items (``phase == "final_answer"``); reasoning/"thinking" items are
    intentionally ignored so only the real answer reaches the caller.

    When the gateway rejects the request with ``model_not_found`` (some
    models — deepseek-*, glm-* — only speak Chat Completions), the request is
    retried against the Chat Completions endpoint with a translated payload
    (see ``_chat_completions_fallback``).
    """

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        model: str,
        # See DEFAULT_MAX_OUTPUT_TOKENS: the gateway model spends tokens on
        # chain-of-thought BEFORE emitting function_call items, and a truncated
        # function_call arrives as cut-off JSON arguments. Keep headroom for
        # agents that write whole files in single tool calls.
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
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
                if not self._is_model_not_found(response):
                    # Surface the upstream error body (the gateway's message
                    # is the only way to diagnose rejected payloads).
                    raise self._upstream_error("Responses", response) from None
                # The model exists but only speaks Chat Completions — retry
                # the whole request against the /chat/completions endpoint.
                return await self._chat_completions_fallback(
                    messages, tools, tool_choice
                )
            data = response.json()

        usage = data.get("usage") or {}
        return GenerationResult(
            text=self._extract_answer(data),
            tool_calls=self._extract_tool_calls(data),
            tokens_in=int(usage.get("input_tokens") or 0),
            tokens_out=int(usage.get("output_tokens") or 0),
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
    def _normalize_arguments(arguments: str) -> str:
        """Return canonical JSON for tool-call arguments.

        The gateway's structured function-call output is already valid JSON,
        but models sometimes hand-craft retries with raw newlines/tabs inside
        JSON strings, which breaks the client's parser. Repair by escaping raw
        control characters, re-parsing, and re-serializing when possible; keep
        the raw value when the arguments are irreparable.
        """
        candidates = [
            arguments,
            arguments.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"),
        ]
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, dict):
                return json.dumps(parsed)
        return arguments

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
                    "arguments": ResponsesClient._normalize_arguments(
                        item.get("arguments") or ""
                    ),
                }
            )
        return calls

    @staticmethod
    def _is_model_not_found(response: httpx.Response) -> bool:
        """True when the gateway rejects the model as unknown / protocol-less.

        Some gateway models only speak Chat Completions; the Responses
        endpoint answers 404 with ``error.code == "model_not_found"``.
        """
        if getattr(response, "status_code", None) != 404:
            return False
        try:
            code = (response.json().get("error") or {}).get("code")
        except (ValueError, AttributeError):
            return False
        return code == "model_not_found"

    @staticmethod
    def _upstream_error(protocol: str, response: httpx.Response) -> httpx.HTTPStatusError:
        """Build an error that carries the upstream body for diagnosis."""
        return httpx.HTTPStatusError(
            f"Upstream {protocol} API {response.status_code}: {response.text[:500]}",
            request=response.request,
            response=response,
        )

    async def _chat_completions_fallback(
        self,
        messages: List[BaseMessage],
        tools: Optional[list] = None,
        tool_choice: Optional[dict] = None,
    ) -> GenerationResult:
        """Retry a ``model_not_found`` rejection on the Chat Completions API.

        The payload is translated (``input`` items -> ``messages``,
        ``instructions`` -> system role, ``max_output_tokens`` ->
        ``max_tokens``, Responses tools -> Chat ``function`` wrapper) and the
        chat response is normalized back into a ``GenerationResult``.
        """
        url = Config.chat_completions_url()
        payload = self._build_chat_completions_payload(messages, tools, tool_choice)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                raise self._upstream_error("Chat Completions", response) from None
            data = response.json()

        usage = data.get("usage") or {}
        return GenerationResult(
            text=self._extract_chat_answer(data),
            tool_calls=self._extract_chat_tool_calls(data),
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
        )

    def _build_chat_completions_payload(
        self,
        messages: List[BaseMessage],
        tools: Optional[list] = None,
        tool_choice: Optional[dict] = None,
    ) -> dict:
        """Translate LangChain messages + tools into a Chat Completions body.

        Mirrors ``_build_input_items`` but for the Chat protocol: system
        messages become the system role, assistant tool calls replay as
        ``message.tool_calls``, and tool results carry their ``call_id``
        pairing as ``tool_call_id``.
        """
        chat_messages: list[dict] = []
        for message in messages:
            text = self._message_text(message)
            role = getattr(message, "type", "")

            if role == "system":
                if text:
                    chat_messages.append({"role": "system", "content": text})
                continue

            if role in ("human", "user"):
                if text:
                    chat_messages.append({"role": "user", "content": text})
                continue

            if role == "tool":
                chat_messages.append(
                    {
                        "role": "tool",
                        "content": text,
                        "tool_call_id": getattr(message, "tool_call_id", None) or "",
                    }
                )
                continue

            if role in ("ai", "assistant"):
                entry: dict = {"role": "assistant"}
                if text:
                    entry["content"] = text
                calls = getattr(message, "tool_calls", None) or []
                if calls:
                    entry["tool_calls"] = [
                        {
                            "id": call.get("id") or f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": call.get("name") or "",
                                "arguments": (
                                    json.dumps(call["args"])
                                    if isinstance(call.get("args"), dict)
                                    else str(call.get("args") or "{}")
                                ),
                            },
                        }
                        for index, call in enumerate(calls)
                    ]
                chat_messages.append(entry)
                continue

            if text:
                chat_messages.append({"role": "user", "content": f"[{role}] {text}"})

        payload: dict = {
            "model": self._model,
            "messages": chat_messages,
            "max_tokens": self._max_output_tokens,
        }
        chat_tools = self._normalize_chat_tools(tools)
        if chat_tools:
            payload["tools"] = chat_tools
        chat_tool_choice = self._normalize_chat_tool_choice(tool_choice)
        if chat_tool_choice is not None:
            payload["tool_choice"] = chat_tool_choice
        return payload

    @staticmethod
    def _normalize_chat_tools(tools: Optional[list]) -> list[dict]:
        """Convert Responses-format tool specs into Chat Completions specs.

        Accepts both the Responses shape (``{"type": "function", "name", ...}``)
        and the Chat wrapper (``{"type": "function", "function": {...}}``) and
        normalizes to the Chat shape.
        """
        normalized: list[dict] = []
        for tool in tools or []:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                continue
            if isinstance(tool.get("function"), dict):
                fn = tool["function"]
                function: dict = {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", "") or "",
                    "parameters": fn.get("parameters") or {},
                }
                strict = fn.get("strict")
            elif tool.get("name"):
                function = {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", "") or "",
                    "parameters": tool.get("parameters") or {},
                }
                strict = tool.get("strict")
            else:
                continue
            if strict is not None:
                function["strict"] = strict
            normalized.append({"type": "function", "function": function})
        return normalized

    @staticmethod
    def _normalize_chat_tool_choice(tool_choice: Optional[dict]) -> Optional[dict]:
        """Convert a Responses tool_choice into Chat Completions format.

        ``{"type": "function", "name": X}`` becomes the Chat
        ``{"type": "function", "function": {"name": X}}`` shape; strings and
        already-normalized dicts pass through unchanged.
        """
        if tool_choice is None or isinstance(tool_choice, str):
            return tool_choice
        if not isinstance(tool_choice, dict):
            return None
        if isinstance(tool_choice.get("function"), dict):
            return tool_choice
        name = tool_choice.get("name")
        if name:
            return {"type": "function", "function": {"name": name}}
        return None

    @staticmethod
    def _extract_chat_answer(data: dict) -> str:
        """Return the assistant message content from a chat completion."""
        choices = data.get("choices") or []
        if not choices:
            return ""
        content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "".join(parts)
        return ""

    @staticmethod
    def _extract_chat_tool_calls(data: dict) -> list[dict]:
        """Return ``message.tool_calls`` as {id, name, arguments} items."""
        choices = data.get("choices") or []
        if not choices:
            return []
        message = choices[0].get("message") or {}
        calls: list[dict] = []
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            calls.append(
                {
                    "id": call.get("id") or "",
                    "name": fn.get("name", "") if isinstance(fn, dict) else "",
                    "arguments": ResponsesClient._normalize_arguments(
                        fn.get("arguments") or ""
                    ),
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
