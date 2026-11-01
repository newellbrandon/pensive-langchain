"""FastAPI routes — OpenAI-compatible endpoints for OpenWebUI."""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from admin.service import get_system_prompt
from agent.graph import build_graph
from api.admin_routes import router as admin_router
from api.models import ChatCompletionRequest, ChatMessage, ModelObject, ModelsListResponse
from config import Config, configure_langsmith
from persistence.mongo import close, get_client, verify_connection

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pensive API",
    description="LangGraph demo with MongoDB checkpointing and long-term memory",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin_router)

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class _NoCacheStaticFiles(StaticFiles):
    """StaticFiles that forces revalidation on every load.

    Without a cache policy, browsers may serve a stale console bundle (old
    app.js/index.html) even though the server was updated. ``no-cache`` keeps
    ETag-based revalidation so UI changes appear on a normal reload.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


if _STATIC_DIR.exists():
    app.mount("/ui", _NoCacheStaticFiles(directory=_STATIC_DIR, html=True), name="ui")


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")

def _message_text(message: ChatMessage) -> str:
    if message.content is None:
        return ""
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        parts = []
        for block in message.content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return " ".join(parts)
    return str(message.content)

def _header_value(http_request: Request, header_name: str) -> str | None:
    value = http_request.headers.get(header_name)
    if value and value.strip():
        return value.strip()
    return None

def _resolve_user_id(
    request: ChatCompletionRequest,
    http_request: Request,
) -> str:
    """Resolve the memory identity for a chat request.

    ``USER_ID`` in ``.env`` acts as an identity PIN: when set, it wins over
    everything (forwarded OpenWebUI headers, body user) so a deployment can
    pin all clients to one stable memory scope. Without a pin, the forwarded
    ``X-OpenWebUI-User-Id`` header is authoritative, then the body/metadata,
    then the default user.
    """
    if Config.USER_ID:
        return Config.USER_ID

    header_user_id = _header_value(http_request, Config.OPENWEBUI_USER_ID_HEADER)
    if header_user_id:
        return header_user_id

    if isinstance(request.user, str) and request.user.strip():
        return request.user.strip()
    if isinstance(request.user, dict):
        user_id = request.user.get("id")
        if user_id:
            return str(user_id)

    if request.metadata:
        if request.metadata.get("user_id"):
            return str(request.metadata["user_id"])
        if request.metadata.get("user"):
            return str(request.metadata["user"])

    # Fail open for generic OpenAI clients (e.g. an OpenCode coding harness) that
    # don't send a user identity, so requests are never rejected. Memory is then
    # scoped under this stable default user. Set USER_ID in .env to PIN a single
    # identity for every client.
    logger.warning("No user id from headers/body/Config; falling back to default user")
    return "default"

def _resolve_thread_id(
    request: ChatCompletionRequest,
    http_request: Request,
    user_id: str,
) -> str:
    """Map OpenWebUI chat/session to a LangGraph checkpoint thread_id."""
    if request.chat_id:
        return request.chat_id

    header_chat_id = _header_value(http_request, Config.OPENWEBUI_CHAT_ID_HEADER)
    if header_chat_id:
        return header_chat_id

    if request.metadata:
        if request.metadata.get("thread_id"):
            return str(request.metadata["thread_id"])
        if request.metadata.get("chat_id"):
            return str(request.metadata["chat_id"])

    if request.session_id:
        return f"{user_id}:{request.session_id}"

    first_user = next((m for m in request.messages if m.role == "user"), None)
    if first_user:
        seed = f"{user_id}:{_message_text(first_user)}"
        digest = hashlib.sha256(seed.encode()).hexdigest()[:16]
        return f"{user_id}:{digest}"
    return f"{user_id}:default"

def _resolve_session_metadata(
    request: ChatCompletionRequest,
    http_request: Request,
) -> dict[str, str]:
    """Collect OpenWebUI session fields for LangSmith metadata."""
    session: dict[str, str] = {}

    if request.session_id:
        session["session_id"] = request.session_id
    if request.chat_id:
        session["chat_id"] = request.chat_id
    if request.id:
        session["message_id"] = request.id

    header_chat_id = _header_value(http_request, Config.OPENWEBUI_CHAT_ID_HEADER)
    if header_chat_id:
        session.setdefault("chat_id", header_chat_id)

    header_message_id = _header_value(http_request, Config.OPENWEBUI_MESSAGE_ID_HEADER)
    if header_message_id:
        session.setdefault("message_id", header_message_id)

    return session

def _latest_user_query(request: ChatCompletionRequest) -> str:
    for message in reversed(request.messages):
        if message.role == "user":
            text = _message_text(message)
            if text.strip():
                return text
    return ""

# --- OpenAI Responses API helpers ---------------------------------------------

def _responses_input_text(item: dict) -> str:
    """Extract text from a Responses API input item."""
    if "output" in item:
        return _responses_input_text({"content": item.get("output")})
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type in ("input_text", "output_text", "text"):
                    parts.append(str(block.get("text", "")))
        return " ".join(parts)
    return str(content or "")

def _chat_tool_calls(message) -> list[dict]:
    """Convert chat-format tool_calls to LangChain AIMessage tool_calls."""
    converted: list[dict] = []
    for call in message.tool_calls or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        try:
            args = json.loads(function.get("arguments") or "{}")
        except (ValueError, TypeError):
            args = {}
        converted.append(
            {
                "id": call.get("id", ""),
                "name": function.get("name", ""),
                "args": args if isinstance(args, dict) else {},
            }
        )
    return converted

def _chat_messages_to_langchain(body_messages: list) -> list[BaseMessage]:
    """Convert a Chat Completions message list into LangChain messages, preserving
    every role/content (including tool calls and results) so the full harness
    context reaches the model."""
    out: list[BaseMessage] = []
    for message in body_messages:
        text = _message_text(message)
        role = message.role
        if role == "system":
            out.append(SystemMessage(content=text))
        elif role == "assistant":
            out.append(AIMessage(content=text, tool_calls=_chat_tool_calls(message)))
        elif role == "user":
            out.append(HumanMessage(content=text))
        elif role == "tool":
            out.append(ToolMessage(content=text, tool_call_id=message.tool_call_id or ""))
        else:
            out.append(HumanMessage(content=f"[{role}] {text}" if text else text))
    return out

def _responses_function_calls(content_blocks: list) -> list[dict]:
    """Convert Responses ``function_call`` content blocks to LangChain tool_calls."""
    converted: list[dict] = []
    for block in content_blocks or []:
        if not isinstance(block, dict) or block.get("type") != "function_call":
            continue
        try:
            args = json.loads(block.get("arguments") or "{}")
        except (ValueError, TypeError):
            args = {}
        converted.append(
            {
                "id": block.get("call_id", ""),
                "name": block.get("name", ""),
                "args": args if isinstance(args, dict) else {},
            }
        )
    return converted

def _responses_input_messages(body: dict) -> list[BaseMessage]:
    """Convert a Responses API request (instructions + input items) into
    LangChain messages, preserving the full context (tool calls and
    ``function_call_output`` results included)."""
    out: list[BaseMessage] = []
    instructions = body.get("instructions")
    if instructions:
        out.append(SystemMessage(content=str(instructions)))
    raw = body.get("input")
    if isinstance(raw, str):
        out.append(HumanMessage(content=raw))
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "function_call_output":
                out.append(
                    ToolMessage(
                        content=_responses_input_text(item),
                        tool_call_id=item.get("call_id") or "",
                    )
                )
                continue
            content_blocks = item.get("content")
            if isinstance(content_blocks, list):
                text = _responses_input_text(item)
                role = item.get("role") or "user"
                if role == "assistant":
                    out.append(
                        AIMessage(
                            content=text,
                            tool_calls=_responses_function_calls(content_blocks),
                        )
                    )
                elif role == "system":
                    out.append(SystemMessage(content=text))
                else:
                    out.append(HumanMessage(content=text))
                continue
            role = item.get("role") or "user"
            text = _responses_input_text(item)
            if role == "system":
                out.append(SystemMessage(content=text))
            elif role == "assistant":
                out.append(AIMessage(content=text))
            elif role == "user":
                out.append(HumanMessage(content=text))
            else:
                out.append(HumanMessage(content=f"[{role}] {text}" if text else text))
    return out

def _responses_latest_user_query(body: dict) -> str:
    """Return the latest user message text from a Responses API request body."""
    raw = body.get("input")
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        for item in reversed(raw):
            if isinstance(item, dict) and item.get("role") == "user":
                text = _responses_input_text(item).strip()
                if text:
                    return text
    return ""

def _build_responses_response(
    completion_id: str,
    model: str,
    content: str,
    decisions: list[dict] | None = None,
    tool_calls: list[dict] | None = None,
) -> dict:
    """Build a non-streaming OpenAI Responses API response object.

    ``content`` is the model's final answer — the ONLY text in the message.
    Current-turn workflow decisions are rendered on the native reasoning
    channel (a ``reasoning`` output item) so clients show them as a thought
    block, and mirrored in the standard ``metadata`` field for inspection.
    Tool calls the model requested are emitted as ``function_call`` items.
    """
    decisions = decisions or []
    tool_calls = tool_calls or []
    output: list[dict] = []
    if decisions:
        output.append(_reasoning_item(completion_id, _workflow_thought_text(decisions)))
    for index, call in enumerate(tool_calls):
        output.append(
            {
                "id": f"fc_{completion_id}_{index}",
                "type": "function_call",
                "status": "completed",
                "call_id": call.get("id", ""),
                "name": call.get("name", ""),
                "arguments": call.get("arguments", "") or "",
            }
        )
    output.append(
        {
            "id": f"msg_{completion_id}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {"type": "output_text", "annotations": [], "logprobs": [], "text": content}
            ],
            "phase": "final_answer",
        }
    )
    return {
        "id": completion_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "metadata": _workflow_metadata(decisions),
        "output": output,
        "usage": {
            "input_tokens": 0,
            "output_tokens": (len(content) + 3) // 4,
            "total_tokens": (len(content) + 3) // 4,
        },
        "error": None,
    }

def _responses_delta_chunk(completion_id: str, text: str) -> str:
    payload = {
        "type": "response.output_text.delta",
        "item_id": completion_id,
        "output_index": 0,
        "content_index": 0,
        "delta": text,
    }
    return f"data: {json.dumps(payload)}\n\n"

def _word_pieces(text: str) -> list[str]:
    """Split text into word-sized streaming pieces (spaces preserved)."""
    words = text.split(" ")
    return [w if i == 0 else f" {w}" for i, w in enumerate(words)]

def _responses_completed(
    completion_id: str,
    model: str,
    content: str,
    decisions: list[dict] | None = None,
    tool_calls: list[dict] | None = None,
) -> str:
    response = _build_responses_response(
        completion_id, model, content, decisions, tool_calls
    )
    payload = {"type": "response.completed", "response": response}
    return f"data: {json.dumps(payload)}\n\n"

def _responses_output_item_added(completion_id: str, item: dict) -> str:
    payload = {"type": "response.output_item.added", "output_index": 0, "item": item}
    return f"data: {json.dumps(payload)}\n\n"

def _responses_output_item_done(completion_id: str, item: dict) -> str:
    payload = {"type": "response.output_item.done", "output_index": 0, "item": item}
    return f"data: {json.dumps(payload)}\n\n"

def _responses_reasoning_delta(completion_id: str, text: str) -> str:
    payload = {
        "type": "response.reasoning_summary_text.delta",
        "item_id": f"rs_{completion_id}",
        "output_index": 0,
        "content_index": 0,
        "delta": text,
    }
    return f"data: {json.dumps(payload)}\n\n"

def _responses_reasoning_done(completion_id: str, text: str) -> str:
    payload = {
        "type": "response.reasoning_summary_text.done",
        "item_id": f"rs_{completion_id}",
        "output_index": 0,
        "content_index": 0,
        "text": text,
    }
    return f"data: {json.dumps(payload)}\n\n"

def _responses_function_call_item(
    completion_id: str, index: int, call: dict, in_progress: bool = False
) -> dict:
    return {
        "id": f"fc_{completion_id}_{index}",
        "type": "function_call",
        "status": "in_progress" if in_progress else "completed",
        "call_id": call.get("id", ""),
        "name": call.get("name", ""),
        "arguments": "" if in_progress else (call.get("arguments", "") or ""),
    }

def _responses_function_call_added(completion_id: str, index: int, call: dict) -> str:
    item = _responses_function_call_item(completion_id, index, call, in_progress=True)
    payload = {"type": "response.output_item.added", "output_index": index, "item": item}
    return f"data: {json.dumps(payload)}\n\n"

def _responses_function_arguments_delta(completion_id: str, index: int, piece: str) -> str:
    payload = {
        "type": "response.function_call_arguments.delta",
        "item_id": f"fc_{completion_id}_{index}",
        "output_index": index,
        "delta": piece,
    }
    return f"data: {json.dumps(payload)}\n\n"

def _responses_function_arguments_done(completion_id: str, index: int, call: dict) -> str:
    payload = {
        "type": "response.function_call_arguments.done",
        "item_id": f"fc_{completion_id}_{index}",
        "output_index": index,
        "arguments": call.get("arguments", "") or "",
    }
    return f"data: {json.dumps(payload)}\n\n"

def _responses_function_call_done(completion_id: str, index: int, call: dict) -> str:
    item = _responses_function_call_item(completion_id, index, call)
    payload = {"type": "response.output_item.done", "output_index": index, "item": item}
    return f"data: {json.dumps(payload)}\n\n"

def _workflow_metadata(decisions: list[dict]) -> dict[str, str]:
    """Serialize current-turn workflow decisions for response metadata.

    Decisions are diagnostics and must never be part of the assistant message
    text; the model's answer is the final response.
    """
    if not decisions:
        return {}
    return {"workflow": json.dumps(decisions)}

_THOUGHT_HEADER = "**Workflow decisions for this query:**"


def _decision_line(event: dict) -> str:
    """One rendered decision line for the reasoning/thought text."""
    step = event.get("step", "step")
    decision = event.get("decision", "")
    reason = event.get("reason", "")
    label = step.replace("_", " ").title()
    line = f"- {label}: {decision}"
    if reason:
        line += f" — {reason}"
    return line


def _workflow_thought_text(decisions: list[dict]) -> str:
    """Format current-turn decisions as reasoning/thought text.

    Emitted on the native reasoning channel (Responses ``reasoning`` item /
    chat ``reasoning_content``) so OpenAI-compatible clients render it as a
    collapsible thought block — never as part of the final answer. The bold
    header doubles as the thought's title in OpenCode's timeline.
    """
    return "\n".join([_THOUGHT_HEADER, *(_decision_line(d) for d in decisions)])

def _reasoning_item(completion_id: str, text: str, in_progress: bool = False) -> dict:
    """Build a Responses API ``reasoning`` output item carrying the thought."""
    return {
        "id": f"rs_{completion_id}",
        "type": "reasoning",
        "status": "in_progress" if in_progress else "completed",
        "summary": [] if in_progress else [{"type": "summary_text", "text": text}],
    }

def _final_answer(final_state: dict) -> str:
    """Extract the model's answer — the sole content of the final response."""
    answer = final_state.get("response") or ""
    if not answer and final_state.get("messages"):
        last = final_state["messages"][-1]
        # Only a non-empty string message body counts as an answer; tool-call
        # turns legitimately have an empty assistant message.
        content = getattr(last, "content", None)
        if isinstance(content, str) and content:
            answer = content
    return str(answer)

async def _stream_responses(
    graph,
    input_state: dict,
    config: dict,
    completion_id: str,
    model: str,
) -> AsyncIterator[str]:
    """Stream a Responses API response.

    Workflow decisions are streamed on the native reasoning channel
    (``response.reasoning_summary_text.delta``) so clients render them as a
    thought block; the model's final answer is the only output text. Decisions
    are also mirrored on the completed event's metadata.
    """
    final_state: dict = {}
    decisions: list[dict] = []
    reasoning_started = False

    try:
        async for mode, chunk in graph.astream(
            input_state,
            config=config,
            stream_mode=["custom", "values"],
        ):
            if mode == "custom" and isinstance(chunk, dict) and chunk.get("type") == "decision":
                decisions.append(chunk)
                if not reasoning_started:
                    # Open the reasoning item and stream the thought header the
                    # moment the first decision arrives — live, mid-run.
                    yield _responses_output_item_added(
                        completion_id, _reasoning_item(completion_id, "", in_progress=True)
                    )
                    yield _responses_reasoning_delta(completion_id, _THOUGHT_HEADER)
                    reasoning_started = True
                yield _responses_reasoning_delta(completion_id, _decision_line(chunk))
            elif mode == "values" and isinstance(chunk, dict):
                final_state = chunk
    except Exception as exc:
        logger.exception("Graph streaming failed")
        message = f"\n\n**Error:** {type(exc).__name__}: {exc}\n"
        yield _responses_delta_chunk(completion_id, message)
        yield _responses_completed(completion_id, model, message)
        yield "data: [DONE]\n\n"
        return

    content = _final_answer(final_state)
    tool_calls = final_state.get("tool_calls") or []

    if decisions:
        # Close the reasoning item with its complete thought text.
        thought = _workflow_thought_text(decisions)
        yield _responses_reasoning_done(completion_id, thought)
        yield _responses_output_item_done(
            completion_id, _reasoning_item(completion_id, thought)
        )

    # Tool calls the model requested: streamed as function_call output items.
    for index, call in enumerate(tool_calls):
        yield _responses_function_call_added(completion_id, index, call)
        arguments = call.get("arguments", "") or ""
        if arguments:
            for piece in _word_pieces(arguments):
                yield _responses_function_arguments_delta(completion_id, index, piece)
        else:
            yield _responses_function_arguments_delta(completion_id, index, "")
        yield _responses_function_arguments_done(completion_id, index, call)
        yield _responses_function_call_done(completion_id, index, call)

    # Message channel: only the answer, word-by-word.
    if content:
        for piece in _word_pieces(content):
            yield _responses_delta_chunk(completion_id, piece)

    yield _responses_completed(completion_id, model, content, decisions, tool_calls)
    yield "data: [DONE]\n\n"

class _RequestBodyAdapter:
    """Expose Responses API body fields as the ChatCompletionRequest attributes
    that the session-resolution helpers expect (headers remain authoritative)."""

    def __init__(self, body: dict) -> None:
        self.user = body.get("user")
        self.metadata = body.get("metadata")
        self.id = body.get("id")
        self.chat_id = body.get("chat_id")
        self.session_id = body.get("session_id")
        self.messages = []

def _role_chunk(completion_id: str, model: str) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"

def _reasoning_chunk(completion_id: str, model: str, text: str, role: bool = False) -> str:
    """Chat-completions chunk carrying ``delta.reasoning_content`` (the
    DeepSeek/OpenRouter convention for thinking blocks)."""
    delta: dict = {"reasoning_content": text}
    if role:
        delta["role"] = "assistant"
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"

def _chunk_payload(content: str, completion_id: str, model: str) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"

def _final_chunk(completion_id: str, model: str, finish_reason: str = "stop") -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n"

def _tool_call_decl_chunk(
    completion_id: str, model: str, index: int, call: dict
) -> str:
    """First tool-call delta: declares id, type, and function name."""
    delta = {
        "tool_calls": [
            {
                "index": index,
                "id": call.get("id", ""),
                "type": "function",
                "function": {"name": call.get("name", ""), "arguments": ""},
            }
        ]
    }
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"

def _tool_call_arg_chunk(
    completion_id: str, model: str, index: int, piece: str
) -> str:
    """Subsequent tool-call deltas: stream the JSON arguments."""
    delta = {"tool_calls": [{"index": index, "function": {"arguments": piece}}]}
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"

async def _run_graph(graph, input_state: dict, config: dict, on_decision=None):
    """Run the graph once; return (final_state, decisions, error).

    ``decisions`` are collected from the *current* invocation's custom stream
    events so only this turn's workflow steps are surfaced — not the
    historical decisions that accumulate on the checkpointer thread.
    """
    final_state: dict = {}
    decisions: list[dict] = []
    error: Exception | None = None
    try:
        async for mode, chunk in graph.astream(
            input_state,
            config=config,
            stream_mode=["custom", "values"],
        ):
            if mode == "custom" and isinstance(chunk, dict) and chunk.get("type") == "decision":
                decisions.append(chunk)
                if on_decision is not None:
                    await on_decision(chunk)
            elif mode == "values" and isinstance(chunk, dict):
                final_state = chunk
    except Exception as exc:
        logger.exception("Graph execution failed")
        error = exc
    return final_state, decisions, error

async def _stream_graph(
    graph,
    input_state: dict,
    config: dict,
    completion_id: str,
    model: str,
) -> AsyncIterator[str]:
    """Stream the model's final answer (chat completions SSE).

    Workflow decisions stream live as ``delta.reasoning_content`` the moment
    the graph emits them (rendered as a streaming thought block); the answer
    is the only ``content``.
    """
    final_state: dict = {}
    decisions: list[dict] = []
    reasoning_started = False

    try:
        async for mode, chunk in graph.astream(
            input_state,
            config=config,
            stream_mode=["custom", "values"],
        ):
            if mode == "custom" and isinstance(chunk, dict) and chunk.get("type") == "decision":
                decisions.append(chunk)
                if not reasoning_started:
                    yield _reasoning_chunk(
                        completion_id, model, _THOUGHT_HEADER, role=True
                    )
                    reasoning_started = True
                yield _reasoning_chunk(completion_id, model, _decision_line(chunk))
            elif mode == "values" and isinstance(chunk, dict):
                final_state = chunk
    except Exception as exc:
        logger.exception("Graph streaming failed")
        yield _chunk_payload(
            f"\n\n**Error:** {type(exc).__name__}: {exc}\n",
            completion_id,
            model,
        )
        yield _final_chunk(completion_id, model)
        return

    answer = _final_answer(final_state)
    tool_calls = final_state.get("tool_calls") or []

    if not reasoning_started:
        yield _role_chunk(completion_id, model)

    if answer:
        # Stream answer in word-sized chunks for smoother UX
        words = answer.split(" ")
        buffer = ""
        for index, word in enumerate(words):
            piece = word if index == 0 else f" {word}"
            buffer += piece
            if len(buffer) >= 24 or index == len(words) - 1:
                yield _chunk_payload(buffer, completion_id, model)
                buffer = ""

    # Tool calls requested by the model: declaration then arguments deltas.
    for index, call in enumerate(tool_calls):
        yield _tool_call_decl_chunk(completion_id, model, index, call)
        arguments = call.get("arguments", "") or ""
        for piece in (_word_pieces(arguments) if arguments else [""]):
            yield _tool_call_arg_chunk(completion_id, model, index, piece)

    yield _final_chunk(completion_id, model, "tool_calls" if tool_calls else "stop")

def _build_chat_completion(
    completion_id: str,
    model: str,
    content: str,
    decisions: list[dict],
    user_query: str,
    tool_calls: list[dict] | None = None,
) -> dict:
    """Build a non-streaming Chat Completions response.

    The model's answer is the only ``message.content``. Workflow decisions are
    attached as ``message.reasoning_content`` (rendered as a thought block)
    and mirrored in a top-level ``workflow`` field. Tool calls the model
    requested are relayed in the assistant message with a ``tool_calls``
    finish reason.
    """
    message: dict = {"role": "assistant", "content": content}
    if decisions:
        message["reasoning_content"] = _workflow_thought_text(decisions)
    tool_calls = tool_calls or []
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": call.get("id", ""),
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    "arguments": call.get("arguments", "") or "",
                },
            }
            for call in tool_calls
        ]
    payload: dict = {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(user_query) // 4,
            "completion_tokens": len(content) // 4,
            "total_tokens": (len(user_query) + len(content)) // 4,
        },
    }
    if decisions:
        payload["workflow"] = decisions
    return payload


@app.on_event("startup")
async def startup_event() -> None:
    Config.validate()
    configure_langsmith()
    verify_connection()
    app.state.graph = build_graph()
    from agent.graph import set_active_graph

    set_active_graph(app.state.graph)
    logger.info("Pensive API started (db=%s)", Config.MONGODB_DB)

@app.on_event("shutdown")
async def shutdown_event() -> None:
    close()

@app.get("/health")
async def health() -> JSONResponse:
    try:
        get_client().admin.command("ping")
        return JSONResponse({"status": "ok", "mongodb": "connected"})
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": "MongoDB not connected"},
        )

@app.get("/v1/models")
async def list_models() -> ModelsListResponse:
    return ModelsListResponse(
        data=[ModelObject(id="pensive", created=int(time.time()))]
    )

@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    http_request: Request,
):
    user_query = _latest_user_query(body)
    if not user_query:
        raise HTTPException(status_code=400, detail="No user query found in messages")

    user_id = _resolve_user_id(body, http_request)
    thread_id = _resolve_thread_id(body, http_request, user_id)
    session_metadata = _resolve_session_metadata(body, http_request)
    system_prompt = get_system_prompt()
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        },
        "metadata": {
            "thread_id": thread_id,
            "user_id": user_id,
            **session_metadata,
        },
        "tags": ["pensive", "chat"],
        "run_name": f"chat:{thread_id}",
    }
    input_state = {
        "messages": [HumanMessage(content=user_query)],
        "prompt_messages": _chat_messages_to_langchain(body.messages),
        "decisions": [],
        "user_id": user_id,
        "intent": "",
        "retrieved_memories": [],
        "response": "",
        "memory_needed": False,
        "memory_query": "",
        "memory_candidates": [],
        "memory_context": [],
        "tools": body.tools or [],
        "tool_choice": body.tool_choice,
        "tool_calls": [],
        "system_prompt": system_prompt,
        "offer_spawn_tool": True,
    }

    graph = app.state.graph
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    model_name = body.model or "pensive"

    if body.stream:
        return StreamingResponse(
            _stream_graph(graph, input_state, config, completion_id, model_name),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    try:
        final_state, decisions, error = await _run_graph(graph, input_state, config)
    except Exception as exc:
        logger.exception("Graph invocation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    content = _final_answer(final_state)
    return _build_chat_completion(
        completion_id,
        model_name,
        content,
        decisions,
        user_query,
        tool_calls=final_state.get("tool_calls") or [],
    )

@app.post("/v1/responses")
async def responses_completions(body: dict, http_request: Request):
    """OpenAI Responses API compatible endpoint (OpenWebUI 'Responses' connection type)."""
    user_query = _responses_latest_user_query(body)
    if not user_query:
        raise HTTPException(status_code=400, detail="No user query found in input")

    # Reuse chat-completion session resolution; Responses requests rarely carry
    # these in the body, so OpenWebUI headers (X-OpenWebUI-*) are authoritative.
    adapter = _RequestBodyAdapter(body)
    user_id = _resolve_user_id(adapter, http_request)
    thread_id = _resolve_thread_id(adapter, http_request, user_id)
    session_metadata = _resolve_session_metadata(adapter, http_request)
    system_prompt = get_system_prompt()
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        },
        "metadata": {
            "thread_id": thread_id,
            "user_id": user_id,
            **session_metadata,
        },
        "tags": ["pensive", "responses"],
        "run_name": f"responses:{thread_id}",
    }
    input_state = {
        "messages": [HumanMessage(content=user_query)],
        "prompt_messages": _responses_input_messages(body),
        "decisions": [],
        "user_id": user_id,
        "intent": "",
        "retrieved_memories": [],
        "response": "",
        "memory_needed": False,
        "memory_query": "",
        "memory_candidates": [],
        "memory_context": [],
        "tools": body.get("tools") or [],
        "tool_choice": body.get("tool_choice"),
        "tool_calls": [],
        "system_prompt": system_prompt,
        "offer_spawn_tool": True,
    }

    graph = app.state.graph
    completion_id = f"resp_{uuid.uuid4().hex[:24]}"
    model_name = body.get("model") or "pensive"

    if body.get("stream"):
        return StreamingResponse(
            _stream_responses(graph, input_state, config, completion_id, model_name),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    try:
        final_state, decisions, error = await _run_graph(graph, input_state, config)
    except Exception as exc:
        logger.exception("Graph invocation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    content = _final_answer(final_state)
    return _build_responses_response(
        completion_id,
        model_name,
        content,
        decisions,
        tool_calls=final_state.get("tool_calls") or [],
    )
