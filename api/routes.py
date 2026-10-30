"""FastAPI routes — OpenAI-compatible endpoints for OpenWebUI."""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage

from agent.graph import build_graph
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


def _message_text(message: ChatMessage) -> str:
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
    """Resolve OpenWebUI user id from forwarded session headers or request body."""
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

    if Config.USER_ID:
        return Config.USER_ID

    raise HTTPException(
        status_code=400,
        detail=(
            "user_id not found. Enable ENABLE_FORWARD_USER_INFO_HEADERS=True on OpenWebUI "
            f"so {Config.OPENWEBUI_USER_ID_HEADER} is sent, or set USER_ID for local testing."
        ),
    )


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


def _build_responses_response(completion_id: str, model: str, content: str) -> dict:
    """Build a non-streaming OpenAI Responses API response object."""
    return {
        "id": completion_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": [
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
        ],
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


def _responses_completed(completion_id: str, model: str, content: str) -> str:
    response = _build_responses_response(completion_id, model, content)
    payload = {"type": "response.completed", "response": response}
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_responses(
    graph,
    input_state: dict,
    config: dict,
    completion_id: str,
    model: str,
) -> AsyncIterator[str]:
    """Stream a Responses API response as response.output_text.delta events."""
    final_state: dict = {}

    try:
        async for mode, chunk in graph.astream(
            input_state,
            config=config,
            stream_mode=["custom", "values"],
        ):
            if mode == "custom" and isinstance(chunk, dict) and chunk.get("type") == "decision":
                yield _responses_delta_chunk(completion_id, _format_decision_event(chunk))
            elif mode == "values" and isinstance(chunk, dict):
                final_state = chunk
    except Exception as exc:
        logger.exception("Graph streaming failed")
        message = f"\n\n**Error:** {type(exc).__name__}: {exc}\n"
        yield _responses_delta_chunk(completion_id, message)
        yield _responses_completed(completion_id, model, message)
        yield "data: [DONE]\n\n"
        return

    answer = final_state.get("response") or ""
    if not answer and final_state.get("messages"):
        last = final_state["messages"][-1]
        answer = getattr(last, "content", "") or str(last)

    decisions_md = "".join(
        _format_decision_event(d) for d in final_state.get("decisions", [])
    )
    content = f"### Workflow{decisions_md}\n---\n\n{answer}" if decisions_md else answer

    if content:
        words = content.split(" ")
        for index, word in enumerate(words):
            piece = word if index == 0 else f" {word}"
            yield _responses_delta_chunk(completion_id, piece)

    yield _responses_completed(completion_id, model, content)
    yield "data: [DONE]\n\n"


def _format_decision_event(event: dict) -> str:
    step = event.get("step", "step")
    decision = event.get("decision", "")
    reason = event.get("reason", "")
    label = step.replace("_", " ").title()
    return f"\n> **{label}**: {decision} — {reason}\n"


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


def _final_chunk(completion_id: str, model: str) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n"


async def _stream_graph(
    graph,
    input_state: dict,
    config: dict,
    completion_id: str,
    model: str,
) -> AsyncIterator[str]:
    """Stream workflow decisions, then the final answer."""
    final_state: dict = {}

    yield _role_chunk(completion_id, model)

    try:
        async for mode, chunk in graph.astream(
            input_state,
            config=config,
            stream_mode=["custom", "values"],
        ):
            if mode == "custom" and isinstance(chunk, dict) and chunk.get("type") == "decision":
                yield _chunk_payload(_format_decision_event(chunk), completion_id, model)
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

    answer = final_state.get("response") or ""
    if not answer and final_state.get("messages"):
        last = final_state["messages"][-1]
        answer = getattr(last, "content", "") or str(last)

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

    yield _final_chunk(completion_id, model)


@app.on_event("startup")
async def startup_event() -> None:
    Config.validate()
    configure_langsmith()
    verify_connection()
    app.state.graph = build_graph()
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
        "decisions": [],
        "user_id": user_id,
        "intent": "",
        "retrieved_memories": [],
        "response": "",
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
        final_state = await graph.ainvoke(input_state, config=config)
    except Exception as exc:
        logger.exception("Graph invocation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    answer = final_state.get("response") or ""
    if not answer and final_state.get("messages"):
        last = final_state["messages"][-1]
        answer = getattr(last, "content", "") or str(last)

    decisions_md = "".join(
        _format_decision_event(d) for d in final_state.get("decisions", [])
    )
    content = f"### Workflow{decisions_md}\n---\n\n{answer}" if decisions_md else answer

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(user_query) // 4,
            "completion_tokens": len(content) // 4,
            "total_tokens": (len(user_query) + len(content)) // 4,
        },
    }


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
        "decisions": [],
        "user_id": user_id,
        "intent": "",
        "retrieved_memories": [],
        "response": "",
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
        final_state = await graph.ainvoke(input_state, config=config)
    except Exception as exc:
        logger.exception("Graph invocation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    answer = final_state.get("response") or ""
    if not answer and final_state.get("messages"):
        last = final_state["messages"][-1]
        answer = getattr(last, "content", "") or str(last)
    decisions_md = "".join(
        _format_decision_event(d) for d in final_state.get("decisions", [])
    )
    content = f"### Workflow{decisions_md}\n---\n\n{answer}" if decisions_md else answer
    return _build_responses_response(completion_id, model_name, content)
