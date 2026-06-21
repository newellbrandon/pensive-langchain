"""FastAPI routes — OpenAI-compatible endpoints for OpenWebUI."""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
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


def _resolve_thread_id(request: ChatCompletionRequest) -> str:
    if request.metadata and request.metadata.get("thread_id"):
        return str(request.metadata["thread_id"])
    if request.metadata and request.metadata.get("chat_id"):
        return str(request.metadata["chat_id"])

    user_id = request.user or "anonymous"
    first_user = next((m for m in request.messages if m.role == "user"), None)
    if first_user:
        seed = f"{user_id}:{_message_text(first_user)}"
        digest = hashlib.sha256(seed.encode()).hexdigest()[:16]
        return f"{user_id}:{digest}"
    return f"{user_id}:default"


def _resolve_user_id(request: ChatCompletionRequest) -> str:
    return request.user or "anonymous"


def _latest_user_query(request: ChatCompletionRequest) -> str:
    for message in reversed(request.messages):
        if message.role == "user":
            text = _message_text(message)
            if text.strip():
                return text
    return ""


def _format_decision_event(event: dict) -> str:
    step = event.get("step", "step")
    decision = event.get("decision", "")
    reason = event.get("reason", "")
    label = step.replace("_", " ").title()
    return f"\n> **{label}**: {decision} — {reason}\n"


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
async def chat_completions(request: ChatCompletionRequest):
    user_query = _latest_user_query(request)
    if not user_query:
        raise HTTPException(status_code=400, detail="No user query found in messages")

    thread_id = _resolve_thread_id(request)
    user_id = _resolve_user_id(request)
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        },
        "metadata": {
            "thread_id": thread_id,
            "user_id": user_id,
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
    model_name = request.model or "pensive"

    if request.stream:
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
