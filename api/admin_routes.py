"""Pensive admin REST API: memory curation, system prompt, and test agents."""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from admin import service as admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


def _bad_request(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


class MemoryUpdateRequest(BaseModel):
    content: str


class SystemPromptRequest(BaseModel):
    text: str = ""


class AgentRequest(BaseModel):
    name: str
    system_prompt: str = ""
    tools: Optional[list[dict]] = None


class AgentUpdateRequest(BaseModel):
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    tools: Optional[list[dict]] = None


class AgentRunRequest(BaseModel):
    task: str


@router.get("/memory-types")
async def get_memory_types() -> dict:
    return {"types": admin.memory_types()}


@router.get("/memories")
async def list_memories(user_id: str = "default", mem_type: Optional[str] = None) -> dict:
    try:
        rows = admin.list_memories(user_id, mem_type)
    except ValueError as exc:
        raise _bad_request(exc)
    counts = admin.memory_counts(user_id)
    return {"user_id": user_id, "memories": rows, "counts": counts}


@router.put("/memories/{mem_type}/{key}")
async def update_memory(
    mem_type: str, key: str, body: MemoryUpdateRequest, user_id: str = "default"
) -> dict:
    try:
        record = admin.update_memory(user_id, mem_type, key, body.content)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise _bad_request(exc)
    return {"memory": record}


@router.delete("/memories/{mem_type}/{key}")
async def delete_memory(mem_type: str, key: str, user_id: str = "default") -> dict:
    try:
        deleted = admin.delete_memory(user_id, mem_type, key)
    except ValueError as exc:
        raise _bad_request(exc)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"memory not found: {mem_type}/{key}")
    return {"deleted": True}


@router.get("/system-prompt")
async def get_system_prompt() -> dict:
    return {"text": admin.get_system_prompt()}


@router.put("/system-prompt")
async def set_system_prompt(body: SystemPromptRequest) -> dict:
    text = admin.set_system_prompt(body.text)
    return {"text": text}


@router.get("/agents")
async def list_agents() -> dict:
    return {"agents": admin.list_agents()}


@router.post("/agents")
async def create_agent(body: AgentRequest) -> dict:
    try:
        agent = admin.create_agent(body.name, body.system_prompt, body.tools or [])
    except ValueError as exc:
        raise _bad_request(exc)
    return {"agent": agent}


@router.put("/agents/{agent_id}")
async def update_agent(agent_id: str, body: AgentUpdateRequest) -> dict:
    try:
        agent = admin.update_agent(
            agent_id,
            name=body.name,
            system_prompt=body.system_prompt,
            tools=body.tools,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise _bad_request(exc)
    return {"agent": agent}


@router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str) -> dict:
    if not admin.delete_agent(agent_id):
        raise HTTPException(status_code=404, detail=f"agent not found: {agent_id}")
    return {"deleted": True}


@router.post("/agents/{agent_id}/run")
async def run_agent(agent_id: str, body: AgentRunRequest, request: Request) -> dict:
    agent = admin.get_agent(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"agent not found: {agent_id}")
    graph: Any = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="pensive graph not ready")
    try:
        result = await admin.run_agent(agent, body.task, graph)
    except ValueError as exc:
        raise _bad_request(exc)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    try:
        record = admin.record_run(agent_id, body.task, result, executor="test")
        result = dict(result, run_id=record["run_id"])
    except Exception:  # noqa: BLE001 — history is best-effort, never fails the run
        logger.exception("Could not record agent run history")
    return result


@router.get("/agents/{agent_id}/runs")
async def list_agent_runs(agent_id: str) -> dict:
    return {"agent_id": agent_id, "runs": admin.list_runs(agent_id)}


@router.get("/runs")
async def list_recent_runs(limit: int = 25) -> dict:
    """Global execution feed: latest runs across lab agents AND spawned
    subagents (``sub-*``), so delegated work shows up in the console."""
    limit = max(1, min(int(limit), 100))
    return {"runs": admin.list_recent_runs(limit=limit)}


@router.get("/agents/{agent_id}/workspace")
async def agent_workspace(agent_id: str) -> dict:
    return {"agent_id": agent_id, "files": admin.list_workspace_files(agent_id)}


@router.get("/agents/{agent_id}/workspace/file")
async def agent_workspace_file(agent_id: str, path: str) -> dict:
    try:
        content = admin.read_workspace_file(agent_id, path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise _bad_request(exc)
    return {"path": path, "content": content}