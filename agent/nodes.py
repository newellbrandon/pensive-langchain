"""LangGraph agent nodes with four-type long-term memory and streamed decisions."""
from __future__ import annotations

import json
import logging
import uuid
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from admin import service as admin
from admin.executor import PROJECT_TOOL_NAMES, execute_project_tool_call
from glean.client import TOOL_PREFIX, get_glean_client
from agent.state import AgentState
from memory import memories
from utils.llm import get_responses_client

logger = logging.getLogger(__name__)


def _emit_decision(step: str, decision: str, reason: str) -> None:
    writer = get_stream_writer()
    writer({"type": "decision", "step": step, "decision": decision, "reason": reason})


def _latest_user_message(state: AgentState) -> str:
    return memories.latest_user_text(state["messages"])


async def analyze_intent(state: AgentState) -> dict:
    """Decide whether long-term memory is needed and build a search query."""
    context = state.get("prompt_messages") or state["messages"]
    client = get_responses_client()
    plan = await memories.plan_memory_query(client, context)
    intent = "memory" if plan.needs_memory else "general"
    _emit_decision("classify_intent", intent, plan.rationale)
    decision = {
        "step": "classify_intent",
        "decision": intent,
        "reason": plan.rationale,
    }
    return {
        "intent": intent,
        "memory_needed": plan.needs_memory,
        "memory_query": plan.query,
        "decisions": [decision],
    }


def route_after_intent(state: AgentState) -> Literal["search_long_term_memory", "generate_response"]:
    """Route to memory search when the intent analysis decided it is needed."""
    if state.get("memory_needed") and state.get("user_id"):
        return "search_long_term_memory"
    return "generate_response"


async def search_long_term_memory(state: AgentState, runtime: Runtime) -> dict:
    """Search all four memory types for the current user and rerank the hits."""
    user_id = state.get("user_id")
    query = state.get("memory_query") or ""
    store: BaseStore | None = runtime.store

    candidates = memories.retrieve(store, query, user_id, sources=memories.ALL_TYPES)
    hits = memories.rerank(query, candidates) if query else []

    counts = {mem_type: 0 for mem_type in memories.MEMORY_TYPES}
    for hit in hits:
        counts[hit.get("source", "unknown")] = counts.get(hit.get("source", "unknown"), 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
    reason = (
        f"Found {len(hits)} relevant memories ({summary})."
        if summary
        else "No relevant memories found in any type."
    )

    _emit_decision("search_long_term_memory", f"{len(hits)} results", reason)
    decision = {
        "step": "search_long_term_memory",
        "decision": str(len(hits)),
        "reason": reason,
    }
    return {
        "memory_candidates": candidates,
        "memory_context": hits,
        "retrieved_memories": hits,
        "decisions": [decision],
    }


async def generate_response(state: AgentState) -> dict:
    """Generate an assistant response using the full incoming context + retrieved memory.

    Tool definitions from the harness are forwarded to the model so it can
    request read/write/execute operations; any tool calls it emits are relayed
    back to the client in the response.

    When ``offer_server_tools`` is set (main-context chats only), the model is
    also offered ``spawn_subagent`` (delegate to an isolated subagent with its
    own memory scope, workspace, and tools) and the read-only project tools
    (``project_list`` / ``project_read`` over the host projects directory,
    including the OKF bundle). server-executed results are fed back to the
    model; only non-server tool calls are relayed to the client. Subagent runs
    have the flag off, so nesting is impossible.
    """
    query = _latest_user_message(state)
    memory_context = state.get("memory_context") or []

    # Honoring the full request transcript (harness context) is what lets this
    # act as an OpenCode coding model: the client's system prompt, tool results,
    # and conversation history all flow to the model. Pensive adds persistent
    # memory on top and does not override the caller's instructions.
    transcript = state.get("prompt_messages") or state["messages"]
    memory_block = memories.format_context(memory_context)
    prompt_messages: list = []
    system_prompt = state.get("system_prompt") or ""
    if system_prompt:
        prompt_messages.append(SystemMessage(content=system_prompt))
    if memory_block:
        prompt_messages.append(SystemMessage(content=memory_block))
    prompt_messages.extend(transcript)

    client = get_responses_client()
    offer_server = bool(state.get("offer_server_tools"))
    if offer_server:
        # Warm the Glean tool snapshot (cheap: once per minute; no-op when the
        # deployment is not connected) so glean_* schemas reach the model.
        try:
            await get_glean_client().refresh_tools_if_stale()
        except (Exception, BaseExceptionGroup):  # noqa: BLE001
            logger.warning("Glean snapshot refresh failed before tool build", exc_info=True)
    tools = _server_tools(state.get("tools") or None, offer_server)
    tool_choice = state.get("tool_choice")
    spawn_budget = _SPAWN_BUDGET
    server_budget = _SERVER_TOOL_BUDGET
    spawn_used = 0

    content = ""
    tool_calls: list[dict] = []
    tokens_in = 0
    tokens_out = 0
    for _ in range(_SERVER_ITERATION_LIMIT):
        try:
            result = await client.acomplete(
                prompt_messages,
                tools=tools,
                tool_choice=tool_choice,
            )
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            logger.error("LLM generation failed: %s", detail)
            content = (
                "Sorry, I couldn't reach the language model. "
                f"Error: {detail[:300]}"
            )
            reason = f"Generation failed: {detail[:200]}"
            _emit_decision("generate_response", "respond", reason)
            return _generate_result(
                content,
                [],
                [{"step": "generate_response", "decision": "respond", "reason": reason}],
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )

        tokens_in += getattr(result, "tokens_in", 0)
        tokens_out += getattr(result, "tokens_out", 0)
        content = result.text
        tool_calls = result.tool_calls or []
        server_calls = [c for c in tool_calls if _is_server_call(c)]
        relay_calls = [c for c in tool_calls if not _is_server_call(c)]
        if not server_calls:
            break

        # Execute server tools (spawns + project browsing) and feed results back.
        handled: list[dict] = []
        exhausted: list[dict] = []
        for call in server_calls:
            is_spawn = _call_name(call) == SPAWN_SUBAGENT_NAME
            if server_budget > 0 and (not is_spawn or spawn_used < spawn_budget):
                handled.append(call)
                server_budget -= 1
                spawn_used += int(is_spawn)
            else:
                exhausted.append(call)
        for call in handled:
            digest = await _execute_server_call(call, state)
            prompt_messages.append(
                AIMessage(content="", tool_calls=[_to_langchain_tool_call(call)])
            )
            prompt_messages.append(
                ToolMessage(content=digest, tool_call_id=call.get("id") or "")
            )
        for call in exhausted:
            prompt_messages.append(
                AIMessage(content="", tool_calls=[_to_langchain_tool_call(call)])
            )
            prompt_messages.append(
                ToolMessage(
                    content=(
                        "[error] server tool budget exhausted: at most 2 "
                        "spawn_subagent and 8 server tool calls per turn; "
                        "combine tasks or finish with what you have."
                    ),
                    tool_call_id=call.get("id") or "",
                )
            )
            _emit_decision(
                "server_tools",
                "skipped",
                "budget exhausted: at most 2 spawns and 8 server tool calls per turn",
            )

    if tool_calls:
        reason = f"Requested {len(tool_calls)} tool call(s)."
    else:
        reason = f"Generated answer ({len(content)} chars)."
    _emit_decision("generate_response", "tool_calls" if tool_calls else "respond", reason)
    decision = {
        "step": "generate_response",
        "decision": "tool_calls" if tool_calls else "respond",
        "reason": reason,
    }
    return _generate_result(content, relay_calls, [decision], tokens_in=tokens_in, tokens_out=tokens_out)


def _call_name(call: dict) -> str:
    return str(call.get("name") or "")


SPAWN_SUBAGENT_NAME = "spawn_subagent"
_SPAWN_BUDGET = 2
_SERVER_TOOL_BUDGET = 8
# More LLM rounds than the server-tool budget, so a model that pages through
# multiple tool results can actually finish; the budget still bounds total work.
_SERVER_ITERATION_LIMIT = _SERVER_TOOL_BUDGET + 4
_SPAWN_DIGEST_ANSWER_LIMIT = 4000
_SPAWN_DIGEST_OUTPUT_LIMIT = 200

SERVER_TOOL_NAMES = frozenset({SPAWN_SUBAGENT_NAME, *PROJECT_TOOL_NAMES})

SPAWN_SUBAGENT_TOOL: dict = {
    "type": "function",
    "function": {
        "name": SPAWN_SUBAGENT_NAME,
        "description": (
            "Delegate a self-contained task to an isolated subagent. The subagent "
            "gets its own memory scope, workspace, and tools (file editing, shell, "
            "web search) so this conversation stays clean; only a short digest of "
            "its final answer returns. Pass tools: [] for the defaults. At most 2 "
            "spawns per turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The full task for the subagent"},
                "system_prompt": {"type": "string", "description": "Optional persona/instructions for the subagent"},
                "tools": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Optional OpenAI function definitions; empty = built-in sandbox + web tools",
                },
            },
            "required": ["task"],
        },
    },
}


def _named_tool(name: str) -> dict:
    for tool in admin.DEFAULT_AGENT_TOOLS:
        if (tool.get("function") or {}).get("name") == name:
            return tool
    raise KeyError(f"missing built-in tool: {name}")


PROJECT_LIST_TOOL = _named_tool("project_list")
PROJECT_READ_TOOL = _named_tool("project_read")


def _server_tools(client_tools: list[dict] | None, offer_server: bool) -> list[dict]:
    """Client tool defs plus the server tools (unless the client defines its own)."""
    tools = [t for t in (client_tools or []) if isinstance(t, dict)]
    if not offer_server:
        return tools
    defined = {
        (t.get("function") or {}).get("name") or t.get("name")
        for t in tools
    }
    for tool in (SPAWN_SUBAGENT_TOOL, PROJECT_LIST_TOOL, PROJECT_READ_TOOL):
        name = (tool.get("function") or {}).get("name")
        if name not in defined:
            tools.append(tool)
    # Glean tools (discovered MCP surface, prefixed glean_*) — only when the
    # deployment is connected; failures surface as [error] results at call time.
    for tool in get_glean_client().tool_schemas():
        name = (tool.get("function") or {}).get("name")
        if name and name not in defined:
            tools.append(tool)
    return tools


def _is_server_call(call: dict) -> bool:
    name = _call_name(call)
    return name in SERVER_TOOL_NAMES or name.startswith(TOOL_PREFIX)


async def _execute_server_call(call: dict, state: AgentState) -> str:
    """Execute one server tool call and return the result fed back to the model."""
    name = _call_name(call)
    if name == SPAWN_SUBAGENT_NAME:
        return await _execute_spawn(call, state)
    if name.startswith(TOOL_PREFIX):
        try:
            args = json.loads(call.get("arguments") or "{}")
            if not isinstance(args, dict):
                args = {}
        except (ValueError, TypeError):
            return (
                "[error] glean tool arguments are not valid JSON; reissue the "
                "call with valid JSON"
            )
        try:
            await get_glean_client().refresh_tools_if_stale()
        except Exception as exc:  # noqa: BLE001
            logger.warning("glean refresh failed: %s", exc)
        return (await get_glean_client().call_tool(name, args))["output"]
    executed = execute_project_tool_call(call)
    return executed["output"]


async def _execute_spawn(call: dict, state: AgentState) -> str:
    """Run one spawn_subagent call; returns the digest fed back to the model."""
    # Imported here to avoid the agent.graph <-> agent.nodes import cycle.
    from agent.graph import active_graph

    try:
        args = json.loads(call.get("arguments") or "{}")
        if not isinstance(args, dict):
            args = {}
    except (ValueError, TypeError):
        return (
            "[error] spawn_subagent arguments are not valid JSON; reissue the "
            "call with valid JSON (task is a string, tools optional)."
        )

    task = str(args.get("task") or "").strip()
    if not task:
        return "[error] spawn_subagent requires a non-empty 'task' string"
    graph = active_graph()
    if graph is None:
        return "[error] subagent execution unavailable: the graph is not ready"

    raw_tools = args.get("tools")
    tools = (
        [t for t in raw_tools if isinstance(t, dict)][:12]
        if isinstance(raw_tools, list)
        else None
    )
    agent_id = f"sub-{uuid.uuid4().hex[:12]}"
    subagent = {
        "id": agent_id,
        "name": admin.autogen_agent_name(),
        "system_prompt": str(args.get("system_prompt") or "").strip()
        or (
            "You are a focused subagent that completes the delegated task using "
            "the available tools. Work autonomously, verify your result, and give "
            "a concise final answer."
        ),
        "tools": tools or admin.DEFAULT_AGENT_TOOLS,
    }
    try:
        result = await admin.run_agent(subagent, task, graph)
    except Exception as exc:
        logger.exception("subagent execution failed")
        return f"[error] subagent run failed: {type(exc).__name__}: {exc}"
    try:
        admin.record_run(
            agent_id,
            task,
            result,
            agent_def={
                "name": subagent["name"],
                "system_prompt": subagent["system_prompt"],
                "tools": subagent["tools"],
            },
            executor="pensive",
        )
    except Exception:
        logger.exception("could not record subagent run history")

    answer = str(result.get("answer") or "")[:_SPAWN_DIGEST_ANSWER_LIMIT]
    calls = result.get("tool_calls") or []
    summary = "\n".join(
        f"- {c.get('name')}: {str(c.get('output') or '')[: _SPAWN_DIGEST_OUTPUT_LIMIT]}"
        for c in calls
    )
    _emit_decision(
        "spawn_subagent",
        "ok",
        f"subagent {agent_id} finished: {len(calls)} tool call(s), "
        f"answer {len(answer)} chars",
    )
    return (
        f"[ok] subagent {agent_id} ({subagent['name']}) finished after "
        f"{len(calls)} executed tool call(s); full artifacts are in its workspace "
        f"({result.get('workspace') or 'n/a'}).\n"
        f"SUBAGENT ANSWER:\n{answer}\n"
        + (f"TOOL ACTIVITY:\n{summary}\n" if summary else "")
    )


def _to_langchain_tool_call(call: dict) -> dict:
    """Convert a model tool call (id/name/arguments) to LangChain form for replay."""
    try:
        args = json.loads(call.get("arguments") or "{}")
    except (ValueError, TypeError):
        args = {}
    return {
        "id": call.get("id") or "",
        "name": call.get("name") or "",
        "args": args if isinstance(args, dict) else {},
    }


def _generate_result(
    content: str,
    tool_calls: list[dict],
    decisions: list[dict],
    *,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> dict:
    """Build the generate node's return value."""
    assistant_message = AIMessage(content=content)
    if tool_calls:
        assistant_message = AIMessage(
            content=content,
            tool_calls=[
                {
                    "id": call.get("id", ""),
                    "name": call.get("name", ""),
                    "args": _parse_arguments(call.get("arguments", "")),
                }
                for call in tool_calls
            ],
        )
    return {
        "response": content,
        "tool_calls": tool_calls,
        "messages": [assistant_message],
        "decisions": decisions,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }


def _parse_arguments(arguments: str) -> dict:
    try:
        parsed = json.loads(arguments or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


async def save_memory(state: AgentState, runtime: Runtime) -> dict:
    """Extract durable facts from the user message and persist them by type."""
    query = _latest_user_message(state)
    user_id = state.get("user_id")
    store: BaseStore | None = runtime.store

    if store is None or not user_id:
        _emit_decision("save_memory", "skipped", "No store available.")
        return {
            "decisions": [
                {"step": "save_memory", "decision": "skipped", "reason": "No store available."}
            ]
        }

    client = get_responses_client()
    saved = await memories.extract_and_save(client, store, query, user_id)

    if saved:
        reason = f"Saved {saved} memory item(s)."
        decision_label = "saved"
    else:
        reason = "No durable fact to save."
        decision_label = "skipped"

    _emit_decision("save_memory", decision_label, reason)
    return {
        "decisions": [
            {"step": "save_memory", "decision": decision_label, "reason": reason}
        ]
    }