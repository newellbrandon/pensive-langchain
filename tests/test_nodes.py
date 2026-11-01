"""Tests for generate_response's server-side spawn_subagent loop."""
from __future__ import annotations

import asyncio
import json
import unittest
from unittest import mock

from langchain_core.messages import HumanMessage, ToolMessage

from admin import service as admin
from agent import nodes


def _run(coro):
    return asyncio.run(coro)


def _state(offer=True, tools=None):
    return {
        "messages": [HumanMessage(content="delegate the research")],
        "prompt_messages": [HumanMessage(content="delegate the research")],
        "decisions": [],
        "user_id": "u1",
        "intent": "",
        "retrieved_memories": [],
        "response": "",
        "memory_needed": False,
        "memory_query": "",
        "memory_candidates": [],
        "memory_context": [],
        "tools": tools or [],
        "tool_choice": None,
        "tool_calls": [],
        "system_prompt": "",
        "offer_spawn_tool": offer,
    }


class _Result:
    def __init__(self, text, tool_calls):
        self.text = text
        self.tool_calls = tool_calls


class _FakeClient:
    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []

    async def acomplete(self, messages, tools=None, tool_choice=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        step = self.steps[min(len(self.calls) - 1, len(self.steps) - 1)]
        return _Result(step.get("text", ""), step.get("tool_calls", []))


class _FakeSubGraph:
    """Graph the subagent runs on (driven by admin.run_agent)."""

    def __init__(self, response):
        self.response = response
        self.invocations = []

    async def astream(self, input_state, config=None, stream_mode=None):
        self.invocations.append({"input_state": input_state, "config": config})
        yield ("values", self.response)


class _BoomGraph:
    async def astream(self, input_state, config=None, stream_mode=None):
        raise RuntimeError("subgraph exploded")


def _spawn_call(call_id, task, **extra):
    args = {"task": task, **extra}
    return {"id": call_id, "name": "spawn_subagent", "arguments": json.dumps(args)}


class SpawnSubagentTest(unittest.TestCase):
    def setUp(self):
        self.decisions = []
        self.recorded_runs = []

        def capture(step, decision, reason):
            self.decisions.append((step, decision, reason))

        self.client_patcher = mock.patch("agent.nodes.get_responses_client")
        self.mock_get_client = self.client_patcher.start()
        self.addCleanup(self.client_patcher.stop)
        self.emit_patcher = mock.patch("agent.nodes._emit_decision", side_effect=capture)
        self.emit_patcher.start()
        self.addCleanup(self.emit_patcher.stop)
        self.graph_patcher = mock.patch("agent.graph.active_graph")
        self.mock_graph = self.graph_patcher.start()
        self.addCleanup(self.graph_patcher.stop)
        # record_run would hit the real Mongo connection on the host — stub it.
        self.record_patcher = mock.patch(
            "agent.nodes.admin.record_run",
            side_effect=lambda agent_id, task, result: self.recorded_runs.append((agent_id, task)),
        )
        self.record_patcher.start()
        self.addCleanup(self.record_patcher.stop)

    def run_node(self, state, steps, graph=None):
        self.mock_get_client.return_value = _FakeClient(steps)
        self.mock_graph.return_value = graph
        return _run(nodes.generate_response(state)), self.mock_get_client.return_value

    def tool_names(self, client):
        names = []
        for tool in client.calls[0]["tools"]:
            if isinstance(tool, dict):
                names.append((tool.get("function") or {}).get("name") or tool.get("name"))
        return names

    def test_server_tool_offered_when_flag_on(self):
        result, client = self.run_node(_state(), [{"text": "ok"}])
        self.assertIn("spawn_subagent", self.tool_names(client))
        self.assertEqual(result["response"], "ok")
        self.assertEqual(result["tool_calls"], [])

    def test_no_server_tool_without_flag(self):
        client_tool = {"type": "function", "function": {"name": "edit", "description": "d"}}
        result, client = self.run_node(_state(offer=False, tools=[client_tool]), [{"text": "ok"}])
        self.assertNotIn("spawn_subagent", self.tool_names(client))
        self.assertIn("edit", self.tool_names(client))

    def test_client_tool_calls_still_relayed(self):
        call = {"id": "c1", "name": "read_file", "arguments": json.dumps({"path": "x.txt"})}
        result, _ = self.run_node(_state(tools=[]), [{"text": "", "tool_calls": [call]}])
        self.assertEqual(result["tool_calls"], [call])

    def test_spawn_executes_isolated_subagent_and_returns_digest(self):
        graph = _FakeSubGraph({"response": "research digest: HNSW is the index", "tool_calls": [], "decisions": []})
        steps = [
            {"text": "", "tool_calls": [_spawn_call("s1", "research vector search")]},
            {"text": "Here is the summary of the research.", "tool_calls": []},
        ]
        result, client = self.run_node(_state(), steps, graph=graph)
        self.assertEqual(result["response"], "Here is the summary of the research.")
        self.assertEqual(result["tool_calls"], [])

        # subagent ran in complete isolation: agent:sub-* memory scope + thread
        inv = graph.invocations[0]["input_state"]
        self.assertTrue(inv["user_id"].startswith("agent:sub-"))
        thread = graph.invocations[0]["config"]["configurable"]["thread_id"]
        self.assertTrue(thread.startswith("agent:sub-"), thread)
        # subagents never see the spawn tool (no nesting)
        self.assertFalse(inv["offer_spawn_tool"])
        self.assertEqual(inv["tools"], admin.DEFAULT_AGENT_TOOLS)

        # digest was fed back to the main model as a ToolMessage
        second = client.calls[1]["messages"]
        tool_msgs = [m for m in second if isinstance(m, ToolMessage)]
        self.assertTrue(any("[ok] subagent" in m.content for m in tool_msgs))
        self.assertTrue(any("HNSW is the index" in m.content for m in tool_msgs))
        # decision streamed for the UI
        self.assertTrue(any(d[0] == "spawn_subagent" and d[1] == "ok" for d in self.decisions))
        # run history recorded under the ephemeral subagent id
        self.assertEqual(len(self.recorded_runs), 1)
        self.assertTrue(self.recorded_runs[0][0].startswith("sub-"))
        self.assertEqual(self.recorded_runs[0][1], "research vector search")

    def test_spawn_budget_caps_at_two(self):
        graph = _FakeSubGraph({"response": "r", "tool_calls": [], "decisions": []})
        steps = [
            {"text": "", "tool_calls": [_spawn_call("s1", "t1")]},
            {"text": "", "tool_calls": [_spawn_call("s2", "t2")]},
            {"text": "", "tool_calls": [_spawn_call("s3", "t3")]},
            {"text": "all delegated", "tool_calls": []},
        ]
        result, client = self.run_node(_state(), steps, graph=graph)
        self.assertEqual(result["response"], "all delegated")
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(len(graph.invocations), 2)  # only two spawns executed
        fourth = client.calls[3]["messages"]
        err = [m for m in fourth if isinstance(m, ToolMessage) and "budget exhausted" in m.content]
        self.assertTrue(err)
        self.assertTrue(any(d[0] == "spawn_subagent" and d[1] == "skipped" for d in self.decisions))

    def test_missing_task_returns_error_result(self):
        graph = _FakeSubGraph({"response": "r", "tool_calls": [], "decisions": []})
        call = {"id": "s1", "name": "spawn_subagent", "arguments": "{}"}
        steps = [{"text": "", "tool_calls": [call]}, {"text": "done"}]
        result, client = self.run_node(_state(), steps, graph=graph)
        self.assertEqual(result["response"], "done")
        self.assertEqual(len(graph.invocations), 0)
        second = client.calls[1]["messages"]
        err = [m for m in second if isinstance(m, ToolMessage) and "task" in m.content]
        self.assertTrue(err)

    def test_spawn_failure_becomes_error_result(self):
        steps = [
            {"text": "", "tool_calls": [_spawn_call("s1", "t")]},
            {"text": "recovered here", "tool_calls": []},
        ]
        result, client = self.run_node(_state(), steps, graph=_BoomGraph())
        self.assertEqual(result["response"], "recovered here")
        second = client.calls[1]["messages"]
        err = [m for m in second if isinstance(m, ToolMessage) and "subagent run failed" in m.content]
        self.assertTrue(err)