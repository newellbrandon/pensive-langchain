"""Tests for the admin management layer: memory curation, system prompt,
and isolated test agents."""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from config import Config
from admin import service as admin
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def _run(coro):
    return asyncio.run(coro)


def _now():
    return datetime.now(timezone.utc).isoformat()


class FakeCursor(list):
    def sort(self, key=None, direction=1):
        return self


class FakeCollection:
    def __init__(self, docs=None):
        self.docs = list(docs or [])
        self.calls = []

    def find(self, query):
        self.calls.append(("find", query))

        def match(doc):
            doc_value = doc.get("value")
            for field, value in query.items():
                if field == "value.user_id":
                    if not isinstance(doc_value, dict) or doc_value.get("user_id") != value:
                        return False
                elif field == "value.type":
                    if not isinstance(doc_value, dict) or doc_value.get("type") != value:
                        return False
                elif field == "key" and doc.get("key") != value:
                    return False
                elif field == "_id" and doc.get("_id") != value:
                    return False
                elif field == "agent_id" and doc.get("agent_id") != value:
                    return False
            return True

        return FakeCursor([d for d in self.docs if match(d)])

    def find_one(self, query):
        rows = self.find(query)
        return rows[0] if rows else None

    def insert_one(self, doc):
        self.calls.append(("insert_one", doc))
        self.docs.append(doc)
        return doc

    def update_one(self, query, update, upsert=False):
        self.calls.append(("update_one", query, update))
        rows = self.find(query)
        if rows:
            doc = rows[0]
            for field, value in update.get("$set", {}).items():
                doc[field] = value
            return 1
        if upsert:
            doc = dict(query)
            for field, value in update.get("$set", {}).items():
                doc[field] = value
            self.docs.append(doc)
        return 0

    def replace_one(self, query, doc, upsert=False):
        rows = self.find(query)
        if rows:
            rows[0].clear()
            rows[0].update(doc)
        elif upsert:
            self.docs.append(dict(doc))
        return rows[0] if rows else None

    def delete_one(self, query):
        rows = self.find(query)
        if rows:
            self.docs.remove(rows[0])
            return 1
        return 0


class FakeStore:
    def __init__(self):
        self.calls = []
        self.data = {}

    def put(self, namespace, key, value, index=None, **kwargs):
        self.calls.append(("put", namespace, key, value))
        self.data[(tuple(namespace), key)] = value

    def delete(self, namespace, key):
        self.calls.append(("delete", namespace, key))
        self.data.pop((tuple(namespace), key), None)


def _memory_doc(key, mem_type="semantic", user_id="u1", label="name", content="Brandon"):
    return {
        "_id": f"{mem_type}:{key}",
        "namespace": ["users", user_id, "memories", mem_type],
        "key": key,
        "value": {
            "content": content,
            "user_id": user_id,
            "type": mem_type,
            "label": label,
            "recorded_at": "2026-10-01T00:00:00+00:00",
            "updated_at": "2026-10-02T00:00:00+00:00",
        },
        "created_at": "2026-10-01T00:00:00+00:00",
        "updated_at": "2026-10-02T00:00:00+00:00",
    }


class SystemPromptTest(unittest.TestCase):
    def test_default_falls_back_to_config(self):
        coll = FakeCollection()
        self.assertEqual(admin.get_system_prompt(collection=coll), Config.SYSTEM_PROMPT)

    def test_set_persists_and_get_returns_override(self):
        coll = FakeCollection()
        admin.set_system_prompt("You are a test agent.", collection=coll)
        self.assertEqual(admin.get_system_prompt(collection=coll), "You are a test agent.")
        self.assertTrue(any(c[0] == "update_one" for c in coll.calls))


class MemoryCurationTest(unittest.TestCase):
    def setUp(self):
        self.coll = FakeCollection(
            [
                _memory_doc("k1"),
                _memory_doc("k2", mem_type="taxonomic", content="MongoDB"),
                _memory_doc("k3", user_id="other", content="someone else"),
            ]
        )
        self.store = FakeStore()

    def test_list_filters_by_user_and_type(self):
        rows = admin.list_memories(
            "u1", mem_type="semantic", collection=self.coll, store=self.store
        )
        self.assertEqual([r["content"] for r in rows], ["Brandon"])
        rows_all = admin.list_memories("u1", collection=self.coll, store=self.store)
        self.assertEqual(len(rows_all), 2)

    def test_memory_counts_are_unfiltered_per_type(self):
        counts = admin.memory_counts("u1", collection=self.coll, store=self.store)
        self.assertEqual(counts["semantic"], 1)
        self.assertEqual(counts["taxonomic"], 1)
        self.assertEqual(counts["episodic"], 0)

    def test_memory_counts_ignore_legacy_string_values(self):
        legacy = {
            "_id": "legacy",
            "key": "k9",
            "value": "{'content': 'old', 'user_id': 'u1', 'type': 'episodic'}",
        }
        self.coll.docs.append(legacy)
        counts = admin.memory_counts("u1", collection=self.coll, store=self.store)
        self.assertEqual(counts["episodic"], 0)

    def test_update_memory_reembeds_via_store_put(self):
        record = admin.update_memory(
            "u1", "semantic", "k1", "Brandon Alexander",
            collection=self.coll, store=self.store,
        )
        self.assertEqual(record["content"], "Brandon Alexander")
        put_calls = [c for c in self.store.calls if c[0] == "put"]
        self.assertEqual(len(put_calls), 1)
        namespace, key, value = put_calls[0][1], put_calls[0][2], put_calls[0][3]
        # Same namespace + key so the vector record is OVERWRITTEN (re-embed).
        self.assertEqual(namespace, ("users", "u1", "memories", "semantic"))
        self.assertEqual(key, "k1")
        self.assertEqual(value["content"], "Brandon Alexander")
        self.assertEqual(value["label"], "name")
        self.assertTrue(value["updated_at"])

    def test_update_unknown_memory_raises(self):
        with self.assertRaises(KeyError):
            admin.update_memory(
                "u1", "semantic", "missing", "x",
                collection=self.coll, store=self.store,
            )

    def test_update_rejects_unknown_type(self):
        with self.assertRaises(ValueError):
            admin.update_memory(
                "u1", "bogus", "k1", "x",
                collection=self.coll, store=self.store,
            )

    def test_delete_removes_memory(self):
        deleted = admin.delete_memory(
            "u1", "semantic", "k1", collection=self.coll, store=self.store
        )
        self.assertTrue(deleted)
        self.assertIn(("delete", ("users", "u1", "memories", "semantic"), "k1"), self.store.calls)
        self.assertFalse(
            admin.delete_memory("u1", "semantic", "missing", collection=self.coll, store=self.store)
        )


class AgentRegistryTest(unittest.TestCase):
    def setUp(self):
        self.coll = FakeCollection()
        self.store = FakeStore()

    def test_crud_roundtrip(self):
        agent = admin.create_agent(
            "Formatter", "You format code.", {"fmt": {"type": "function"}},
            collection=self.coll,
        )
        self.assertTrue(agent["id"])
        self.assertEqual(agent["name"], "Formatter")
        fetched = admin.get_agent(agent["id"], collection=self.coll)
        self.assertEqual(fetched["system_prompt"], "You format code.")
        updated = admin.update_agent(
            agent["id"], system_prompt="Format with black.", collection=self.coll
        )
        self.assertEqual(updated["system_prompt"], "Format with black.")
        self.assertTrue(admin.delete_agent(agent["id"], collection=self.coll))
        self.assertIsNone(admin.get_agent(agent["id"], collection=self.coll))

    def test_list_agents(self):
        admin.create_agent("A", "p1", [], collection=self.coll)
        admin.create_agent("B", "p2", [], collection=self.coll)
        agents = admin.list_agents(collection=self.coll)
        self.assertEqual([a["name"] for a in agents], ["A", "B"])

    def test_blank_name_autogenerates(self):
        agent = admin.create_agent("  ", "p", [], collection=self.coll)
        self.assertTrue(agent["name"].strip())
        self.assertNotEqual(agent["name"], "  ")
        self.assertIn("-", agent["name"])


class AgentHistoryTest(unittest.TestCase):
    def setUp(self):
        self.coll = FakeCollection()
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_run_and_list_runs_newest_first(self):
        first = admin.record_run(
            "ag_1", "build a diary", {"answer": "v1", "decisions": [], "tool_calls": [{"name": "write_file", "output": "[ok]"}], "workspace": "/w"},
            collection=self.coll,
        )
        second = admin.record_run(
            "ag_1", "fix the bug", {"answer": "v2", "decisions": [], "tool_calls": [], "workspace": "/w"},
            collection=self.coll,
        )
        self.assertTrue(first["run_id"])
        self.assertEqual(first["task"], "build a diary")
        self.assertEqual(first["answer"], "v1")
        self.assertEqual(first["status"], "completed")

        runs = admin.list_runs("ag_1", collection=self.coll)
        self.assertEqual([r["run_id"] for r in runs], [second["run_id"], first["run_id"]])
        self.assertEqual(runs[0]["answer"], "v2")
        self.assertEqual(runs[0]["tool_calls"], [])

    def test_record_run_keeps_step_history(self):
        steps = [
            {
                "response": "writing now",
                "decisions": [{"step": "generate_response", "decision": "tool_calls"}],
                "tool_calls": [{"name": "write_file", "arguments": '{"path": "a.txt"}', "output": "[ok] wrote a.txt"}],
            }
        ]
        admin.record_run("ag_1", "t", {"answer": "done", "steps": steps}, collection=self.coll)
        runs = admin.list_runs("ag_1", collection=self.coll)
        self.assertEqual(runs[0]["steps"], steps)
        self.assertEqual(runs[0]["steps"][0]["tool_calls"][0]["name"], "write_file")

    def test_list_runs_scoped_to_agent(self):
        admin.record_run("ag_1", "t1", {"answer": "a"}, collection=self.coll)
        admin.record_run("ag_2", "t2", {"answer": "b"}, collection=self.coll)
        runs = admin.list_runs("ag_1", collection=self.coll)
        self.assertEqual([r["task"] for r in runs], ["t1"])

    def test_list_workspace_files_returns_relative_paths_and_skips_caches(self):
        ws = self.workspace / "ag_1"
        ws.mkdir(parents=True)
        (ws / "diary.py").write_text("print(1)")
        (ws / "tests").mkdir(parents=True)
        (ws / "tests" / "test_diary.py").write_text("def test(): pass")
        (ws / "__pycache__").mkdir()
        (ws / "__pycache__" / "diary.cpython-313.pyc").write_bytes(b"x")
        (ws / ".pytest_cache").mkdir()
        files = admin.list_workspace_files("ag_1", base=self.workspace)
        paths = [f["path"] for f in files]
        self.assertIn("diary.py", paths)
        self.assertIn("tests/test_diary.py", paths)
        self.assertNotIn("__pycache__/diary.cpython-313.pyc", paths)
        entry = next(f for f in files if f["path"] == "diary.py")
        self.assertEqual(entry["size"], 8)

    def test_read_workspace_file_returns_content_and_blocks_escape(self):
        ws = self.workspace / "ag_1"
        ws.mkdir(parents=True)
        (ws / "diary.py").write_text("print(1)")
        content = admin.read_workspace_file("ag_1", "diary.py", base=self.workspace)
        self.assertEqual(content, "print(1)")
        with self.assertRaises(FileNotFoundError):
            admin.read_workspace_file("ag_1", "missing.py", base=self.workspace)
        with self.assertRaises(ValueError):
            admin.read_workspace_file("ag_1", "../evil.txt", base=self.workspace)


class _FakeGraph:
    def __init__(self, decisions=None, final_state=None):
        self.decisions = decisions or []
        self.final_state = final_state or {"response": "done"}
        self.seen = {}

    async def astream(self, input_state, config=None, stream_mode=None):
        self.seen["input_state"] = input_state
        self.seen["config"] = config
        for decision in self.decisions:
            yield ("custom", {"type": "decision", **decision})
        yield ("values", self.final_state)


class AgentRunTest(unittest.TestCase):
    def setUp(self):
        self.agent = {
            "id": "ag_123",
            "name": "Tester",
            "system_prompt": "You are a strict tester.",
            "tools": admin.DEFAULT_AGENT_TOOLS,
        }

    def test_run_uses_isolated_scope_agent_prompt_and_tools(self):
        graph = _FakeGraph(
            decisions=[{"step": "classify_intent", "decision": "general", "reason": "unit"}],
            final_state={"response": "all green"},
        )
        result = _run(admin.run_agent(self.agent, "Test the math module", graph))

        self.assertEqual(result["answer"], "all green")
        self.assertEqual(result["decisions"][0]["step"], "classify_intent")
        state = graph.seen["input_state"]
        # Isolation: agent memory scope, not the main user's.
        self.assertEqual(state["user_id"], "agent:ag_123")
        self.assertEqual(state["system_prompt"], "You are a strict tester.")
        self.assertEqual(state["tools"], self.agent["tools"])
        # Messages must be LangChain messages (the client chokes on raw strings).
        self.assertTrue(isinstance(state["messages"][0], HumanMessage))
        self.assertTrue(isinstance(state["prompt_messages"][0], HumanMessage))
        config = graph.seen["config"]
        self.assertTrue(config["configurable"]["thread_id"].startswith("agent:ag_123:"))


class _ScriptedGraph:
    """Yields scripted final states, one per invocation."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.invocations = []

    async def astream(self, input_state, config=None, stream_mode=None):
        self.invocations.append(input_state)
        state = self.responses[min(len(self.invocations) - 1, len(self.responses) - 1)]
        yield ("values", state)


class AgentToolLoopTest(unittest.TestCase):
    def test_executes_tool_calls_and_feeds_results_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = {
                "id": "ag_loop",
                "name": "Builder",
                "system_prompt": "p",
                "tools": admin.DEFAULT_AGENT_TOOLS,
            }
            graph = _ScriptedGraph(
                [
                    {
                        "response": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "name": "write_file",
                                "arguments": json.dumps({"path": "diary.txt", "content": "secret"}),
                            }
                        ],
                    },
                    {"response": "done"},
                ]
            )
            result = _run(admin.run_agent(agent, "write a diary", graph, workspace_base=Path(tmp)))

            self.assertEqual(result["answer"], "done")
            # the tool call was executed inside the agent workspace
            self.assertEqual((Path(tmp) / "ag_loop" / "diary.txt").read_text(), "secret")
            self.assertEqual(len(graph.invocations), 2)
            # the second invocation replays the tool result
            second = graph.invocations[1]["prompt_messages"]
            types = [m.__class__.__name__ for m in second]
            self.assertIn("AIMessage", types)
            self.assertIn("ToolMessage", types)
            tool_result = next(m for m in second if isinstance(m, ToolMessage))
            self.assertIn("diary.txt", tool_result.content)

    def test_loop_caps_at_max_steps(self):
        agent = {"id": "ag_cap", "name": "Loop", "system_prompt": "p", "tools": admin.DEFAULT_AGENT_TOOLS}
        always_call = {
            "response": "",
            "tool_calls": [
                {"id": "c1", "name": "read_file", "arguments": json.dumps({"path": "x.txt"})}
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            graph = _ScriptedGraph([always_call])
            result = _run(
                admin.run_agent(agent, "loop", graph, max_steps=2, workspace_base=Path(tmp))
            )
            self.assertEqual(len(graph.invocations), 3)
            self.assertIn("answer", result)

    def test_unknown_tool_becomes_error_result_not_crash(self):
        agent = {"id": "ag_unk", "name": "U", "system_prompt": "p", "tools": admin.DEFAULT_AGENT_TOOLS}
        graph = _ScriptedGraph(
            [
                {
                    "response": "",
                    "tool_calls": [{"id": "c1", "name": "nope_tool", "arguments": "{}"}],
                },
                {"response": "recovered"},
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = _run(admin.run_agent(agent, "t", graph, workspace_base=Path(tmp)))
        self.assertEqual(result["answer"], "recovered")
        second = graph.invocations[1]["prompt_messages"]
        tool_result = next(m for m in second if isinstance(m, ToolMessage))
        self.assertIn("unknown tool", tool_result.content)

    def test_retries_transient_upstream_failures(self):
        agent = {"id": "ag_retry", "name": "R", "system_prompt": "p", "tools": admin.DEFAULT_AGENT_TOOLS}
        failure = "Sorry, I couldn't reach the language model. Error: RemoteProtocolError"
        graph = _ScriptedGraph([{"response": failure}, {"response": failure}, {"response": "ok at last"}])
        with tempfile.TemporaryDirectory() as tmp:
            result = _run(admin.run_agent(agent, "t", graph, workspace_base=Path(tmp)))
        self.assertEqual(result["answer"], "ok at last")
        self.assertEqual(len(graph.invocations), 3)

    def test_retries_steps_with_unparseable_tool_arguments(self):
        agent = {"id": "ag_args", "name": "A", "system_prompt": "p", "tools": admin.DEFAULT_AGENT_TOOLS}
        bad_call = {
            "response": "",
            "tool_calls": [{"id": "c1", "name": "write_file", "arguments": '{"path": "x.py", "content": "unterminated'}],
        }
        graph = _ScriptedGraph([bad_call, bad_call, {"response": "built it"}])
        with tempfile.TemporaryDirectory() as tmp:
            result = _run(admin.run_agent(agent, "t", graph, workspace_base=Path(tmp)))
        self.assertEqual(result["answer"], "built it")
        # both broken steps were retried with a fresh sample instead of
        # polluting the transcript or burning executed tool calls
        self.assertEqual(len(graph.invocations), 3)
        self.assertEqual(result["tool_calls"], [])

    def test_run_records_step_by_step_history(self):
        agent = {"id": "ag_steps", "name": "S", "system_prompt": "p", "tools": admin.DEFAULT_AGENT_TOOLS}
        graph = _ScriptedGraph(
            [
                {
                    "response": "writing now",
                    "tool_calls": [{"id": "c1", "name": "write_file", "arguments": json.dumps({"path": "a.txt", "content": "x"})}],
                },
                {"response": "done"},
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = _run(admin.run_agent(agent, "t", graph, workspace_base=Path(tmp)))
        self.assertEqual(result["answer"], "done")
        # one step per executed tool iteration, with FULL arguments/outputs
        self.assertEqual(len(result["steps"]), 1)
        step = result["steps"][0]
        self.assertEqual(step["response"], "writing now")
        self.assertEqual(step["tool_calls"][0]["name"], "write_file")
        self.assertIn("a.txt", step["tool_calls"][0]["output"])
        parsed_args = json.loads(step["tool_calls"][0]["arguments"])
        self.assertEqual(parsed_args, {"path": "a.txt", "content": "x"})


class ExecutorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_write_and_read_roundtrip(self):
        write = admin.execute_tool_call(
            {"id": "c1", "name": "write_file", "arguments": json.dumps({"path": "app.py", "content": "print(1)"})},
            self.workspace,
        )
        self.assertIn("[ok]", write["output"])
        read = admin.execute_tool_call(
            {"id": "c2", "name": "read_file", "arguments": json.dumps({"path": "app.py"})},
            self.workspace,
        )
        self.assertIn("print(1)", read["output"])
        self.assertEqual((self.workspace / "app.py").read_text(), "print(1)")

    def test_write_without_path_returns_actionable_error(self):
        result = admin.execute_tool_call(
            {"id": "c1", "name": "write_file", "arguments": json.dumps({"content": "x"})},
            self.workspace,
        )
        self.assertIn("[error]", result["output"])
        self.assertIn("path is required", result["output"])

    def test_append_creates_and_grows_file(self):
        first = admin.execute_tool_call(
            {"id": "c1", "name": "append_file", "arguments": json.dumps({"path": "diary.py", "content": "part1\n"})},
            self.workspace,
        )
        self.assertIn("[ok]", first["output"])
        second = admin.execute_tool_call(
            {"id": "c2", "name": "append_file", "arguments": json.dumps({"path": "diary.py", "content": "part2\n"})},
            self.workspace,
        )
        self.assertIn("[ok]", second["output"])
        self.assertEqual((self.workspace / "diary.py").read_text(), "part1\npart2\n")

    def test_append_without_path_returns_actionable_error(self):
        result = admin.execute_tool_call(
            {"id": "c1", "name": "append_file", "arguments": json.dumps({"content": "x"})},
            self.workspace,
        )
        self.assertIn("[error]", result["output"])
        self.assertIn("path is required", result["output"])

    def test_malformed_arguments_report_json_error(self):
        result = admin.execute_tool_call(
            {"id": "c1", "name": "write_file", "arguments": '{"path": "x.py", "content": "unterminated' },
            self.workspace,
        )
        self.assertIn("[error]", result["output"])
        self.assertIn("not valid JSON", result["output"])
        self.assertIn("reissue", result["output"].lower())

    def test_path_escape_is_blocked(self):
        result = admin.execute_tool_call(
            {"id": "c1", "name": "write_file", "arguments": json.dumps({"path": "../evil.txt", "content": "x"})},
            self.workspace,
        )
        self.assertIn("[error]", result["output"])
        self.assertFalse((self.workspace.parent / "evil.txt").exists())

    def test_run_command_executes_in_workspace(self):
        result = admin.execute_tool_call(
            {"id": "c1", "name": "run_command", "arguments": json.dumps({"command": "echo hello-from-sandbox"})},
            self.workspace,
        )
        self.assertIn("hello-from-sandbox", result["output"])

    def test_unknown_tool_returns_error_output(self):
        result = admin.execute_tool_call(
            {"id": "c1", "name": "bogus", "arguments": "{}"}, self.workspace
        )
        self.assertIn("[error]", result["output"])

    def test_default_tools_include_web_search_and_extract(self):
        names = [t["function"]["name"] for t in admin.DEFAULT_AGENT_TOOLS]
        self.assertIn("web_search", names)
        self.assertIn("web_extract", names)

    def test_default_tools_include_project_browsing(self):
        names = [t["function"]["name"] for t in admin.DEFAULT_AGENT_TOOLS]
        self.assertIn("project_list", names)
        self.assertIn("project_read", names)

    def test_web_search_formats_results(self):
        class FakeDDGS:
            def __init__(self, timeout=None):
                self.timeout = timeout

            def text(self, query, max_results=5):
                self.seen = (query, max_results)
                return [
                    {"title": "MongoDB Atlas", "href": "https://www.mongodb.com/atlas", "body": "The easiest way to run MongoDB."},
                    {"title": "Second", "href": "https://example.com/2", "body": "Another result body."},
                ]

        with mock.patch("admin.executor.DDGS", FakeDDGS):
            result = admin.execute_tool_call(
                {"id": "c1", "name": "web_search", "arguments": json.dumps({"query": "mongodb atlas"})},
                self.workspace,
            )
        self.assertIn("[ok]", result["output"])
        self.assertIn("2 web result(s)", result["output"])
        self.assertIn("MongoDB Atlas", result["output"])
        self.assertIn("https://www.mongodb.com/atlas", result["output"])

    def test_web_search_requires_query(self):
        result = admin.execute_tool_call(
            {"id": "c1", "name": "web_search", "arguments": "{}"}, self.workspace
        )
        self.assertIn("[error]", result["output"])
        self.assertIn("query is required", result["output"])

    def test_web_search_failure_is_actionable_error(self):
        class BoomDDGS:
            def __init__(self, timeout=None):
                pass

            def text(self, query, max_results=5):
                raise RuntimeError("rate limited")

        with mock.patch("admin.executor.DDGS", BoomDDGS):
            result = admin.execute_tool_call(
                {"id": "c1", "name": "web_search", "arguments": json.dumps({"query": "x"})},
                self.workspace,
            )
        self.assertIn("[error]", result["output"])
        self.assertIn("retry", result["output"].lower())
        self.assertNotIn("Traceback", result["output"])

    def test_web_extract_returns_truncated_content(self):
        class FakeDDGS:
            def __init__(self, timeout=None):
                pass

            def extract(self, url, fmt="text_markdown"):
                return {"url": url, "content": "# Hello\n" + "x" * 10000}

        with mock.patch("admin.executor.DDGS", FakeDDGS):
            result = admin.execute_tool_call(
                {"id": "c1", "name": "web_extract", "arguments": json.dumps({"url": "https://example.com"})},
                self.workspace,
            )
        self.assertIn("[ok]", result["output"])
        self.assertIn("# Hello", result["output"])
        self.assertLess(len(result["output"]), 9000)

    def test_project_list_and_read_within_projects_root(self):
        root = Path(self.tmp.name) / "projects"
        (root / "okf" / "concepts").mkdir(parents=True)
        (root / "okf" / "concepts" / "memory.md").write_text("# Memory\n\nA durable fact.")
        (root / "okf" / "index.md").write_text("# Bundle index")

        listed = admin.execute_tool_call(
            {"id": "c1", "name": "project_list", "arguments": json.dumps({"path": "okf"})},
            self.workspace,
            projects_root=root,
        )
        self.assertIn("[ok]", listed["output"])
        self.assertIn("index.md", listed["output"])
        self.assertIn("concepts/memory.md", listed["output"])

        read = admin.execute_tool_call(
            {"id": "c2", "name": "project_read", "arguments": json.dumps({"path": "okf/concepts/memory.md"})},
            self.workspace,
            projects_root=root,
        )
        self.assertIn("A durable fact.", read["output"])
        self.assertIn("[ok]", read["output"])

    def test_project_tools_block_escape_and_require_path(self):
        root = Path(self.tmp.name) / "projects"
        (root / "okf").mkdir(parents=True)
        (root / "okf" / "index.md").write_text("x")

        escaped = admin.execute_tool_call(
            {"id": "c1", "name": "project_read", "arguments": json.dumps({"path": "../secret.txt"})},
            self.workspace,
            projects_root=root,
        )
        self.assertIn("[error]", escaped["output"])
        self.assertNotIn("Traceback", escaped["output"])

        no_path = admin.execute_tool_call(
            {"id": "c2", "name": "project_read", "arguments": "{}"},
            self.workspace,
            projects_root=root,
        )
        self.assertIn("[error]", no_path["output"])

    def test_project_tools_disabled_without_root(self):
        result = admin.execute_tool_call(
            {"id": "c1", "name": "project_list", "arguments": "{}"},
            self.workspace,
            projects_root=None,
        )
        self.assertIn("[error]", result["output"])
        self.assertIn("not configured", result["output"].lower())


# --- admin API routes (standalone app, no DB startup) ------------------------

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.admin_routes import router as admin_router

_api_app = FastAPI()
_api_app.include_router(admin_router)
_api_app.state.graph = _FakeGraph(final_state={"response": "isolated answer"})

_client = TestClient(_api_app)


def test_memory_types_describe_all_four():
    response = _client.get("/api/admin/memory-types")
    assert response.status_code == 200
    types = response.json()["types"]
    assert [t["type"] for t in types] == ["semantic", "episodic", "taxonomic", "procedural"]
    by_type = {t["type"]: t for t in types}
    assert "User facts and preferences" in by_type["semantic"]["description"]
    assert "Session events" in by_type["episodic"]["description"]
    assert "corporate policies" in by_type["taxonomic"]["description"]
    assert "skills" in by_type["procedural"]["description"].lower()


def test_list_memories_delegates_to_service(monkeypatch):
    monkeypatch.setattr(
        admin, "list_memories",
        lambda user_id, mem_type=None: [
            {"key": "k1", "type": "semantic", "label": "name",
             "content": "Brandon", "recorded_at": None, "updated_at": None}
        ],
    )
    monkeypatch.setattr(
        admin, "memory_counts",
        lambda user_id: {"semantic": 1, "episodic": 0, "taxonomic": 0, "procedural": 0},
    )
    response = _client.get("/api/admin/memories?user_id=u1&mem_type=semantic")
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "u1"
    assert body["memories"][0]["content"] == "Brandon"
    assert body["counts"]["semantic"] == 1


def test_update_memory_maps_errors(monkeypatch):
    monkeypatch.setattr(
        admin, "update_memory",
        lambda user_id, mem_type, key, content: {"key": key, "content": content},
    )
    ok = _client.put("/api/admin/memories/semantic/k1?user_id=u1", json={"content": "new"})
    assert ok.status_code == 200
    assert ok.json()["memory"]["content"] == "new"

    monkeypatch.setattr(admin, "update_memory", lambda *a, **k: (_ for _ in ()).throw(KeyError("missing")))
    not_found = _client.put("/api/admin/memories/semantic/k1", json={"content": "x"})
    assert not_found.status_code == 404

    monkeypatch.setattr(admin, "update_memory", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad type")))
    bad = _client.put("/api/admin/memories/bogus/k1", json={"content": "x"})
    assert bad.status_code == 400


def test_delete_memory_404_when_missing(monkeypatch):
    monkeypatch.setattr(admin, "delete_memory", lambda *a, **k: False)
    response = _client.delete("/api/admin/memories/semantic/k1?user_id=u1")
    assert response.status_code == 404
    monkeypatch.setattr(admin, "delete_memory", lambda *a, **k: True)
    assert _client.delete("/api/admin/memories/semantic/k1?user_id=u1").status_code == 200


def test_system_prompt_roundtrip(monkeypatch):
    monkeypatch.setattr(admin, "get_system_prompt", lambda: "Hello")
    assert _client.get("/api/admin/system-prompt").json()["text"] == "Hello"
    monkeypatch.setattr(admin, "set_system_prompt", lambda text: text)
    response = _client.put("/api/admin/system-prompt", json={"text": "New persona"})
    assert response.status_code == 200
    assert response.json()["text"] == "New persona"


def test_agent_crud_routes(monkeypatch):
    created = {"id": "ag_1", "name": "N", "system_prompt": "p", "tools": [], "created_at": None, "updated_at": None}
    monkeypatch.setattr(admin, "create_agent", lambda *a, **k: created)
    response = _client.post("/api/admin/agents", json={"name": "N", "system_prompt": "p"})
    assert response.status_code == 200
    assert response.json()["agent"]["id"] == "ag_1"

    monkeypatch.setattr(admin, "delete_agent", lambda agent_id: True)
    assert _client.delete("/api/admin/agents/ag_1").status_code == 200
    monkeypatch.setattr(admin, "delete_agent", lambda agent_id: False)
    assert _client.delete("/api/admin/agents/ag_1").status_code == 404


def test_agent_run_route_uses_graph_and_returns_result(monkeypatch):
    agent = {"id": "ag_1", "name": "T", "system_prompt": "p", "tools": []}
    monkeypatch.setattr(admin, "get_agent", lambda agent_id: agent)

    async def fake_run(agent, task, graph):
        return {"answer": "running", "decisions": [], "tool_calls": []}

    monkeypatch.setattr(admin, "run_agent", fake_run)
    response = _client.post("/api/admin/agents/ag_1/run", json={"task": "do it"})
    assert response.status_code == 200
    assert response.json()["answer"] == "running"


def test_agent_run_missing_agent_404(monkeypatch):
    monkeypatch.setattr(admin, "get_agent", lambda agent_id: None)
    response = _client.post("/api/admin/agents/nope/run", json={"task": "x"})
    assert response.status_code == 404


def test_agent_run_records_history(monkeypatch):
    agent = {"id": "ag_1", "name": "T", "system_prompt": "p", "tools": []}
    monkeypatch.setattr(admin, "get_agent", lambda agent_id: agent)
    calls = {}

    async def fake_run(agent, task, graph):
        return {"answer": "done", "decisions": [], "tool_calls": [], "workspace": "/w"}

    def fake_record(agent_id, task, result):
        calls["recorded"] = (agent_id, task, result)
        return {"run_id": "r1"}

    monkeypatch.setattr(admin, "run_agent", fake_run)
    monkeypatch.setattr(admin, "record_run", fake_record)
    response = _client.post("/api/admin/agents/ag_1/run", json={"task": "do it"})
    assert response.status_code == 200
    assert response.json()["run_id"] == "r1"
    agent_id, task, result = calls["recorded"]
    assert agent_id == "ag_1" and task == "do it" and result["answer"] == "done"


def test_agent_run_recording_failure_does_not_break_run(monkeypatch):
    agent = {"id": "ag_1", "name": "T", "system_prompt": "p", "tools": []}
    monkeypatch.setattr(admin, "get_agent", lambda agent_id: agent)

    async def fake_run(agent, task, graph):
        return {"answer": "done", "decisions": [], "tool_calls": []}

    monkeypatch.setattr(admin, "run_agent", fake_run)
    monkeypatch.setattr(admin, "record_run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    response = _client.post("/api/admin/agents/ag_1/run", json={"task": "do it"})
    assert response.status_code == 200
    assert response.json()["answer"] == "done"


def test_agent_runs_route_lists_history(monkeypatch):
    runs = [{"run_id": "r1", "task": "do it", "answer": "done", "status": "completed", "created_at": "t"}]
    monkeypatch.setattr(admin, "list_runs", lambda agent_id, limit=20: runs)
    response = _client.get("/api/admin/agents/ag_1/runs")
    assert response.status_code == 200
    assert response.json()["runs"] == runs
    assert response.json()["agent_id"] == "ag_1"


def test_workspace_routes_list_files_and_read_content(monkeypatch):
    files = [{"path": "diary.py", "size": 8}]
    monkeypatch.setattr(admin, "list_workspace_files", lambda agent_id, base=None: files)
    listed = _client.get("/api/admin/agents/ag_1/workspace")
    assert listed.status_code == 200
    assert listed.json()["files"] == files

    monkeypatch.setattr(admin, "read_workspace_file", lambda agent_id, path, base=None: "print(1)")
    content = _client.get("/api/admin/agents/ag_1/workspace/file?path=diary.py")
    assert content.status_code == 200
    assert content.json()["content"] == "print(1)"

    monkeypatch.setattr(
        admin, "read_workspace_file",
        lambda agent_id, path, base=None: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    assert _client.get("/api/admin/agents/ag_1/workspace/file?path=nope.py").status_code == 404

    monkeypatch.setattr(
        admin, "read_workspace_file",
        lambda agent_id, path, base=None: (_ for _ in ()).throw(ValueError("escapes")),
    )
    assert _client.get("/api/admin/agents/ag_1/workspace/file?path=../evil.py").status_code == 400


def test_ui_static_served_with_no_cache_headers():
    """The console bundle must never be served stale: browsers revalidate and
    pick up UI updates without a hard refresh."""
    from api.routes import app as real_app

    client = TestClient(real_app)
    for path in ("/ui/", "/ui/app.js"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert "no-cache" in response.headers.get("cache-control", ""), path