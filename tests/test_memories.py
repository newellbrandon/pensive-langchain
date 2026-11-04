"""Tests for the four-type (semantic/episodic/taxonomic/procedural) memory module.

Covers Pensive's MongoDBStore per-type namespaces and the Responses API client.
"""
from __future__ import annotations

import asyncio
import json
import os
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from memory import memories as mem
from memory.memories import MemoryPlan


def _run(coro):
    return asyncio.run(coro)


class FakeSearchItem:
    def __init__(self, key, value, score=None):
        self.key = key
        self.value = value
        self.score = score


class FakeStore:
    def __init__(self, chunks_by_namespace=None):
        self.chunks = chunks_by_namespace or {}
        self.calls: list[dict] = []
        self.data: dict = {}

    def search(self, namespace, query=None, limit=10, **kwargs):
        self.calls.append({"namespace": namespace, "query": query, "limit": limit})
        return self.chunks.get(namespace, [])

    def put(self, namespace, key, value, index=None, **kwargs):
        self.data.setdefault(namespace, {})[key] = value
        self.calls.append({"op": "put", "namespace": namespace, "key": key, "value": value})

    def delete(self, namespace, key):
        self.calls.append({"op": "delete", "namespace": namespace, "key": key})
        if self.data.get(namespace):
            self.data[namespace].pop(key, None)


class FakeClient:
    def __init__(self, result: str | Exception):
        self.result = result
        self.calls: list = []

    async def agenerate(self, messages):
        self.calls.append(messages)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _item(src: str, text: str, score=None, *, recorded_at=None, superseded=False):
    value = {
        "content": text,
        "user_id": "u1",
        "type": src,
        "label": f"{src}_x",
        "recorded_at": recorded_at,
        **({"superseded_by_version_id": "v2"} if superseded else {}),
    }
    return FakeSearchItem(f"k_{src}", value, score)


class MemTypeTest(unittest.TestCase):
    def test_all_four_types(self):
        self.assertEqual(set(mem.MEMORY_TYPES), {"semantic", "episodic", "taxonomic", "procedural"})


class RetrieveTest(unittest.TestCase):
    def test_searches_all_four_sources(self):
        store = FakeStore()
        # store returns nothing; assert all 4 namespaces were queried
        store.chunks = {}
        _ = mem.retrieve(store, "training plan", "u1")
        self.assertEqual(
            [c["namespace"][-1] for c in store.calls],
            ["semantic", "episodic", "taxonomic", "procedural"],
        )

    def test_maps_chunks_and_drops_empty_and_superseded(self):
        namespace = ("users", "u1", "memories", "semantic")
        store = FakeStore(
            {
                namespace: [
                    _item("semantic", "User name: Brandon", 0.9,
                          recorded_at="2026-10-26T16:56:00+00:00"),
                    _item("semantic", "", 0.5),
                    _item("semantic", "User name: Old", 0.7, superseded=True),
                ]
            }
        )
        hits = mem.retrieve(store, "who", "u1", sources=["semantic"])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["text"], "User name: Brandon")
        self.assertEqual(hits[0]["source"], "semantic")
        self.assertEqual(hits[0]["similarity_score"], 0.9)
        self.assertEqual(hits[0]["timestamp"], "2026-10-26T16:56:00+00:00")

    def test_returns_empty_without_store(self):
        self.assertEqual(mem.retrieve(None, "q", "u1"), [])


class RerankTest(unittest.TestCase):
    CANDIDATES = [
        {"text": "a", "source": "semantic", "similarity_score": 0.9, "timestamp": None},
        {"text": "b", "source": "episodic", "similarity_score": 0.6, "timestamp": None},
    ]

    def test_maps_voyage_and_attaches_relevance(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("authorization")
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"data": [{"index": 1, "relevance_score": 0.8}]})

        with patch.dict(os.environ, {"VOYAGE_API_KEY": "vk"}):
            hits = mem.rerank("q", self.CANDIDATES, top_k=1, transport=httpx.MockTransport(handler))

        self.assertEqual(captured["body"]["documents"], ["a", "b"])
        self.assertEqual([h["text"] for h in hits], ["b"])
        self.assertEqual(hits[0]["relevance_score"], 0.8)

    def test_falls_back_to_similarity_without_key(self):
        with patch.dict(os.environ, {"VOYAGE_API_KEY": ""}):
            hits = mem.rerank("q", self.CANDIDATES, top_k=2)
        self.assertEqual([h["text"] for h in hits], ["a", "b"])
        self.assertTrue(all(h["relevance_score"] is None for h in hits))

    def test_skips_call_without_candidates(self):
        self.assertEqual(mem.rerank("q", []), [])


class FormatTest(unittest.TestCase):
    def test_chronological_newest_first(self):
        hits = [
            {"text": "old", "source": "semantic", "timestamp": "2026-10-20T00:00:00+00:00"},
            {"text": "new", "source": "semantic", "timestamp": "2026-10-26T00:00:00+00:00"},
            {"text": "none", "source": "semantic", "timestamp": None},
        ]
        ordered = mem.chronological(hits)
        self.assertEqual([h["text"] for h in ordered], ["new", "old", "none"])

    def test_format_context_dated(self):
        hits = [{"text": "User name: Brandon", "source": "semantic",
                 "timestamp": "2026-10-26T14:30:00+00:00"}]
        block = mem.format_context(hits, now=datetime(2026, 10, 26, 17, 0, tzinfo=timezone.utc))
        self.assertIn("Current time: 2026-10-26 17:00 UTC.", block)
        self.assertIn("- [2026-10-26 14:30 UTC, semantic] User name: Brandon", block)


class PlanTest(unittest.TestCase):
    def test_intent_prompt_instructs_research_classification(self):
        # The planner must recognize requests needing current/external/
        # multi-source information as intent == "research" so the graph can
        # delegate them deterministically.
        self.assertIn("research", mem.INTENT_PROMPT)

    def test_falls_back_to_no_memory_search_on_llm_failure(self):
        client = FakeClient(RuntimeError("boom"))
        plan = _run(mem.plan_memory_query(client, [HumanMessage(content="inspire me")]))
        self.assertIsInstance(plan, MemoryPlan)
        # A failed analysis must not claim memory is needed.
        self.assertFalse(plan.needs_memory)
        self.assertEqual(plan.query, "inspire me")
        self.assertEqual(plan.rationale, "Intent analysis unavailable")

    def test_classify_context_is_text_only_across_tool_turns(self):
        client = FakeClient(
            json.dumps({
                "intent": "general",
                "needs_memory": False,
                "query": "",
                "rationale": "simple",
            })
        )
        transcript = [
            HumanMessage(content="read the file"),
            AIMessage(
                content="",
                tool_calls=[{"id": "call_1", "name": "read", "args": {"path": "README.md"}}],
            ),
            ToolMessage(content="<file>", tool_call_id="call_1"),
            HumanMessage(content="what does it say?"),
        ]
        plan = _run(mem.plan_memory_query(client, transcript))
        self.assertEqual(plan.intent, "general")
        sent = client.calls[-1]
        # The classify context must be text-only: truncated tool turns would
        # split function_call/function_call_output pairs and fail upstream (the
        # gateway rejects outputs without a matching call).
        self.assertTrue(all(not getattr(m, "tool_calls", None) for m in sent))
        self.assertTrue(all(not isinstance(m, ToolMessage) for m in sent))
        self.assertTrue(
            any("what does it say?" in str(getattr(m, "content", "")) for m in sent)
        )

    def test_parses_json_plan(self):
        payload = json.dumps({
            "intent": "wants inspiration",
            "needs_memory": True,
            "query": "user interests",
            "rationale": "r",
        })
        client = FakeClient(payload)
        plan = _run(mem.plan_memory_query(client, [HumanMessage(content="inspire me")]))
        self.assertEqual(plan.intent, "wants inspiration")
        self.assertTrue(plan.needs_memory)
        self.assertEqual(plan.query, "user interests")


class ExtractTest(unittest.TestCase):
    def test_parses_typed_memories_and_saves_per_type(self):
        payload = json.dumps({
            "memories": [
                {"type": "semantic", "text": "User name is Brandon", "label": "name"},
                {"type": "episodic", "text": "Discussed MongoDB today", "label": "today"},
            ]
        })
        client = FakeClient(payload)
        store = FakeStore()
        n = _run(mem.extract_and_save(client, store, "My name is Brandon and we discussed MongoDB", "u1"))
        self.assertEqual(n, 2)
        put_namespaces = [c["namespace"][-1] for c in store.calls if c.get("op") == "put"]
        self.assertEqual(set(put_namespaces), {"semantic", "episodic"})
        # same label overwrites -> supersession by key
        n2 = _run(mem.extract_and_save(client, store, "My name is Brandon", "u1"))
        self.assertEqual(n2, 2)


class SaveTest(unittest.TestCase):
    def test_put_into_type_namespace_with_metadata(self):
        store = FakeStore()
        mem.save_memories(
            store, "u1",
            [{"type": "taxonomic", "text": "Trekking is hiking over 5 days", "label": "trekking"}],
        )
        puts = [c for c in store.calls if c.get("op") == "put"]
        self.assertEqual(puts[0]["namespace"], ("users", "u1", "memories", "taxonomic"))
        self.assertEqual(puts[0]["value"]["content"], "Trekking is hiking over 5 days")
        self.assertEqual(puts[0]["value"]["type"], "taxonomic")
        self.assertIn("recorded_at", puts[0]["value"])

    def test_save_updates_near_duplicate_instead_of_new_key(self):
        store = FakeStore(
            {
                ("users", "u1", "memories", "semantic"): [
                    FakeSearchItem(
                        "k1",
                        {
                            "content": "Brandon prefers Glean for internal search.",
                            "user_id": "u1",
                            "type": "semantic",
                            "label": "search prefs",
                            "recorded_at": "2026-10-01T00:00:00+00:00",
                        },
                        score=0.93,
                    )
                ]
            }
        )
        saved = mem.save_memories(
            store, "u1",
            [{
                "type": "semantic",
                "text": "Brandon prefers Glean for internal search. He uses it daily.",
                "label": "preferred search tool",
            }],
        )
        self.assertEqual(saved, 1)
        puts = [c for c in store.calls if c.get("op") == "put"]
        # same record, not a new key: exact label drift converges onto one memory
        self.assertEqual(len(puts), 1)
        self.assertEqual(puts[0]["key"], "k1")
        self.assertEqual(puts[0]["value"]["label"], "search prefs")
        self.assertEqual(puts[0]["value"]["recorded_at"], "2026-10-01T00:00:00+00:00")
        self.assertIn("He uses it daily", puts[0]["value"]["content"])
        self.assertIn("updated_at", puts[0]["value"])

    def test_save_keeps_new_key_without_near_duplicate(self):
        store = FakeStore()
        mem.save_memories(
            store, "u1",
            [{"type": "semantic", "text": "Likes dark mode.", "label": "Theme"}],
        )
        puts = [c for c in store.calls if c.get("op") == "put"]
        self.assertEqual(len(puts), 1)

    def test_save_normalizes_labels_before_keying(self):
        store = FakeStore()
        mem.save_memories(
            store, "u1",
            [{"type": "semantic", "text": "Professional focus.", "label": "Professional Focus"}],
        )
        mem.save_memories(
            store, "u1",
            [{"type": "semantic", "text": "Professional focus.", "label": "professional focus"}],
        )
        keys = {c["key"] for c in store.calls if c.get("op") == "put"}
        self.assertEqual(len(keys), 1)

    def test_merge_texts_combines_unique_lines(self):
        self.assertEqual(mem.merge_texts("alpha", "alpha"), "alpha")
        self.assertEqual(mem.merge_texts("alpha", "alpha\nbeta"), "alpha\nbeta")
        self.assertEqual(mem.merge_texts("alpha\ngamma", "beta\nalpha"), "alpha\ngamma\nbeta")
        self.assertEqual(mem.merge_texts("", "new"), "new")

    def test_merge_texts_collapses_near_identical_lines_keeping_longer(self):
        merged = mem.merge_texts(
            "Prefers to create files and run tests automatically",
            "Prefers to create files and run tests automatically without further questions",
        )
        # one line — the fuller version — not a literal concatenation
        self.assertEqual(merged.count("\n"), 0)
        self.assertIn("without further questions", merged)
        self.assertNotIn("automatically\n", merged)

    def test_merge_texts_keeps_distinct_facts(self):
        merged = mem.merge_texts("Likes dark mode", "Works at MongoDB")
        self.assertEqual(merged, "Likes dark mode\nWorks at MongoDB")

    def test_consolidate_memories_returns_llm_polish(self):
        client = FakeClient('"CSX batches data every 30 minutes and needs near-real-time intelligence."')
        text = _run(mem.consolidate_memories(client, ["CSX batches at 30-minute intervals", "CSX needs near-real-time intelligence"]))
        self.assertIn("30 minutes", text)

    def test_consolidate_memories_falls_back_on_failure(self):
        client = FakeClient(RuntimeError("model down"))
        text = _run(mem.consolidate_memories(client, ["some fragment"]))
        self.assertEqual(text, "")

    def test_lexical_similarity_compares_token_overlap(self):
        self.assertEqual(mem.lexical_similarity("a b c", "a b c"), 1.0)
        self.assertEqual(mem.lexical_similarity("a b", "c d"), 0.0)
        self.assertGreater(mem.lexical_similarity("prefers glean search daily", "prefers glean search sometimes"), 0.5)
        self.assertLess(mem.lexical_similarity("prefers glean search daily", "prefers glean search sometimes"), 1.0)


if __name__ == "__main__":
    unittest.main()