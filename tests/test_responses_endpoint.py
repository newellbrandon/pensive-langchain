"""Tests for the OpenAI Responses API endpoint (POST /v1/responses).

covers:
- extracting the latest user query from Responses `input` (items or raw string)
- building a Responses `response` object (message / output_text item)
- streaming SSE events (response.output_text.delta + response.completed)
"""
from __future__ import annotations

from api.routes import (
    _responses_latest_user_query,
    _build_responses_response,
    _responses_delta_chunk,
    _responses_completed,
    _final_answer,
    _stream_responses,
    _stream_graph,
    _workflow_thought_text,
    _build_chat_completion,
    _chat_messages_to_langchain,
    _responses_input_messages,
    _resolve_user_id,
)
from api.models import ChatMessage
from config import Config
from fastapi import Request as FastAPIRequest
from starlette.datastructures import Headers
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import asyncio
import json


def _item(role: str, text: str) -> dict:
    return {
        "role": role,
        "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}],
    }


class TestResponsesUserQuery:
    def test_extracts_last_user_text_from_items(self):
        body = {"input": [_item("user", "first"), _item("assistant", "a"), _item("user", "second?")]}
        assert _responses_latest_user_query(body) == "second?"

    def test_accepts_raw_string_input(self):
        assert _responses_latest_user_query({"input": "hello there"}) == "hello there"

    def test_empty_when_no_user(self):
        assert _responses_latest_user_query({"input": [_item("assistant", "hi")]}) == ""
        assert _responses_latest_user_query({"input": []}) == ""


class TestResponsesObject:
    def test_message_output_item_shape(self):
        resp = _build_responses_response(
            "resp_abc", "pensive", "Remember that MongoDB is my favorite database"
        )
        assert resp["id"] == "resp_abc"
        assert resp["object"] == "response"
        assert resp["status"] == "completed"
        assert resp["model"] == "pensive"
        assert len(resp["output"]) == 1
        item = resp["output"][0]
        assert item["type"] == "message"
        assert item["role"] == "assistant"
        assert item["phase"] == "final_answer"
        assert item["content"][0]["type"] == "output_text"
        assert item["content"][0]["text"] == "Remember that MongoDB is my favorite database"


class TestResponsesStreaming:
    def test_delta_chunk_is_output_text_delta(self):
        line = _responses_delta_chunk("resp_abc", "Hello")
        assert line.startswith("data: ")
        event = json.loads(line[len("data: "):])
        assert event["type"] == "response.output_text.delta"
        assert event["item_id"] == "resp_abc"
        assert event["delta"] == "Hello"

    def test_completed_event_embeds_final_response(self):
        line = _responses_completed("resp_abc", "pensive", "final!")
        event = json.loads(line[len("data: "):])
        assert event["type"] == "response.completed"
        assert event["response"]["object"] == "response"
        text = event["response"]["output"][0]["content"][0]["text"]
        assert text == "final!"


class TestFinalResponsePurity:
    def test_answer_only_in_content(self):
        # Workflow decisions must never be embedded in the message content:
        # the model's answer is the final response, verbatim.
        final_state = {
            "response": "Yes — Pensive is ready.",
            "decisions": [{"step": "classify_intent", "decision": "memory", "reason": "old turn"}],
        }
        assert _final_answer(final_state) == "Yes — Pensive is ready."

    def test_answer_falls_back_to_last_message(self):
        final_state = {"messages": [AIMessage(content="fallback answer")]}
        assert _final_answer(final_state) == "fallback answer"

    def test_no_workflow_text_without_decisions(self):
        assert _final_answer({"response": "Just an answer"}) == "Just an answer"

    def test_decisions_live_in_metadata_and_reasoning_not_content(self):
        current_turn = [
            {"type": "decision", "step": "classify_intent", "decision": "general", "reason": "this turn"},
            {"type": "decision", "step": "save_memory", "decision": "saved", "reason": "saved 1 item"},
        ]
        resp = _build_responses_response(
            "resp_1", "pensive", "Hello!", decisions=current_turn
        )
        # The message content is pure.
        message = next(i for i in resp["output"] if i["type"] == "message")
        assert message["content"][0]["text"] == "Hello!"
        assert "classify_intent" not in message["content"][0]["text"]
        # Decisions are carried in the reasoning item and the metadata field.
        reasoning = next(i for i in resp["output"] if i["type"] == "reasoning")
        assert reasoning["summary"][0]["type"] == "summary_text"
        assert json.loads(resp["metadata"]["workflow"]) == current_turn

    def test_metadata_empty_without_decisions(self):
        resp = _build_responses_response("resp_1", "pensive", "Hello!")
        assert resp["metadata"] == {}

    def test_completed_event_carries_decisions_in_metadata(self):
        decisions = [
            {"type": "decision", "step": "classify_intent", "decision": "memory", "reason": "q"},
        ]
        line = _responses_completed("resp_abc", "pensive", "Answer", decisions=decisions)
        event = json.loads(line[len("data: "):])
        assert event["type"] == "response.completed"
        message = next(i for i in event["response"]["output"] if i["type"] == "message")
        assert message["content"][0]["text"] == "Answer"
        assert json.loads(event["response"]["metadata"]["workflow"]) == decisions


class TestThoughtRendering:
    DECISIONS = [
        {"type": "decision", "step": "classify_intent", "decision": "memory", "reason": "explicit recall"},
        {"type": "decision", "step": "search_memories", "decision": "2 results", "reason": "semantic=2"},
        {"type": "decision", "step": "generate_response", "decision": "respond", "reason": "generated"},
        {"type": "decision", "step": "save_memory", "decision": "saved", "reason": "saved 1 item"},
    ]

    def test_workflow_thought_text_lists_steps(self):
        text = _workflow_thought_text(self.DECISIONS)
        assert "Classify Intent" in text
        assert "Search Memories" in text
        assert "Save Memory" in text
        assert "explicit recall" in text

    def test_responses_output_leads_with_reasoning_item(self):
        resp = _build_responses_response("resp_1", "pensive", "Hello!", decisions=self.DECISIONS)
        assert resp["output"][0]["type"] == "reasoning"
        summary = resp["output"][0]["summary"][0]
        assert summary["type"] == "summary_text"
        assert "Classify Intent" in summary["text"]
        assert "Save Memory" in summary["text"]
        message = resp["output"][1]
        assert message["type"] == "message"
        assert message["content"][0]["text"] == "Hello!"

    def test_responses_no_reasoning_item_without_decisions(self):
        resp = _build_responses_response("resp_1", "pensive", "Hello!")
        assert len(resp["output"]) == 1
        assert resp["output"][0]["type"] == "message"

    def test_stream_responses_emits_reasoning_summary_then_answer(self):
        graph = _FakeGraph(self.DECISIONS, {"response": "Yes — done."})
        chunks = asyncio.run(_collect(_stream_responses(graph, {}, {}, "resp_1", "pensive")))

        events = []
        for chunk in chunks:
            if not chunk.startswith("data: "):
                continue
            line = chunk[len("data: "):].strip()
            if line == "[DONE]":
                continue
            events.append(json.loads(line))

        types = [e["type"] for e in events]
        assert "response.output_item.added" in types
        assert "response.reasoning_summary_text.delta" in types
        assert "response.reasoning_summary_text.done" in types
        thought = "".join(
            e["delta"] for e in events if e["type"] == "response.reasoning_summary_text.delta"
        )
        assert "Classify Intent" in thought
        assert "Save Memory" in thought
        # The answer is still the only output_text content.
        answer = "".join(
            e["delta"] for e in events if e["type"] == "response.output_text.delta"
        )
        assert answer == "Yes — done."
        assert "Workflow" not in answer

        completed = next(e for e in events if e["type"] == "response.completed")
        assert completed["response"]["output"][0]["type"] == "reasoning"
        message = completed["response"]["output"][1]
        assert message["type"] == "message"
        assert message["content"][0]["text"] == "Yes — done."

    def test_chat_completion_message_has_reasoning_content(self):
        payload = _build_chat_completion(
            "chatcmpl_1", "pensive", "Hello!", self.DECISIONS, user_query="hi"
        )
        message = payload["choices"][0]["message"]
        assert message["content"] == "Hello!"
        assert "Save Memory" in message["reasoning_content"]
        assert payload["workflow"] == self.DECISIONS

    def test_chat_completion_no_reasoning_without_decisions(self):
        payload = _build_chat_completion("chatcmpl_1", "pensive", "Hello!", [], user_query="hi")
        message = payload["choices"][0]["message"]
        assert message["content"] == "Hello!"
        assert "reasoning_content" not in message
        assert "workflow" not in payload

    def test_chat_stream_emits_reasoning_content_then_content(self):
        graph = _FakeGraph(self.DECISIONS, {"response": "Yes — done."})
        chunks = asyncio.run(_collect(_stream_graph(graph, {}, {}, "chatcmpl_1", "pensive")))

        deltas = []
        for chunk in chunks:
            for line in chunk.split("\n\n"):
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):].strip()
                if data == "[DONE]":
                    continue
                event = json.loads(data)
                choices = event.get("choices") or []
                for choice in choices:
                    if isinstance(choice.get("delta"), dict):
                        deltas.append(choice["delta"])

        reasoning = "".join(d.get("reasoning_content", "") or "" for d in deltas)
        content = "".join(d.get("content", "") or "" for d in deltas)
        assert "Classify Intent" in reasoning
        assert "Save Memory" in reasoning
        assert content == "Yes — done."


class _FakeGraph:
    """Minimal graph double: yields decisions first, then the final state."""

    def __init__(self, decisions, final_state):
        self._decisions = decisions
        self._final_state = final_state

    async def astream(self, input_state, config=None, stream_mode=None):
        for decision in self._decisions:
            yield ("custom", {"type": "decision", **decision})
        yield ("values", self._final_state)


async def _collect(agen):
    return [chunk async for chunk in agen]


class TestStreamingPurity:
    def test_stream_responses_output_text_is_answer_only(self):
        decisions = [
            {"step": "classify_intent", "decision": "memory", "reason": "recall"},
            {"step": "save_memory", "decision": "saved", "reason": "saved 1 item"},
        ]
        graph = _FakeGraph(decisions, {"response": "Yes — done."})
        chunks = asyncio.run(_collect(_stream_responses(graph, {}, {}, "resp_1", "pensive")))

        events = []
        for chunk in chunks:
            if not chunk.startswith("data: "):
                continue
            line = chunk[len("data: "):].strip()
            if line == "[DONE]":
                continue
            events.append(json.loads(line))

        deltas = "".join(
            e["delta"] for e in events if e["type"] == "response.output_text.delta"
        )
        assert deltas == "Yes — done."
        assert "Workflow" not in deltas
        assert "Classify Intent" not in deltas
        assert "Save Memory" not in deltas

        completed = next(e for e in events if e["type"] == "response.completed")
        meta = json.loads(completed["response"]["metadata"]["workflow"])
        assert meta[0]["step"] == "classify_intent"
        assert meta[1]["step"] == "save_memory"
        message = next(i for i in completed["response"]["output"] if i["type"] == "message")
        assert message["content"][0]["text"] == "Yes — done."


class TestContextPreservation:
    def test_fails_open_to_default_user(self, monkeypatch):
        from api.models import ChatCompletionRequest
        from starlette.datastructures import Headers

        monkeypatch.setattr(Config, "USER_ID", "")
        http = FastAPIRequest(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": Headers({}).raw,
                "query_string": b"",
                "server": ("", 80),
                "client": ("", 0),
                "scheme": "http",
            }
        )
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        assert _resolve_user_id(req, http) == "default"
    def test_chat_messages_preserve_all_roles(self):
        msgs = [
            ChatMessage(role="system", content="You edit code."),
            ChatMessage(role="user", content="Fix the bug"),
            ChatMessage(role="assistant", content="I will add a test."),
            ChatMessage(role="tool", content="contents of /tmp/file.py"),
        ]
        converted = _chat_messages_to_langchain(msgs)
        assert isinstance(converted[0], SystemMessage)
        assert converted[0].content == "You edit code."
        assert isinstance(converted[1], HumanMessage)
        assert converted[1].content == "Fix the bug"
        assert isinstance(converted[2], AIMessage)
        assert converted[2].content == "I will add a test."
        assert isinstance(converted[3], HumanMessage)
        assert "contents of /tmp/file.py" in converted[3].content

    def test_responses_input_preserves_instructions_and_roles(self):
        body = {
            "instructions": "You are a coding agent.",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "fix it"}]},
                {"role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
            ],
        }
        converted = _responses_input_messages(body)
        assert isinstance(converted[0], SystemMessage)
        assert converted[0].content == "You are a coding agent."
        assert isinstance(converted[1], HumanMessage)
        assert converted[1].content == "fix it"
        assert isinstance(converted[2], AIMessage)
        assert converted[2].content == "done"