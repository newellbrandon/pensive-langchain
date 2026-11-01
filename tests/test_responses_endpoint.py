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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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

    def test_answer_is_empty_for_tool_call_turns(self):
        # A tool-only assistant turn has no answer text; it must not fall back
        # to the stringified message object.
        final_state = {"response": "", "messages": [AIMessage(content="")]}
        assert _final_answer(final_state) == ""

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

    def test_chat_stream_streams_decisions_live(self):
        event = asyncio.Event()
        probe = _LiveProbeGraph(
            [{"step": "classify_intent", "decision": "general", "reason": "first"}],
            {"response": "Final answer"},
            event,
        )

        async def collect(agen):
            chunks = []
            async for chunk in agen:
                chunks.append(chunk)
                if b"reasoning_content" in chunk.encode():
                    event.set()
            return chunks

        chunks = asyncio.run(collect(_stream_graph(probe, {}, {}, "chatcmpl_1", "pensive")))
        assert not probe.timed_out, "decision deltas must be streamed while the graph is running"

        deltas = []
        for chunk in chunks:
            for line in chunk.split("\n\n"):
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):].strip()
                if data == "[DONE]":
                    continue
                for choice in json.loads(data).get("choices") or []:
                    if isinstance(choice.get("delta"), dict):
                        deltas.append(choice["delta"])
        reasoning = [d.get("reasoning_content", "") for d in deltas if d.get("reasoning_content")]
        content = "".join(d.get("content", "") or "" for d in deltas)
        assert "Classify Intent" in "".join(reasoning)
        assert content == "Final answer"

    def test_stream_responses_streams_decisions_live(self):
        event = asyncio.Event()
        probe = _LiveProbeGraph(
            [{"step": "classify_intent", "decision": "general", "reason": "first"}],
            {"response": "Final answer"},
            event,
        )

        async def collect(agen):
            chunks = []
            async for chunk in agen:
                chunks.append(chunk)
                if b"response.reasoning_summary_text.delta" in chunk.encode():
                    event.set()
            return chunks

        chunks = asyncio.run(collect(_stream_responses(probe, {}, {}, "resp_1", "pensive")))
        assert not probe.timed_out, "decision deltas must be streamed while the graph is running"

        events = []
        for chunk in chunks:
            if not chunk.startswith("data: "):
                continue
            line = chunk[len("data: "):].strip()
            if line == "[DONE]":
                continue
            events.append(json.loads(line))
        thought = "".join(
            e["delta"] for e in events if e["type"] == "response.reasoning_summary_text.delta"
        )
        assert "Classify Intent" in thought
        answer = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
        assert answer == "Final answer"

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


class _LiveProbeGraph:
    """Proves decisions are streamed while the graph is still running.

    After yielding a decision event, the graph waits for the stream consumer to
    receive that decision's reasoning delta before continuing. A buffering
    implementation (which only emits after the graph finishes) deadlocks the
    probe and sets ``timed_out``.
    """

    def __init__(self, decisions, final_state, saw_delta):
        self._decisions = decisions
        self._final_state = final_state
        self._saw_delta = saw_delta  # asyncio.Event set by the consumer
        self.timed_out = False

    async def astream(self, input_state, config=None, stream_mode=None):
        for decision in self._decisions:
            self._saw_delta.clear()
            yield ("custom", {"type": "decision", **decision})
            try:
                await asyncio.wait_for(self._saw_delta.wait(), timeout=3)
            except asyncio.TimeoutError:
                self.timed_out = True
                break
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


class TestToolCallRelay:
    TOOL_CALLS = [{"id": "call_1", "name": "read", "arguments": '{"path": "README.md"}'}]

    def test_chat_message_carries_tool_calls(self):
        payload = _build_chat_completion(
            "chatcmpl_1",
            "pensive",
            "",
            [],
            user_query="read the file",
            tool_calls=self.TOOL_CALLS,
        )
        message = payload["choices"][0]["message"]
        assert message["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": '{"path": "README.md"}'},
            }
        ]
        assert payload["choices"][0]["finish_reason"] == "tool_calls"

    def test_responses_output_includes_function_call_item(self):
        resp = _build_responses_response(
            "resp_1", "pensive", "", decisions=[], tool_calls=self.TOOL_CALLS
        )
        first = resp["output"][0]
        assert first["type"] == "function_call"
        assert first["call_id"] == "call_1"
        assert first["name"] == "read"
        assert first["arguments"] == '{"path": "README.md"}'
        assert first["status"] == "completed"
        assert resp["output"][1]["type"] == "message"

    def test_chat_stream_emits_tool_call_deltas(self):
        graph = _FakeGraph([], {"response": "", "tool_calls": self.TOOL_CALLS})
        chunks = asyncio.run(_collect(_stream_graph(graph, {}, {}, "chatcmpl_1", "pensive")))

        deltas = []
        finish_reasons = []
        for chunk in chunks:
            for line in chunk.split("\n\n"):
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):].strip()
                if data == "[DONE]":
                    continue
                event = json.loads(data)
                for choice in event.get("choices") or []:
                    if isinstance(choice.get("delta"), dict):
                        deltas.append(choice["delta"])
                    if choice.get("finish_reason"):
                        finish_reasons.append(choice["finish_reason"])

        tool_chunks = [d for d in deltas if d.get("tool_calls")]
        assert tool_chunks, "must stream tool_calls deltas"
        assert tool_chunks[0]["tool_calls"][0]["id"] == "call_1"
        assert tool_chunks[0]["tool_calls"][0]["function"]["name"] == "read"
        arguments = "".join(
            c["tool_calls"][0]["function"].get("arguments", "")
            for c in tool_chunks
        )
        assert arguments == '{"path": "README.md"}'
        assert finish_reasons and finish_reasons[-1] == "tool_calls"
        # no answer text was streamed
        assert "".join(d.get("content", "") or "" for d in deltas) == ""

    def test_stream_responses_emits_function_call_events(self):
        graph = _FakeGraph([], {"response": "", "tool_calls": self.TOOL_CALLS})
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
        added = next(e for e in events if e["type"] == "response.output_item.added")
        assert added["item"]["type"] == "function_call"
        assert added["item"]["name"] == "read"
        arguments = "".join(
            e["delta"]
            for e in events
            if e["type"] == "response.function_call_arguments.delta"
        )
        assert arguments == '{"path": "README.md"}'
        assert any(e["type"] == "response.function_call_arguments.done" for e in events)

        completed = next(e for e in events if e["type"] == "response.completed")
        assert completed["response"]["output"][0]["type"] == "function_call"

    def test_chat_messages_preserve_tool_calls_and_results(self):
        msgs = [
            ChatMessage(role="user", content="read the file"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path": "README.md"}'},
                    }
                ],
            ),
            ChatMessage(role="tool", content="first line: # Pensive", tool_call_id="call_1"),
        ]
        converted = _chat_messages_to_langchain(msgs)
        assistant = converted[1]
        assert assistant.tool_calls[0]["id"] == "call_1"
        assert assistant.tool_calls[0]["name"] == "read"
        assert assistant.tool_calls[0]["args"] == {"path": "README.md"}
        assert converted[2].tool_call_id == "call_1"
        assert converted[2].content == "first line: # Pensive"

    def test_chat_messages_allow_null_content_tool_call_turn(self):
        # OpenCode sends assistant tool-call turns with content: null.
        msgs = [
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            ),
            ChatMessage(role="tool", content="<file contents>", tool_call_id="call_1"),
        ]
        converted = _chat_messages_to_langchain(msgs)
        assert converted[0].content == ""
        assert converted[0].tool_calls[0]["name"] == "read"
        assert converted[0].tool_calls[0]["args"] == {}
        assert converted[1].tool_call_id == "call_1"

    def test_responses_input_preserves_function_call_output(self):
        body = {
            "input": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "function_call", "call_id": "call_1", "name": "read", "arguments": '{"path": "a"}'}
                    ],
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "contents"},
            ]
        }
        converted = _responses_input_messages(body)
        assert converted[0].tool_calls[0]["id"] == "call_1"
        assert converted[0].tool_calls[0]["name"] == "read"
        assert converted[1].tool_call_id == "call_1"
        assert converted[1].content == "contents"


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

    def test_user_id_pin_overrides_forwarded_headers(self, monkeypatch):
        from api.models import ChatCompletionRequest
        from starlette.datastructures import Headers

        monkeypatch.setattr(Config, "USER_ID", "default")
        http = FastAPIRequest(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": Headers({"X-OpenWebUI-User-Id": "some-uuid"}).raw,
                "query_string": b"",
                "server": ("", 80),
                "client": ("", 0),
                "scheme": "http",
            }
        )
        # Pin wins even against a forwarded OpenWebUI header + body user.
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")], user="body-user"
        )
        assert _resolve_user_id(req, http) == "default"

    def test_forwarded_header_used_when_pin_unset(self, monkeypatch):
        from api.models import ChatCompletionRequest
        from starlette.datastructures import Headers

        monkeypatch.setattr(Config, "USER_ID", "")
        http = FastAPIRequest(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": Headers({"X-OpenWebUI-User-Id": "some-uuid"}).raw,
                "query_string": b"",
                "server": ("", 80),
                "client": ("", 0),
                "scheme": "http",
            }
        )
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        assert _resolve_user_id(req, http) == "some-uuid"
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
        assert isinstance(converted[3], ToolMessage)
        assert converted[3].content == "contents of /tmp/file.py"

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