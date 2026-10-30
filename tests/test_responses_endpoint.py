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
    _assemble_content,
    _chat_messages_to_langchain,
    _responses_input_messages,
    _resolve_user_id,
)
from api.models import ChatMessage
from config import Config
from fastapi import Request as FastAPIRequest
from starlette.datastructures import Headers
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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


class TestAssembleContent:
    def test_uses_only_supplied_decisions(self):
        # Decisions passed in represent the CURRENT turn only — historical
        # decisions are not replayed even if they live on the thread state.
        final_state = {
            "response": "Hello!",
            "decisions": [{"step": "classify_intent", "decision": "memory", "reason": "old turn"}],
        }
        current_turn = [{"step": "classify_intent", "decision": "general", "reason": "this turn"}]
        content = _assemble_content(final_state, current_turn)
        assert "this turn" in content or "general" in content
        assert "old turn" not in content

    def test_no_workflow_header_without_decisions(self):
        content = _assemble_content({"response": "Just an answer"}, [])
        assert content == "Just an answer"


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