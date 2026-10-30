"""Tests for the OpenAI Responses API client used to reach the AI gateway.

covers:
- URL normalization (LLM_URI may or may not include the /responses suffix)
- request shape (model, instructions, input) and Authorization header
- extraction of the final assistant answer, ignoring reasoning/"thinking" items
- tool definition forwarding, function-call extraction, and result replay
"""
from __future__ import annotations

import asyncio

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from config import Config
from langchain_core.messages import HumanMessage, SystemMessage
from utils.llm import ResponsesClient, get_responses_client


def _fake_http(payload: dict):
    """Return (session, factory) where session.post returns a response with payload."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)

    session = AsyncMock()
    session.post = AsyncMock(return_value=resp)

    async def _aenter(*_args, **_kwargs):
        return session

    async def _aexit(*_args, **_kwargs):
        return False

    cm = MagicMock()
    cm.__aenter__ = _aenter
    cm.__aexit__ = _aexit

    factory = MagicMock(return_value=cm)
    return session, factory


def _run(coro):
    return asyncio.run(coro)


def _sample_response(final_text: str = "pensive", with_reasoning: bool = True) -> dict:
    output = []
    if with_reasoning:
        output.append(
            {
                "id": "msg_reasoning_1",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": "the model's private thinking"},
                            {"type": "summary_text", "text": "should be ignored"}],
            }
        )
    output.append(
        {
            "id": "msg_ans_1",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": final_text}],
            "phase": "final_answer",
        }
    )
    return {"id": "resp_1", "object": "response", "status": "completed", "output": output}


class TestResponsesUrl:
    def test_appends_suffix_when_missing(self, monkeypatch):
        monkeypatch.setattr(Config, "LLM_URI", "https://gateway.example/v1")
        assert Config.responses_url() == "https://gateway.example/v1/responses"

    def test_strips_redundant_suffix(self, monkeypatch):
        monkeypatch.setattr(Config, "LLM_URI", "https://gateway.example/v1/responses")
        assert Config.responses_url() == "https://gateway.example/v1/responses"

    def test_strips_trailing_slash_before_suffix(self, monkeypatch):
        monkeypatch.setattr(Config, "LLM_URI", "https://gateway.example/v1/")
        assert Config.responses_url() == "https://gateway.example/v1/responses"


class TestResponsesClient:
    def test_builds_request_with_bearer_auth_instructions_and_input(self):
        payload = _sample_response("hi", with_reasoning=False)
        session, factory = _fake_http(payload)
        client = ResponsesClient(
            url="https://gateway.example/v1/responses",
            api_key="secret-key",
            model="gpt-5.6-luna",
        )

        with patch("httpx.AsyncClient", factory):
            text = _run(
                client.agenerate(
                    [
                        SystemMessage(content="Be concise."),
                        HumanMessage(content="Say hi"),
                    ]
                )
            )

        assert text == "hi"
        session.post.assert_awaited_once()
        args, kwargs = session.post.await_args
        assert args[0] == "https://gateway.example/v1/responses"
        assert kwargs["headers"]["Authorization"] == "Bearer secret-key"
        body = kwargs["json"]
        assert body["model"] == "gpt-5.6-luna"
        assert body["instructions"] == "Be concise."
        assert body["input"] == [
            {"role": "user", "content": [{"type": "input_text", "text": "Say hi"}]}
        ]

    def test_extracts_final_answer_and_ignores_reasoning_items(self):
        session, factory = _fake_http(_sample_response("hello pensive", with_reasoning=True))
        client = ResponsesClient(
            url="https://gateway.example/v1/responses",
            api_key="secret-key",
            model="gpt-5.6-luna",
        )

        with patch("httpx.AsyncClient", factory):
            text = _run(client.agenerate([HumanMessage(content="Hello")]))

        # reasoning summary text must NOT leak into the answer
        assert text == "hello pensive"
        assert "thinking" not in text
        assert "ignored" not in text

    def test_returns_empty_string_when_no_message_output(self):
        session, factory = _fake_http({"id": "resp_1", "output": []})
        client = ResponsesClient(url="https://g/v1/responses", api_key="k", model="m")

        with patch("httpx.AsyncClient", factory):
            text = _run(client.agenerate([HumanMessage(content="Hi")]))

        assert text == ""

    def test_get_responses_client_uses_configured_values(self, monkeypatch):
        monkeypatch.setattr(Config, "LLM_URI", "https://g.example/v1/responses")
        monkeypatch.setattr(Config, "LLM_KEY", "k")
        monkeypatch.setattr(Config, "LLM_MODEL", "m")
        client = get_responses_client()
        assert client._url == "https://g.example/v1/responses"
        assert client._api_key == "k"
        assert client._model == "m"

    def test_preserves_tool_and_unknown_role_content(self):
        from langchain_core.messages import ToolMessage

        payload = _sample_response("ok", with_reasoning=False)
        session, factory = _fake_http(payload)
        client = ResponsesClient(url="https://g/v1/responses", api_key="k", model="m")

        with patch("httpx.AsyncClient", factory):
            _run(
                client.agenerate(
                    [
                        HumanMessage(content="check the file"),
                        ToolMessage(content="<file contents>", tool_call_id="call_1"),
                    ]
                )
            )

        body = session.post.await_args.kwargs["json"]
        # Tool results must replay as proper function_call_output items so the
        # upstream Responses API sees the call/result pairing.
        assert body["input"][1] == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "<file contents>",
        }


def _sample_tool_call_response() -> dict:
    return {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "id": "fc_1",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_abc",
                "name": "read",
                "arguments": '{"path": "README.md"}',
            },
            {
                "id": "msg_1",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": ""}],
                "phase": "final_answer",
            },
        ],
    }


class TestResponsesClientTools:
    def test_forwards_tools_and_tool_choice(self):
        payload = _sample_response("ok", with_reasoning=False)
        session, factory = _fake_http(payload)
        client = ResponsesClient(url="https://g/v1/responses", api_key="k", model="m")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {"type": "object"},
                },
            }
        ]
        with patch("httpx.AsyncClient", factory):
            _run(
                client.acomplete(
                    [HumanMessage(content="read the file")], tools=tools, tool_choice="auto"
                )
            )

        body = session.post.await_args.kwargs["json"]
        assert body["tools"] == [
            {
                "type": "function",
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object"},
            }
        ]
        assert body["tool_choice"] == "auto"

    def test_passes_responses_format_tools_through(self):
        payload = _sample_response("ok", with_reasoning=False)
        session, factory = _fake_http(payload)
        client = ResponsesClient(url="https://g/v1/responses", api_key="k", model="m")

        tools = [{"type": "function", "name": "exec", "description": "Run a command"}]
        with patch("httpx.AsyncClient", factory):
            _run(client.acomplete([HumanMessage(content="run it")], tools=tools))

        body = session.post.await_args.kwargs["json"]
        assert body["tools"] == tools

    def test_extracts_tool_calls_together_with_answer(self):
        session, factory = _fake_http(_sample_tool_call_response())
        client = ResponsesClient(url="https://g/v1/responses", api_key="k", model="m")

        with patch("httpx.AsyncClient", factory):
            result = _run(client.acomplete([HumanMessage(content="read the file")]))

        assert result.text == ""
        assert result.tool_calls == [
            {"id": "call_abc", "name": "read", "arguments": '{"path": "README.md"}'}
        ]

    def test_error_surfaces_upstream_response_body(self):
        resp = MagicMock()
        resp.status_code = 400
        resp.text = '{"error": {"message": "bad function_call"}}'
        resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Bad Request", request=MagicMock(), response=resp
            )
        )
        resp.request = MagicMock()

        session = AsyncMock()
        session.post = AsyncMock(return_value=resp)

        async def _aenter(*_args, **_kwargs):
            return session

        async def _aexit(*_args, **_kwargs):
            return False

        cm = MagicMock()
        cm.__aenter__ = _aenter
        cm.__aexit__ = _aexit
        factory = MagicMock(return_value=cm)

        client = ResponsesClient(url="https://g/v1/responses", api_key="k", model="m")
        with patch("httpx.AsyncClient", factory):
            with pytest.raises(httpx.HTTPStatusError) as excinfo:
                _run(client.acomplete([HumanMessage(content="Hi")]))

        assert "bad function_call" in str(excinfo.value)

    def test_replays_assistant_tool_calls_and_results(self):
        from langchain_core.messages import AIMessage, ToolMessage

        payload = _sample_response("done", with_reasoning=False)
        session, factory = _fake_http(payload)
        client = ResponsesClient(url="https://g/v1/responses", api_key="k", model="m")

        with patch("httpx.AsyncClient", factory):
            _run(
                client.acomplete(
                    [
                        HumanMessage(content="read README"),
                        AIMessage(
                            content="",
                            tool_calls=[{"id": "call_1", "name": "read", "args": {"path": "README.md"}}],
                        ),
                        ToolMessage(content="line 1: # Pensive", tool_call_id="call_1"),
                    ]
                )
            )

        body = session.post.await_args.kwargs["json"]
        items = body["input"]
        # Assistant tool calls replay as top-level function_call input items
        # (the gateway rejects function_call content blocks), followed by the
        # function_call_output result paired by call_id.
        assert len(items) == 3
        assert items[0]["role"] == "user"
        assert items[1] == {
            "type": "function_call",
            "call_id": "call_1",
            "name": "read",
            "arguments": '{"path": "README.md"}',
            "status": "completed",
        }
        assert items[2] == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "line 1: # Pensive",
        }