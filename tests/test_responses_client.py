"""Tests for the OpenAI Responses API client used to reach the AI gateway.

covers:
- URL normalization (LLM_URI may or may not include the /responses suffix)
- request shape (model, instructions, input) and Authorization header
- extraction of the final assistant answer, ignoring reasoning/"thinking" items
"""
from __future__ import annotations

import asyncio

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