"""Tests for the Glean MCP integration: OAuth2, token cache, MCP client, routes."""
from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from glean.oauth import OAuth2Client, ClientRegistration, TokenSet
from glean.token_cache import TokenCache
from glean import client as glean_client_mod


def _run(coro):
    return asyncio.run(coro)


class FakeResponse:
    def __init__(self, status_code=200, body=None, headers=None):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.headers = headers or {}

    def json(self):
        return self._body

    @property
    def text(self):
        return json.dumps(self._body)


class FakeHttp:
    """Scripted httpx stand-in: responds from a queue or a handler."""

    def __init__(self, handler=None):
        self.handler = handler
        self.calls = []

    async def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.handler("GET", url, kwargs) if self.handler else FakeResponse()

    async def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.handler("POST", url, kwargs) if self.handler else FakeResponse()


class TokenCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_save_and_load_roundtrip(self):
        cache = TokenCache(self.tmp / "tokens.json")
        cache.save(
            access_token="at-1",
            refresh_token="rt-1",
            expires_at=time.time() + 3600,
            client_id="cid",
        )
        loaded = TokenCache(self.tmp / "tokens.json").load()
        self.assertEqual(loaded["access_token"], "at-1")
        self.assertEqual(loaded["refresh_token"], "rt-1")
        self.assertEqual(loaded["client_id"], "cid")

    def test_missing_file_loads_none(self):
        self.assertIsNone(TokenCache(self.tmp / "nope.json").load())

    def test_clear_removes_file(self):
        path = self.tmp / "tokens.json"
        TokenCache(path).save(access_token="a", refresh_token="r", expires_at=1, client_id="c")
        TokenCache(path).clear()
        self.assertIsNone(TokenCache(path).load())


class OAuth2ClientTest(unittest.TestCase):
    def setUp(self):
        self.http = FakeHttp(
            handler=lambda method, url, kwargs: {
                ("GET", "https://mcp.example/.well-known/oauth-protected-resource"): FakeResponse(
                    body={"authorization_servers": ["https://auth.example"]}
                ),
                ("GET", "https://auth.example/.well-known/oauth-authorization-server"): FakeResponse(
                    body={
                        "authorization_endpoint": "https://auth.example/authorize",
                        "token_endpoint": "https://auth.example/token",
                        "registration_endpoint": "https://auth.example/register",
                    }
                ),
            }.get((method, url), FakeResponse(status_code=404))
        )
        self.oauth = OAuth2Client(
            http=self.http,
            mcp_url="https://mcp.example/mcp/default",
            client_name="Test Client",
            scope="SEARCH DOCUMENTS",
            redirect_uri="http://localhost:8000/glean/callback",
        )

    def test_discover_reads_protected_resource(self):
        metadata = _run(self.oauth.discover())
        self.assertEqual(metadata.authorization_server, "https://auth.example")
        self.assertEqual(metadata.registration_url, "https://auth.example/register")

    def test_register_posts_client_metadata(self):
        metadata = _run(self.oauth.discover())
        self.http.handler = lambda method, url, kwargs: (
            FakeResponse(
                body={
                    "client_id": "cid-1",
                    "client_secret": "secret-1",
                    "token_endpoint": "https://auth.example/token",
                    "authorization_endpoint": "https://auth.example/authorize",
                }
            )
            if url == "https://auth.example/register"
            else FakeResponse(status_code=404)
        )
        reg = _run(self.oauth.register(metadata))
        self.assertEqual(reg.client_id, "cid-1")
        self.assertEqual(reg.client_secret, "secret-1")
        posted = self.http.calls[-1]
        self.assertEqual(posted[0], "POST")
        self.assertEqual(posted[1], "https://auth.example/register")
        self.assertIn("redirect_uris", posted[2]["json"])
        self.assertIn("client_name", posted[2]["json"])

    def test_authorization_url_contains_pkce_and_state(self):
        metadata = _run(self.oauth.discover())
        reg = ClientRegistration(
            client_id="cid", token_endpoint="https://auth.example/token",
            authorization_endpoint="https://auth.example/authorize",
        )
        url = self.oauth.authorization_url(reg, "state-9", "verifier-abc")
        self.assertIn("response_type=code", url)
        self.assertIn("client_id=cid", url)
        self.assertIn("redirect_uri=", url)
        self.assertIn("state=state-9", url)
        self.assertIn("code_challenge=", url)
        self.assertIn("code_challenge_method=S256", url)

    def test_exchange_code_returns_token_set(self):
        metadata = _run(self.oauth.discover())
        reg = ClientRegistration(
            client_id="cid", client_secret="sec",
            token_endpoint="https://auth.example/token",
            authorization_endpoint="https://auth.example/authorize",
        )
        self.http.handler = lambda method, url, kwargs: FakeResponse(
            body={"access_token": "at", "refresh_token": "rt", "expires_in": 3600}
        )
        tokens = _run(self.oauth.exchange(reg, "code-1", "verifier"))
        self.assertEqual(tokens.access_token, "at")
        self.assertEqual(tokens.refresh_token, "rt")
        body = self.http.calls[-1][2]["data"]
        self.assertEqual(body["grant_type"], "authorization_code")
        self.assertEqual(body["code"], "code-1")

    def test_refresh_uses_refresh_token_grant(self):
        reg = ClientRegistration(
            client_id="cid", client_secret="sec",
            token_endpoint="https://auth.example/token",
            authorization_endpoint="https://auth.example/authorize",
        )
        self.http.handler = lambda method, url, kwargs: FakeResponse(
            body={"access_token": "at2", "refresh_token": "rt2", "expires_in": 3600}
        )
        tokens = _run(self.oauth.refresh(reg, "rt-old"))
        self.assertEqual(tokens.access_token, "at2")
        body = self.http.calls[-1][2]["data"]
        self.assertEqual(body["grant_type"], "refresh_token")
        self.assertEqual(body["refresh_token"], "rt-old")


class FakeMCPSession:
    def __init__(self, tools, call_results=None):
        self.tools = tools
        self.call_results = call_results or {}
        self.initialized = False
        self.calls = []

    async def __aenter__(self):
        self.initialized = True
        return self

    async def __aexit__(self, *exc):
        pass

    async def initialize(self):
        self.initialized = True

    async def list_tools(self):
        return {"tools": self.tools}

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        result = self.call_results.get(name, {"content": [{"type": "text", "text": f"result for {name}"}]})
        return result


class GleanClientTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        cache = TokenCache(self.tmp / "tokens.json")
        cache.save(access_token="at-1", refresh_token="rt-1", expires_at=time.time() + 3600, client_id="cid")
        self.oauth = mock.Mock(spec=OAuth2Client)
        self.cache = cache
        self.session = None

    def make_client(self, tools=None, call_results=None, session=None):
        session = session or FakeMCPSession(tools=tools or [], call_results=call_results or {})
        self.session = session

        async def runner(handler):
            async with session:
                return await handler(session)

        return glean_client_mod.GleanClient(
            http=mock.Mock(),
            mcp_url="https://mcp.example/mcp/default",
            oauth=self.oauth,
            cache=self.cache,
            session_runner=runner,
        )

    def test_status_without_token_reports_not_connected(self):
        TokenCache(self.tmp / "tokens.json").clear()
        client = self.make_client()
        status = _run(client.status())
        self.assertFalse(status["connected"])
        self.assertFalse(status["has_token"])

    def test_list_tools_connects_and_snapshots(self):
        session = FakeMCPSession(
            tools=[{"name": "search", "description": "Search Glean", "inputSchema": {"type": "object"}}]
        )
        client = self.make_client(session=session)
        tools = _run(client.list_tools())
        self.assertEqual(tools[0]["name"], "search")
        self.assertTrue(session.initialized)
        # snapshot served without a new connection
        self.assertEqual(client.tool_schemas()[0]["function"]["name"], "glean_search")

    def test_call_tool_returns_ok_formatted_result(self):
        client = self.make_client(
            tools=[{"name": "search", "description": "Search Glean", "inputSchema": {"type": "object"}}],
            call_results={"search": {"content": [{"type": "text", "text": "hit one"}]}},
        )
        _run(client.list_tools())
        result = _run(client.call_tool("glean_search", {"query": "x"}))
        self.assertIn("[ok]", result["output"])
        self.assertIn("hit one", result["output"])
        self.assertEqual(self.session.calls[0][0], "search")

    def test_call_tool_without_connection_returns_error(self):
        TokenCache(self.tmp / "tokens.json").clear()
        client = self.make_client()
        result = _run(client.call_tool("glean_search", {}))
        self.assertIn("[error]", result["output"])

    def test_call_tool_unknown_prefixed_name_returns_error(self):
        client = self.make_client()
        result = _run(client.call_tool("glean_nope", {}))
        self.assertIn("[error]", result["output"])