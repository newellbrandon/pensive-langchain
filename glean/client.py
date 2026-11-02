"""Glean MCP client wrapper: token lifecycle, MCP session, tool snapshot.

Agents see Glean tools under the ``glean_`` prefix (``glean_search``,
``glean_read_document``, ...) with OpenAI function schemas derived from the
discovered MCP tool definitions. Every call returns a plain ``{name, output}``
result with an ``[ok]``/``[error]`` prefix so failures are model-visible and
never raise.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from glean.oauth import ClientRegistration, OAuth2Error, OAuth2Client
from glean.token_cache import TokenCache
from glean.transport import Httpx2Transport

logger = logging.getLogger(__name__)

TOOL_PREFIX = "glean_"
_CALL_OUTPUT_LIMIT = 8000
_SNAPSHOT_TTL_SECONDS = 60.0


def _block_text(block: Any) -> str:
    if isinstance(block, dict):
        if block.get("type") in ("text", "output_text", "input_text"):
            return str(block.get("text") or "")
        return ""
    return str(getattr(block, "text", None) or "")


def _call_text(result: Any) -> str:
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")
    parts = [_block_text(b) for b in (content or [])]
    parts = [p for p in parts if p]
    return "\n".join(parts) or "<no text content>"


class GleanClient:
    """Lazy MCP client with a cached tool snapshot and session."""

    def __init__(
        self,
        *,
        http: Any = None,
        mcp_url: str,
        oauth: OAuth2Client,
        cache: TokenCache,
        session_runner: Any = None,
    ) -> None:
        self._http = http or Httpx2Transport(timeout=60)
        self.mcp_url = mcp_url
        self.oauth = oauth
        self.cache = cache
        self._session_runner = session_runner
        self._snapshot: list[dict] = []
        self._snapshot_at = 0.0
        self._tool_map: dict[str, str] = {}

    # --- token lifecycle ------------------------------------------------------

    def _token_valid(self, data: Optional[dict]) -> bool:
        return bool(data) and float(data.get("expires_at") or 0) > time.time() + 60

    def has_credentials(self) -> bool:
        data = self.cache.load()
        if self._token_valid(data):
            return True
        return bool(
            data
            and data.get("refresh_token")
            and data.get("token_endpoint")
            and data.get("client_id")
            and data.get("expires_at")
        )

    async def _ensure_token(self) -> None:
        data = self.cache.load()
        if self._token_valid(data):
            return
        if not (
            data
            and data.get("refresh_token")
            and data.get("token_endpoint")
            and data.get("client_id")
        ):
            raise OAuth2Error(
                "Glean is not connected: run the console Connect flow or "
                "scripts/glean_auth.py first"
            )
        registration = ClientRegistration(
            client_id=str(data["client_id"]),
            client_secret=str(data.get("client_secret") or "") or None,
            token_endpoint=str(data["token_endpoint"]),
            authorization_endpoint="",
        )
        tokens = await self.oauth.refresh(registration, str(data["refresh_token"]))
        self.cache.save(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token or str(data.get("refresh_token") or ""),
            expires_at=time.time() + tokens.expires_in,
            client_id=str(data["client_id"]),
            token_endpoint=str(data["token_endpoint"]),
            client_secret=str(data.get("client_secret") or ""),
        )
        logger.info("Glean access token refreshed")

    # --- MCP session ----------------------------------------------------------

    async def _run_with_session(
        self, handler: Any
    ) -> Any:
        """Open one MCP session and run ``handler(session)`` in a single task.

        The streamable-HTTP context and ClientSession must live in one task
        (anyio cancel scopes are task-bound); routing the lifecycle through an
        async-context-manager boundary breaks that.
        """
        await self._ensure_token()
        data = self.cache.load()
        if self._session_runner is not None:
            return await self._session_runner(handler)

        import httpx2
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        http_client = httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {data['access_token']}"},
            timeout=90,
        )
        try:
            async with streamable_http_client(
                self.mcp_url, http_client=http_client
            ) as streams:
                async with ClientSession(
                    streams[0], streams[1], read_timeout_seconds=120
                ) as session:
                    await session.initialize()
                    return await handler(session)
        finally:
            await http_client.aclose()

    async def close(self) -> None:
        self._snapshot = []
        self._tool_map = {}
        self._snapshot_at = 0.0

    # --- tools ----------------------------------------------------------------

    async def list_tools(self) -> list[dict]:
        """Discover tools; returns original-name dicts and refreshes the snapshot."""
        result = await self._run_with_session(lambda session: session.list_tools())
        if isinstance(result, dict):
            raw = list(result.get("tools") or [])
        else:
            raw = list(getattr(result, "tools", None) or [])
        snapshot: list[dict] = []
        tool_map: dict[str, str] = {}
        for item in raw:
            name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else "")
            description = getattr(item, "description", None) or (
                item.get("description") if isinstance(item, dict) else ""
            )
            schema = getattr(item, "inputSchema", None) or (
                item.get("inputSchema") if isinstance(item, dict) else {}
            )
            if not name:
                continue
            params = dict(schema) if isinstance(schema, dict) else {"type": "object"}
            params.setdefault("type", "object")
            params.setdefault("properties", {})
            prefixed = f"{TOOL_PREFIX}{name}"
            tool_map[prefixed] = name
            snapshot.append(
                {
                    "type": "function",
                    "function": {
                        "name": prefixed,
                        "description": f"Glean ({name}): {description}"[:500],
                        "parameters": params,
                    },
                }
            )
        self._snapshot = snapshot
        self._tool_map = tool_map
        self._snapshot_at = time.time()
        logger.info("Glean: %d tool(s) discovered", len(snapshot))
        return [
            {
                "name": name,
                "description": desc,
                "inputSchema": schema,
            }
            for item in raw
            for name, desc, schema in [
                (
                    getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else ""),
                    getattr(item, "description", None) or (item.get("description") if isinstance(item, dict) else ""),
                    getattr(item, "inputSchema", None) or (item.get("inputSchema") if isinstance(item, dict) else {}),
                )
            ]
            if name
        ]

    def tool_schemas(self) -> list[dict]:
        """Prefixed OpenAI function schemas from the latest snapshot."""
        return list(self._snapshot)

    def tool_names(self) -> list[str]:
        return [s["function"]["name"] for s in self._snapshot]

    async def refresh_tools_if_stale(self) -> None:
        """Rediscover tools when the snapshot is old (best-effort)."""
        if self._snapshot and time.time() - self._snapshot_at < _SNAPSHOT_TTL_SECONDS:
            return
        try:
            await self.list_tools()
        except (Exception, BaseExceptionGroup):  # noqa: BLE001
            logger.warning("Glean tool refresh failed", exc_info=True)

    async def call_tool(self, prefixed_name: str, arguments: dict) -> dict:
        """Call a Glean tool by its prefixed name; never raises."""
        original = self._tool_map.get(prefixed_name)
        if original is None:
            try:
                await self.list_tools()
                original = self._tool_map.get(prefixed_name)
            except (Exception, BaseExceptionGroup):  # noqa: BLE001
                logger.warning("Glean reconnect failed", exc_info=True)
        if original is None:
            available = ", ".join(sorted(self.tool_names())) or "none"
            return {
                "name": prefixed_name,
                "output": (
                    f"[error] unknown glean tool {prefixed_name}; discovered "
                    f"tools: {available}"
                ),
            }
        try:
            result = await self._run_with_session(
                lambda session: session.call_tool(original, arguments or {})
            )
        except (Exception, BaseExceptionGroup) as exc:  # noqa: BLE001
            return {
                "name": prefixed_name,
                "output": (
                    f"[error] glean tool {original} failed: {type(exc).__name__}: "
                    f"{str(exc)[:300]}"
                ),
            }
        text = _call_text(result)
        if len(text) > _CALL_OUTPUT_LIMIT:
            text = text[:_CALL_OUTPUT_LIMIT] + "\n…[truncated]"
        return {
            "name": prefixed_name,
            "output": f"[ok] glean {original} returned:\n{text}",
        }

    async def status(self) -> dict:
        data = self.cache.load()
        return {
            "connected": self._token_valid(data),
            "has_token": bool(data and data.get("access_token")),
            "expires_at": data.get("expires_at") if data else None,
            "tool_count": len(self._snapshot),
            "tools": self.tool_names(),
            "mcp_url": self.mcp_url,
        }


# --- singleton wiring ---------------------------------------------------------

_client: Optional[GleanClient] = None


def reset_glean_client() -> None:
    global _client
    _client = None


def get_glean_client() -> GleanClient:
    """Lazily built singleton bound to the deployment configuration."""
    global _client
    if _client is None:
        from config import Config

        _client = GleanClient(
            mcp_url=Config.GLEAN_MCP_URL,
            oauth=OAuth2Client(
                http=Httpx2Transport(timeout=60),
                mcp_url=Config.GLEAN_MCP_URL,
                client_name=Config.GLEAN_CLIENT_NAME,
                scope=Config.GLEAN_SCOPE,
                redirect_uri=Config.GLEAN_REDIRECT_URI,
            ),
            cache=TokenCache(Path(Config.GLEAN_TOKEN_CACHE)),
        )
    return _client