"""HTTP transport for the Glean OAuth/MCP clients (httpx2, as installed with mcp)."""
from __future__ import annotations

from typing import Any, Optional

try:
    import httpx2 as httpx_mod
except ImportError:  # pragma: no cover — older environments
    import httpx as httpx_mod


class Httpx2Transport:
    """Minimal async get/post adapter around httpx2.AsyncClient."""

    def __init__(self, headers: Optional[dict] = None, timeout: float = 30.0) -> None:
        self._client = httpx_mod.AsyncClient(
            headers=headers or {}, timeout=timeout, follow_redirects=False
        )

    async def get(self, url: str, **kwargs: Any):
        return await self._client.get(url, **kwargs)

    async def post(self, url: str, **kwargs: Any):
        return await self._client.post(url, **kwargs)

    async def aclose(self) -> None:
        await self._client.aclose()

    def raw_client(self) -> Any:
        """The underlying AsyncClient (used to inject Authorization into MCP)."""
        return self._client