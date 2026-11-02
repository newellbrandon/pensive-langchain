"""Glean admin routes: connection status, OAuth flow, and tool calls.

The OAuth authorization-code + PKCE dance is served from the console:
    POST /api/admin/glean/auth      -> {auth_url, state} (open in a browser)
    GET  /glean/callback?code&state -> exchanges, caches, redirects to /ui
The redirect URI (http://localhost:8000/glean/callback) reuses the published
API port, so no extra ports are needed for the loopback.
"""
from __future__ import annotations

import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from glean.client import get_glean_client, reset_glean_client
from glean.oauth import OAuth2Client, new_code_verifier

router = APIRouter(prefix="/api/admin/glean", tags=["admin"])
callback_router = APIRouter(tags=["glean"])

# state -> {"verifier", "registration", "expires_at"} for in-flight auths.
_pending_auth: dict[str, dict] = {}
_PENDING_TTL_SECONDS = 600


class GleanToolCallRequest(BaseModel):
    name: str
    arguments: dict = {}


def _expire_pending() -> None:
    now = time.time()
    for state in [s for s, p in _pending_auth.items() if p.get("expires_at", 0) < now]:
        _pending_auth.pop(state, None)


@router.get("/status")
async def glean_status() -> dict:
    client = get_glean_client()
    status = await client.status()
    _expire_pending()
    return {
        **status,
        "pending_auth": bool(_pending_auth),
        "scope": client.oauth.scope,
    }


@router.post("/auth")
async def glean_start_auth() -> dict:
    client = get_glean_client()
    status = await client.status()
    if status["connected"]:
        return {"already_connected": True, **status}
    _expire_pending()

    oauth: OAuth2Client = client.oauth
    try:
        metadata = await oauth.discover()
        registration = await oauth.register(metadata)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"Glean OAuth setup failed: {type(exc).__name__}: {exc}",
        ) from exc

    state = uuid.uuid4().hex[:24]
    verifier = new_code_verifier()
    _pending_auth[state] = {
        "verifier": verifier,
        "registration": {
            "client_id": registration.client_id,
            "client_secret": registration.client_secret or "",
            "token_endpoint": registration.token_endpoint,
            "authorization_endpoint": registration.authorization_endpoint,
        },
        "expires_at": time.time() + _PENDING_TTL_SECONDS,
    }
    auth_url = oauth.authorization_url(registration, state, verifier)
    return {"auth_url": auth_url, "state": state}


@callback_router.get("/glean/callback")
async def glean_callback(code: str, state: str) -> RedirectResponse:
    pending = _pending_auth.pop(state, None)
    if pending is None:
        return RedirectResponse(url="/ui?glean=error")
    from glean.oauth import ClientRegistration

    client = get_glean_client()
    registration = ClientRegistration(
        client_id=pending["registration"]["client_id"],
        client_secret=pending["registration"]["client_secret"] or None,
        token_endpoint=pending["registration"]["token_endpoint"],
        authorization_endpoint=pending["registration"]["authorization_endpoint"],
    )
    try:
        tokens = await client.oauth.exchange(registration, code, pending["verifier"])
    except Exception:  # noqa: BLE001
        return RedirectResponse(url="/ui?glean=error")
    client.cache.save(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token or "",
        expires_at=time.time() + tokens.expires_in,
        client_id=registration.client_id,
        token_endpoint=registration.token_endpoint,
        client_secret=registration.client_secret or "",
    )
    reset_glean_client()
    return RedirectResponse(url="/ui?glean=connected")


@router.post("/refresh")
async def glean_refresh() -> dict:
    client = get_glean_client()
    if not client.has_credentials():
        raise HTTPException(status_code=400, detail="Glean is not connected")
    try:
        await client._ensure_token()  # noqa: SLF001 — deliberate refresh trigger
        reset_glean_client()
        return {"refreshed": True, **(await get_glean_client().status())}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"refresh failed: {exc}") from exc


@router.post("/disconnect")
async def glean_disconnect() -> dict:
    client = get_glean_client()
    await client.close()
    client.cache.clear()
    reset_glean_client()
    return {"disconnected": True}


@router.post("/tools/call")
async def glean_tools_call(body: GleanToolCallRequest, request: Request) -> dict:
    client = get_glean_client()
    if not client.has_credentials():
        raise HTTPException(status_code=400, detail="Glean is not connected")
    await client.refresh_tools_if_stale()
    result = await client.call_tool(body.name, body.arguments or {})
    return {"name": result["name"], "output": result["output"]}