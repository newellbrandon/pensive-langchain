"""Minimal OAuth2 (RFC 6749) client for MCP servers with dynamic registration.

Implements the MCP OAuth flow: resource discovery via
``/.well-known/oauth-protected-resource``, dynamic client registration, the
authorization-code + PKCE dance, and token refresh. Everything is driven
through an injectable HTTP client so tests never touch the network.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlencode, urljoin

logger = logging.getLogger(__name__)


class OAuth2Error(Exception):
    """OAuth2 flow failure with a user-actionable message."""


@dataclass
class OAuthMetadata:
    authorization_server: str
    registration_url: Optional[str] = None
    token_endpoint: Optional[str] = None
    authorization_endpoint: Optional[str] = None


@dataclass
class ClientRegistration:
    client_id: str
    token_endpoint: str
    authorization_endpoint: str
    client_secret: Optional[str] = None


@dataclass
class TokenSet:
    access_token: str
    refresh_token: Optional[str]
    expires_in: int
    created_at: float = field(default_factory=time.time)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def new_code_verifier() -> str:
    return _b64url(secrets.token_bytes(48))


def code_challenge(verifier: str) -> str:
    return _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


class OAuth2Client:
    """MCP dynamic-client OAuth2 helper (authorization-code + PKCE)."""

    def __init__(
        self,
        *,
        http: Any,
        mcp_url: str,
        client_name: str,
        scope: str,
        redirect_uri: str,
    ) -> None:
        self._http = http
        self.mcp_url = mcp_url
        self.client_name = client_name
        self.scope = scope
        self.redirect_uri = redirect_uri

    def _origin(self) -> str:
        from urllib.parse import urlsplit

        parts = urlsplit(self.mcp_url)
        return f"{parts.scheme}://{parts.netloc}"

    async def _get_json(self, url: str) -> dict:
        response = await self._http.get(url)
        if response.status_code != 200:
            raise OAuth2Error(
                f"discovery failed for {url}: HTTP {response.status_code}"
            )
        body = response.json()
        if not isinstance(body, dict):
            raise OAuth2Error(f"expected JSON object from {url}")
        return body

    async def discover(self) -> OAuthMetadata:
        """Locate the authorization server and endpoints for the MCP server."""
        resource = await self._get_json(
            urljoin(self._origin() + "/", ".well-known/oauth-protected-resource")
        )
        servers = resource.get("authorization_servers") or []
        if not servers:
            raise OAuth2Error(
                "Glean MCP resource did not advertise an authorization server"
            )
        auth_server = str(servers[0])

        metadata: dict = {}
        try:
            metadata = await self._get_json(
                urljoin(auth_server + "/", ".well-known/oauth-authorization-server")
            )
        except OAuth2Error as exc:
            logger.warning("oauth authorization-server metadata unavailable: %s", exc)

        return OAuthMetadata(
            authorization_server=auth_server,
            registration_url=metadata.get("registration_endpoint")
            or urljoin(auth_server + "/", "register"),
            token_endpoint=metadata.get("token_endpoint")
            or urljoin(auth_server + "/", "token"),
            authorization_endpoint=metadata.get("authorization_endpoint")
            or urljoin(auth_server + "/", "authorize"),
        )

    async def register(self, metadata: OAuthMetadata) -> ClientRegistration:
        """Dynamically register this client with the authorization server."""
        payload = {
            "client_name": self.client_name,
            "redirect_uris": [self.redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "client_secret_post",
            "scope": self.scope,
        }
        try:
            response = await self._http.post(
                metadata.registration_url, json=payload
            )
        except Exception as exc:
            raise OAuth2Error(
                f"client registration request failed: {type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code not in (200, 201):
            raise OAuth2Error(
                f"client registration rejected (HTTP {response.status_code}); "
                "the tenant may not permit dynamic client registration — ask the "
                "Glean administrator to approve the OAuth client/DCR policy"
            )
        body = response.json()
        if not isinstance(body, dict) or not body.get("client_id"):
            raise OAuth2Error("client registration response missing client_id")
        return ClientRegistration(
            client_id=str(body["client_id"]),
            client_secret=str(body["client_secret"]) if body.get("client_secret") else None,
            token_endpoint=str(body.get("token_endpoint") or metadata.token_endpoint or ""),
            authorization_endpoint=str(
                body.get("authorization_endpoint") or metadata.authorization_endpoint or ""
            ),
        )

    def authorization_url(
        self, registration: ClientRegistration, state: str, verifier: str
    ) -> str:
        query = urlencode(
            {
                "response_type": "code",
                "client_id": registration.client_id,
                "redirect_uri": self.redirect_uri,
                "scope": self.scope,
                "state": state,
                "code_challenge": code_challenge(verifier),
                "code_challenge_method": "S256",
            }
        )
        return f"{registration.authorization_endpoint}?{query}"

    async def _token_request(self, endpoint: str, form: dict) -> TokenSet:
        if registration_client_secret := form.pop("client_secret", ""):
            form["client_secret"] = registration_client_secret
        try:
            response = await self._http.post(
                endpoint,
                data=form,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        except Exception as exc:
            raise OAuth2Error(
                f"token request failed: {type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code != 200:
            raise OAuth2Error(
                f"token endpoint rejected the request (HTTP {response.status_code})"
            )
        body = response.json()
        if not isinstance(body, dict) or not body.get("access_token"):
            raise OAuth2Error("token response missing access_token")
        return TokenSet(
            access_token=str(body["access_token"]),
            refresh_token=str(body["refresh_token"]) if body.get("refresh_token") else None,
            expires_in=int(body.get("expires_in") or 3600),
        )

    async def exchange(
        self, registration: ClientRegistration, code: str, verifier: str
    ) -> TokenSet:
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": registration.client_id,
            "code_verifier": verifier,
        }
        if registration.client_secret:
            form["client_secret"] = registration.client_secret
        return await self._token_request(registration.token_endpoint, form)

    async def refresh(
        self, registration: ClientRegistration, refresh_token: str
    ) -> TokenSet:
        form = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": registration.client_id,
        }
        if registration.client_secret:
            form["client_secret"] = registration.client_secret
        return await self._token_request(registration.token_endpoint, form)