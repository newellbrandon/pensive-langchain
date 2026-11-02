"""One-shot interactive Glean OAuth for local testing.

Runs the real MCP dynamic-client OAuth flow against the configured Glean
endpoint: discovery -> client registration -> prints an authorization URL ->
listens on http://localhost:8765/callback -> exchanges the code -> saves the
token cache to the configured path (default ./.glean-tokens.json).

Usage:
    python scripts/glean_auth.py [--cache PATH] [--url https://mongodb-be.glean.com/mcp/default]

Sign in with the printed URL (MongoDB Okta SSO). The script waits up to 10
minutes for the callback.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlsplit

from glean.oauth import OAuth2Client, code_challenge, new_code_verifier
from glean.token_cache import TokenCache
from glean.transport import Httpx2Transport

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("glean_auth")

REDIRECT_HOST = "127.0.0.1"
REDIRECT_PORT = 8765
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/callback"

_state: dict = {}


def _make_handler(
    ready: threading.Event, received: dict, arm_fallback: Callable[[], None]
) -> type[BaseHTTPRequestHandler]:
    RELAY_PAGE = b"""<!doctype html><meta charset="utf-8"><title>Glean auth</title>
<h2>Glean auth complete - you can close this tab.</h2>
<script>
  const params = new URLSearchParams(location.hash.slice(1));
  if (params.get("code") || params.get("error")) {
    fetch("/callback", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(Object.fromEntries(params))});
  } else if (!location.search) {
    document.body.textContent = "No code received in this redirect. Check the terminal.";
  }
</script>"""

    class CallbackHandler(BaseHTTPRequestHandler):
        def _respond(self, params: dict) -> None:
            logger.info(
                "callback params: %s",
                {k: ("***" if k == "code" else v) for k, v in params.items()},
            )
            received["code"] = params.get("code")
            received["state"] = params.get("state")
            received["error"] = params.get("error")
            ready.set()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(RELAY_PAGE)))
            self.end_headers()
            self.wfile.write(RELAY_PAGE)

        def do_GET(self):  # noqa: N802
            logger.info("callback GET path: %s", self.path)
            query = parse_qs(urlsplit(self.path).query)
            params = {key: values[0] for key, values in query.items() if values}
            if not params:
                # No query at all: the code is likely in the URL fragment, which
                # the browser cannot send. Serve the relay page (it POSTs the
                # fragment back) and arm a fallback in case that never happens.
                arm_fallback()
                logger.info("callback GET with empty query; serving fragment relay page")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(RELAY_PAGE)))
                self.end_headers()
                self.wfile.write(RELAY_PAGE)
                return
            self._respond(params)

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8", "replace")
            logger.info("callback POST body: %s", raw[:200])
            try:
                params = json.loads(raw) if raw else {}
            except ValueError:
                params = {}
            self._respond(params)

        def log_message(self, *args):  # silence
            pass

    return CallbackHandler


def _arm_fallback(
    holder: dict, ready: threading.Event, received: dict
) -> Callable[[], None]:
    armed = {"done": False}

    def arm() -> None:
        if armed["done"]:
            return
        armed["done"] = True

        def fallback() -> None:
            if ready.is_set():
                return
            received["error"] = "no code arrived after the fragment relay (timeout)"
            ready.set()
            if holder.get("server") is not None:
                holder["server"].shutdown()

        threading.Timer(60.0, fallback).start()

    return arm


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=".glean-tokens.json")
    parser.add_argument("--url", default="https://mongodb-be.glean.com/mcp/default")
    parser.add_argument(
        "--scope",
        default="SEARCH DOCUMENTS ENTITIES",
        help="OAuth scopes (Glean MCP requires the MCP scope)",
    )
    args = parser.parse_args()

    transport = Httpx2Transport(timeout=30)
    oauth = OAuth2Client(
        http=transport,
        mcp_url=args.url,
        client_name="Pensive Local Glean Agent",
        scope=args.scope,
        redirect_uri=REDIRECT_URI,
    )

    print("== step 1/4: discovery ==")
    metadata = await oauth.discover()
    print(f"   authorization server: {metadata.authorization_server}")
    print(f"   registration:         {metadata.registration_url}")

    print("== step 2/4: client registration ==")
    registration = await oauth.register(metadata)
    print(f"   client_id:            {registration.client_id}")
    print(f"   client_secret:        {'yes' if registration.client_secret else 'no'}")
    print(f"   token endpoint:       {registration.token_endpoint}")

    print("== step 3/4: authorization URL ==")
    verifier = new_code_verifier()
    state = f"pensive-{int(time.time())}"
    auth_url = oauth.authorization_url(registration, state, verifier)
    print(f"   OPEN THIS URL AND SIGN IN:\n\n   {auth_url}\n")

    ready = threading.Event()
    received: dict = {}
    holder: dict = {}
    server = ThreadingHTTPServer(
        (REDIRECT_HOST, REDIRECT_PORT),
        _make_handler(ready, received, _arm_fallback(holder, ready, received)),
    )
    holder["server"] = server
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"   waiting for callback on {REDIRECT_URI} (10 min timeout)...")

    event = threading.Event()

    async def wait_callback() -> dict:
        while not ready.is_set():
            await asyncio.sleep(0.2)
        server.shutdown()
        return received

    try:
        result = await asyncio.wait_for(wait_callback(), timeout=600)
    except asyncio.TimeoutError:
        print("!! timed out waiting for the browser callback")
        return 1

    if result.get("error"):
        print(f"!! authorization error: {result['error']}")
        return 1
    code = result.get("code")
    if not code:
        print("!! callback arrived without a code")
        return 1
    if result.get("state") != state:
        print("!! state mismatch in callback (possible CSRF) — aborting")
        return 1

    print("== step 4/4: exchanging code ==")
    tokens = await oauth.exchange(registration, code, verifier)
    cache = TokenCache(Path(args.cache))
    cache.save(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token or "",
        expires_at=time.time() + tokens.expires_in,
        client_id=registration.client_id,
        token_endpoint=registration.token_endpoint,
        client_secret=registration.client_secret or "",
    )
    print(f"   access token valid for {tokens.expires_in}s")
    print(f"   refresh token:        {'yes' if tokens.refresh_token else 'no'}")
    print(f"   cache written to:     {Path(args.cache).resolve()}")
    await transport.aclose()
    print("   AUTH COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))